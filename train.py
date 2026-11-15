"""Train one configuration and keep the checkpoint with the best validation PSNR.

    python train.py --config dwt
    python train.py --config dwt --epochs 40 --samples_per_branch 1500     # reduced budget
    python train.py --config dwt --resume checkpoints/dwt_seed20/latest.pth

Validation uses the held-out training scenes from data/splits.py, never the test set.
"""
import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from configs import CONFIGS
from data.dataset import WEATHERS, FlareEvalDataset, WeatherEvalDataset, WeatherPairDataset
from data.splits import branch_paths, load_manifest
from data.weather_labels import filename_weather_labels
from networks.model import ALL_WEATHER_LABELS
from utils.checkpoints import capture_rng, load_checkpoint, model_from_config, restore_rng, save_checkpoint
from utils.evaluation import evaluate_weather


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', choices=list(CONFIGS), default='baseline')
    p.add_argument('--data_root', default='data/AllWeatherNight')
    p.add_argument('--manifest', default=None, help='default: <data_root>/split_manifest.json')
    p.add_argument('--out_dir', default='checkpoints')
    p.add_argument('--log_dir', default='logs')
    p.add_argument('--weights_dir', default='losses/weights', help='vgg16 + ADDS depth weights')
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--cosine_period', type=int, default=None, help='default: epochs // 2, as in the paper (100/50)')
    p.add_argument('--samples_per_branch', type=int, default=None, help='iterations per epoch (official: 5000)')
    p.add_argument('--val_every', type=int, default=5)
    p.add_argument('--protocol', choices=['native', 'official'], default='native')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--print_every', type=int, default=200)
    p.add_argument('--no_aux_losses', action='store_true', help='drop VGG + depth losses (cheaper; use for all arms)')
    p.add_argument('--resume', default=None)
    p.add_argument('--max_hours', type=float, default=0, help='stop after the epoch that would pass this limit (resume later)')
    return p


def make_config(args):
    cfg = replace(CONFIGS[args.config], seed=args.seed)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs, cosine_period=max(1, args.epochs // 2))
    if args.cosine_period is not None:
        cfg = replace(cfg, cosine_period=args.cosine_period)
    if args.samples_per_branch is not None:
        cfg = replace(cfg, samples_per_branch=args.samples_per_branch)
    if args.no_aux_losses:
        cfg = replace(cfg, lam_vgg=0.0, lam_depth=0.0)
    return cfg


def weather_targets(names, device):
    t = torch.zeros(len(names), len(ALL_WEATHER_LABELS), device=device)
    for i, name in enumerate(names):
        for w in filename_weather_labels(name):
            t[i, ALL_WEATHER_LABELS.index(w)] = 1
    return t


def build_aux_losses(cfg, weights_dir, device):
    vgg_loss = depth_loss = None
    if cfg.lam_vgg:
        from torchvision import models
        from losses.perceptual import LossNetwork
        vgg = models.vgg16(weights=None)
        vgg.load_state_dict(torch.load(Path(weights_dir) / 'vgg16-397923af.pth', map_location='cpu'))
        vgg_loss = LossNetwork(vgg.features[:16]).to(device).eval().requires_grad_(False)
    if cfg.lam_depth:
        from losses.image_losses import DepthConsistencyLoss
        depth_loss = DepthConsistencyLoss(Path(weights_dir) / 'encoder.pth',
                                          Path(weights_dir) / 'depth.pth').to(device)
    return vgg_loss, depth_loss


def train(args):
    cfg = make_config(args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    manifest = load_manifest(args.manifest or Path(args.data_root) / 'split_manifest.json')
    records = manifest['records']
    for w in WEATHERS:
        unlabeled = [n for n in records[w]['train'] if not filename_weather_labels(n)]
        if unlabeled:
            raise ValueError(f'cannot read weather labels from file names such as {unlabeled[:3]}')

    dirs = {w: branch_paths(args.data_root, 'train', w) for w in WEATHERS}
    train_set = WeatherPairDataset(
        dirs, {w: records[w]['train'] for w in WEATHERS}, crop_size=cfg.crop_size,
        samples_per_branch=cfg.samples_per_branch, seed=cfg.seed, augmentation=cfg.augmentation,
        augmentation_probability=cfg.augmentation_probability,
        min_strength=cfg.min_strength, max_strength=cfg.max_strength, flare_probability=cfg.flare_data)
    shuffle_gen = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=shuffle_gen,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loaders = {w: DataLoader(WeatherEvalDataset(dirs[w], records[w]['val']), batch_size=1,
                                 num_workers=args.num_workers) for w in WEATHERS}
    flare_val = FlareEvalDataset([Path(dirs[w][1]) / n for w in WEATHERS for n in records[w]['val']], cfg.seed)
    if len(flare_val):  # synthetic flare-only val pairs, so checkpoint selection sees flare too
        val_loaders['flare'] = DataLoader(flare_val, batch_size=1, num_workers=args.num_workers)

    net = model_from_config(cfg.as_dict()).to(device)
    vgg_loss, depth_loss = build_aux_losses(cfg, args.weights_dir, device)
    optimizer = Adam(net.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.cosine_period, T_mult=1)

    run_dir = Path(args.out_dir) / f'{cfg.name}_seed{cfg.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        if ckpt['config'] != cfg.as_dict():
            raise ValueError('resume checkpoint was trained with a different config')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, best = ckpt['epoch'] + 1, ckpt['best']
        restore_rng(ckpt['rng'])
    elif (run_dir / 'history.jsonl').exists():
        raise FileExistsError(f'{run_dir} already has a run; use --resume or another --out_dir')
    (run_dir / 'config.json').write_text(json.dumps(cfg.as_dict(), indent=2))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(Path(args.log_dir) / run_dir.name))
    n_params = sum(p.numel() for p in net.parameters())
    print(f'{cfg.name} seed={cfg.seed} on {device}: {n_params / 1e6:.3f}M params, '
          f'{len(train_loader)} steps/epoch, {cfg.epochs} epochs')

    start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        train_set.set_epoch(epoch)
        shuffle_gen.manual_seed(cfg.seed * 1000 + epoch)
        scheduler.step(epoch)
        sums, t0 = {}, time.time()
        for it, branches in enumerate(train_loader):
            inp, gt, imap, rmap = [torch.cat([b[i] for b in branches]).to(device) for i in range(4)]
            names = [n for b in branches for n in b[4]]

            out, class_logits, l2_reg, lb_loss = net(inp, imap, rmap)
            losses = {'recon': F.smooth_l1_loss(out, gt),  # official code uses smooth L1
                      'bce': cfg.lam_bce * F.binary_cross_entropy_with_logits(class_logits, weather_targets(names, device)),
                      'balance': cfg.lam_balance * lb_loss,
                      'router_l2': l2_reg}
            if vgg_loss is not None:
                losses['vgg'] = cfg.lam_vgg * vgg_loss(out, gt)
            if depth_loss is not None:
                losses['depth'] = cfg.lam_depth * depth_loss(out, gt)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'loss is {loss.item()} at epoch {epoch}, step {it}')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            if (it + 1) % args.print_every == 0:
                print(f'  epoch {epoch} step {it + 1}/{len(train_loader)} loss {loss.item():.4f} '
                      f'({time.time() - t0:.0f}s)')

        record = {'epoch': epoch, 'seconds': round(time.time() - t0, 1),
                  'lr': optimizer.param_groups[0]['lr'],
                  'losses': {k: v / len(train_loader) for k, v in sums.items()}}
        for k, v in record['losses'].items():
            writer.add_scalar(f'train/{k}', v, epoch)

        is_best = False
        if (epoch + 1) % args.val_every == 0 or epoch == cfg.epochs - 1:
            val = {w: evaluate_weather(net, loader, device, args.protocol) for w, loader in val_loaders.items()}
            record['val'] = {w: {'psnr': v['psnr'], 'ssim': v['ssim'], 'n': v['n']} for w, v in val.items()}
            record['val_psnr'] = sum(v['psnr'] for v in val.values()) / len(val)
            for w, v in val.items():
                writer.add_scalar(f'val/{w}_psnr', v['psnr'], epoch)
            writer.add_scalar('val/mean_psnr', record['val_psnr'], epoch)
            is_best = best is None or record['val_psnr'] > best['val_psnr']
            if is_best:
                best = {'epoch': epoch, 'val_psnr': record['val_psnr'], 'val': record['val']}
            print(f"epoch {epoch}: val PSNR {'/'.join(val)} = "
                  + ' / '.join(f"{record['val'][w]['psnr']:.3f}" for w in val)
                  + f", mean {record['val_psnr']:.3f}" + (' (best)' if is_best else ''))

        ckpt = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'config': cfg.as_dict(),
                'epoch': epoch, 'best': best, 'rng': capture_rng()}
        save_checkpoint(ckpt, run_dir / 'latest.pth')
        if is_best:
            save_checkpoint({'model': net.state_dict(), 'config': cfg.as_dict(), 'epoch': epoch,
                             'val': best['val']}, run_dir / 'best.pth')
        with open(run_dir / 'history.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f"epoch {epoch} done in {record['seconds']}s, losses "
              + ', '.join(f'{k}={v:.4f}' for k, v in record['losses'].items()))

        # stop before the session limit if the next epoch would not fit; resume from latest.pth
        epoch_time = time.time() - t0
        if args.max_hours and epoch < cfg.epochs - 1 and time.time() - start + epoch_time > args.max_hours * 3600:
            print(f'time limit: stopped after epoch {epoch}, resume with --resume {run_dir / "latest.pth"}')
            writer.close()
            return None

    writer.close()
    summary = {**best, 'config': cfg.as_dict(), 'checkpoint': str(run_dir / 'best.pth')}
    (run_dir / 'best.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    train(get_parser().parse_args())
