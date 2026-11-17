"""Verification script to test 1 training step/epoch and evaluation on sample data."""
import sys
from pathlib import Path

# Add project root and tests dir to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / 'tests'))

import tempfile
import shutil
import torch

from test_correctness import make_fixture
from data.splits import build_manifest, branch_paths
from data.prepare_dataset import TEST_SUBSETS
import train as training
import evaluate
import compare

def main():
    print("=== Step 1: Generating Sample Multi-Weather Nighttime Fixture ===")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        make_fixture(data_dir, scenes=6, size=(64, 64))
        manifest_path = data_dir / 'split_manifest.json'
        build_manifest(data_dir, manifest_path, val_fraction=0.33)
        print("Sample data generated with valid scene splits and Retinex priors.")

        print("\n=== Step 2: Testing 1 Epoch of Training (DWT Flare Model) ===")
        out_dir = root / 'checkpoints'
        args = [
            '--config', 'dwt_fd',
            '--data_root', str(data_dir),
            '--manifest', str(manifest_path),
            '--epochs', '1',
            '--samples_per_branch', '2',
            '--val_every', '1',
            '--num_workers', '0',
            '--no_aux_losses',
            '--out_dir', str(out_dir),
            '--log_dir', str(root / 'logs')
        ]
        parser = training.get_parser()
        with torch.no_grad():
            pass
        
        # Override crop_size for fast testing on tiny fixture
        with torch.random.fork_rng():
            old_crop = training.CONFIGS['dwt_fd'].crop_size
            training.CONFIGS['dwt_fd'] = training.replace(training.CONFIGS['dwt_fd'], crop_size=32)
            try:
                training.train(parser.parse_args(args))
            finally:
                training.CONFIGS['dwt_fd'] = training.replace(training.CONFIGS['dwt_fd'], crop_size=old_crop)

        ckpt_path = out_dir / 'dwt_fd_seed20' / 'best.pth'
        assert ckpt_path.exists(), "Checkpoint best.pth was not created!"
        print(f"Training successfully completed 1 epoch! Saved checkpoint: {ckpt_path}")

        print("\n=== Step 3: Testing Inference & Evaluation Harness ===")
        # Populate fake test subsets from train data
        for subset in TEST_SUBSETS:
            for src, dst in zip(branch_paths(data_dir, 'train', 'rain'), branch_paths(data_dir, 'test', subset)):
                Path(dst).mkdir(parents=True, exist_ok=True)
                for f in sorted(Path(src).glob('*.png'))[:2]:
                    shutil.copy2(f, dst)

        eval_out = root / 'results' / 'eval_sample.json'
        sys.argv = [
            'evaluate.py',
            '--checkpoint', str(ckpt_path),
            '--data_root', str(data_dir),
            '--subsets', 'single_flare', 'rain_scene',
            '--num_workers', '0',
            '--out_json', str(eval_out)
        ]
        res = evaluate.main()
        print(f"Evaluation succeeded on subsets {list(res['scores'].keys())}!")
        for s, sc in res['scores'].items():
            print(f"  [{s}] PSNR: {sc['psnr']:.2f} dB, SSIM: {sc['ssim']:.4f}")

    print("\n=== All Training and Evaluation Verification Checks Passed! ===")

if __name__ == '__main__':
    main()
