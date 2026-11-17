"""Unit tests. CPU only, no dataset or pretrained weights needed.

    python -m unittest discover -s tests -v
"""
import io
import os
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.augmentation import attenuate_residual
from data.dataset import FlareEvalDataset, WeatherPairDataset
from data.flare import ALPHA, add_flare
from data.prepare_dataset import extract_images
from data.retinex import decompose
from data.splits import branch_paths, build_manifest
from data.weather_labels import filename_weather_labels, scene_id
from networks.flare import FlareSuppression, HaarDWT
from networks.model import DWS, build_clearnight
from utils.metrics import compute_psnr, compute_ssim, region_psnr

WEATHER_TAGS = {'rain': ['hazerain', 'raindrop', 'haze', 'rain'],
                'snow': ['hazesnow', 'snow'],
                'drop': ['hazeraindrop', 'hazedrop', 'drop']}


def make_fixture(root, scenes=6, size=(40, 48)):
    """Tiny fake AllWeatherNight train set with the real naming scheme."""
    for weather, tags in WEATHER_TAGS.items():
        dirs = branch_paths(root, 'train', weather)
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        for i in range(scenes):
            gt = np.random.default_rng(i).integers(0, 180, (*size, 3), dtype=np.uint8)
            gt[4:9, 6:11] = 255  # a street lamp, so flare synthesis has a light source
            inp = np.clip(gt.astype(np.int16) + 30, 0, 255).astype(np.uint8)
            rmap, imap = decompose(inp, 10)
            name = f'bdd-{i}_{tags[i % len(tags)]}.png'
            for d, img in zip(dirs, (inp, gt, imap, rmap)):
                Image.fromarray(img).save(Path(d) / name)


def adam_steps(module, inputs, steps):
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    for _ in range(steps):
        out = module(*inputs)[0]
        opt.zero_grad()
        (out - 1).square().mean().backward()
        opt.step()


class SpecificityInitTests(unittest.TestCase):
    def test_official_init_never_trains_units(self):
        torch.manual_seed(0)
        block = DWS(24, 4, specificity_init='official')
        x = torch.rand(2, 24, 32, 32)
        adam_steps(block, (x, torch.rand_like(x)), steps=5)
        for unit in block.moe.units:
            self.assertEqual(unit.conv.weight.abs().sum().item(), 0)
        for layer in list(block.decoder)[:-1]:
            if hasattr(layer, 'weight'):
                self.assertEqual(layer.weight.abs().sum().item(), 0)
        self.assertEqual(block.decoder[-1].weight.abs().sum().item(), 0)
        self.assertNotEqual(block.decoder[-1].bias.abs().sum().item(), 0)  # only a constant is learned

    def test_fixed_init_starts_at_zero_and_trains(self):
        torch.manual_seed(0)
        block = DWS(24, 4, specificity_init='fixed')
        x = torch.rand(2, 24, 32, 32)
        self.assertEqual(block(x, x)[0].abs().max().item(), 0)
        before = [u.conv.weight.clone() for u in block.moe.units]
        adam_steps(block, (x, torch.rand_like(x)), steps=3)
        changed = sum(not torch.equal(b, u.conv.weight) for b, u in zip(before, block.moe.units))
        self.assertGreater(changed, 0)


class FlareTests(unittest.TestCase):
    def test_haar_dwt_is_orthonormal(self):
        dwt = HaarDWT()
        x = torch.rand(2, 5, 8, 12, dtype=torch.float64)
        bands = dwt.to(torch.float64)(x)
        self.assertEqual(bands.shape, (2, 5, 4, 4, 6))
        torch.testing.assert_close(dwt.inverse(bands), x)
        torch.testing.assert_close(bands.square().sum(), x.square().sum())  # energy preserved

    def test_flat_image_has_no_detail(self):
        bands = HaarDWT()(torch.full((1, 3, 8, 8), 0.7))
        self.assertLess(bands[:, :, 1:].abs().max().item(), 1e-6)
        torch.testing.assert_close(bands[:, :, 0], torch.full((1, 3, 4, 4), 1.4))

    def test_starts_as_identity_and_trains(self):
        torch.manual_seed(4)
        module = FlareSuppression(24)
        x, light = torch.rand(2, 24, 32, 32), torch.rand(2, 1, 32, 32)
        torch.testing.assert_close(module(x, light), x)
        gate_before = module.gate[0].weight.clone()
        opt = torch.optim.Adam(module.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad()
            (module(x, light) - 1).square().mean().backward()
            if module.glow[-1].weight.abs().sum() == 0:
                self.assertGreater(module.glow[-1].weight.grad.abs().sum().item(), 0)
            opt.step()
        self.assertFalse(torch.equal(gate_before, module.gate[0].weight))

    def test_gradcheck(self):
        torch.manual_seed(5)
        module = FlareSuppression(3, gate_hidden=4).double()
        torch.nn.init.normal_(module.glow[-1].weight, std=0.1)
        x = torch.rand(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
        light = torch.rand(1, 1, 4, 4, dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda t: module(t, light), (x,)))

    def test_full_model(self):
        for flare in (False, True):
            torch.manual_seed(3)
            net = build_clearnight(flare=flare).eval()
            x = torch.rand(1, 3, 35, 51)  # not a multiple of 16
            with torch.no_grad():
                self.assertEqual(net(x, x, x)[0].shape, x.shape)
            net.train()
            x = torch.rand(2, 3, 32, 32)
            adam_steps(net, (x, x, x), steps=1)
            net.zero_grad()
            (net(x, x, x)[0] - 1).square().mean().backward()
            for block in (net.moe1, net.moe2, net.moe3, net.moe4, net.moe5):
                grad = sum(u.conv.weight.grad.abs().sum().item() for u in block.moe.units)
                self.assertGreater(grad, 0)
            if flare:
                self.assertGreater(net.flare.glow[0].weight.grad.abs().sum().item(), 0)


class DataTests(unittest.TestCase):
    def test_labels_from_real_file_names(self):
        cases = {'bdd-1001_hazerain.png': ['haze', 'rain'],
                 'bdd-1001_raindrop.png': ['rain', 'drop'],
                 'dark-7_hazeraindrop.png': ['haze', 'rain', 'drop'],
                 'bdd-3_drop.png': ['drop'],
                 'dark-1050_hazesnow2.png': ['haze', 'snow'],
                 'bdd-1001_flare.png': []}
        for name, labels in cases.items():
            self.assertEqual(filename_weather_labels(name), labels, name)
        self.assertEqual(scene_id('dark-1050_hazesnow2.png'), 'dark-1050')

    def test_split_keeps_scenes_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fixture(tmp, scenes=10)
            m = build_manifest(tmp, Path(tmp) / 'split.json', val_fraction=0.2)
            for weather, rec in m['records'].items():
                train_scenes = {scene_id(n) for n in rec['train']}
                val_scenes = {scene_id(n) for n in rec['val']}
                self.assertFalse(train_scenes & val_scenes)
                self.assertEqual(val_scenes, set(m['val_scenes']))
            with self.assertRaises(FileExistsError):
                build_manifest(tmp, Path(tmp) / 'split.json')

    def test_zip_subfolders_are_skipped(self):
        png = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(png, format='PNG')
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'single_haze.zip'
            with zipfile.ZipFile(archive, 'w') as zf:
                zf.writestr('single_haze/bdd-1_haze.png', png.getvalue())
                zf.writestr('single_haze/no-exposure/bdd-1_(3).png', png.getvalue())
                zf.writestr('single_haze\\bdd-2_haze.png', png.getvalue())  # Windows-style path
            self.assertEqual(extract_images(archive, Path(tmp) / 'out'), ['bdd-1_haze.png', 'bdd-2_haze.png'])

    def test_attenuation_stays_between_input_and_target(self):
        clean = np.zeros((48, 64, 3))
        clean[:, 32:] = 0.8
        degraded = np.clip(clean + 0.15, 0, 1)
        for mode in ('light', 'uniform'):
            x = attenuate_residual(degraded, clean, mode, random.Random(7))
            self.assertTrue(np.all(x >= clean - 1e-6) and np.all(x <= degraded + 1e-6))
        light = attenuate_residual(degraded, clean, 'light', random.Random(7)) - clean
        self.assertGreater(light[:, 48:].mean(), light[:, :16].mean())  # more residual where it is bright


class FlareDataTests(unittest.TestCase):
    def test_flare_follows_eq1(self):
        clean = np.full((90, 160, 3), 0.1, np.float32)
        clean[40:44, 70:74] = 1.0
        x = add_flare(clean, random.Random(1))
        glow = x - ALPHA * clean
        self.assertGreaterEqual(glow.min(), -1e-6)                       # additive
        self.assertGreater(glow[30:55, 60:85].mean(), glow[:, :20].mean())  # strongest around the lamp
        self.assertIsNone(add_flare(np.full((90, 160, 3), 0.3, np.float32), random.Random(1)))  # no lights

    def test_flare_pairs_in_training_and_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fixture(tmp, scenes=6)
            dirs = {w: branch_paths(tmp, 'train', w) for w in WEATHER_TAGS}
            ds = WeatherPairDataset(dirs, crop_size=32, samples_per_branch=0, flare_probability=1.0)
            inp, gt, imap, rmap, name = ds[0][0]
            self.assertTrue(name.endswith('_flare.png'))
            self.assertEqual(filename_weather_labels(name), [])
            self.assertGreaterEqual((inp - gt * ALPHA).min().item(), -1 / 255)
            gts = [Path(dirs['rain'][1]) / n for n in sorted(os.listdir(dirs['rain'][1]))]
            val = FlareEvalDataset(gts)
            self.assertEqual(len(val), len({scene_id(p.name) for p in gts}))
            self.assertTrue(torch.equal(val[0][0], val[0][0]))  # fixed, not random


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        a = torch.zeros(1, 3, 32, 32)
        self.assertAlmostEqual(compute_psnr(a, a + 0.1), 20, places=5)
        self.assertAlmostEqual(compute_ssim(a + 0.1, a + 0.1), 1, places=8)
        light = torch.rand(1, 1, 32, 32)
        for value in region_psnr(a, a + 0.1, light):
            self.assertAlmostEqual(value, 20, places=5)


if __name__ == '__main__':
    unittest.main()
