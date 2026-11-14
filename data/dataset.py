"""Paired datasets. Folder layout follows the official code:
<branch>/ (degraded), <branch>_gt/, <branch>1/ (illumination prior), <branch>2/ (reflectance prior).
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import attenuate_residual
from .flare import add_flare
from .retinex import decompose
from .weather_labels import filename_weather_labels, scene_id

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
WEATHERS = ('rain', 'snow', 'drop')


def image_names(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'missing folder: {folder}')
    names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not names:
        raise ValueError(f'no images in {folder}')
    return names


def check_pairs(dirs, files=None):
    """Make sure every file exists in the input, gt and both prior folders."""
    files = list(files) if files is not None else image_names(dirs[0])
    for folder in dirs:
        missing = set(files) - set(image_names(folder))
        if missing:
            raise ValueError(f'{len(missing)} files missing in {folder}, e.g. {sorted(missing)[:3]}')
    return files


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class WeatherPairDataset(Dataset):
    """Training set. Each item is one random crop from each of the rain/snow/drop branches,
    like the official my_dataset. Crops (and augmentation) are seeded by (seed, epoch, index)
    so runs with different configs see exactly the same patches."""

    def __init__(self, dirs_by_weather, files_by_weather=None, crop_size=256, samples_per_branch=5000,
                 seed=20, augmentation='none', augmentation_probability=0.5,
                 min_strength=0.4, max_strength=0.9, flare_probability=0.0):
        if crop_size % 16:
            raise ValueError('crop_size must be a multiple of 16')
        self.dirs = dirs_by_weather
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0
        self.augmentation = augmentation
        self.aug_p = augmentation_probability
        self.strength = (min_strength, max_strength)
        self.flare_p = flare_probability

        self.files = {}
        for w in WEATHERS:
            names = check_pairs(self.dirs[w], files_by_weather[w] if files_by_weather else None)
            if 0 < samples_per_branch < len(names):
                # fixed subset per run, as with the official fix_sample
                names = random.Random(f'{seed}:subset:{w}').sample(names, samples_per_branch)
            self.files[w] = names
        self.length = max(len(v) for v in self.files.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def _load(self, weather, index):
        names = self.files[weather]
        name = names[index % len(names)]
        img, gt, imap, rmap = [read_rgb(Path(d) / name) for d in self.dirs[weather]]

        rng = random.Random(f'{self.seed}:flare:{self.epoch}:{weather}:{index}')
        x = add_flare(gt / 255.0, rng) if self.flare_p and rng.random() < self.flare_p else None
        if x is not None:
            # flare-only pair: target is the clean image, no weather label
            img = np.rint(x * 255).astype(np.uint8)
            rmap, imap = decompose(img, sigma=10)
            name = f'{scene_id(name)}_flare.png'
        elif self.augmentation != 'none':
            rng = random.Random(f'{self.seed}:aug:{self.epoch}:{weather}:{index}')
            # raindrops are refraction, not an additive layer, so leave them alone
            if 'drop' not in filename_weather_labels(name) and rng.random() < self.aug_p:
                x = attenuate_residual(img / 255.0, gt / 255.0, self.augmentation, rng, *self.strength)
                img = np.rint(x * 255).astype(np.uint8)
                rmap, imap = decompose(img, sigma=10)  # priors must come from the new input

        h, w = img.shape[:2]
        rng = random.Random(f'{self.seed}:crop:{self.epoch}:{weather}:{index}')
        y0 = rng.randint(0, h - self.crop_size)
        x0 = rng.randint(0, w - self.crop_size)
        crop = (slice(y0, y0 + self.crop_size), slice(x0, x0 + self.crop_size))
        return tuple(to_tensor(a[crop]) for a in (img, gt, imap, rmap)) + (name,)

    def __getitem__(self, index):
        return tuple(self._load(w, index) for w in WEATHERS)


class WeatherEvalDataset(Dataset):
    """Full images at native resolution, sorted by name."""

    def __init__(self, dirs, files=None):
        self.dirs = dirs
        self.files = sorted(check_pairs(dirs, files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = self.files[index]
        return tuple(to_tensor(read_rgb(Path(d) / name)) for d in self.dirs) + (name,)


class FlareEvalDataset(Dataset):
    """Fixed flare-only validation pairs, one per scene, made from clean validation images."""

    def __init__(self, gt_paths, seed=20):
        self.items, seen = [], set()
        for path in sorted(gt_paths, key=lambda p: Path(p).name):
            scene = scene_id(Path(path).name)
            if scene in seen:
                continue
            seen.add(scene)
            gt = read_rgb(path)
            if add_flare(gt / 255.0, random.Random(f'{seed}:flareval:{scene}')) is not None:
                self.items.append((path, scene))
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, scene = self.items[index]
        gt = read_rgb(path)
        x = add_flare(gt / 255.0, random.Random(f'{self.seed}:flareval:{scene}'))
        img = np.rint(x * 255).astype(np.uint8)
        rmap, imap = decompose(img, sigma=10)
        return tuple(to_tensor(a) for a in (img, gt, imap, rmap)) + (f'{scene}_flare.png',)
