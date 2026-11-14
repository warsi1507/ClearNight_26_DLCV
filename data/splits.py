"""Hold out a validation set from the training data, grouped by scene.

The same clean image (e.g. bdd-123) appears in several degraded versions and in
more than one branch (rain / snow / drop), so we split by scene id, not by file.
The official test set is never used for picking checkpoints.

    python -m data.splits --data_root data/AllWeatherNight
"""
import argparse
import json
import random
from pathlib import Path

from .dataset import WEATHERS, image_names
from .weather_labels import scene_id


def branch_paths(root, split, weather):
    """(input, gt, illumination prior, reflectance prior) folders, official naming."""
    base = Path(root) / split / weather
    return str(base), str(base) + '_gt', str(base) + '1', str(base) + '2'


def build_manifest(root, output, val_fraction=0.1, split_seed=2026):
    files = {w: image_names(Path(root) / 'train' / w) for w in WEATHERS}
    scenes = sorted({scene_id(n) for names in files.values() for n in names})
    random.Random(split_seed).shuffle(scenes)
    n_val = max(1, round(len(scenes) * val_fraction))
    val_scenes = set(scenes[:n_val])

    records = {}
    for weather, names in files.items():
        records[weather] = {
            'train': [n for n in names if scene_id(n) not in val_scenes],
            'val': [n for n in names if scene_id(n) in val_scenes],
        }
        if not records[weather]['train'] or not records[weather]['val']:
            raise ValueError(f'empty train or val list for {weather}, try another --split_seed')

    manifest = {'split_seed': split_seed, 'val_fraction': val_fraction,
                'val_scenes': sorted(val_scenes), 'records': records}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; delete it on purpose if you want a new split')
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/AllWeatherNight')
    parser.add_argument('--out', default=None)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--split_seed', type=int, default=2026)
    args = parser.parse_args()
    out = args.out or Path(args.data_root) / 'split_manifest.json'
    m = build_manifest(args.data_root, out, args.val_fraction, args.split_seed)
    print({w: {k: len(v) for k, v in r.items()} for w, r in m['records'].items()})
