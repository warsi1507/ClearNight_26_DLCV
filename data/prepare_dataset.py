"""Download AllWeatherNight from Hugging Face and unpack it into the folder layout we use.

    python -m data.prepare_dataset --precompute_retinex

Hugging Face layout (checked Oct 2026): train/{rain,snow,drop}[_gt].zip and
test/{rain_scene,snow_scene,single_haze,single_rain,single_drop,single_snow,single_flare}[_gt].zip.
The real-world test zip (AllWeatherNight_real.zip, ~740 MB, no ground truth) is skipped.

Note: test/single_haze.zip also contains a 'no-exposure/' subfolder with 100
extra images that have no ground truth. We only take files from the top-level
folder of every zip, so those are ignored.
"""
import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

TRAIN_SUBSETS = ('rain', 'snow', 'drop')
TEST_SUBSETS = ('rain_scene', 'snow_scene', 'single_haze', 'single_rain',
                'single_drop', 'single_snow', 'single_flare')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def subsets_for(split):
    return TRAIN_SUBSETS if split == 'train' else TEST_SUBSETS


def extract_images(archive, destination):
    """Extract images that sit directly in the zip's top folder, flattened into destination."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names, skipped = [], 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            path = PurePosixPath(info.filename.replace('\\', '/'))  # zips were made on Windows
            if info.is_dir() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if len(path.parts) > 2 or '__MACOSX' in path.parts or path.name.startswith('._'):
                skipped += 1
                continue
            with zf.open(info) as src, open(destination / path.name, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            names.append(path.name)
    if skipped:
        print(f'  {Path(archive).name}: skipped {skipped} files in subfolders')
    return names


def organize(raw_dir, out_dir, splits):
    for split in splits:
        for subset in subsets_for(split):
            inp = extract_images(Path(raw_dir) / split / f'{subset}.zip', Path(out_dir) / split / subset)
            gt = extract_images(Path(raw_dir) / split / f'{subset}_gt.zip', Path(out_dir) / split / f'{subset}_gt')
            if set(inp) != set(gt):
                raise ValueError(f'{split}/{subset}: input and gt file names do not match')
            print(f'{split}/{subset}: {len(inp)} pairs')


def precompute(root, splits):
    from data.retinex import process_folder
    for split in splits:
        for subset in subsets_for(split):
            base = Path(root) / split / subset
            print(f'retinex priors for {split}/{subset}')
            process_folder(str(base), str(base) + '2', str(base) + '1', sigma=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo_id', default='YuetongLiu/AllWeatherNight')
    parser.add_argument('--raw_dir', default='data/_raw_AllWeatherNight')
    parser.add_argument('--out_dir', default='data/AllWeatherNight')
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    parser.add_argument('--skip_download', action='store_true', help='zips are already in --raw_dir')
    parser.add_argument('--precompute_retinex', action='store_true')
    args = parser.parse_args()

    if not args.skip_download:
        from huggingface_hub import snapshot_download
        wanted = [f'{split}/{subset}{suffix}.zip' for split in args.splits
                  for subset in subsets_for(split) for suffix in ('', '_gt')]
        snapshot_download(repo_id=args.repo_id, repo_type='dataset',
                          local_dir=args.raw_dir, allow_patterns=wanted)
    organize(args.raw_dir, args.out_dir, args.splits)
    if args.precompute_retinex:
        precompute(args.out_dir, args.splits)


if __name__ == '__main__':
    main()
