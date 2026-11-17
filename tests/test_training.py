"""End-to-end check on a tiny fake dataset: train, interrupt, resume, evaluate, compare."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import compare
import evaluate
import train as training
from data.prepare_dataset import TEST_SUBSETS
from data.splits import branch_paths, build_manifest
from test_correctness import make_fixture


class Interrupted(Exception):
    pass


class PipelineTest(unittest.TestCase):
    def test_train_resume_evaluate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            make_fixture(data, scenes=8)
            build_manifest(data, data / 'split_manifest.json', val_fraction=0.25)
            argv = ['--config', 'dwt', '--data_root', str(data), '--epochs', '2',
                    '--samples_per_branch', '2', '--val_every', '1', '--num_workers', '0',
                    '--no_aux_losses', '--log_dir', str(root / 'logs')]
            parser = training.get_parser()
            with patch.dict(training.CONFIGS, {'dwt': training.replace(training.CONFIGS['dwt'], crop_size=32)}):
                training.train(parser.parse_args(argv + ['--out_dir', str(root / 'a')]))
                full = torch.load(root / 'a/dwt_seed20/latest.pth', weights_only=False)

                real_save = training.save_checkpoint
                def save_then_stop(ckpt, path):
                    real_save(ckpt, path)
                    if Path(path).name == 'latest.pth' and ckpt['epoch'] == 0:
                        raise Interrupted
                with patch('train.save_checkpoint', save_then_stop), self.assertRaises(Interrupted):
                    training.train(parser.parse_args(argv + ['--out_dir', str(root / 'b')]))
                latest = str(root / 'b/dwt_seed20/latest.pth')
                training.train(parser.parse_args(argv + ['--out_dir', str(root / 'b'), '--resume', latest]))
                resumed = torch.load(latest, weights_only=False)

            # a time limit stops after one epoch without best.json; resuming finishes the run
            with patch.dict(training.CONFIGS, {'dwt': training.replace(training.CONFIGS['dwt'], crop_size=32)}):
                c_args = argv + ['--out_dir', str(root / 'c')]
                self.assertIsNone(training.train(parser.parse_args(c_args + ['--max_hours', '1e-9'])))
                self.assertFalse((root / 'c/dwt_seed20/best.json').exists())
                training.train(parser.parse_args(c_args + ['--resume', str(root / 'c/dwt_seed20/latest.pth')]))
                self.assertTrue((root / 'c/dwt_seed20/best.json').exists())

            for key, value in full['model'].items():
                self.assertTrue(torch.equal(value, resumed['model'][key]), key)
            self.assertEqual(full['best'], resumed['best'])

            for subset in TEST_SUBSETS:  # reuse two training images as a fake test set
                for src, dst in zip(branch_paths(data, 'train', 'rain'), branch_paths(data, 'test', subset)):
                    Path(dst).mkdir(parents=True)
                    for f in sorted(Path(src).glob('*.png'))[:2]:
                        shutil.copy2(f, dst)
            outputs = []
            for run in ('a', 'b'):
                out = str(root / f'results/{run}.json')
                ckpt = str(root / run / 'dwt_seed20' / 'best.pth')
                with patch('sys.argv', ['evaluate.py', '--checkpoint', ckpt, '--data_root', str(data),
                                        '--num_workers', '0', '--out_json', out]):
                    result = evaluate.main()
                self.assertEqual(set(result['scores']), set(TEST_SUBSETS))
                outputs.append(out)
            with patch('sys.argv', ['compare.py', *outputs, '--out', str(root / 'results/cmp.md')]):
                compare.main()
            table = (root / 'results/cmp.md').read_text()
            self.assertIn('rain_scene', table)
            self.assertIn('input (no processing)', table)


if __name__ == '__main__':
    unittest.main()
