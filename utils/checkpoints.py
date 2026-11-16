"""Checkpoint helpers. A checkpoint stores the model config, so evaluation rebuilds the right network."""
import random
from pathlib import Path

import numpy as np
import torch

from networks.model import build_clearnight

MODEL_KEYS = ('flare', 'specificity_init', 'moe_num_units', 'moe_top_k',
              'router_noise_std', 'router_l2_lambda')


def model_from_config(config):
    return build_clearnight(**{k: config[k] for k in MODEL_KEYS if k in config})


def save_checkpoint(checkpoint, path):
    # write to a temp file first so a crash never leaves a half-written latest.pth
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def capture_rng():
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
