"""Experiment configurations. Everything not listed in a config uses the paper's values."""
from dataclasses import asdict, dataclass, replace


@dataclass
class ExperimentConfig:
    name: str
    # model
    specificity_init: str = 'fixed'   # fixed | official
    flare: bool = False               # Haar DWT flare suppression
    moe_num_units: int = 25
    moe_top_k: int = 10
    router_noise_std: float = 0.01
    router_l2_lambda: float = 0.01    # official code: 0.01 * sum of router parameter norms
    # training (paper / official script)
    epochs: int = 100
    cosine_period: int = 50
    batch_size: int = 1               # per branch, so 3 crops per step
    crop_size: int = 256
    learning_rate: float = 2e-4
    samples_per_branch: int = 5000    # official fix_sample
    lam_vgg: float = 0.1
    lam_bce: float = 0.001
    lam_balance: float = 0.01
    lam_depth: float = 0.02
    grad_clip_norm: float = 0.0
    flare_data: float = 0.0           # probability of replacing a crop by a synthetic flare-only pair
    # optional residual-attenuation augmentation (not part of the main study)
    augmentation: str = 'none'        # none | uniform | light
    augmentation_probability: float = 0.5
    min_strength: float = 0.4
    max_strength: float = 0.9
    seed: int = 20

    def as_dict(self):
        return asdict(self)


# Main study
OFFICIAL = ExperimentConfig(name='official', specificity_init='official')  # released code behaviour
BASELINE = ExperimentConfig(name='baseline')                               # ClearNight with working WDS
DWT = replace(BASELINE, name='dwt', flare=True)                            # proposed
BASELINE_FD = replace(BASELINE, name='baseline_fd', flare_data=0.2)       # + synthetic flare pairs
DWT_FD = replace(DWT, name='dwt_fd', flare_data=0.2)

# Optional extra experiment (earlier idea, see EXPLAINER.md)
UNIFORM = replace(BASELINE, name='uniform', augmentation='uniform')
LCRA = replace(BASELINE, name='lcra', augmentation='light')

CONFIGS = {c.name: c for c in (OFFICIAL, BASELINE, DWT, BASELINE_FD, DWT_FD, UNIFORM, LCRA)}
