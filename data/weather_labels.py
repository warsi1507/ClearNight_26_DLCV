"""Weather labels and scene ids from AllWeatherNight file names.

Names look like  bdd-123_hazeraindrop.png  or  dark-1001_hazesnow2.png :
<scene id>_<weather words glued together>. In this dataset 'raindrop' means
rain streak + raindrop (the Rain Scene "RS+RD" combination), and a pure raindrop
image is called '..._drop', so plain substring matching is correct here
(it is also what the official training script does).
"""
from pathlib import Path

WEATHER_WORDS = ('haze', 'snow', 'rain', 'drop')


def scene_id(name):
    return Path(name).stem.split('_', 1)[0]


def filename_weather_labels(name):
    stem = Path(name).stem.lower()
    tag = stem.split('_', 1)[1] if '_' in stem else stem
    return [w for w in WEATHER_WORDS if w in tag]
