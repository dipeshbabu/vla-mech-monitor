"""Loading intervention dictionaries.

In the original mechanistic-steering-vlas codebase, intervention dictionaries are
stored in YAML as:

dict_name:
  neuron_id: [tokens...]
  neuron_id: [tokens...]

Where dict_name encodes the layer index as the last underscore-separated number
(e.g., 'up_10_full', 'careful_5', etc.).

For the gate_proj hook code, we need the format:

{layer_idx: {neuron_id: weight}}

We ignore the token lists at runtime and treat each listed neuron as weight=1.0.
"""

from __future__ import annotations

import re
from typing import Dict

import yaml


_LAYER_RE = re.compile(r".*_(\d+)(?:_full)?$")


def load_intervention_dict(dict_name: str, config_path: str) -> Dict[int, Dict[int, float]]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if dict_name == "blank" or dict_name is None:
        return {}

    if dict_name not in config:
        raise KeyError(f"Dictionary '{dict_name}' not found in {config_path}. Available: {list(config.keys())[:10]}...")

    m = _LAYER_RE.match(dict_name)
    if m is None:
        raise ValueError(
            f"Could not parse layer index from dict_name='{dict_name}'. Expected suffix like '_10' or '_10_full'."
        )
    layer_idx = int(m.group(1))

    neuron_map = config[dict_name]
    if not isinstance(neuron_map, dict):
        raise ValueError(f"Bad dict format for {dict_name}: expected mapping neuron_id -> token_list")

    out: Dict[int, Dict[int, float]] = {layer_idx: {}}
    for k in neuron_map.keys():
        try:
            nid = int(k)
        except Exception:
            # YAML may parse numeric keys as int already
            nid = int(str(k))
        out[layer_idx][nid] = 1.0
    return out
