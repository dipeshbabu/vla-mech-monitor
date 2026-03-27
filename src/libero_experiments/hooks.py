"""Neuron intervention hooks.

We intervene on the *input* to each MLP down-projection ("down_proj") by setting
selected hidden dimensions to a specific value.

For closed-loop interventions, `coef` can be a callable (returning a float) so
the strength can be updated online.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Union

import torch.nn.functional as F


CoefT = Union[float, Callable[[], float]]


def _coef_provider(coef: CoefT) -> Callable[[], float]:
    if callable(coef):
        return coef

    def _const() -> float:
        return float(coef)

    return _const


def apply_gate_proj_hooks(
    model,
    flat_index_dict: Dict[int, List[str]],
    intermediate_size: int = 11008,
    coef: CoefT = 1.0,
):
    values_per_layer = {}
    for flat_idx in flat_index_dict:
        layer_idx = flat_idx // intermediate_size
        neuron_idx = flat_idx % intermediate_size
        values_per_layer.setdefault(layer_idx, []).append(neuron_idx)

    print("\nNeurons selected for activation:\n")
    for layer, neurons in values_per_layer.items():
        print(f"  Layer {layer}: Neurons {neurons}")
    print()

    get_coef = _coef_provider(coef)

    def down_proj_hook(neuron_ids):
        def hook_fn(module, input, output):
            # Input: activations of shape (batch, seq_len, num_value_vectors) (where for openvla num_value_vectors = 11008)
            # Output: FFN output of shape (batch, seq_len, embedding_dim) (where for openvla embedding_dim = 4096)
            # print(f"⚡️ Hook triggered: setting {len(neuron_ids)} neurons to {coef_val}")
            coef_val = get_coef()
            modified_input = input[0]
            modified_input[..., neuron_ids] = coef_val
            output = F.linear(modified_input, module.weight, module.bias)
            return output
        return hook_fn

    hooks = []
    decoder_layers = model.language_model.model.layers
    for layer_idx, neuron_ids in values_per_layer.items():
        layer = decoder_layers[layer_idx]
        hook = layer.mlp.down_proj.register_forward_hook(
            down_proj_hook(neuron_ids)
        )
        hooks.append(hook)

    return hooks
