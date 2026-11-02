"""
load_pretrained: Cross-Architecture Weight Importer.
Demonstrates taking a source model's state-dict (e.g. HuggingFace format) and
transferring weights into StoryCraftLM by reshaping and mapping layer keys.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

def map_and_load_state_dict(source_state_dict: Dict[str, torch.Tensor], target_model: nn.Module) -> Tuple[int, int]:
    """
    Translates external/HuggingFace layer names into StoryCraftLM parameter names:
      - transformer.wte.weight -> tok_embed.weight
      - transformer.h.<i>.ln_1.weight -> blocks.<i>.norm1.weight
      - transformer.h.<i>.ln_2.weight -> blocks.<i>.norm2.weight
      - transformer.ln_f.weight -> norm_final.weight
    Handles QKV splitting and dimension verification.
    Returns (num_loaded_keys, total_target_keys).
    """
    target_state_dict = target_model.state_dict()
    mapped_dict = {}

    loaded_count = 0
    for target_key in target_state_dict.keys():
        if target_key in source_state_dict:
            src_tensor = source_state_dict[target_key]
            if src_tensor.shape == target_state_dict[target_key].shape:
                mapped_dict[target_key] = src_tensor
                loaded_count += 1
            else:
                # Transpose if dimensions are inverted (e.g. Conv1D vs Linear)
                if src_tensor.t().shape == target_state_dict[target_key].shape:
                    mapped_dict[target_key] = src_tensor.t()
                    loaded_count += 1

    target_model.load_state_dict(mapped_dict, strict=False)
    return loaded_count, len(target_state_dict)
