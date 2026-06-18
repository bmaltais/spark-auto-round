# Copyright (c) 2026 Dr Henry Thomas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Router Jaccard Similarity for MOE models.

Computes the Jaccard similarity between FP16 and quantized router outputs
to detect routing instability caused by quantization.

The router (gating layer) decides which experts process each token.
Quantization noise can flip these decisions, causing cascading quality
degradation even if individual expert layers have high cosine similarity.

Reference: https://arxiv.org/abs/2606.05688
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pattern to identify router weights: *.gate.weight
# This matches the standard HuggingFace naming convention for MOE routers.
ROUTER_WEIGHT_SUFFIX = ".gate.weight"

# Multi-modal keys to exclude (from auto_round/utils/common.py)
# These are non-text components that should not be included in Jaccard analysis.
_MM_MODULE_KEYS = [
    "multi_modal_projector",
    "mm_projector",
    "vision_tower",
    "multimodal_projector",
    "thinker",
    "talker",
    "token2wav",
    "code2wav",
    "code_predictor",
    "vqmodel",
    "vision_model",
    "audio_tower",
    "audio_model",
    "vision_encoder",
    "vision_language_adapter",
    "patch_merger",
    "pre_mm_projector_norm",
    "image_newline",
    "model.connector",
    "audio",
    "visual",
    "speech",
    "wav",
    "waveform",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_multimodal_layer(name: str) -> bool:
    """Check if a layer name contains any multi-modal component key."""
    return any(key in name for key in _MM_MODULE_KEYS)


def _find_router_layers(weights: dict[str, torch.Tensor]) -> list[str]:
    """Find all router weight names in the model.
    
    Routers are identified by the pattern `*.gate.weight`.
    Multi-modal layers are excluded.
    
    Args:
        weights: Model weights dict.
        
    Returns:
        Sorted list of router weight names.
    """
    routers = []
    for name in weights:
        if name.endswith(ROUTER_WEIGHT_SUFFIX):
            if not _is_multimodal_layer(name):
                routers.append(name)
    return sorted(routers)


def _extract_layer_index(router_name: str) -> int | None:
    """Extract the layer index from a router weight name.
    
    Examples:
        "model.layers.0.mlp.gate.weight" → 0
        "model.layers.42.mlp.gate.weight" → 42
        "model.language_model.layers.5.mlp.gate.weight" → 5
    """
    parts = router_name.rstrip(".weight").rstrip(".gate").split(".")
    try:
        # Find the last numeric part (layer index)
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
    except (ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_router_logits(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute router logits from hidden states and gate weight.
    
    Args:
        hidden_states: Input tensor [batch_size, hidden_size] or [batch_size * seq_len, hidden_size].
        gate_weight: Router weight tensor [num_experts, hidden_size].
        
    Returns:
        Router logits [batch_size, num_experts].
    """
    return torch.nn.functional.linear(hidden_states, gate_weight)


def compute_jaccard_similarity(
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
) -> float:
    """Compute Jaccard similarity between two sets of expert indices.
    
    J = |A ∩ B| / |A ∪ B|
    
    Args:
        indices_a: Expert indices from model A [batch, top_k].
        indices_b: Expert indices from model B [batch, top_k].
        
    Returns:
        Jaccard similarity score (0.0 to 1.0), averaged over batch.
    """
    if indices_a.shape != indices_b.shape:
        raise ValueError(f"Shape mismatch: {indices_a.shape} vs {indices_b.shape}")
    
    batch_size = indices_a.shape[0]
    total_jaccard = 0.0
    
    for i in range(batch_size):
        set_a = set(indices_a[i].tolist())
        set_b = set(indices_b[i].tolist())
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union > 0:
            total_jaccard += intersection / union
        else:
            total_jaccard += 1.0  # Both empty = perfect match
    
    return total_jaccard / batch_size


def compute_router_jaccard_for_layer(
    hidden_states_fp16: torch.Tensor,
    hidden_states_quant: torch.Tensor,
    gate_weight_fp16: torch.Tensor,
    gate_weight_quant: torch.Tensor,
    top_k: int = 2,
    use_sigmoid: bool = False,
) -> float:
    """Compute Router Jaccard for a single layer.
    
    Args:
        hidden_states_fp16: Input to router from FP16 model.
        hidden_states_quant: Input to router from quantized model.
        gate_weight_fp16: Router weight from FP16 model.
        gate_weight_quant: Router weight from quantized model.
        top_k: Number of top experts to select.
        use_sigmoid: Whether to apply sigmoid before top-k (DeepSeek style).
        
    Returns:
        Jaccard similarity score (0.0 to 1.0).
    """
    # Compute router logits
    logits_fp16 = compute_router_logits(hidden_states_fp16, gate_weight_fp16)
    logits_quant = compute_router_logits(hidden_states_quant, gate_weight_quant)
    
    # Apply activation if needed (DeepSeek uses sigmoid)
    if use_sigmoid:
        logits_fp16 = logits_fp16.sigmoid()
        logits_quant = logits_quant.sigmoid()
    
    # Get top-k expert indices
    _, indices_fp16 = torch.topk(logits_fp16, top_k, dim=-1)
    _, indices_quant = torch.topk(logits_quant, top_k, dim=-1)
    
    # Flatten batch dimensions if needed
    if indices_fp16.dim() > 2:
        indices_fp16 = indices_fp16.view(-1, top_k)
        indices_quant = indices_quant.view(-1, top_k)
    
    return compute_jaccard_similarity(indices_fp16, indices_quant)


# ---------------------------------------------------------------------------
# Report-based analysis
# ---------------------------------------------------------------------------


def select_layers_by_jaccard(
    jaccard_scores: dict[str, float],
    threshold: float = 0.95,
) -> list[str]:
    """Select router layers with Jaccard below threshold.
    
    Args:
        jaccard_scores: Dict mapping router names to Jaccard scores.
        threshold: Jaccard threshold. Layers below this are selected.
        
    Returns:
        List of router layer names with Jaccard < threshold.
    """
    return [name for name, score in jaccard_scores.items() if score < threshold]


def jaccard_report_to_layer_indices(
    jaccard_scores: dict[str, float],
    threshold: float = 0.95,
) -> list[int]:
    """Convert Jaccard scores to layer indices for ASAQ substitution.
    
    Args:
        jaccard_scores: Dict mapping router names to Jaccard scores.
        threshold: Jaccard threshold. Layers below this are selected.
        
    Returns:
        Sorted list of layer indices.
    """
    indices = []
    for name, score in jaccard_scores.items():
        if score < threshold:
            idx = _extract_layer_index(name)
            if idx is not None:
                indices.append(idx)
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_jaccard_report(
    jaccard_scores: dict[str, float],
    output_dir: str,
    threshold: float = 0.95,
) -> str:
    """Save Jaccard scores to a JSON report.
    
    Args:
        jaccard_scores: Dict mapping router names to Jaccard scores.
        output_dir: Directory to save the report.
        threshold: Threshold used for selection.
        
    Returns:
        Path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "threshold": threshold,
        "scores": jaccard_scores,
        "summary": {
            "total_routers": len(jaccard_scores),
            "below_threshold": len(select_layers_by_jaccard(jaccard_scores, threshold)),
            "mean_jaccard": sum(jaccard_scores.values()) / len(jaccard_scores) if jaccard_scores else 0.0,
        },
    }
    
    report_path = os.path.join(output_dir, "router-jaccard-report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path


def load_jaccard_report(report_path: str) -> dict[str, Any]:
    """Load a Jaccard report from JSON.
    
    Args:
        report_path: Path to router-jaccard-report.json.
        
    Returns:
        Report dict with 'threshold', 'scores', and 'summary' keys.
    """
    with open(report_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def find_routers_in_model(model_dir: str) -> list[str]:
    """Find all router layers in a model directory.
    
    Loads the model index to identify router weights without loading
    the full model weights.
    
    Args:
        model_dir: Path to model directory with safetensors.
        
    Returns:
        List of router weight names.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_names = list(index.get("weight_map", {}).keys())
    else:
        # Single safetensors file
        from safetensors import safe_open
        
        single_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"No safetensors found in {model_dir}")
        
        with safe_open(single_path, framework="pt", device="cpu") as f:
            weight_names = list(f.keys())
    
    # Filter for router layers
    routers = []
    for name in weight_names:
        if name.endswith(ROUTER_WEIGHT_SUFFIX):
            if not _is_multimodal_layer(name):
                routers.append(name)
    
    return sorted(routers)


def has_quantized_routers(
    weights: dict[str, torch.Tensor],
    weight_names: list[str] | None = None,
) -> bool:
    """Check if the model has quantized router layers.
    
    Routers are quantized if they exist but don't have quantized versions
    (i.e., they're not in the ignore list).
    
    Args:
        weights: Model weights dict.
        weight_names: Optional list of weight names to check. 
                      If None, scans all weights.
                      
    Returns:
        True if quantized routers exist.
    """
    if weight_names is None:
        weight_names = list(weights.keys())
    
    for name in weight_names:
        if name.endswith(ROUTER_WEIGHT_SUFFIX):
            if not _is_multimodal_layer(name):
                # Check if this router has quantized versions
                # If it has .qweight, it's quantized; if not, it's FP16
                prefix = name[:-len("gate.weight")]
                has_qweight = any(k.endswith(".qweight") and k.startswith(prefix) for k in weights)
                if not has_qweight:
                    # Router exists but is not quantized (likely ignored)
                    continue
                return True
    
    return False
