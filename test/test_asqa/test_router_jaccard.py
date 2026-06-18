"""Unit tests for auto_round.asqa.router_jaccard — no CUDA or real models required."""

from __future__ import annotations

import json
import os

import pytest
import torch

from auto_round.asqa.router_jaccard import (
    _extract_layer_index,
    _find_router_layers,
    _is_multimodal_layer,
    compute_jaccard_similarity,
    compute_router_jaccard_for_layer,
    find_routers_in_model,
    jaccard_report_to_layer_indices,
    load_jaccard_report,
    save_jaccard_report,
    select_layers_by_jaccard,
)


# ---------------------------------------------------------------------------
# _is_multimodal_layer
# ---------------------------------------------------------------------------


class TestIsMultimodalLayer:
    def test_vision_tower(self):
        assert _is_multimodal_layer("model.vision_tower.layers.0.weight") is True

    def test_audio_tower(self):
        assert _is_multimodal_layer("model.audio_tower.layers.0.weight") is True

    def test_mm_projector(self):
        assert _is_multimodal_layer("model.mm_projector.weight") is True

    def test_thinker(self):
        assert _is_multimodal_layer("thinker.visual.weight") is True

    def test_text_layer(self):
        assert _is_multimodal_layer("model.layers.0.mlp.gate.weight") is False

    def test_expert_layer(self):
        assert _is_multimodal_layer("model.layers.0.mlp.experts.0.gate_proj.weight") is False


# ---------------------------------------------------------------------------
# _find_router_layers
# ---------------------------------------------------------------------------


class TestFindRouterLayers:
    def test_finds_routers(self):
        weights = {
            "model.layers.0.mlp.gate.weight": torch.randn(10, 10),
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(10, 10),
            "model.layers.1.mlp.gate.weight": torch.randn(10, 10),
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.randn(10, 10),
        }
        result = _find_router_layers(weights)
        assert result == [
            "model.layers.0.mlp.gate.weight",
            "model.layers.1.mlp.gate.weight",
        ]

    def test_excludes_multimodal(self):
        weights = {
            "model.layers.0.mlp.gate.weight": torch.randn(10, 10),
            "model.vision_tower.layers.0.mlp.gate.weight": torch.randn(10, 10),
            "model.audio_tower.layers.0.mlp.gate.weight": torch.randn(10, 10),
        }
        result = _find_router_layers(weights)
        assert result == ["model.layers.0.mlp.gate.weight"]

    def test_empty_weights(self):
        assert _find_router_layers({}) == []

    def test_no_routers(self):
        weights = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(10, 10),
        }
        assert _find_router_layers(weights) == []


# ---------------------------------------------------------------------------
# _extract_layer_index
# ---------------------------------------------------------------------------


class TestExtractLayerIndex:
    def test_standard_format(self):
        assert _extract_layer_index("model.layers.0.mlp.gate.weight") == 0
        assert _extract_layer_index("model.layers.42.mlp.gate.weight") == 42

    def test_language_model_prefix(self):
        assert _extract_layer_index("model.language_model.layers.5.mlp.gate.weight") == 5

    def test_no_index(self):
        assert _extract_layer_index("model.mlp.gate.weight") is None


# ---------------------------------------------------------------------------
# compute_jaccard_similarity
# ---------------------------------------------------------------------------


class TestComputeJaccardSimilarity:
    def test_identical_sets(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[1, 2], [3, 4]])
        assert compute_jaccard_similarity(a, b) == 1.0

    def test_no_overlap(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        assert compute_jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        # Set A = {1, 2}, Set B = {2, 3}
        # Intersection = {2}, Union = {1, 2, 3}
        # Jaccard = 1/3
        a = torch.tensor([[1, 2]])
        b = torch.tensor([[2, 3]])
        result = compute_jaccard_similarity(a, b)
        assert abs(result - 1 / 3) < 1e-6

    def test_averages_over_batch(self):
        # Batch 0: perfect match (J=1.0)
        # Batch 1: no overlap (J=0.0)
        # Average = 0.5
        a = torch.tensor([[1, 2], [1, 2]])
        b = torch.tensor([[1, 2], [3, 4]])
        result = compute_jaccard_similarity(a, b)
        assert abs(result - 0.5) < 1e-6

    def test_shape_mismatch_raises(self):
        a = torch.tensor([[1, 2]])
        b = torch.tensor([[1, 2, 3]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_jaccard_similarity(a, b)


# ---------------------------------------------------------------------------
# compute_router_jaccard_for_layer
# ---------------------------------------------------------------------------


class TestComputeRouterJaccardForLayer:
    def test_identical_weights(self):
        """Same weights should give Jaccard = 1.0."""
        hidden = torch.randn(8, 64)
        gate = torch.randn(10, 64)
        
        result = compute_router_jaccard_for_layer(hidden, hidden, gate, gate, top_k=2)
        assert result == 1.0

    def test_different_weights(self):
        """Very different weights should give lower Jaccard."""
        hidden = torch.randn(8, 64)
        gate_a = torch.randn(10, 64)
        gate_b = torch.randn(10, 64) * 10  # Scale up significantly
        
        result = compute_router_jaccard_for_layer(hidden, hidden, gate_a, gate_b, top_k=2)
        assert result < 1.0

    def test_with_sigmoid(self):
        """Test sigmoid activation (DeepSeek style)."""
        hidden = torch.randn(8, 64)
        gate = torch.randn(10, 64)
        
        result = compute_router_jaccard_for_layer(
            hidden, hidden, gate, gate, top_k=2, use_sigmoid=True
        )
        assert result == 1.0


# ---------------------------------------------------------------------------
# select_layers_by_jaccard / jaccard_report_to_layer_indices
# ---------------------------------------------------------------------------


class TestSelectByJaccard:
    def test_selects_below_threshold(self):
        scores = {
            "model.layers.0.mlp.gate.weight": 0.99,
            "model.layers.1.mlp.gate.weight": 0.85,
            "model.layers.2.mlp.gate.weight": 0.95,
            "model.layers.3.mlp.gate.weight": 0.70,
        }
        result = select_layers_by_jaccard(scores, threshold=0.95)
        assert result == [
            "model.layers.1.mlp.gate.weight",
            "model.layers.3.mlp.gate.weight",
        ]

    def test_empty_when_all_above(self):
        scores = {
            "model.layers.0.mlp.gate.weight": 0.99,
            "model.layers.1.mlp.gate.weight": 0.98,
        }
        assert select_layers_by_jaccard(scores, threshold=0.95) == []

    def test_report_to_indices(self):
        scores = {
            "model.layers.5.mlp.gate.weight": 0.80,
            "model.layers.2.mlp.gate.weight": 0.75,
            "model.layers.8.mlp.gate.weight": 0.90,
        }
        result = jaccard_report_to_layer_indices(scores, threshold=0.95)
        assert result == [2, 5, 8]  # Sorted


# ---------------------------------------------------------------------------
# save/load jaccard report
# ---------------------------------------------------------------------------


class TestJaccardReportIO:
    def test_save_and_load(self, tmp_path):
        scores = {
            "model.layers.0.mlp.gate.weight": 0.99,
            "model.layers.1.mlp.gate.weight": 0.85,
        }
        
        report_path = save_jaccard_report(scores, str(tmp_path), threshold=0.95)
        assert os.path.exists(report_path)
        
        loaded = load_jaccard_report(report_path)
        assert loaded["threshold"] == 0.95
        assert loaded["scores"] == scores
        assert loaded["summary"]["total_routers"] == 2
        assert loaded["summary"]["below_threshold"] == 1


# ---------------------------------------------------------------------------
# find_routers_in_model
# ---------------------------------------------------------------------------


class TestFindRoutersInModel:
    def test_reads_from_index(self, tmp_path):
        """Test reading router names from model.safetensors.index.json."""
        index = {
            "weight_map": {
                "model.layers.0.mlp.gate.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.mlp.gate.weight": "model-00002-of-00002.safetensors",
                "model.vision_tower.layers.0.weight": "model-00001-of-00002.safetensors",
            },
            "metadata": {},
        }
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)
        
        result = find_routers_in_model(str(tmp_path))
        assert result == [
            "model.layers.0.mlp.gate.weight",
            "model.layers.1.mlp.gate.weight",
        ]

    def test_raises_for_missing_model(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_routers_in_model(str(tmp_path))
