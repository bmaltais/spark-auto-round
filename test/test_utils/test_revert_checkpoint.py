"""Unit tests for revert_checkpoint_conversion_mapping — no CUDA required."""

import pytest

from auto_round.utils.common import revert_checkpoint_conversion_mapping, preserve_original_visual_block_name


class TestRevertCheckpointConversionMapping:
    """Test that revert_checkpoint_conversion_mapping handles complex regex patterns."""

    def test_simple_prefix_reversion(self):
        """Simple prefix pattern should be reverted correctly."""
        mapping = {
            "^model.language_model": "model",
        }
        result = revert_checkpoint_conversion_mapping(
            "model.language_model.layers.0.self_attn.q_proj.weight", mapping
        )
        assert result == "model.layers.0.self_attn.q_proj.weight"

    def test_comma_separated_names(self):
        """Comma-separated names should be split and reverted individually."""
        mapping = {
            "^model.language_model": "model",
        }
        result = revert_checkpoint_conversion_mapping(
            "model.language_model.layers.0.weight,model.language_model.layers.1.weight",
            mapping,
        )
        assert result == "model.layers.0.weight,model.layers.1.weight"

    def test_no_match_returns_original(self):
        """If no pattern matches, return the original name."""
        mapping = {
            "^model.language_model": "model",
        }
        result = revert_checkpoint_conversion_mapping(
            "visual.encoder.layers.0.weight", mapping
        )
        assert result == "visual.encoder.layers.0.weight"

    def test_upstream_strips_caret_anchor(self):
        """Upstream strips ^ anchor before matching."""
        mapping = {
            "^model.layers": "model.language_model.layers",
        }
        result = revert_checkpoint_conversion_mapping(
            "model.layers.0.weight", mapping
        )
        # Should match after stripping ^
        assert result == "model.language_model.layers.0.weight"

    def test_multiple_target_patterns(self):
        """Multiple target patterns for same source should try each."""
        mapping = {
            "^model": ["model.language_model", "model.visual"],
        }
        # First target pattern should match
        result = revert_checkpoint_conversion_mapping(
            "model.layers.0.weight", mapping
        )
        # Should use first matching target
        assert result == "model.language_model.layers.0.weight"

    def test_string_target_converted_to_list(self):
        """String target should be converted to list internally."""
        mapping = {
            "^model.language_model": "model",
        }
        result = revert_checkpoint_conversion_mapping(
            "model.language_model.layers.0.weight", mapping
        )
        assert result == "model.layers.0.weight"

    def test_empty_mapping_returns_original(self):
        """Empty mapping should return original name."""
        mapping = {}
        result = revert_checkpoint_conversion_mapping(
            "model.layers.0.weight", mapping
        )
        assert result == "model.layers.0.weight"

    def test_single_name_no_comma(self):
        """Single name without comma should be processed directly."""
        mapping = {
            "^model.language_model": "model",
        }
        result = revert_checkpoint_conversion_mapping(
            "model.language_model.layers.0.weight", mapping
        )
        assert result == "model.layers.0.weight"


class TestPreserveOriginalVisualBlockName:
    """Test preserve_original_visual_block_name handles edge cases."""

    def test_preserve_noop_case(self):
        """preserve_original_visual_block_name handles no-op reversion."""
        # When reversion doesn't change the name, we should not get duplicates
        original = "model.language_model.layers.0.weight"
        reverted = "model.language_model.layers.0.weight"  # no-op
        result = preserve_original_visual_block_name(original, reverted)
        # Should contain the name once (from the original_part branch)
        parts = [p.strip() for p in result.split(",")]
        assert len(parts) == 1
        assert parts[0] == "model.language_model.layers.0.weight"
