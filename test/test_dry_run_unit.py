"""Unit tests for --dry-run mode (no GPU required).

Tests parameter threading through the factory chain and CLI parser.
"""

import inspect

import pytest


class TestDryRunParameterThreading:
    """Test that dry_run parameter flows through the factory chain."""

    def test_dry_run_kwarg_flows_through_factory(self):
        """dry_run should flow through auto_round_factory in **kwargs."""
        from auto_round.compressors.entry import auto_round_factory

        sig = inspect.signature(auto_round_factory)
        # dry_run is not an explicit parameter, so it flows via **kwargs
        assert "kwargs" in sig.parameters, "auto_round_factory should accept **kwargs"

    def test_base_compressor_accepts_kwargs(self):
        """BaseCompressor should accept dry_run via **kwargs."""
        from auto_round.compressors.base import BaseCompressor

        sig = inspect.signature(BaseCompressor.__init__)
        # dry_run is not an explicit parameter — it's in **kwargs
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert has_var_keyword, "BaseCompressor.__init__ should accept **kwargs"

    def test_auto_round_factory_accepts_kwargs(self):
        """AutoRound() factory should accept dry_run kwarg."""
        from auto_round.autoround import AutoRound

        sig = inspect.signature(AutoRound)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert has_var_keyword, "AutoRound() should accept **kwargs"

    def test_data_driven_compressor_accepts_kwargs(self):
        """DataDrivenCompressor should accept dry_run via **kwargs."""
        from auto_round.compressors.data_driven import DataDrivenCompressor

        sig = inspect.signature(DataDrivenCompressor.__init__)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        assert has_var_keyword, "DataDrivenCompressor.__init__ should accept **kwargs"

    def test_dry_run_popped_in_base_compressor_init(self):
        """dry_run should be popped from kwargs in BaseCompressor.__init__."""
        # Verify the source code contains the pop statement
        import auto_round.compressors.base as base_mod

        source = inspect.getsource(base_mod.BaseCompressor.__init__)
        assert 'kwargs.pop("dry_run"' in source or "kwargs.pop('dry_run'" in source, (
            "BaseCompressor.__init__ should pop 'dry_run' from kwargs"
        )

    def test_dry_run_stored_as_instance_attr(self):
        """dry_run should be stored as self.dry_run in BaseCompressor."""
        import auto_round.compressors.base as base_mod

        source = inspect.getsource(base_mod.BaseCompressor.__init__)
        assert "self.dry_run" in source, (
            "BaseCompressor.__init__ should store dry_run as self.dry_run"
        )

    def test_save_config_dry_run_method_exists(self):
        """_save_config_dry_run method should exist on BaseCompressor."""
        from auto_round.compressors.base import BaseCompressor

        assert hasattr(BaseCompressor, "_save_config_dry_run"), (
            "BaseCompressor should have _save_config_dry_run method"
        )

    def test_quantize_and_save_checks_dry_run(self):
        """quantize_and_save should check self.dry_run."""
        import auto_round.compressors.base as base_mod

        source = inspect.getsource(base_mod.BaseCompressor.quantize_and_save)
        assert "self.dry_run" in source, (
            "quantize_and_save should check self.dry_run"
        )
        assert "_save_config_dry_run" in source, (
            "quantize_and_save should call _save_config_dry_run when dry_run is True"
        )


class TestDryRunCLIParser:
    """Test the CLI parser accepts --dry-run flag."""

    def test_parser_accepts_dry_run(self):
        """BasicArgumentParser should accept --dry-run flag."""
        from auto_round.__main__ import BasicArgumentParser

        parser = BasicArgumentParser()
        args = parser.parse_args(["some-model", "--dry-run"])
        assert args.dry_run is True

    def test_parser_dry_run_default_false(self):
        """--dry-run should default to False."""
        from auto_round.__main__ import BasicArgumentParser

        parser = BasicArgumentParser()
        args = parser.parse_args(["some-model"])
        assert args.dry_run is False

    def test_dry_run_threaded_to_autoround(self):
        """CLI should pass dry_run to AutoRound constructor."""
        import auto_round.__main__ as main_mod

        source = inspect.getsource(main_mod.tune)
        assert "dry_run=args.dry_run" in source, (
            "tune() should pass dry_run=args.dry_run to AutoRound()"
        )
