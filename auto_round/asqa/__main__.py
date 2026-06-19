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
"""CLI entry point for spark-asqa-substitute."""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for ASAQ layer substitution.

    Selection precedence:

    1. ``--layers 54,58`` — explicit manual layer indices
    2. ``--top-n 5`` — pick N worst layers from quantization report
    3. Neither — read report, substitute all layers that failed quality checks

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="spark-asqa-substitute",
        description="Substitute quantized layers back to FP16 in a quantized model.",
        epilog=(
            "Layer selection precedence:\n"
            "  1. --layers 54,58      explicit manual indices\n"
            "  2. --top-n 5           pick N worst from quantization report\n"
            "  3. (default)           substitute all layers that failed quality checks"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="Model name or path (e.g. Qwen/Qwen3.6-27B). "
        "Used to infer quantized model path, FP16 model, and output directory.",
    )

    # Layer selection (mutually exclusive)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to substitute to FP16 (e.g. 54,58). "
        "Overrides report-based selection.",
    )
    selection.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Pick the N worst layers from the quantization report. "
        "Worst = lowest cosine_sim (default) or psnr_db.",
    )

    # Report-based selection options
    parser.add_argument(
        "--metric",
        choices=["cosine", "psnr"],
        default="cosine",
        help="Metric for --top-n ranking (default: cosine). "
        "cosine: lower cosine_sim = worse. psnr: lower psnr_db = worse.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Select all layers below this quality threshold. "
        "For cosine: layers with cosine_sim < threshold. "
        "For psnr: layers with psnr_db < threshold.",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory override. Default: ./models/{name}-int4-ASAQ",
    )
    return parser.parse_args(argv)


def parse_layer_indices(layers_str: str) -> list[int]:
    """Parse comma-separated layer indices string.

    Args:
        layers_str: e.g. ``"54,58"`` or ``"54, 58"``

    Returns:
        Sorted list of unique layer indices.

    Raises:
        ValueError: If any index is not a valid integer or duplicates exist.
    """
    if not layers_str or not layers_str.strip():
        raise ValueError("No layer indices provided.")

    try:
        indices = [int(x.strip()) for x in layers_str.split(",")]
    except ValueError:
        raise ValueError(
            f"Invalid layer indices: '{layers_str}'. "
            f"Expected comma-separated integers (e.g. 54,58)."
        ) from None

    if not indices:
        raise ValueError("No layer indices provided.")

    # Check for duplicates
    unique = sorted(set(indices))
    if len(unique) != len(indices):
        from collections import Counter

        dupes = [i for i, c in Counter(indices).items() if c > 1]
        raise ValueError(f"Duplicate layer indices: {dupes}")

    return unique


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(argv: list[str] | None = None) -> None:
    """Main entry point for spark-asqa-substitute CLI.

    Orchestrates the full substitution workflow:
    1. Parse CLI arguments
    2. Select layers (manual, top-N, or report-based)
    3. Infer paths from model name
    4. Load quantized model
    5. Load FP16 layers (lazy)
    6. Substitute layers
    7. Save model
    8. Copy config files
    9. Update quantization config
    10. Generate report
    11. Run smoke test
    """
    from auto_round.asqa.substitute import (
        copy_config_files,
        compute_model_size,
        generate_asaq_report,
        infer_paths,
        load_fp16_layers,
        load_quantized_weights,
        save_model,
        select_layers_from_report,
        substitute_layers,
        update_quantization_config,
    )

    args = parse_args(argv)

    # Infer paths first (need quantized_path for report-based selection)
    try:
        quantized_path, fp16_model_id, default_output_dir = infer_paths(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Layer selection ---
    if args.layers is not None:
        # Explicit manual selection
        layer_indices = parse_layer_indices(args.layers)
        print(f"Layer selection: manual ({args.layers})")
    elif args.top_n is not None:
        # Top-N from report
        layer_indices = select_layers_from_report(
            quantized_path, top_n=args.top_n, metric=args.metric
        )
        print(f"Layer selection: top-{args.top_n} worst by {args.metric}")
    elif args.threshold is not None:
        # Threshold from report
        layer_indices = select_layers_from_report(
            quantized_path, threshold=args.threshold, metric=args.metric
        )
        print(f"Layer selection: threshold < {args.threshold} ({args.metric})")
    else:
        # Default: all failed layers from report
        layer_indices = select_layers_from_report(quantized_path)
        print("Layer selection: all failed layers (from report)")

    if not layer_indices:
        print("Nothing to substitute — exiting.")
        sys.exit(0)

    output_dir = args.output_dir or default_output_dir

    print(f"Quantized model: {quantized_path}")
    print(f"FP16 model:      {fp16_model_id}")
    print(f"Output:          {output_dir}")
    print()

    # Validate layer indices against model
    # (will be checked during substitution, but print info now)
    print(f"Substituting {len(layer_indices)} layers to FP16:")
    for idx in layer_indices:
        print(f"  model.language_model.layers.{idx}")
    print()

    # Load quantized model
    print("Loading quantized model...")
    weights, config = load_quantized_weights(quantized_path)
    original_size = compute_model_size(weights)

    # Load FP16 layers (lazy)
    print("Loading FP16 layers (lazy)...")
    fp16_layers = load_fp16_layers(fp16_model_id, layer_indices)

    # Substitute
    print("Substituting layers...")
    weights = substitute_layers(weights, fp16_layers, layer_indices)
    new_size = compute_model_size(weights)

    # Size report
    size_diff_gb = (new_size - original_size) / (1024**3)
    pct = ((new_size - original_size) / original_size) * 100
    print(
        f"Model size: {original_size / 1024**3:.1f} GB → {new_size / 1024**3:.1f} GB "
        f"({size_diff_gb:+.1f} GB, {pct:+.1f}%)"
    )
    print()

    # Save model
    print(f"Saving to {output_dir}...")
    save_model(weights, config, output_dir)

    # Copy config files
    copy_config_files(quantized_path, output_dir)

    # Update quantization config (pass weights to mark substituted layers as FP16)
    update_quantization_config(output_dir, layer_indices, weights)

    # Generate report
    report_path = generate_asaq_report(quantized_path, output_dir, layer_indices)
    print(f"Report: {report_path}")

    print()
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    run()
