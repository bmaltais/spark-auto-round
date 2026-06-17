#!/usr/bin/env python3
"""Fix v14.1 regression: wrong layer name prefix in quantized model configs.

The bug: revert_checkpoint_conversion_mapping() stripped the ^ regex anchor,
causing 'model.language_model.layers' to be saved as 'model.layers' in
quantization_config.json, config.json, and extra_config keys.

Safetensors tensors are fine — only config files need fixing.

Usage:
    python fix-v14.1-layer-prefix.py <quantized_model_dir> <source_model_dir>

Example:
    python fix-v14.1-layer-prefix.py \
        ~/models/Qwen3.6-35B-A3B-int4-AutoRound \
        Qwen/Qwen3.6-35B-A3B
"""

import json
import os
import shutil
import sys
from pathlib import Path


def fix_config_keys(d):
    """Recursively fix model.layers -> model.language_model.layers in dict keys and values."""
    if not isinstance(d, dict):
        return 0
    fixed = 0
    new_d = {}
    for k, v in d.items():
        new_key = k

        # Fix literal keys: model.layers.X -> model.language_model.layers.X
        if k.startswith("model.layers."):
            new_key = "model.language_model.layers." + k[len("model.layers."):]
            fixed += 1

        # Fix regex keys: .*model\.layers\. -> .*model\.language_model\.layers\.
        elif ".*model\\.layers\\." in k:
            new_key = k.replace(".*model\\.layers\\.", ".*model\\.language_model\\.layers\\.")
            fixed += 1

        # Fix double-prefix from previous bad fix attempts
        elif k.startswith("model.language_model.model.layers."):
            new_key = "model.language_model.layers." + k[len("model.language_model.model.layers."):]
            fixed += 1

        # Fix string values (block_name_to_quantize)
        if isinstance(v, str):
            if v == "model.layers":
                v = "model.language_model.layers"
                fixed += 1

        # Fix list values
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str) and item == "model.layers":
                    v[i] = "model.language_model.layers"
                    fixed += 1

        # Recurse into nested dicts
        if isinstance(v, dict):
            fixed += fix_config_keys(v)

        new_d[new_key] = v

    d.clear()
    d.update(new_d)
    return fixed


def fix_safetensors_index(index_path):
    """Fix weight_map keys in model.safetensors.index.json if needed."""
    with open(index_path) as f:
        idx = json.load(f)

    wm = idx.get("weight_map", {})
    bad_keys = [k for k in wm if k.startswith("model.layers.")]

    if not bad_keys:
        return 0

    for k in list(wm.keys()):
        if k.startswith("model.layers."):
            wm["model.language_model." + k] = wm.pop(k)

    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)

    return len(bad_keys)


def copy_missing_files(source_dir, target_dir):
    """Copy processor/preprocessor configs from source model if missing."""
    files_to_copy = [
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    copied = []
    for fname in files_to_copy:
        src = Path(source_dir) / fname
        dst = Path(target_dir) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(fname)
    return copied


def main():
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    model_dir = Path(sys.argv[1]).expanduser()
    source_dir = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else None

    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)

    print(f"Fixing: {model_dir}\n")

    # 1. Fix quantization_config.json
    qc_path = model_dir / "quantization_config.json"
    if qc_path.exists():
        with open(qc_path) as f:
            qc = json.load(f)
        fixed = fix_config_keys(qc)
        with open(qc_path, "w") as f:
            json.dump(qc, f, indent=2)
        print(f"  quantization_config.json: fixed {fixed} entries")
        print(f"    block_name_to_quantize: {qc.get('block_name_to_quantize')}")
    else:
        print("  quantization_config.json: not found, skipping")

    # 2. Fix config.json (has nested quantization_config)
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        fixed = fix_config_keys(cfg)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  config.json: fixed {fixed} entries")
    else:
        print("  config.json: not found, skipping")

    # 3. Fix model.safetensors.index.json
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        fixed = fix_safetensors_index(idx_path)
        print(f"  model.safetensors.index.json: fixed {fixed} weight_map keys")
    else:
        print("  model.safetensors.index.json: not found, skipping")

    # 4. Copy missing processor files from source model
    if source_dir and source_dir.is_dir():
        copied = copy_missing_files(source_dir, model_dir)
        for f in copied:
            print(f"  Copied {f} from source model")
        if not copied:
            print("  No missing processor files to copy")
    elif source_dir:
        print(f"  Source model not found: {source_dir}")
        print("  (processor_config.json and preprocessor_config.json may be missing)")
    else:
        print("  No source model specified — skipping file copy")
        print("  If processor_config.json is missing, copy it manually from the source model")

    print("\nDone!")


if __name__ == "__main__":
    main()
