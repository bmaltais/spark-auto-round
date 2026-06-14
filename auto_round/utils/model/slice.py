# Copyright (c) 2025 Intel Corporation
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
"""Model slicing and layer detection utilities."""
import os

import torch

from auto_round.logger import logger


def find_layers_from_config(model_dir: str, class_names: list[str] | None = None) -> dict[str, str]:
    """Detect layers of given class names by loading the model on ``device='meta'``.

    Only ``config.json`` is required — no weights are read.

    For regular models the root directory is checked.  For diffusion-style
    repos (no root ``config.json`` but a ``transformer/`` subfolder), only the
    ``transformer/`` subfolder is checked.

    Args:
        model_dir: Local directory containing ``config.json``, or a diffusion
            repo root whose ``transformer/`` subfolder contains ``config.json``.
        class_names: Class names to look for, matched against
            ``type(module).__name__``.  Defaults to
            ``["Embedding", "Conv1d", "Conv1D"]``.

    Returns:
        ``{class_name: [layer_name, ...]}`` for every matched module.
        Returns an empty dict on any failure.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModel

    if class_names is None:
        class_names = ["Embedding", "Conv1d", "Conv1D"]
    if isinstance(class_names, str):
        class_names = [class_names]
    target = set(class_names)

    if not os.path.exists(model_dir):
        model_dir = snapshot_download(
            repo_id=model_dir,
            allow_patterns=["**/config.json"],
        )

    dirs: list[tuple[str, str]] = []
    if os.path.exists(os.path.join(model_dir, "config.json")):
        dirs.append(("", model_dir))
    else:
        transformer_dir = os.path.join(model_dir, "transformer")
        if os.path.isdir(transformer_dir) and os.path.exists(os.path.join(transformer_dir, "config.json")):
            dirs.append(("", transformer_dir))

    result: dict[str, str] = {}
    for prefix, config_dir in dirs:
        try:
            with torch.device("meta"):
                config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
                model = AutoModel.from_config(config, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load model from {config_dir} for layer detection. Skipping. Error: {e}")
            continue
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if any(t.lower() in cls_name.lower() for t in target):
                full_name = f"{prefix}.{name}" if prefix else name
                if cls_name not in result:
                    result[cls_name] = [full_name]
                else:
                    result[cls_name].append(full_name)
        del model
    return result
