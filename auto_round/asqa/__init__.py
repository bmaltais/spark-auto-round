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

"""ASAQ — Adaptive Sensitivity-Aware Quantization post-processing utilities."""

from auto_round.asqa.router_jaccard import (
    compute_jaccard_similarity,
    compute_router_jaccard_for_layer,
    find_routers_in_model,
    has_quantized_routers,
    jaccard_report_to_layer_indices,
    load_jaccard_report,
    save_jaccard_report,
    select_layers_by_jaccard,
)
from auto_round.asqa.substitute import (
    copy_config_files,
    compute_model_size,
    generate_asaq_report,
    infer_paths,
    load_fp16_layers,
    load_quantized_weights,
    save_model,
    select_layers_from_report,
    smoke_test,
    substitute_layers,
    update_quantization_config,
)

__all__ = [
    # router_jaccard
    "compute_jaccard_similarity",
    "compute_router_jaccard_for_layer",
    "find_routers_in_model",
    "has_quantized_routers",
    "jaccard_report_to_layer_indices",
    "load_jaccard_report",
    "save_jaccard_report",
    "select_layers_by_jaccard",
    # substitute
    "copy_config_files",
    "compute_model_size",
    "generate_asaq_report",
    "infer_paths",
    "load_fp16_layers",
    "load_quantized_weights",
    "save_model",
    "select_layers_from_report",
    "smoke_test",
    "substitute_layers",
    "update_quantization_config",
]
