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
"""Configuration for spark-auto-round quantization."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Optional, Union

import torch


@dataclass
class SARConfig:
    """Single configuration dataclass for spark-auto-round.

    Replaces the old ExtraConfig/TuningExtraConfig/SchemeExtraConfig hierarchy.
    All parameters are flat — no nested config objects.
    """
    # Tuning parameters
    amp: bool = True
    disable_opt_rtn: bool | None = None
    enable_alg_ext: bool = False
    enable_minmax_tuning: bool = True
    enable_norm_bias_tuning: bool = False
    enable_quanted_input: bool = True
    enable_deterministic_algorithms: bool = False
    lr: float = None
    lr_scheduler: Callable = None
    minmax_lr: float = None
    nblocks: int = 1
    to_quant_block_names: Union[str, list, None] = None
    scale_dtype: str = "fp16"

    # Scheme parameters
    bits: int = None
    group_size: int = None
    sym: bool = None
    data_type: str = None
    act_bits: int = None
    act_group_size: int = None
    act_sym: bool = None
    act_data_type: str = None
    act_dynamic: bool = None
    super_bits: int = None
    super_group_size: int = None
    static_kv_dtype: Union[str, torch.dtype] = None
    static_attention_dtype: Union[str, torch.dtype] = None
    quant_lm_head: bool = False
    ignore_layers: str = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# Backward-compatible aliases
ExtraConfig = SARConfig
TuningExtraConfig = None  # Will cause ImportError if used directly — that's the point
SchemeExtraConfig = None
