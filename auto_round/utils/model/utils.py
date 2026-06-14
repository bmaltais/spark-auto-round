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
"""General utility functions used across model and device modules."""
from collections import UserDict
from typing import Union

import torch
import transformers

from auto_round.logger import logger


def convert_dtype_str2torch(str_dtype):
    """Converts a string dtype to its corresponding PyTorch dtype."""
    if isinstance(str_dtype, torch.dtype) or str_dtype is None:
        return str_dtype
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "float32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16" or str_dtype == "float16":
        return torch.float16
    elif str_dtype == "bf16" or str_dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported string dtype '{str_dtype}' for conversion to torch dtype.")


def convert_dtype_torch2str(dtype):
    """Converts a PyTorch dtype to its corresponding string representation."""
    if isinstance(dtype, str) or dtype is None:
        return dtype
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        raise ValueError(f"Unsupported PyTorch dtype '{dtype}' for conversion to string dtype.")


def convert_dtype_torch2str_hf(dtype):
    """Converts a PyTorch dtype to its corresponding huggingface string dtype."""
    if dtype is None:
        return dtype
    if isinstance(dtype, str):
        if "float" not in dtype and "int" not in dtype:
            dtype = convert_dtype_str2torch(dtype)
        else:
            return dtype
    str_dtype = str(dtype)
    if "." not in str_dtype:
        raise ValueError(f"Unsupported pytorch dtype '{dtype}' for conversion to huggingface str dtype")
    str_dtype = str_dtype.split(".")[1]
    return str_dtype


def check_start_with_block_name(name: str, block_name_to_quantize: list):
    """Checks if the given layer name starts with any of the block names to be quantized."""
    for block_name in block_name_to_quantize:
        if name.startswith(block_name):
            return True
    return False


def get_nested_attr(module, attr_name: str):
    """Recursively get nested attribute (e.g., 'orig_layer.act_max')."""
    attrs = attr_name.split(".")
    for attr in attrs:
        if not hasattr(module, attr):
            return None
        module = getattr(module, attr)
    return module


def set_nested_attr(module, attr_name: str, value):
    """Recursively set nested attribute (e.g., 'orig_layer.act_max' = value)."""
    attrs = attr_name.split(".")
    for attr in attrs[:-1]:
        if not hasattr(module, attr):
            return None  # No need to set act_max for fp layers
        module = getattr(module, attr)
    setattr(module, attrs[-1], value)


def check_to_quantized(config):
    """Checks if the configuration is valid for quantization (bits <= 8)."""
    from auto_round.schemes import QuantizationScheme

    if isinstance(config, (dict, QuantizationScheme)):
        bits = config.get("bits", None)
        act_bits = config.get("act_bits", None)
    elif hasattr(config, "orig_layer"):
        bits = getattr(config.orig_layer, "bits", None)
        act_bits = getattr(config.orig_layer, "act_bits", None)
    else:
        bits = getattr(config, "bits", None)
        act_bits = getattr(config, "act_bits", None)

    bits = int(bits) if bits is not None else 16
    act_bits = int(act_bits) if act_bits is not None else 16

    return bits <= 8 or act_bits <= 8


def check_seqlen_compatible(input_seqlen, tokenizer=None, model=None):
    """Check whether the input sequence length is within the limits."""
    if model is not None and hasattr(model, "config"):
        model_config = model.config
        if hasattr(model_config, "max_position_embeddings") and input_seqlen > model_config.max_position_embeddings:
            raise ValueError(
                f"seqlen({input_seqlen}) exceeds model.config.max_position_embeddings("
                f"{model_config.max_position_embeddings}). Please lowering '--seqlen'"
            )
    if tokenizer is not None and hasattr(tokenizer, "model_max_length") and input_seqlen > tokenizer.model_max_length:
        raise ValueError(
            f"seqlen({input_seqlen}) exceeds tokenizer.model_max_length({tokenizer.model_max_length}). "
            "Please oncider Consider lowering the '--seqlen' or increasing tokenizer.model_max_length."
        )


def get_attr(module, key):
    """Get attribute (including parameters like `...weight`) by dotted key."""
    name_list = key.split(".")
    for name in name_list:
        if module is None:
            return None
        module = getattr(module, name, None)
    return module


def set_attr(model, key, new_attr):
    """Set attribute (including parameters like `...weight`) by dotted key."""
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if not hasattr(module, name):
            return
        module = getattr(module, name)
    setattr(module, name_list[-1], new_attr)


def get_module(module, key):
    """Get module from model by key name using PyTorch native API."""
    try:
        return module.get_submodule(key)
    except (AttributeError, KeyError):
        return None


def set_module(model, key, new_module):
    """Set new module into model by key name using PyTorch native API."""
    try:
        model.set_submodule(key, new_module)
    except (AttributeError, KeyError):
        return


def get_layer_features(layer):
    """Extracts input and output feature dimensions for supported layers."""
    from auto_round.utils import deepspeed_exists

    if deepspeed_exists:
        from deepspeed.module_inject import LinearAllreduce, LinearLayer
    if type(layer) == torch.nn.Linear:
        return layer.in_features, layer.out_features
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        return layer.weight.shape[0], layer.weight.shape[1]
    elif isinstance(layer, torch.nn.Embedding):
        return layer.num_embeddings, layer.embedding_dim
    elif deepspeed_exists and type(layer) in (LinearLayer, LinearAllreduce):
        return layer.weight.shape[1], layer.weight.shape[0]
    elif "FP8Linear" in layer.__class__.__name__:
        return layer.in_features, layer.out_features
    return None, None


def get_common_prefix(paths):
    """Find the common prefix of dotted paths."""
    split_paths = [path.split(".") for path in paths]
    common_prefix = split_paths[0]
    for path in split_paths[1:]:
        common_prefix = [comp for comp, other in zip(common_prefix, path) if comp == other]
    return ".".join(common_prefix)


def unsupported_meta_device(model):
    """Checks if the model is a valid model for auto_round."""
    target_device = None
    for param in model.parameters():
        if target_device is None:
            target_device = param.device
        if param.device != target_device:
            if param.device.type == "meta" or target_device.type == "meta":
                return True
    if target_device.type == "meta":
        if hasattr(model, "path"):
            return False
        else:
            return True
    return False


def to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device."""
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_device(inp, device))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res
    return input


def mv_module_from_gpu(module):
    """Moves module from gpu to cpu."""
    if hasattr(module, "device"):
        if module.device.type in ("cpu", "meta"):
            return module

    has_meta = any(p.device.type == "meta" for p in module.parameters())
    if not has_meta:
        has_meta = any(b.device.type == "meta" for b in module.buffers())

    if has_meta:
        for _, child in module.named_children():
            mv_module_from_gpu(child)
        for attr_name in list(module._parameters.keys()):
            p = module._parameters[attr_name]
            if p is not None and p.device.type != "meta" and p.device.type != "cpu":
                module._parameters[attr_name] = torch.nn.Parameter(p.to("cpu"), requires_grad=p.requires_grad)
        for attr_name in list(module._buffers.keys()):
            b = module._buffers[attr_name]
            if b is not None and b.device.type != "meta" and b.device.type != "cpu":
                module._buffers[attr_name] = b.to("cpu")
        return module

    return module.to("cpu")


def safe_device_move_with_meta_handling(
    model,
    target_device="cpu",
    *,
    materialize_meta=None,
    logger=None,
):
    """Move model to target device, handling meta parameters and buffers correctly."""
    target_type = torch.device(target_device).type

    if materialize_meta is not None:
        materialize_meta(model)

    meta_count = 0

    for p in model.parameters():
        if p.device.type in (target_type, "meta"):
            meta_count += p.device.type == "meta"
            continue
        p.data = p.data.to(target_device)

    for m in model.modules():
        for n, b in m._buffers.items():
            if b is None or b.device.type in (target_type, "meta"):
                meta_count += b is not None and b.device.type == "meta"
                continue
            m._buffers[n] = b.to(target_device)

    if meta_count and logger is not None:
        logger.warning(f"{meta_count} tensors still on meta device after movement")

    return model


def to_dtype(input, dtype=torch.float32):
    """Moves input data to the specified data type."""
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_dtype(input[inp], dtype)
    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_dtype(inp, dtype))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res
    return input
