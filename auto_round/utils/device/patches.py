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
"""Compile patches and device map functions."""
import os
import re
from itertools import combinations
from typing import Any, Callable, Optional, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory

from auto_round.logger import logger
from auto_round.utils.device.detect import (
    detect_device,
    detect_device_count,
    parse_available_devices,
)
from auto_round.utils.device.memory import (
    clear_memory,
    get_device_memory,
    memory_monitor,
)


def _bump_dynamo_cache_limit(min_size: Optional[int] = None):
    """Raise torch._dynamo cache/recompile limits."""
    try:
        if min_size is None:
            from auto_round import envs
            min_size = envs.AR_DYNAMO_CACHE_SIZE_LIMIT
        from torch._dynamo import config as _dynamo_config

        for attr in ("cache_size_limit", "accumulated_cache_size_limit", "recompile_limit"):
            if hasattr(_dynamo_config, attr) and getattr(_dynamo_config, attr) < min_size:
                setattr(_dynamo_config, attr, min_size)
    except Exception:
        pass


def compile_func_on_cuda_or_cpu(func):
    _bump_dynamo_cache_limit()
    return torch.compile(func)


def compile_func(
    fun: Union[torch.nn.Module, Callable], device: Union[str, torch.device, int]
) -> Union[torch.nn.Module, Callable]:
    """Compile function on the specified device."""
    return compile_func_on_cuda_or_cpu(fun)


def is_numba_available():  # pragma: no cover
    """Check if Numba is available."""
    try:
        import numba
        return True
    except ImportError:
        return False


def _is_tbb_installed():  # pragma: no cover
    import importlib.metadata
    try:
        importlib.metadata.version("tbb")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _is_tbb_configured():  # pragma: no cover
    try:
        from numba.np.ufunc.parallel import _check_tbb_version_compatible
        _check_tbb_version_compatible()
        return True
    except ImportError as e:
        logger.warning_once(f"TBB not available: {e}")
        return False


def is_tbb_available():  # pragma: no cover
    """Check if TBB is available."""
    if not _is_tbb_installed():
        logger.warning_once("TBB is not installed, please install it with `pip install tbb`.")
        return False
    if not _is_tbb_configured():
        logger.warning_once(
            (
                "TBB is installed but not configured correctly. \n"
                "Please add the TBB library path to `LD_LIBRARY_PATH`, "
                "for example: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/`."
            )
        )
        return False
    return True


def can_pack_with_numba():  # pragma: no cover
    """Check if Numba and TBB are available for packing."""
    if not is_numba_available():
        logger.warning_once("Numba is not installed, please install it with `pip install numba`.")
        return False
    if not is_tbb_available():
        return False
    return True


def get_major_device(device_map: Union[None, str, torch.device, int, dict]) -> str:
    """Get the primary device from a device map."""
    if device_map is None or isinstance(device_map, (str, torch.device, int)):
        device = detect_device(device_map)
        return device

    if isinstance(device_map, dict) and device_map:
        tmp_devices = []
        for val in device_map.values():
            if isinstance(val, (str, torch.device, int)):
                tmp_device = detect_device(val)
                tmp_device = tmp_device.split(":")[0]
                tmp_devices.append(tmp_device)
        tmp_devices = list(set(tmp_devices))
        device = None
        for tmp_device in tmp_devices:
            if tmp_device != "cpu":
                device = tmp_device
                break
        if device is None:
            device = tmp_devices[0]
        if len(tmp_devices) > 1:
            logger.warning_once(
                f"there are multiple device types in the device_map, "
                f"please make sure they are correct,use the first none-cpu device {device} as the core device "
            )

        return device
    logger.warning_once(f"device_map should be [str, torch.device, int, dict], but got {type(device_map)}")
    return "cpu"


def set_tuning_device_for_layer(model, name: str, device: str) -> None:
    """Sets the device for a module if it matches the given name."""
    from auto_round.utils.model import get_module

    module = get_module(model, name)
    if hasattr(module, "tuning_device") and module.tuning_device != device:
        logger.warning(
            f"multiple devices have been set for layer {name}, keeping original device {module.tuning_device}"
        )
    else:
        module.tuning_device = device


def set_non_auto_device_map(
    model: torch.nn.Module, device_map: Union[str, int, dict], quant_layer_names: Union[None, list, tuple] = None
) -> None:
    """Set device map for non-auto configurations."""
    if not device_map or device_map == "auto" or isinstance(device_map, int):
        return
    from auto_round.utils.model import get_module

    if isinstance(device_map, str):
        if "," in device_map:
            return
        device_map = device_map.replace(" ", "")
        infos = device_map.split(",")
        device_map_dict = {}
        for info in infos:
            if ":" not in info:
                continue
            index = info.find(":")
            key = info[:index]
            value = info[index + 1:]
            device_map_dict[key] = value
        device_map = device_map_dict
    if isinstance(device_map, dict) and not any("." in k for k in device_map):
        return
    if quant_layer_names is not None:
        names = quant_layer_names
    else:
        names = [
            n for n, m in model.named_modules() if len(list(m.children())) == 0
        ]
    for key, device in device_map.items():
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        device = detect_device(device)
        if key in names:
            module = get_module(model, key)
            module.tuning_device = device
        else:
            matching_names = [name for name in names if re.match(key, name)]
            for name in matching_names:
                set_tuning_device_for_layer(model, name, device)
            if not matching_names:
                logger.warning(f"{key} in `device_map` dose not match any modules, please have a check")


def _allocate_layers_to_devices(
    layer_memory_dict: dict, device_memory: dict, gpu_devices: list, mem_per_param: float
) -> tuple[dict, list]:
    """Allocates layers to devices using a load-balancing strategy."""
    device_map = {}
    names = []
    layer_names_in_order = list(layer_memory_dict.keys())
    layer_order = {name: idx for idx, name in enumerate(layer_names_in_order)}
    sorted_layers = sorted(layer_memory_dict.items(), key=lambda x: (-x[1]["param_memory"], -layer_order[x[0]]))
    num_devices = len(gpu_devices)

    def find_best_device(layer_name, estimated_memory, layer_idx):
        if layer_idx < num_devices - 1:
            return gpu_devices[-(layer_idx + 1)]

        best_device = None
        best_score = float("-inf")
        current_layer_order = layer_order[layer_name]

        for device in gpu_devices:
            if device_memory[device] < estimated_memory:
                continue

            memory_score = device_memory[device] / estimated_memory

            continuity_bonus = 0
            for offset in [-1, 1]:
                neighbor_idx = current_layer_order + offset
                if 0 <= neighbor_idx < len(layer_names_in_order):
                    neighbor_name = layer_names_in_order[neighbor_idx]
                    if neighbor_name in device_map and device_map[neighbor_name] == device:
                        continuity_bonus += 1.0

            total_score = memory_score + continuity_bonus
            if total_score > best_score:
                best_score = total_score
                best_device = device

        return best_device or max(gpu_devices, key=lambda d: device_memory[d])

    for layer_idx, (layer_name, mem_info) in enumerate(sorted_layers):
        names.append(layer_name)
        estimated_memory = mem_info["param_memory"] * mem_per_param
        best_device = find_best_device(layer_name, estimated_memory, layer_idx)
        device_map[layer_name] = best_device
        device_memory[best_device] -= estimated_memory

    ordered_device_map = {name: device_map[name] for name in layer_names_in_order if name in device_map}
    return ordered_device_map, names


def get_first_available_attr(obj, attr_names: list[str], default=None):
    """Get the first available attribute from a list of attribute names."""
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)
        if value is not None:
            return value
    return default


def get_moe_memory_ratio(block: torch.nn.Module) -> float:
    """Calculate the memory ratio for MoE models."""
    from auto_round.utils.model import is_moe_layer

    for name, module in block.named_modules():
        if not is_moe_layer(module):
            continue

        config = getattr(block, "config", None)
        if config is None:
            break

        num_experts_per_tok = get_first_available_attr(
            config, ["num_experts_per_tok", "moe_num_active_primary_experts"]
        )

        if num_experts_per_tok is None:
            moe_topk = getattr(config, "moe_topk", None)
            if moe_topk is not None and isinstance(moe_topk, (list, tuple)) and len(moe_topk) > 0:
                num_experts_per_tok = moe_topk[0]
            elif moe_topk is not None:
                num_experts_per_tok = moe_topk

        if num_experts_per_tok is None:
            break

        num_experts = get_first_available_attr(
            config, ["num_local_experts", "num_experts", "moe_num_primary_experts", "n_routed_experts"]
        )

        if num_experts is not None and num_experts > 0:
            moe_ratio = num_experts_per_tok / num_experts
            logger.debug(
                f"MoE detected: {num_experts_per_tok}/{num_experts} experts active per token, "
                f"activation memory ratio: {moe_ratio:.2f}"
            )
            logger.debug(f"Using MoE memory ratio: {moe_ratio:.4f}")
            return moe_ratio, True
        break

    return 1.0, False


def estimate_tuning_block_mem(
    block: torch.nn.Module, input_ids: list[torch.Tensor], batch_size: int
) -> tuple[dict, float]:
    """Calculates the memory consumption of a specific block in the model."""
    from auto_round.utils.model import check_to_quantized, get_layer_features, get_module, is_moe_layer

    layer_memory_dict = {}

    seq_len = input_ids[0].shape[1] if input_ids and len(input_ids[0].shape) >= 2 else 1
    element_size = input_ids[0].element_size() if input_ids else 2

    moe_ratio, has_moe = get_moe_memory_ratio(block)

    for name, module in block.named_modules():
        if check_to_quantized(module):
            enable_act_quant = module.act_bits <= 8
            layer_name = name
            param_size = module.weight.nbytes
            param_memory_gb = param_size / 1024**3
            param_memory_gb *= 2

            in_features, out_features = get_layer_features(module)
            if in_features is not None and out_features is not None:
                output_size = batch_size * seq_len * out_features * element_size
                output_memory_gb = output_size / 1024**3

                if enable_act_quant:
                    input_size = batch_size * seq_len * in_features * element_size
                    input_memory_gb = input_size / 1024**3
                    param_memory_gb += input_memory_gb
            else:
                output_memory_gb = 0.0

            if has_moe:
                pparent_module = get_module(block, layer_name.rsplit(".", 2)[0]) if "." in layer_name else block
                is_moe_expert = "expert" in layer_name.lower() and isinstance(pparent_module, torch.nn.ModuleList)
            else:
                is_moe_expert = False

            layer_memory_dict[layer_name] = {
                "param_memory": param_memory_gb * 2,
                "output_memory": output_memory_gb * 2,
                "is_moe_expert": is_moe_expert,
            }

    block_input_output_memory = 2 * sum(tensor.nbytes for tensor in input_ids) / 1024**3

    layer_activation_memory = 0.0
    for layer_name, info in layer_memory_dict.items():
        if info.get("is_moe_expert", False):
            layer_activation_memory += info["output_memory"] * moe_ratio
        else:
            layer_activation_memory += info["output_memory"]

    additional_memory = layer_activation_memory + 1
    if has_moe:
        moe_additional_memory = additional_memory * 6
        additional_memory += moe_additional_memory

    return layer_memory_dict, layer_activation_memory, block_input_output_memory, additional_memory


def set_auto_device_map_for_block_with_tuning(
    block: torch.nn.Module,
    device_map,
    input_ids: list[torch.Tensor],
    low_gpu_mem_usage: bool = False,
    batch_size: int = 8,
    output_device: str | torch.device = None,
    card_0_threshold: float = 0.9,
):
    """Automatically sets the device map for the block based on available GPUs and memory constraints."""
    from auto_round.utils.model import get_module

    card_0_in_high_risk, loss_device = False, output_device
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_name = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        num_devices = torch.xpu.device_count()
        device_name = "xpu"
    else:
        return card_0_in_high_risk, loss_device

    if not (
        device_map == "auto" or ((isinstance(device_map, str) and "," in device_map)) or num_devices > 1
    ):
        block = block.to(output_device)
        return card_0_in_high_risk, loss_device

    device_list = None
    if isinstance(device_map, str) and "," in device_map:
        device_list = [int(dev) for dev in device_map.split(",") if dev.isdigit()]

    if device_list:
        gpu_devices = [f"{device_name}:{i}" for i in device_list]
        device_0 = gpu_devices[0]
        device_1 = gpu_devices[1]
    else:
        gpu_devices = [f"{device_name}:{i}" for i in range(num_devices)]
        device_0 = f"{device_name}:0"
        device_1 = f"{device_name}:1"

    device_0_memory = get_device_memory(device_list[0] if device_list else 0)
    device_1_memory = get_device_memory(device_list[1] if device_list else 1)
    layer_memory_dict, layer_activation_memory, block_input_output_memory, additional_memory = (
        estimate_tuning_block_mem(block, input_ids, batch_size)
    )
    loss_memory = block_input_output_memory / 2
    if low_gpu_mem_usage:
        block_input_output_memory = 0

    total_block_param_memory = sum(info["param_memory"] for info in layer_memory_dict.values())

    card_0_used_memory = block_input_output_memory + layer_activation_memory + additional_memory
    logger.debug(f"Card 0 used memory details [Estimated]: {card_0_used_memory} GB")
    logger.debug(f"  Block input output cache memory: {block_input_output_memory} GB")
    logger.debug(f"  Quantized layer outputs memory: {layer_activation_memory} GB")
    logger.debug(f"  Additional_memory from other ops: {additional_memory} GB")

    card_0_left_memory = max(0, (device_0_memory - card_0_used_memory))
    card_0_in_high_risk = card_0_used_memory / device_0_memory >= card_0_threshold
    card_1_left_memory = max(0, device_1_memory - loss_memory) if card_0_in_high_risk else device_1_memory
    loss_device = device_1 if card_0_in_high_risk else output_device

    total_available_memory = card_0_left_memory + card_1_left_memory
    for i in range(2, len(gpu_devices)):
        device_idx = device_list[i] if device_list else i
        total_available_memory += get_device_memory(device_idx)

    total_params = total_block_param_memory
    mem_per_param = total_available_memory / total_params

    device_memory = {device_0: card_0_left_memory}
    for i in range(1, len(gpu_devices)):
        device_idx = device_list[i] if device_list else i
        device_memory[gpu_devices[i]] = get_device_memory(device_idx)

    device_map, names = _allocate_layers_to_devices(layer_memory_dict, device_memory, gpu_devices, mem_per_param)

    logger.debug(f"Auto device map for block: {device_map}")
    set_non_auto_device_map(block, device_map, names)

    output_device = device_0 if output_device is None else output_device
    for name, module in block.named_modules():
        if name not in names:
            has_params = any(True for _ in module.parameters(recurse=False))
            has_buffers = any(True for _ in module.buffers(recurse=False))
            if has_params or has_buffers:
                module = module.to(output_device)

    return card_0_in_high_risk, loss_device


def partition_dict_numbers(number_dict, n):
    """Partition a dictionary of numbers into N groups with approximately equal sums."""
    if n > len(number_dict):
        groups = []
        for key, value in number_dict.items():
            groups.append({key: value})
        for _ in range(n - len(number_dict)):
            groups.append({})
        return groups

    if n == len(number_dict):
        return [{key: value} for key, value in number_dict.items()]

    items = list(number_dict.items())
    result = []
    remaining = items.copy()

    def find_optimal_subset(arr, target):
        best_subset = []
        best_diff = float("inf")

        for r in range(1, len(arr) + 1):
            for combo in combinations(arr, r):
                current_sum = sum(value for _, value in combo)
                current_diff = abs(current_sum - target)

                if current_diff == 0:
                    return list(combo)

                if current_diff < best_diff and current_sum <= sum(value for _, value in number_dict.values()):
                    best_diff = current_diff
                    best_subset = list(combo)

        return best_subset

    for i in range(n - 1):
        if not remaining:
            break

        remaining_target = sum(value for _, value in remaining) / (n - i)
        subset = find_optimal_subset(remaining, remaining_target)

        result.append(dict(subset))

        for item in subset:
            remaining.remove(item)

    result.append(dict(remaining))

    return result


def dispatch_model_block_wise(model: torch.nn.Module, device_map: str, max_mem_ratio=0.9):
    """Dispatch model block-wise across devices."""
    if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
        import accelerate
        accelerate.hooks.remove_hook_from_submodules(model)
    no_split_modules = getattr(model, "_no_split_modules", [])
    devices = parse_available_devices(device_map)
    if len(devices) == 1:
        model.to(devices[0])
        return model

    max_memory = get_max_memory()
    new_max_memory = {}
    if "cpu" not in devices:
        devices.append("cpu")
    for device in devices:
        if ":" in device:
            device = int(device.split(":")[-1])
        elif device == "cpu":
            device = "cpu"
        elif isinstance(device, str):
            device = 0
        else:
            raise ValueError(f"Unsupported device {device} in device_map: {device_map}")
        new_max_memory[device] = max_memory[device] * max_mem_ratio
    new_max_memory = get_balanced_memory(
        model,
        max_memory=new_max_memory,
        no_split_module_classes=no_split_modules,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = infer_auto_device_map(model, max_memory=new_max_memory, no_split_module_classes=no_split_modules)
    if len(devices) > 1 and "cpu" in device_map.values():
        logger.warning(
            "Some layers are offloaded to cpu, which may severely impact calibration speed."
            " Please consider using more cards."
        )

    model = dispatch_model(model, device_map=device_map)

    return model


def set_avg_auto_device_map(model: torch.nn.Module, device_map):
    """Set average auto device map for the model."""
    from auto_round.utils.model import get_block_names, get_layer_features, get_module

    block_name_list = get_block_names(model)
    device_list = parse_available_devices(device_map)
    gpu_devices = []
    for device in device_list:
        if device.startswith("hpu") and len(device_list) > 1:
            logger.warning_once("Auto-scheme does not support multiple HPUs.")
        if device.startswith("cpu") or device.startswith("hpu"):
            continue
        gpu_devices.append(device)
    num_devices = len(gpu_devices)
    if num_devices <= 1:
        return

    for block_names in block_name_list:
        for block_name in block_names:
            params_dict = {}
            block_module = get_module(model, block_name)
            for n, m in block_module.named_modules():
                in_features, out_features = get_layer_features(m)
                if in_features is None:
                    continue
                params_dict[n] = in_features * out_features

            res_list = partition_dict_numbers(params_dict, num_devices)
            device_index = 0
            for res in res_list:
                for key in res.keys():
                    set_tuning_device_for_layer(block_module, key, gpu_devices[device_index])
                device_index += 1


def dispatch_model_by_all_available_devices(
    model: torch.nn.Module, device_map: Union[str, int, dict, None]
) -> torch.nn.Module:
    """Dispatch model across all available devices."""
    from auto_round.utils.device.detect import DEVICE_ENVIRON_VARIABLE_MAPPING

    device_type = detect_device()
    if device_type in DEVICE_ENVIRON_VARIABLE_MAPPING:
        existing_env = os.environ.get(DEVICE_ENVIRON_VARIABLE_MAPPING[device_type])
        if existing_env is None:
            logger.warning_once(
                "`get_balanced_memory` is used here, but no environment variable "
                + "is set to specify device visibility. This may lead to OOM issue even the memory "
                + "is large enough."
            )

    is_diffusion_pipeline = False
    try:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline

        if isinstance(model, DiffusionPipeline):
            is_diffusion_pipeline = True
    except ImportError:
        pass
    if is_diffusion_pipeline:
        pipe = model
        _device_map = 0 if device_map is None else device_map
        devices = parse_available_devices(_device_map)
        main_attr = next(
            (attr for attr in ("transformer", "unet") if isinstance(getattr(pipe, attr, None), torch.nn.Module)),
            None,
        )
        if main_attr is None or len(devices) == 1:
            pipe.to(devices[0] if devices else "cuda:0")
            return pipe
        main_model = getattr(pipe, main_attr)
        primary_device = devices[0]
        comp_device = devices[-1]
        for attr, component in pipe.components.items():
            if attr == main_attr:
                continue
            if not isinstance(component, torch.nn.Module):
                continue
            if hasattr(component, "dtype") and component.dtype != main_model.dtype:
                try:
                    component.to(dtype=main_model.dtype)
                except Exception:
                    pass
            try:
                component.to(comp_device)
            except (NotImplementedError, RuntimeError):
                pass

        from auto_round.utils.common import normalize_no_split_modules

        no_split_modules = normalize_no_split_modules(getattr(main_model, "_no_split_modules", []))

        dispatched = dispatch_model_block_wise(main_model, device_map)
        setattr(pipe, main_attr, dispatched)

        unique_devices = set()
        if hasattr(dispatched, "hf_device_map"):
            unique_devices = {v for v in dispatched.hf_device_map.values() if v not in ("cpu", "disk")}
        if len(unique_devices) <= 1:
            execution_device = primary_device
            try:
                execution_device = next(dispatched.parameters()).device
            except StopIteration:
                execution_device = torch.device(primary_device)

            if hasattr(dispatched, "_hf_hook") and hasattr(dispatched._hf_hook, "execution_device"):
                dispatched._hf_hook.execution_device = execution_device

            if not getattr(dispatched, "_autoround_align_inputs_hook_installed", False):
                _first_param_device = execution_device
                _pipeline_device = torch.device(comp_device)

                def _align_all_inputs_pre_hook(module, args, kwargs):
                    try:
                        target = next(module.parameters()).device
                    except StopIteration:
                        target = _first_param_device
                    new_args = tuple(a.to(target) if isinstance(a, torch.Tensor) else a for a in args)
                    new_kwargs = {k: v.to(target) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                    return new_args, new_kwargs

                def _move_outputs_back_hook(module, input, output):
                    def _to_device(obj, device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(device) if obj.device != device else obj
                        if isinstance(obj, (tuple, list)):
                            converted = [_to_device(o, device) for o in obj]
                            return type(obj)(converted)
                        if isinstance(obj, dict):
                            return {k: _to_device(v, device) for k, v in obj.items()}
                        return obj

                    return _to_device(output, _pipeline_device)

                dispatched.register_forward_pre_hook(_align_all_inputs_pre_hook, with_kwargs=True)
                dispatched.register_forward_hook(_move_outputs_back_hook)
                dispatched._autoround_align_inputs_hook_installed = True

        return pipe

    if device_map is None:
        device_map = 0

    from auto_round.utils.common import normalize_no_split_modules

    no_split_modules = normalize_no_split_modules(getattr(model, "_no_split_modules", []))
    if device_map == "auto":
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=no_split_modules,
        )
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_modules)
        model = dispatch_model(model, device_map=device_map)
        return model

    devices = parse_available_devices(device_map)

    if len(devices) == 1:
        model.to(devices[0])
        return model

    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=no_split_modules,
    )

    new_max_memory = {}
    for device in devices:
        if ":" in device:
            device = int(device.split(":")[-1])
        elif device == "cpu":
            device = "cpu"
        elif isinstance(device, str):
            device = 0
        else:
            raise ValueError(f"Unsupported device {device} in device_map: {device_map}")
        new_max_memory[device] = max_memory[device]
    if hasattr(model, "tie_weights") and callable(model.tie_weights):
        model.tie_weights()
    device_map = infer_auto_device_map(model, max_memory=new_max_memory, no_split_module_classes=no_split_modules)
    model = dispatch_model(model, device_map=device_map)
    return model
