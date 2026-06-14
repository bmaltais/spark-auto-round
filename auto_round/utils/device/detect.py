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
"""Device detection and package availability utilities."""
import os
import re
from contextlib import ContextDecorator
from functools import lru_cache
from typing import Union

import cpuinfo
import torch

from auto_round.logger import logger


DEVICE_ENVIRON_VARIABLE_MAPPING = {
    "cuda": "CUDA_VISIBLE_DEVICES",
}


def is_package_available(package_name: str) -> bool:
    """Check if the package exists in the environment without importing."""
    from importlib.util import find_spec

    package_spec = find_spec(package_name)
    return package_spec is not None


def is_autoround_exllamav2_available():
    """Checks if the AutoRound ExLlamaV2 kernels are available."""
    try:
        from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
    except ImportError:
        return False
    return True


def check_is_cpu(device):
    """Check if the device is a CPU."""
    return device == torch.device("cpu") or device == "cpu"


def detect_device_count():
    """Detects the number of available CUDA devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def detect_device(device: Union[None, str, int, torch.device] = None) -> str:
    """Detects the appropriate computation device.

    Args:
        device (str, int, or torch.device, optional): The desired device.

    Returns:
        str: The device to use for computations.
    """

    def is_valid_digit(s):
        try:
            num = int(s)
            return 0 <= num
        except Exception:
            return False

    dev_idx = None
    if is_valid_digit(device):
        dev_idx = int(device)
        device = "auto"
    if isinstance(device, str) and "," in device:
        device_list = [int(dev) for dev in device.split(",") if dev.isdigit()]
        dev_idx = device_list[0] if device_list else None
        device = "auto"
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if dev_idx is not None and str(device) != "cpu":
            device = str(device) + f":{dev_idx}"
        return str(device)
    elif isinstance(device, torch.device):
        device = str(device)
    elif isinstance(device, str):
        if device == "tp":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = device
    return device


def get_device_and_parallelism(device: Union[str, torch.device, int, dict]) -> tuple[str, bool]:
    """Get device string and whether parallelism is needed."""
    if device is None:
        device = detect_device(device)
        return device, False
    if isinstance(device, dict):
        unique_devices = set(device.values())
        if len(unique_devices) == 1:
            device = next(iter(unique_devices))
        else:
            device = "auto"
    if isinstance(device, str):
        if device in ["cuda"]:
            device = detect_device(device)
            parallelism = False
            return device, parallelism
        else:
            device = re.sub("cuda:", "", device)
            devices = device.replace(" ", "").split(",")
    elif isinstance(device, int):
        devices = [str(device)]
    else:
        devices = [device]
    if all(s.isdigit() for s in devices) and len(devices) > 1 and torch.cuda.is_available():
        device = "cuda"
        parallelism = True
    elif device == "auto":
        device = detect_device(device)
        parallelism = True
    else:
        device = detect_device(device)
        parallelism = False
    return device, parallelism


def set_cuda_visible_devices(device: str):
    """Set CUDA_VISIBLE_DEVICES environment variable."""
    if device == "cuda":
        devices = ["0"]
    elif device == "auto":
        return
    else:
        devices = device.replace(" ", "").split(",")
    devices = [device.split(":")[-1] for device in devices]
    if all(s.isdigit() for s in devices):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            current_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            current_visible_devices = current_visible_devices.split(",")
            indices = [int(device) for device in devices]
            try:
                pick_device = [current_visible_devices[i] for i in indices]
            except Exception:
                raise ValueError(
                    "Invalid '--device' value: It must be smaller than the number of available devices."
                    " For example, with CUDA_VISIBLE_DEVICES=4,5, "
                    "--device 0,1 is valid, but --device 4,5 is not supported."
                )
            visible_devices = ",".join(pick_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device


class override_cuda_device_capability(ContextDecorator):
    """Context manager/decorator to temporarily override CUDA capability checks."""

    def __init__(self, major: int = 100, minor: int = 1) -> None:
        self.major = major
        self.minor = minor
        self._orig_func = None

    def __enter__(self):
        self._orig_func = torch.cuda.get_device_capability

        def _override_capability(*_args, **_kwargs):
            return self.major, self.minor

        torch.cuda.get_device_capability = _override_capability
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self._orig_func is not None:
            torch.cuda.get_device_capability = self._orig_func
            self._orig_func = None
        return False


def get_packing_device(device: str | torch.device | None = "auto") -> torch.device:
    """Selects the packing device."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        try:
            return torch.device(device)
        except Exception as e:
            raise ValueError(f"Invalid device string: {device}") from e

    raise TypeError(f"Unsupported device type: {type(device)} ({device})")


def is_auto_device_mapping(device_map: str | int | dict | None):
    """Check if device_map indicates auto device mapping."""
    if device_map is None or isinstance(device_map, int):
        return False
    elif device_map == "auto":
        return True
    elif isinstance(device_map, str) and "," in device_map:
        return True
    elif isinstance(device_map, dict):
        return False
    else:
        return False


class CpuInfo(object):
    """Get CPU Info."""

    def __init__(self):
        self._bf16 = False
        info = cpuinfo.get_cpu_info()
        if "arch" in info and "X86" in info["arch"]:
            cpuid = cpuinfo.CPUID()
            max_extension_support = cpuid.get_max_extension_support()
            if max_extension_support >= 7:
                eax = cpuid._run_asm(
                    b"\xb9\x01\x00\x00\x00",
                    b"\xb8\x07\x00\x00\x00" b"\x0f\xa2" b"\xc3",
                )
                self._bf16 = bool(eax & (1 << 5))

    @property
    def bf16(self):
        return self._bf16


@torch._dynamo.disable()
@lru_cache(None)
def is_hpex_available():
    """Check if HPU (Habana) is available. Always False in this CUDA-only fork."""
    return False


@lru_cache(maxsize=None)
def is_gaudi2():
    """Gaudi is not supported in this CUDA-only fork."""
    return False


def parse_available_devices(device_map: Union[str, torch.device, int, dict, None] = None) -> list:
    """Parse the device map and return a list of all available devices."""
    device_types = []
    if torch.cuda.is_available():
        device_types.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_types.append("xpu")
    if hasattr(torch, "hpu") and is_hpex_available():
        device_types.append("hpu")

    if not device_types:
        device_types = ["cpu"]

    if device_map is None:
        if "cuda" in device_types:
            return ["cuda:0"]
        elif "xpu" in device_types:
            return ["xpu:0"]
        elif "hpu" in device_types:
            return ["hpu:0"]
        else:
            return ["cpu"]

    if isinstance(device_map, torch.device):
        dev_type = device_map.type
        index = device_map.index
        if dev_type == "cpu":
            return ["cpu"]
        if index is None:
            index = 0
        return [f"{dev_type}:{index}"]

    if isinstance(device_map, int):
        device_type = device_types[0]
        return [f"{device_type}:{device_map}"] if device_type != "cpu" else ["cpu"]

    if isinstance(device_map, str) and ":" in device_map and "," in device_map:
        pairs = [p.strip() for p in device_map.split(",") if ":" in p]
        devices = []
        for pair in pairs:
            try:
                key, *value_parts = pair.split(":")
                value = ":".join(value_parts).strip()
                if value.isdigit() and device_types[0] != "cpu":
                    value = device_types[0] + ":" + value
                devices.append(value)
            except ValueError:
                continue
        return devices

    if isinstance(device_map, str):
        device_map = device_map.strip()
        if device_map.lower() == "cpu":
            return ["cpu"]
        if device_map.lower() == "auto":
            device_count = detect_device_count()
            if "cuda" in device_types:
                return [f"cuda:{i}" for i in range(device_count)]
            elif "xpu" in device_types:
                return [f"xpu:{i}" for i in range(device_count)]
            elif "hpu" in device_types:
                return [f"hpu:{i}" for i in range(device_count)]
            else:
                return ["cpu"]
        parts = [x.strip() for x in device_map.split(",") if x.strip()]
        parsed = []
        for p in parts:
            if p.isdigit():
                device_type = device_types[0]
                parsed.append(f"{device_type}:{p}" if device_type != "cpu" else "cpu")
            elif p in device_types and ":" not in p:
                if p.lower() == "cpu":
                    parsed.append("cpu")
                else:
                    parsed.append(f"{p}:0")
            else:
                parsed.append(p)
        return list(set(parsed))

    if isinstance(device_map, dict):
        devices = set()
        for v in device_map.values():
            devices.update(parse_available_devices(v))
        return sorted(devices)

    raise TypeError(f"Unsupported device_map type: {type(device_map)}")
