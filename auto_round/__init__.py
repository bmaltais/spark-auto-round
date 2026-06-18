# Copyright (c) 2023 Intel Corporation
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
from auto_round.autoround import AutoRound
from auto_round.schemes import QuantizationScheme
from auto_round.utils import LazyImport
from auto_round.utils import monkey_patch
# WARNING: monkey_patch() modifies transformers internals at import time.
# This includes patching AutoModelForCausalLM and related classes.
# See AGENTS.md for details.
monkey_patch()

from .version import __version__
