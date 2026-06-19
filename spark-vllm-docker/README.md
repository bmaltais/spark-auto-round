# How to run local models with spark-vllm-docker

Scripts for running and testing quantized models locally. Please copy these scripts to your `spark-vllm-docker` directory.

## Quick start

```bash
# Run a model interactively
./run-local-recipe.sh local-qwen3.6-27b-sar

# Run a model + automated benchmark test and save results to a model path
./run-local-test.sh --recipe local-qwen3.6-27b-sar --path ~/test-27b
```

## Prerequisites

- `spark-vllm-docker` installed
- `tool-eval-bench` installed (for `--probe` and `--perf` commands)
- Models in `~/models/`

## Recipes

Recipe YAML files in `recipes/` define how to serve each model:

| Recipe | Model | Notes |
|--------|-------|-------|
| `local-qwen3.6-0.8b` | Qwen3.5-0.8B-int4-AutoRound | for testing |
| `qwen3.6-0.8b` | Qwen/Qwen3.5-0.8B | for comparison |

## Example

Here is an example recipe for running Qwen 3.6 27b

```yaml
# Recipe: Qwen3.6-27B-int4-AutoRound

recipe_version: "1"
name: Qwen3.6-27B-int4-AutoRound
description: vLLM serving Qwen3.6-27B-int4-AutoRound

# No HF download needed - model is already local
# model: null (or omit entirely)

solo_only: true

# Container image to use
container: vllm-node-tf5

build_args:
  - --tf5

mods:
  - mods/fix-qwen3.6-enhanced-chat-template

# Default settings (can be overridden via CLI)
defaults:
  port: 8000
  host: 0.0.0.0
  max_model_len: 128K
  gpu_memory_utilization: 0.60
  max_num_batched_tokens: 32768
  max-num-seqs: 4
  served_model_name: qwen3.6-27b
  generation_config: '{"temperature": 0.7}'
  speculative_mtp: '{"method": "mtp", "num_speculative_tokens": 3}'
  speculative_dflash: '{"method": "dflash", "model":"z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 5}'

# Environment variables
env:
  TORCH_MATMUL_PRECISION: high
  NVIDIA_FORWARD_COMPAT: '1'
  NVIDIA_DISABLE_REQUIRE: '1'
  CUDA_DEVICE_MAX_CONNECTIONS: '1'
  VLLM_MARLIN_USE_ATOMIC_ADD: '1'
  VLLM_ENFORCE_STRICT_TOOL_CALLING: false
  FLASHINFER_DISABLE_VERSION_CHECK: 1
  CUTE_DSL_ARCH: sm_121a
  HF_HUB_OFFLINE: 1
  TRANSFORMERS_OFFLINE: 1

# The vLLM serve command template
command: |
  vllm serve /models/Qwen3.6-27B-int4-AutoRound \
  --served-model-name {served_model_name} \
  --max-model-len {max_model_len} \
  --gpu-memory-utilization {gpu_memory_utilization} \
  --max-num-batched-tokens {max_num_batched_tokens} \
  --max-num-seqs {max-num-seqs} \
  --optimization-level 3 \
  --performance-mode throughput \
  --port {port} \
  --host {host} \
  --load-format fastsafetensors \
  --attention-backend flash_attn \
  --speculative-config '{speculative_mtp}' \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --default-chat-template-kwargs '{{"preserve_thinking":true}}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --generation-config auto \
  --override-generation-config '{generation_config}' \
  --chat-template qwen3.6-enhanced.jinja
```

- `TORCH_MATMUL_PRECISION: high` faster linear layers during the prefill phase without sacrificing overall model accuracy.
- `NVIDIA_FORWARD_COMPAT: 1` Instructs the NVIDIA container use newer CUDA toolkit user-space library.
- `NVIDIA_DISABLE_REQUIRE: 1` Ignore rigid cuda>=12.x mismatch errors on the sm_121 Spark hardware.
- `VLLM_MARLIN_USE_ATOMIC_ADD: 1` ensure the tensor parallelism with unified memory and GPTQ 4-bit model weights
- `CUDA_DEVICE_MAX_CONNECTIONS: 1` DGX spark has 1 GPU
- `VLLM_USE_FLASHINFER_SAMPLER: 1`  improve throughput by sampling natively on the GPU.
- `VLLM_ENFORCE_STRICT_TOOL_CALLING: false` allow Qwen non-json tool call responses.
- `FLASHINFER_DISABLE_VERSION_CHECK=1`  prevent FlashInfer from intentionally crashing due to non-standard build version strings.

**NOTE:** Prefix caching and speculative decoding appear to be incompatible and result in poor test scores. However prefix caching greatly improves latency and time to first token. To achieve the recorded benchmark scores I used `--no-enable-prefix-caching` with `speculative_mtp` and `speculative_dflash` speculative decoding. For production workloads I really need prefix caching enabled so I omit `--speculative-config`.

## Qwen 3.5 and 3.6 Tool Call Fix

This makes tool calling relatively flawless.

Download [chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.6-3.6-chat-template-fix) for Qwen 3.6 and 3.6

- [qwen3.6-enhanced.jinja](https://github.com/allanchan339/vLLM-Qwen3-3.6-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)
- [qwen3.6-enhanced.jinja](https://github.com/allanchan339/vLLM-Qwen3-3.6-3.6-chat-template-fix/blob/main/chat-template/qwen3.6-enhanced.jinja)

For Qwen 3.5 create a mod directory in `spark-vllm-docker/mods/fix-qwen3.5-enhanced-chat-template` with the following files:

- `qwen3.5-enhanced.jinja`
- `run.sh`

```bash
#!/bin/bash
set -e
cp qwen3.5-enhanced.jinja $WORKSPACE_DIR/qwen3.5-enhanced.jinja
echo "=======> to apply chat template, use --chat-template qwen3.5-enhanced.jinja"
```

For Qwen 3.6 create a mod directory in `spark-vllm-docker/mods/fix-qwen3.6-enhanced-chat-template` with the following files:

- `qwen3.6-enhanced.jinja`
- `run.sh`

```bash
#!/bin/bash
set -e
cp qwen3.6-enhanced.jinja $WORKSPACE_DIR/qwen3.6-enhanced.jinja
echo "=======> to apply chat template, use --chat-template qwen3.6-enhanced.jinja"
```

Use either:

`--tool-call-parser qwen3_coder` for OpenCode
`--tool-call-parser qwen3_xml` for other coding harnesses

**IMPORTANT!**


## `run-local-recipe.sh` Script

Starts a vLLM server using a recipe YAML file, with models mounted from a local directory.

```bash
./run-local-recipe.sh <recipe-name> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir <path>` | `~/models` | Directory containing model folders |

**Examples:**

```bash
# Run Qwen3.6-27B with AutoRound quantization
./run-local-recipe.sh local-qwen3.6-27b-sar

# Use a custom models directory
./run-local-recipe.sh local-qwen3.6-27b-sar --model-dir /data/models

# Interactive: container stays running, press Ctrl+C to stop
```

The script:
1. Mounts `~/models/` into the container at `/models/`
2. Lists available models and validates they have `config.json`
3. Starts the vLLM server using the recipe's command template

## `run-local-test.sh` Script

Automated test runner: starts vLLM, waits for it to be ready, runs benchmarks, and cleans up.

```bash
./run-local-test.sh [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --recipe <name>` | `qwen3.6-27b` | Recipe to run |
| `-p, --path <path>` | `~/test-0.8b` | Test output directory |
| `-t, --timeout <secs>` | `300` | Max wait for vLLM to be ready |

**Example:**

```bash
# Full automated test
./run-local-test.sh -r local-qwen3.6-27b-sar -p ~/test-qwen36

# Quick test with shorter timeout
./run-local-test.sh -r local-qwen3.6-27b-sar -t 120
```

The script:
1. Starts vLLM server in background
2. Polls `tool-eval-bench --probe` until ready (or times out)
3. Runs `tool-eval-bench --perf` to collect latency/throughput metrics
4. Stops the container automatically on exit

## `run-test.sh` Script

Same as `run-local-test.sh` but calls `run-recipe.sh` instead of `run-local-recipe.sh` (uses HuggingFace cache instead of local mount).
