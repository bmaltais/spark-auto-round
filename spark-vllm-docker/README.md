# How to run local models with spark-vllm-docker

Scripts for running and testing quantized models locally with vLLM in Docker on DGX Spark.

## Quick start

```bash
# Run a model interactively (starts vLLM, keeps container running)
./run-local-recipe.sh local-qwen3.5-0.8b-ar --solo

# Run a model + automated benchmark test
./run-local-test.sh -r local-qwen3.5-0.8b-ar
```

## Prerequisites

- `spark-vllm-docker` installed
- `tool-eval-bench` installed (for `--probe` and `--perf` commands)
- Models in `~/models/`

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
# Run Qwen3.5-0.8B with AutoRound quantization
./run-local-recipe.sh local-qwen3.5-0.8b-ar

# Use a custom models directory
./run-local-recipe.sh local-qwen3.5-0.8b-ar --model-dir /data/models

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
| `-r, --recipe <name>` | `qwen3.5-0.8b` | Recipe to run |
| `-p, --path <path>` | `~/test-0.8b` | Test output directory |
| `-t, --timeout <secs>` | `300` | Max wait for vLLM to be ready |

**Example:**

```bash
# Full automated test
./run-local-test.sh -r local-qwen3.5-0.8b-ar -p ~/test-qwen35

# Quick test with shorter timeout
./run-local-test.sh -r local-qwen3.5-0.8b-ar -t 120
```

The script:
1. Starts vLLM server in background
2. Polls `tool-eval-bench --probe` until ready (or times out)
3. Runs `tool-eval-bench --perf` to collect latency/throughput metrics
4. Stops the container automatically on exit

## `run-test.sh` Script

Same as `run-local-test.sh` but calls `run-recipe.sh` instead of `run-local-recipe.sh` (uses HuggingFace cache instead of local mount).

## Recipes

Recipe YAML files in `recipes/` define how to serve each model:

| Recipe | Model | Notes |
|--------|-------|-------|
| `local-qwen3.5-0.8b-ar` | Qwen3.5-0.8B | AutoRound INT4 quantization |
| `local-qwen3.5-0.8b-git` | Qwen3.5-0.8B | Git version (reference) |
| `local-qwen3.5-0.8b-oc` | Qwen3.5-0.8B | OpenCanvas version |
| `local-qwen3.5-0.8b-pile` | Qwen3.5-0.8B | Pile-trained version |

