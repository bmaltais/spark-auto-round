# spark-auto-round

![Version](https://img.shields.io/badge/version-0.14.2-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![CUDA](https://img.shields.io/badge/CUDA-required-orange)
![GB10](https://img.shields.io/badge/hardware-GB10-purple)

> int4 AutoRound quantization for GB10 hardware

# NOTE

This is new software under active development. I am working my way up from Qwen 0.8b -> 27B -> 35B, 122B -> Gemma etc. I will post updates with verified results on specific models here:  

| Model | Tested | Score | t/s | Quantization Time |
|-------|--------|-------|-----|-------------------|
| Qwen 3.5 0.8b      | ✔︎ | 67 | 205.6 | |
| Qwen 3.6 27b       | ✔︎ | 92 | 26.4  | |
| Qwen 3.6 35b a3b   | ✔︎ | 91 | 64.7  | 11:38 |
| Qwen 3.5 122b a10b |   |    | |
| Gemma 4 12b        |   |    | |

## What is this?

**Spark Auto Round** is an optimally pre-configured int4 AutoRound quantization command line tool that is straightforward to use -- no tweaking necessary. This is a trimmed-down version of Intel's [auto-round](https://github.com/intel/auto-round) focused on **CUDA**, `torch.compile`, and **int4 AutoRound (W4A16)** targeting the **DGX Spark - GB10 128GiB unified memory** architecture.

**Spark ASAQ Substitute** is an experimental companion tool that performs Adaptive Sensitivity-Aware Quantization by taking layer-wise Cosine Similarity, Peak Signal-to-Noise Ratio and for MOE models Router Jaccard Similarity, to replace sensitive layers with FP16 layers from the original model.

## Who it's for?

Intel’s AutoRound works exceptionally well on the DGX Spark and its GB10 siblings. AutoRound has been a popular go-to quantization method because of its combination of memory footprint, vllm support, performance and inference quality. However, the original [auto-round](https://github.com/intel/auto-round) codebase is more of a research project than a production codebase. This fork attempts to provide GB10 users a version of `auto-round` that is focused on their architecture and quality expectations, and tuned for the models they typically run as daily drivers.

## What is AutoRound?

Intel’s AutoRound is a technique used to quantize 16-bit models down to 4-bit. AutoRound uses signed gradient descent to jointly optimize weight rounding and clipping ranges. Mixture-of-Experts models are notoriously sensitive to quantization. AutoRound preserves the “distribution” of the weights rather than just the values, keeping the MoE logic intact even at 4-bit. The weights effectively halve the model size compared to FP8. Subsequently the Blackwell GPU needs less bandwidth to pull these weights from the unified pool. Once they reach the GPU, the Tensor Cores dequantizes iNT4 weights into bfloat16 on-the-fly for the actual math, giving the speed of 4-bit with the precision of 16-bit calculations. int4 AutoRound quantization allows large models to run with ample room for speculative decoding and the KV cache.

## Why not NVFP4?

To run comparative benchmarks and compare and contrast quantized models we need the best version of each quantization technique for reference. This is my attempt to provide the GB10 community with optimal int4 AutoRound models.

## Features

- **Simple CLI**: Easy-to-use command-line interface i.e. `spark-auto-round <model>`
- **GB10 Optimized**: Whole-model quantization with 128GB unified memory, or automatic fallback to block-by-block loading for large models that don't fit in memory
- **torch.compile**: Always enabled for faster quantization on CUDA
- **New Datasets** including OpenCode Instruct and updated Github Code Clean
- **Adaptive Sensitivity-Aware Quantization:** A companion tool that replaces sensitive layers with with fp16 layers from the original model.

## Installation

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install from GitHub
uv pip install git+https://github.com/whpthomas/spark-auto-round.git

# Or for development
git clone https://github.com/whpthomas/spark-auto-round.git
cd spark-auto-round
uv pip install -e .
```

## Quick Start

```bash
spark-auto-round <model>
spark-asaq-substitute <model>
```

The quantized model is saved to `./models/{model}-int4-AutoRound` by default. For example, quantizing `Qwen/Qwen3.6-27B` produces `./models/Qwen3.6-27B-int4-AutoRound/`. The ASAQ model is saved to `./models/{model}-int4-ASAQ` by default.

## Iteratively optimized using Qwen 3.5 0.8b

The dense *Qwen 3.5 0.8B* model was used as a testbed to optimize Spark Auto Round (SAR). Using this [test setup and methodology](docs/optimization.md) we achieved Tool Eval Bench score parity with the unquantized bf16 model. While these results are encouraging, these are complex system and there are many confounding factors that need to be considered. They only demonstrate that for one 0.8B model, optimal settings were found that achieved test score parity with the original bf16 model. Whether these optimal settings generalize to other models requires further research and is under active investigation.

## Performance with Qwen 3.6 27b

Spark auto round repeatedly achieved a [92/100](docs/test-score.md) tool-eval-bench score with the Nvidia's OpenCode Instruct dataset.

- Quantization command: `spark-auto-round --dataset "opencode-instruct" Qwen/Qwen3.6-27B`
- MTP averages ~26.4 t/s with `num_speculative_tokens: 3` for longer context and agentic coding
- DFlash averages ~38.1 t/s with `num_speculative_tokens: 6` for shorter context and instruction following

| # | Model | Scheme | Dataset | Score | t/s | Rating | P/F | Tokens |
|---|-------|--------|---------|-------|-----|--------|-----|--------|
|🥇 | **qwen3.6-27b-sar-oc-mpt** | **int4** | OpenCode Instruct | **92** | 26.4 | ★★★★★ | 59/9/1 | 284K |
|🥈 | qwen3.6-27b-sar-oc-dflash | int4 | OpenCode Instruct | 90 | **38.1** | ★★★★★ | 57/10/2 | 265K |
|🥉 | qwen/qwen3.6-27b-fp8 | fp8 | - | 88 | 18.1 | ★★★★ | 57/8/4 | 275K |
| 4 | qwen3.6-27b-sar-oc | int4 | OpenCode Instruct | 88 | 12.5 | ★★★★ | 57/8/4 | 275K |
| 5 | qwen3.6-27b-sar-git-mtp | int4 | Github Code Clean | 86 | 26.2 | ★★★★ | 54/10/5 | 268K |
| 6 | qwen/qwen3.6-27b | bf16 | - | 83 | 11.4 | ★★★★ | 53/9/7 | 243K |

## Performance with Qwen 3.6 35b a3b

| # | Model            | Scheme  | Dataset | Score | t/s  | Rating | P/F    | Tokens |
|---|------------------|---------|---------|-------|------|--------|--------|--------|
|🥇 | qwen3.6-35b-sar-bf16 | int4 | OpenCode Instruct | 93 | 65.4 | ★★★★★ | 60/8/1 | 275K |
|🥈 | qwen3.6-35b-sar-mtp | int4 | OpenCode Instruct | 91 | 78.7 | ★★★★★ | 58/10/1 | 272K |
|🥉 | qwen3.6-35b-sar  | int4    | OpenCode Instruct | 91 | 64.7 | ★★★★★ | 59/8/2 | 283K |
| 4 | qwen3.6-35b-sar-pc | int4  | OpenCode Instruct | 91 | 64.5 | ★★★★★ | 59/8/2 | 284K |
| 5 | qwen/qwen3.6-35b | bf16    | - | 91 | 28.3 | ★★★★★ | 59/8/2 | 292K |
| 6 | qwen/qwen3.6-35b-fp8 | fp8 | - | 90 | 48.0 | ★★★★★ | 58/8/3 | 264K |
| 7 | qwen3.6-35b-sar-dflash | int4 | OpenCode Instruct | 88 | 97.1 | ★★★★ | 55/11/3 | 282K |
| 8 | qwen3.6-35b-sar-pc-mtp-bf16 | int4 | OpenCode Instruct | 47 | 73.4 | ★★ | 28/9/32 | 136K |
| 9 | qwen3.6-35b-sar-pc-mtp | int4 | OpenCode Instruct | 44 | 109.1 | ★★ | 26/9/34 | 131K |
|10 | qwen3.6-35b-sar-pc-dflash | int4 | OpenCode Instruct | 43 | 102.3 | ★★ | 25/9/35 | 133K |

### Legend

- `-sar` Spark Auto Round
- `-pc` Prefix Caching
- `-mtp` MTP=2 Speculative Decoding (MTP=1 with `-pc`)
- `-dflash` DFlash=5 Speculative Decoding
- `-bf16` bfloat16 data type

### Scripts and Recipes

For transparency and convenience, and so my results can be independently replicated and verified. All test scripts and recipes are shared in the [spark-vllm-docker/](spark-vllm-docker) sub-directory. These can be used with the DGX [Spark vllm docker](https://github.com/eugr/spark-vllm-docker) community supported tool.

### Examples

```bash
# Optimal quantization
spark-auto-round Qwen/Qwen3.6-27B

# Fast parameters
spark-auto-round Qwen/Qwen3.5-122B-A10B \
    --iters 200 \
    --nsamples 128 \
    --output_dir ./models/fast

# Disable torch.compile (if causing issues)
spark-auto-round Qwen/Qwen3.6-35B-A3B --disable_torch_compile

# Perform adaptive sensitivity-aware quantization
spark-asaq-substitute Qwen/Qwen3.6-27B
```

## CLI Reference

```
spark-auto-round <model> [options]
```

### Basic Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model` | (required) | Model path or HuggingFace model ID |
| `--group_size` | 128 | Group size for weight quantization |
| `--iters` | 1000 | Tuning iterations per block |
| `--nsamples` | 512 | Number of calibration samples |
| `--seqlen` | 2048 | Calibration sequence length |
| `--batch_size` | 8 | Calibration batch size |
| `--output_dir` | ./models | Output directory |
| `--dataset` | opencode-instruct | Calibration dataset |
| `--disable_torch_compile` | (disabled) | Disable torch.compile (enabled by default) |

### Tuning Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | auto | Learning rate (auto-calculated if not set) |
| `--minmax_lr` | auto | MinMax learning rate (uses --lr if not set) |

### Scheme Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--quant_lm_head` | false | Quantize the lm_head layer |
| `--ignore_layers` | "" | Layers to skip (comma-separated) |
| `--layer_config` | null | Per-layer config JSON |

### Other Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dtype` | null | Model dtype for loading |
| `--seed` | 42 | Random seed |
| `--adam` | false | Use Adam optimizer |
| `--mllm` | false | Force multimodal mode |

## Datasets

| Alias | Content | Notes |
|-------|---------|-------------|
| **[opencode-instruct](https://huggingface.co/nvidia/OpenCodeInstruct)** | Code instructions + responses | Packs short sequences; best for coding models |
| [github-code-clean](https://huggingface.co/codeparrot/github-code-clean) | Source code | Downloads random parquet shards for diversity |
| [pile-10k](https://huggingface.co/NeelNanda/pile-10k) | English general text | Classic calibration dataset |
| [CCI3-HQ](https://huggingface.co/BAAI/CCI3-HQ) | Chinese web text | Streaming; good for Chinese models |
| [pile-val-backup](https://modelscope.cn/datasets/swift/pile-val-backup) | English general text | Requires `pip install modelscope` |
| [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Chat dialogues | Splits: `train_sft`, `test_sft` |
| [Ultra-FineWeb](https://huggingface.co/openbmb/Ultra-FineWeb) | Web pages | Splits: `en`, `zh` |
| [mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | Python problems + code | Text + code concatenated |
| [AudioCaps](https://github.com/cdjkim/audiocaps) | Sound/music captions | Good for audio-related models |
| [new-title-chinese](https://huggingface.co/datasets/madao33/new-title-chinese) | Chinese news headlines | For Chinese NLP tasks |

## Supported Format

- `auto_round` (default) — HuggingFace-compatible format using `auto_round:auto_gptq` backend

## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA GPU required (DGX Spark GB10 recommended)
- 128 GB unified memory recommended for large models

Quantization runs on single GB10 GPU — there is no CPU fallback. The CLI hardcodes `device=cuda:0`.

## License

Apache License 2.0

## Spark-Auto-Round Contributions

- [@whpthomas](https://github.com/whpthomas)

## Acknowledgments

Based on [auto-round](https://github.com/intel/auto-round) by Intel.

## References

- [auto-round](https://github.com/intel/auto-round) - *Advanced quantization toolkit designed for Large Language Models*
- [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) - *Docker configuration and startup scripts to run vLLM on DGX Spark*
- [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench/) - *A tool-calling quality benchmark for evaluating LLM tool-use in agentic workflows*
- [chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) - *Stable tool calling with enhanced chat template*