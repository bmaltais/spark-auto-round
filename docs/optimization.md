# Optimization

We download an use Int4 AutoRound models form [Hugging Face](https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound), but very little knowledge of their provenance. So when we make comparisons with other quantization techniques, how do we know whether the model we are using is optimal for our particular use case?

To run comparative benchmarks and compare and contrast quantized models we need the best version of each quantization technique for reference.

## Methodology

The dense *Qwen 3.5 0.8B* model was used as a testbed to optimize Spark AutoRound (SAR). The *Qwen 3.5 0.8B* model was chosen because its recent, and within the family of models that are commonly used for inference on teh DGX spark. It also takes about 45 minutes to quantize, making it a practical choice for iterative testing. With each iteration SAR was invoked with the following command:

```bash
spark-auto-round Qwen/Qwen3.5-0.8B
```

The program generates a quantization report in the model directory that highlights sensitive layers.

```txt
Sensitivity Analysis:
─────────────────────────────────────────────────────────────────────────────────────────────────────
Layer                                    Cosine Sim  PSNR (dB)      Iters              uLoss   Status
─────────────────────────────────────────────────────────────────────────────────────────────────────
🟢 model.layers.0                               0.9965       51.1      947        63.87 → 8.49   PASS
🟢 model.layers.1                               0.9966       52.2      947       57.59 → 12.02   PASS
🟢 model.layers.2                               0.9962       51.8      766       59.71 → 15.20   PASS
🟢 model.layers.3                               0.9953       48.9      758       81.82 → 22.47   PASS
🟢 model.layers.4                               0.9955       48.2     1000       65.34 → 23.74   PASS
🟢 model.layers.5                               0.9952       47.8      862       71.02 → 23.95   PASS
🟢 model.layers.6                               0.9950       57.0      950       74.40 → 29.93   PASS
🟢 model.layers.7                               0.9939       50.7      748      100.77 → 34.33   PASS
🟢 model.layers.8                               0.9939       50.9      990       67.47 → 32.36   PASS
🟢 model.layers.9                               0.9937       50.1      978       66.30 → 38.22   PASS
🟢 model.layers.10                              0.9939       50.8      898       68.12 → 36.67   PASS
🟢 model.layers.11                              0.9922       45.0      872      108.81 → 52.56   PASS
🟠 model.layers.12                              0.9926       44.6      776      103.97 → 59.33   WARN
🟢 model.layers.13                              0.9932       45.0      952      124.58 → 71.75   PASS
🟢 model.layers.14                              0.9932       52.4      942     193.11 → 109.75   PASS
🟠 model.layers.15                              0.9923       43.4      952     385.22 → 181.85   WARN
🟠 model.layers.16                              0.9929       44.4      753     443.40 → 232.46   WARN
🟢 model.layers.17                              0.9931       45.1      886     463.46 → 278.33   PASS
🟢 model.layers.18                              0.9935       46.5      949     503.57 → 352.99   PASS
🟢 model.layers.19                              0.9932       46.0      809     953.13 → 494.19   PASS
🟢 model.layers.20                              0.9933       46.0      821     878.47 → 609.21   PASS
🟢 model.layers.21                              0.9926       45.8      888    1235.61 → 777.71   PASS
🟠 model.layers.22                              0.9909       44.5      658   1901.12 → 1121.81   WARN
🟠 model.layers.23                              0.9872       46.1      233   3695.04 → 1959.51   WARN
```

The model was then served with `vllm` using [Spark vllm Docker](https://github.com/eugr/spark-vllm-docker) using the following bash script:

`./run-0.8b-sar.sh`

```bash
#!/bin/bash

docker run -it --name vllm-qwen35 \
    --gpus all --net=host --ipc=host \
    -v ~/models:/models \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e TORCH_MATMUL_PRECISION=high \
    -e NVIDIA_FORWARD_COMPAT=1 \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    vllm-node-tf5 \
    bash -c -i "vllm serve /models/Qwen3.5-0.8B-int4-AutoRound \
    --served-model-name qwen/qwen3.5-0.8b-sar \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 192K \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 16 \
    --load-format fastsafetensors \
    --dtype auto \
    --quantization modelopt \
    --kv-cache-dtype auto \
    --generation-config auto \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --override-generation-config '{\"temperature\": 0}' \
    --attention-backend flash_attn \
    --limit-mm-per-prompt '{\"image\": 4, \"video\": 2}' \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --chat-template /models/qwen3.6-enhanced.jinja \
    --reasoning-parser qwen3"

#    --speculative-config '{\"method\": \"dflash\", \"model\": \"/models/Qwen3.6-27B-DFlash\", \"num_speculative_tokens\": 5}' \
#    --speculative-config '{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": 3}' \

docker container remove vllm-qwen35
```

**NOTE**

SAR data types have been internally optimized so avoid using the following `vllm` parameters, they actually degrade test scores and inference quality.

```
  --kv-cache-dtype fp8
  --kv-cache-dtype fp8_e4m3
  --dtype bfloat16
```

[Tool Eval Bench](https://github.com/SeraphimSerapis/tool-eval-bench) was then used to evaluate the quantization results. The objective was to get SAR as close to matching the score of the unquantized BF16 model as possible.

## Hardcoded Optimal Values

The following values were found to be optimal for the GB10 arthitecture:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Device | `cuda:0` | GB10 target platform |
| Format | `auto_round` | Only relevant format |
| Scheme | `W4A16` | 4-bit symmetric INT weights, 16-bit activations |
| Platform | `hf` | Only HuggingFace models |
| Scale dtype | `bf16` | Optimal for GB10 |
| AMP | enabled | Always enabled for performance |
| MinMax tuning | enabled | Always enabled for quality |
| Quanted input | enabled | Always enabled for accuracy |
| Best MSE | enabled | Always enabled for quality |

## Results

With the test setup and methodology we achieved Tool Eval Bench score parity with the unquantized bf16 model.

| # | Model | Scheme | Dataset | Score | Rating | P/F | Tokens | Runs |
|---|-------|--------|---------|-------|--------|-----|--------|------|
|🥇 | **qwen3.5-0.8b-sar** | **Int4** | OpenCode Instruct | **69** | ★★★  | 41/13/15 | 516K | 3 |
|🥈 | qwen3.5-0.8b-sar | Int4 | github-code-clean | 67 | ★★★  | 39/14/16 | 516K | 3 |
|🥉 | **qwen3.5-0.8b** | **bf16** | - | **67** | ★★★  | 40/13/16 | 571K | 3 |
| 4 | qwen3.5-0.8b-ar | Int4 | pile-10k | 62 | ★★★ⓢ | 37/11/21 | 486K | 4 |
| 5 | qwen3.5-0.8b-sar | Int4 | pile-10k | 62 | ★★★  | 37/11/21 | 537K | 11 |

- `-sar` Spark AutoRound
- `-ar` Intel AutoRound

## Conclusion

These results should **NOT** be interpreted to mean that Spark Auto Round quantized models are equivalent bf16. It only demonstrates that for one 0.8B model, optimal settings were found that achieved test score parity with the original bf16 model. While these results are encouraging, whether these optimal settings generalize to other models requires further research and is under active investigation.
