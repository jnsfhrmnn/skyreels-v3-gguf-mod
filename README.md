# SkyReels-V3 - GGUF Optimized Fork

Fork of [SkyworkAI/SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3) with GGUF quantized model support and performance optimizations for consumer GPUs.

## What's New in This Fork

### GGUF Quantized Model Support
- **On-the-fly dequantization**: Weights stay quantized in VRAM, dequantized per-layer during forward pass
- **Supported quantization types**: Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0
- **GGUF + `--low_vram` combo**: Block offloading for higher resolutions with quantized models
- **Smart offloading**: VAE and Text Encoder automatically offloaded to CPU when GGUF is active

### SageAttention Backend
- Added [SageAttention 2.x](https://github.com/thu-ml/SageAttention) as attention backend
- Priority: FlashAttention > SageAttention > SDPA (PyTorch default)
- Auto-detected at runtime

### Additional
- 1080P resolution config added

## Performance Benchmarks

Tested on **RTX 5090 32GB**, Windows 11, PyTorch 2.8.0+cu128, 8 inference steps.

### 480P (832x464, 5s duration)

| Mode | Model Size | Per Step | Total | Speedup |
|------|-----------|----------|-------|---------|
| `--low_vram` (FP8 + block offload) | ~14 GB | ~43s | 5:46 | 1x (baseline) |
| `--gguf Q8_0` | ~15.9 GB | ~37s | 6:06 | 0.95x |
| `--gguf Q6_K` | ~12 GB | ~17.5s | 2:19 | **2.5x** |
| `--gguf Q4_K_M` | ~8.5 GB | ~17.8s | 2:22 | **2.4x** |

### 720P (1312x688, 5s duration)

| Mode | Per Step | Total | Speedup |
|------|----------|-------|---------|
| `--low_vram` (FP8 + block offload) | ~174s | 23:14 | 1x (baseline) |
| `--gguf Q4_K_M --low_vram` | ~93s | 12:23 | **1.9x** |

> Q6_K at 720P requires more than 32GB VRAM with block offloading.

### Sample Outputs

All samples generated with prompt: *"A cat walking slowly across a garden"*, seed 42, 8 steps.

| Sample | Settings |
|--------|----------|
| [480P Q4_K_M](samples/480p_Q4_K_M_v2v.mp4) | 480P, Q4_K_M, 5s, v2v — **2:22 total** |
| [480P Q6_K](samples/480p_Q6_K_v2v.mp4) | 480P, Q6_K, 5s, v2v — **2:19 total** |
| [480P Q8_0](samples/480p_Q8_0_v2v.mp4) | 480P, Q8_0, 5s, v2v — **6:06 total** |
| [720P Q4_K_M](samples/720p_Q4_K_M_low_vram_v2v.mp4) | 720P, Q4_K_M + low_vram, 5s, v2v — **12:23 total** |

### Key Findings
- **Q4_K_M and Q6_K deliver nearly identical speed** at 480P despite different sizes
- At 480P, the bottleneck shifts from memory bandwidth to compute
- At 720P, Q4_K_M with block offloading is viable; Q6_K is too large
- Q8_0 is slower than FP8 `--low_vram` due to larger model size without block offloading benefit

## Usage

### Prerequisites

```bash
pip install gguf
# Optional: SageAttention for slight speedup
# pip install sageattention  (or install from wheel for Windows)
```

### Download GGUF Models

GGUF models are provided by [vantagewithai/SkyReels-V3-14B-GGUF](https://huggingface.co/vantagewithai/SkyReels-V3-14B-GGUF):

```bash
# Video Extension (v2v)
huggingface-cli download vantagewithai/SkyReels-V3-14B-GGUF v2v/SkyReels-v3-v2v-Q4_K_M.gguf

# Reference to Video (r2v)
huggingface-cli download vantagewithai/SkyReels-V3-14B-GGUF r2v/SkyReels-v3-r2v-Q4_K_M.gguf
```

### Generate Video with GGUF

**480P (recommended for 24GB+ VRAM, no block offloading needed):**
```bash
python generate_video.py \
  --task_type single_shot_extension \
  --gguf path/to/SkyReels-v3-v2v-Q4_K_M.gguf \
  --resolution 480P \
  --duration 5 \
  --prompt "A cat walking slowly across a garden"
```

**720P (requires `--low_vram` for block offloading on 32GB):**
```bash
python generate_video.py \
  --task_type single_shot_extension \
  --gguf path/to/SkyReels-v3-v2v-Q4_K_M.gguf \
  --low_vram \
  --resolution 720P \
  --duration 5 \
  --prompt "A cat walking slowly across a garden"
```

### Recommended Settings by VRAM

| VRAM | Resolution | Command |
|------|-----------|---------|
| 24 GB | 480P | `--gguf Q4_K_M` |
| 32 GB | 480P | `--gguf Q4_K_M` or `Q6_K` (fastest) |
| 32 GB | 720P | `--gguf Q4_K_M --low_vram` |

## How GGUF Loading Works

1. Model skeleton created on `meta` device (zero memory)
2. GGUF tensors read and categorized:
   - **Quantized weights** (Linear/Conv3d): replaced with `GGMLLinear`/`GGMLConv3d` modules
   - **Direct parameters** (norms, embeddings, modulations): loaded as regular tensors
3. During forward pass: quantized weights dequantized on-the-fly per layer
4. VAE and Text Encoder offloaded to CPU, loaded to GPU only when needed

## Modified Files

| File | Changes |
|------|---------|
| `skyreels_v3/modules/gguf_loader.py` | **New** - GGUF loading and dequantization (adapted from ComfyUI-GGUF, Apache-2.0) |
| `generate_video.py` | Added `--gguf` CLI argument |
| `skyreels_v3/modules/__init__.py` | GGUF path handling in `get_transformer()` |
| `skyreels_v3/modules/attention.py` | SageAttention backend support |
| `skyreels_v3/config.py` | 1080P resolution config |
| `skyreels_v3/pipelines/*.py` | GGUF support + VAE/TE offloading in all 4 pipelines |

## Credits & License

- **Original model & code**: [SkyworkAI/SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3) - [Skywork Community License](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20Community%20License.pdf)
- **GGUF quantized models**: [vantagewithai/SkyReels-V3-14B-GGUF](https://huggingface.co/vantagewithai/SkyReels-V3-14B-GGUF)
- **GGUF dequantization logic**: Adapted from [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0)
- **SageAttention**: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)

This fork is distributed under the same [Skywork Community License](https://github.com/SkyworkAI/Skywork/blob/main/Skywork%20Community%20License.pdf) as the original repository. Commercial use is permitted under the license terms.
