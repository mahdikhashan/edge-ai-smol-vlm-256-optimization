## SmolVLM-256M Edge Inference Optimization

Faster inference for SmolVLM-256M on edge GPUs, without custom CUDA or Triton kernels. The repository also reproduces the benchmark of the EVLM power line inspection paper (Farahani et al., EDGE 2026). Tests ran on a Colab GPU.

### Setup

Make sure a CUDA GPU and Python 3 are available, then clone the repository:

```bash
git clone https://github.com/mahdikhashan/edge-ai-smol-lvm-250-optimization
cd edge-ai-smol-lvm-250-optimization
```

Install the packages:

```bash
pip install "transformers>=4.56" pillow numpy pandas onnx tensorrt
```

On the Jetson, TensorRT comes with JetPack, so only `onnx` is needed.

### Notebooks

Open a notebook in `notebooks/` and run it from the top. Each one is independent.

- `smolvlm_256m_cuda_graph_decoding.ipynb`: one image with CUDA-graph decoding.
- `smolvlm_256m_benchmark.ipynb`: bare model, eager and graph decoding, stop at the end token, image preparation, and TensorRT.
- `smolvlm_256m_attention_check.ipynb`: attention implementations.
- `smolvlm_256m_hybrid_attention.ipynb`: `sdpa` for the prefill and `eager` for the decode.
- `smolvlm_256m_gpu_preprocessing.ipynb`: image processors on the CPU and the GPU.
- `smolvlm_256m_eos_stop.ipynb`: when to stop the decode loop.
- `smolvlm_256m_prefill_profile.ipynb`: prefill time of each part of the model.
- `smolvlm_256m_trt_siglip.ipynb`: TensorRT engine for the SigLIP encoder.
- `smolvlm_256m_tile_sweep.ipynb`: fewer image tiles.
- `smolvlm_256m_gguf_q4.ipynb`: 4-bit GGUF model with llama.cpp (issue #1).
- `evlm/evlm.ipynb`: EVLM reproduction on the paper datasets (about 1.4 GB of data).

---

### Results

On one Colab GPU:

- one frame with a 9-token answer: 159 ms with the best setup (5 tiles, fast GPU preprocessing, CUDA-graph decoding, stop at EOS, TensorRT SigLIP). The bare model takes about 954 ms (estimate).
- EVLM benchmark: each VLM call is 6.1 times faster than a paper-like baseline (221 ms and 1355 ms).

### Documentation

- [Files and notebooks](docs/FILES.md)
- [Optimization ranking](docs/RANKING.md)
- [Prefill profile](docs/PREFILL_PROFILE.md)
- [TensorRT engine for SigLIP](docs/TENSORRT.md)
- [EVLM benchmark and ablation](docs/EVLM.md)
- [Changelog](CHANGELOG.md)
