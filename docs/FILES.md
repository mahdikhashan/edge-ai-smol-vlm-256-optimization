## Files and notebooks

| File | Use |
|---|---|
| `notebooks/smolvlm_256m_cuda_graph_decoding.ipynb` | One image with CUDA-graph decoding. Prints the answer. |
| `notebooks/smolvlm_256m_benchmark.ipynb` | Bare model, eager decoding, graph decoding, stop at the end token, and the image preparation time. Also runs with TensorRT. |
| `notebooks/smolvlm_256m_attention_check.ipynb` | Attention implementations: default, `eager`, `sdpa`, and `sdpa` with the eager mask. |
| `notebooks/smolvlm_256m_hybrid_attention.ipynb` | `sdpa` for the prefill and `eager` for the decode graph. |
| `notebooks/smolvlm_256m_gpu_preprocessing.ipynb` | Slow, fast-CPU and fast-GPU image processors. |
| `notebooks/smolvlm_256m_eos_stop.ipynb` | When to stop the decode loop. |
| `notebooks/smolvlm_256m_prefill_profile.ipynb` | Prefill split into SigLIP, connector, decoder and LM head. |
| `notebooks/smolvlm_256m_trt_siglip.ipynb` | TensorRT engine for the SigLIP encoder. |
| `notebooks/smolvlm_256m_tile_sweep.ipynb` | Fewer image tiles: speed and answer. |
| `notebooks/smolvlm_256m_gguf_q4.ipynb` | 4-bit GGUF model with llama.cpp. |
| `evlm/evlm.ipynb` | EVLM reproduction: datasets, PyTorch and llama.cpp runs, ablation and the log of earlier runs. |
| `CHANGELOG.md` | List of changes. |

Each notebook has its earlier cells and outputs at the bottom, under "Log of previous runs". Those cells call the scripts that were removed, so they are not run again.
