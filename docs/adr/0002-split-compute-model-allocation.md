# Split Compute Model Allocation for Low-VRAM Hardware

With the hardware constrained to an NVIDIA RTX 2050 offering 4 GB VRAM, running a generative LLM alongside embedding and NLI models concurrently on the GPU causes immediate out-of-memory errors. We decided to adopt a split-compute strategy:
1. **Generative LLM / Rewriter**: [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) runs in 4-bit quantization on the GPU (~2.2 GB VRAM).
2. **Dense Embedder**: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) executes on CPU threads (768-dimensional multimodal/text vector space).
3. **NLI Cross-Encoder**: [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base) executes purely on CPU threads (~50-80 ms per claim verification).

This prevents CUDA OOM, provides predictable inference concurrency, and eliminates model swapping overhead.
