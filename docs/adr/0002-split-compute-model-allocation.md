# ADR-0002: Split-Compute Model Allocation and Precision Architecture for Edge-Constrained Hardware

## Status
Accepted (Refined with Quantization & Empirical Feasibility Constraints)

## Context
The production deployment environment is constrained to an NVIDIA GeForce RTX 2050 Mobile GPU with exactly 4,096 MiB (4 GB) of VRAM, running on Windows 11 alongside host CPU and RAM. Concurrently executing a generative Large Language Model, a dense embedding model, and an NLI cross-encoder on a 4 GB GPU causes immediate CUDA Out-Of-Memory (OOM) fatal crashes during peak KV-cache allocation or concurrent batching.

Furthermore, naive quantization assumptions pose severe operational risks:
1. **Generator Quantization Footprints:** Standard unquantized `gemma-4-E4B` (BF16) requires ~17.9 GB of memory. Standard `Q4_0` quantized weights alone demand approximately 4.5 GB, which exceeds the total 4 GB physical VRAM capacity of the RTX 2050 before accounting for CUDA runtime context, CUDA driver reservations, or KV cache buffers.
2. **Dense Embedder Asymmetry:** `google/embeddinggemma-2` uses task-specific instruction prefixes. Symmetrically passing raw text to both corpus documents and search queries degrades retrieval performance.
3. **NLI Label Ordering:** `cross-encoder/nli-deberta-v3-base` has an official label index ordering of `[0: contradiction, 1: entailment, 2: neutral]`. Assuming that index 0 corresponds to `entailment` inverts verification verdicts.
4. **Vector Store Dimension Migration:** Prior iterations utilized 384-dimensional embeddings (MiniLM). Upgrading to `google/embeddinggemma-2` (768 dimensions) renders existing 384-dimensional Qdrant collections incompatible.

---

## Decision

### 1. Split-Compute Architecture Allocation
We strictly isolate workloads between the dedicated GPU and the host CPU cores:

| Component | Designated Model & Checkpoint | Target Hardware | Quantization / Architecture | Memory Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Generative LLM / Rewriter** | [`google/gemma-4-E4B-it-qat-mobile-transformers`](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-transformers) | Dedicated GPU (`cuda:0`) | QAT Mobile Transformers (`wNa8o8` mixed-precision with targeted 2-bit decoding layers) | Target: $\le 2.5\text{ GB}$ VRAM (empirical verification mandatory) |
| **Dense Embedder** | [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) | Host CPU Threads (`cpu`) | 270M-parameter text backbone, FP32/BF16 CPU inference | $\sim 1.2\text{ GB}$ Host RAM |
| **NLI Cross-Encoder** | [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | Host CPU Threads (`cpu`) | FP32/INT8 CPU inference | $\sim 500\text{ MB}$ Host RAM |

---

### 2. Generator Quantization & Deployment Specifications
To fit within the 4 GB VRAM envelope on the RTX 2050 Mobile:
- **Discarded Baseline:** Baseline `google/gemma-4-E4B` standard `Q4_0` (~4.5 GB weights) is explicitly prohibited as it cannot load on a 4 GB GPU.
- **Designated Checkpoint:** `google/gemma-4-E4B-it-qat-mobile-transformers`.
  - Employs Google's mobile Quantization-Aware Training (QAT) schema (`wNa8o8`), featuring selective 2-bit decoding layers to compress model weights to approximately 2.2–2.5 GB.
- **Empirical Measurement Caveat:** The system shall **not** assert that the 2.5 GB limit is met in production documentation until actual GPU VRAM allocation is measured on the physical RTX 2050 under active inference and context loads.

---

### 3. Asymmetric Query/Document Embedding Configuration
`google/embeddinggemma-2` requires asymmetric task instruction handling:
- **Corpus Ingestion:** Documents must be encoded via document-specific instructions (`model.encode_document(documents)` or `prompt_name="Document"` / `"search_document: <text>"`).
- **Claim Retrieval:** Claims/queries must be encoded via query-specific instructions (`model.encode_query(claims)` or `prompt_name="SearchQuery"` / `"search_query: <text>"`).
- **Dimensionality:** Output dimensionality is fixed at 768 (Matryoshka Representation Learning dimension truncation supported if lower dimensions are configured).
- **Framework Requirement:** `sentence-transformers >= 6.1.0` (installed: 6.3.0).

---

### 4. NLI Cross-Encoder Label Mapping
Verdicts from `cross-encoder/nli-deberta-v3-base` must be mapped strictly by label name or model configuration `id2label`, using the official index ordering:
```python
LABEL_ORDER = ["contradiction", "entailment", "neutral"]
```
Domain mapping to `ClaimState`:
- `entailment` $\rightarrow$ `ClaimState.SUPPORTED`
- `contradiction` $\rightarrow$ `ClaimState.CONTRADICTED`
- `neutral` $\rightarrow$ `ClaimState.UNVERIFIED`

Index 0 must never be interpreted as entailment.

---

### 5. Vector Store Migration & Collection Metadata
Upgrading the embedder from 384 dimensions to 768 dimensions requires:
- Automatic detection of vector dimensionality mismatches in `DualIndexStore`. If an existing collection does not match the configured embedder dimension (768), the store must recreate the collection.
- Storing embedding model identity (`google/embeddinggemma-2`) and vector dimension (`768`) in the collection configuration and metadata.
- Re-embedding the knowledge corpus using the asymmetric document embedding pipeline.

---

## Consequences
- **Pros:** Completely eliminates GPU CUDA OOM risk by isolating embedding and NLI models to host CPU threads. Prevents retrieval accuracy degradation via asymmetric query/document task instruction prefixes. Prevents catastrophic verification false passes/failures by enforcing exact NLI label mapping.
- **Cons:** Host CPU utilization increases during batch claim verification; CPU inference latency for DeBERTa (~50–80 ms per pair) requires batching during multi-claim verification.
- **Validation Requirement:** Empirical measurement of GPU VRAM usage during real model smoke testing is required before final sign-off.
