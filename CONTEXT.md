# Dehallucination Add-On Module

An add-on middleware and verification gateway that intercepts, scores, and mitigates factual hallucinations in AI-generated text using tiered grounding and local natural language inference.

## Language

**Dehallucination**:
The process of detecting unsupported or contradictory factual claims in machine-generated text and correcting or flagging them against verified knowledge sources.
_Avoid_: Fact-checking, Content filtering, Guardrailing

**Gateway**:
The dual-mode FastAPI service exposing both an OpenAI-compatible reverse proxy (`/v1/chat/completions`) and explicit verification endpoints (`/verify`).
_Avoid_: Wrapper, Proxy-only, Middleware

**Verification Sidecar**:
The programmatic endpoint allowing client applications to submit pre-generated text, prompt, and optional reference context to obtain sentence-level hallucination metrics.
_Avoid_: Scorer, Analyzer

**Local Generator**:
The GPU-hosted local Large Language Model (`google/gemma-4-E4B-it-qat-mobile-transformers` using `wNa8o8` QAT mobile quantization with targeted 2-bit decoding layers, targeting $\le 2.5\text{ GB}$ VRAM) responsible for primary text generation and rewrite mitigations.
_Avoid_: Remote LLM, Cloud model

**NLI Cross-Encoder**:
A natural language inference model (`cross-encoder/nli-deberta-v3-base` with label mapping index `[0: contradiction, 1: entailment, 2: neutral]`) executing on CPU that evaluates premise-hypothesis pairs to assign entailment, neutral, or contradiction probabilities.
_Avoid_: LLM Judge, Semantic similarity scorer

**Dense Embedder**:
The embedding model (`google/embeddinggemma-2`) executing on CPU to generate 768-dimensional semantic vector representations for hybrid retrieval in Qdrant using asymmetric task instruction prefixes (`encode_query` vs `encode_document`).
_Avoid_: Sparse encoder, Tokenizer

**Split Compute Strategy**:
The hardware resource partitioning dedicating the 4 GB GPU to the quantized generator LLM while allocating embedding generation and NLI verification to host CPU threads.
_Avoid_: Full GPU offloading, Pure CPU execution

**Atomic Claim**:
A single, independent factual proposition extracted from the generated text, capable of being evaluated as a standalone hypothesis by the NLI cross-encoder.
_Avoid_: Sentence, Sub-sentence, Token span

**Claim Segmenter**:
The deterministic component combining NLTK sentence tokenization and rule-based independent clause segmentation to produce atomic claims without LLM inference.
_Avoid_: LLM Extractor, Proposition prompt

**Waterfall Retrieval**:
The sequential evidence resolution process attempting Tier 1, then Tier 2, and conditionally Tier 3, halting immediately upon obtaining decisive evidence.
_Avoid_: Federated search, Parallel broadcasting

**Grounding Tier**:
The hierarchical classification of evidence sources: Tier 1 (In-Context Documents), Tier 2 (Local Hybrid KB via Qdrant + BM25), and Tier 3 (External Web Search via DuckDuckGo).
_Avoid_: Search level, Data source

**Unverified Claim**:
An atomic claim for which no sufficient evidence could be retrieved within the allowed retrieval tiers to support or refute it.
_Avoid_: Hallucination (do not conflate unverified with false), Contradiction

**Claim State**:
The discrete verification label assigned to an atomic claim: `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIED`.
_Avoid_: Binary truth value, True/False

**Safety Gate**:
The boolean safety circuit (`is_safe`) that trips to `false` when any single claim exceeds the severe contradiction threshold ($P \ge 0.85$).
_Avoid_: Blocklist, Content filter

**Review Flag**:
The advisory status (`needs_review`) raised when a document contains unverified claims without triggering a hard contradiction safety gate trip.
_Avoid_: Warning, Alert

**Reliability Metrics**:
The trio of normalized percentage rates summarizing a document: Hallucination Rate, Verification Rate, and Unverified Rate.
_Avoid_: Accuracy score, Confidence score

**Mitigation Mode**:
The policy governing Gateway action upon detecting contradicted claims: `flag`, `rewrite`, `redact`, or `block`.
_Avoid_: Error handler, Filter rule

**Targeted Rewrite**:
The correction procedure where the local generator amends only contradicted claims using retrieved evidence premises, preserving supported text and unverified context.
_Avoid_: Full regenerator, Prompt reprompt

**Secondary Verification**:
The re-verification check run against a newly rewritten response to confirm that no secondary hallucinations were introduced during correction.
_Avoid_: Double check, Loop check

**Mitigation Fallback**:
The automatic fallback to `redact` or `block` if a targeted rewrite fails secondary verification.
_Avoid_: Error catch, Default failover

**Ingestion Pipeline**:
The unified subsystem that parses PDF, Markdown, and TXT documents, chunks them into deterministic windows, and updates both vector and keyword indices in lockstep.
_Avoid_: Indexer, Document loader

**Deterministic Chunk ID**:
A content-hashed identifier assigned to a document chunk, identically referenced in Qdrant and BM25 to guarantee exact alignment during rank fusion.
_Avoid_: Row ID, Vector ID

**Reciprocal Rank Fusion (RRF)**:
The rank aggregation algorithm merging dense vector search ranks from Qdrant with sparse keyword ranks from BM25 into a unified relevance list.
_Avoid_: Score averaging, Dot product

**Synthetic SSE Replay**:
The streaming delivery mechanism where an upstream LLM response is buffered, verified, mitigated, and then replayed as simulated Server-Sent Event chunks to maintain client compatibility.
_Avoid_: True streaming, Passthrough streaming

**Time-to-First-Token (TTFT)**:
The elapsed time before the client receives the initial token chunk, intentionally traded off in the Gateway to complete whole-document verification.
_Avoid_: Latency (use TTFT for streaming delay)

**Evaluation Suite**:
The two-tier testing system comprising Tier A (fast `pytest` regressions over ~50 curated end-to-end edge cases) and Tier B (offline empirical benchmarking over FEVER and HaluEval subsets).
_Avoid_: Test runner, Benchmark script

**Retrieval Metrics**:
Ranking evaluations (Recall@K and Mean Reciprocal Rank - MRR) measuring hybrid retrieval accuracy independent of NLI classification.
_Avoid_: Vector score, Search accuracy

**Classification Metrics**:
Statistical performance indicators (Precision, Recall, F1, Confusion Matrix, and ROC-AUC) calculated across disjoint calibration and test splits to tune decision thresholds.
_Avoid_: Model score, Validation loss

## Functional Ownership & Modules

**Module 1: Gateway & Streaming Proxy Engine** (Owner: Student 1):
Exposes FastAPI `/v1/chat/completions` and `/verify`, implements request buffering, and manages Synthetic SSE Replay streaming delivery.

**Module 2: Claim Segmentation & NLI Verification Engine** (Owner: Student 2):
Performs deterministic NLTK/clause extraction, CPU-based DeBERTa-v3 inference, 3-way classification, reliability rate calculations, and Safety Gate logic.

**Module 3: Hybrid Knowledge Ingestion & Retrieval Engine** (Owner: Student 3):
Extracts and chunks PDF/TXT/MD reference documents, coordinates Qdrant and BM25 persistence, executes Reciprocal Rank Fusion, and manages Tier 3 web search.

**Module 4: Mitigation & Secondary Verification Engine** (Owner: Student 4):
Executes GPU-based Gemma targeted rewriting for contradicted claims, enforces single-pass secondary verification, and executes fallback redaction/blocking.
