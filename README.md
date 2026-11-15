# Dehallucination Add-On Module Gateway

An add-on middleware and verification gateway that intercepts, scores, and mitigates factual hallucinations in AI-generated text using tiered waterfall grounding and local Natural Language Inference (NLI).

The system operates in dual mode:
1. **OpenAI-Compatible Reverse Proxy (`POST /v1/chat/completions`)**: Intercepts chat completion requests, buffers the full upstream or local model generation, verifies atomic claims against grounding tiers, applies configurable mitigation policies (`flag`, `rewrite`, `redact`, `block`), and replays sanitized text to clients as standard responses or paced Server-Sent Event (SSE) streams.
2. **Verification Sidecar (`POST /verify`)**: Evaluates pre-generated text against supplied or indexed reference context to output sentence-level verification states, reliability scores, and safety gate decisions.

---

## Architecture & Core Mechanisms

```
Client Application
       │
       ▼ [POST /v1/chat/completions or POST /verify]
┌─────────────────────────────────────────────────────────────┐
│ Gateway Layer (FastAPI)                                     │
│  - Security Header Override Parser                          │
│  - Full Request / Response Buffering (ADR-0008)             │
│  - Synthetic SSE Replay Engine (15ms token pacing)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Deterministic Claim Segmenter                               │
│  - Sentence Boundary Tokenization (NLTK punkt)              │
│  - Syntactic Clause Splitting (deterministic non-LLM)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ [Atomic Claims]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ NLI-Aware Waterfall Grounding Orchestrator                  │
│                                                             │
│  Tier 1: In-Context Document Matching (Exact/Normalized)    │
│    │ (unresolved)                                           │
│    ▼                                                        │
│  Tier 2: Hybrid Local Knowledge Store                       │
│    ├─ Dense Vector Search: Qdrant (google/embeddinggemma-2) │
│    ├─ Lexical Keyword Search: BM25 (rank-bm25)              │
│    └─ Rank Fusion: Reciprocal Rank Fusion (RRF, k=60)       │
│    │ (unresolved & opt-in)                                  │
│    ▼                                                        │
│  Tier 3: External Web Grounding (DuckDuckGo Search)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ [Premise-Hypothesis Pairs]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ NLI Cross-Encoder Verification (CPU)                        │
│  - cross-encoder/nli-deberta-v3-base                        │
│  - Label Probabilities: [Contradiction, Entailment, Neutral]│
│  - Claim States: SUPPORTED, CONTRADICTED, UNVERIFIED        │
│  - Safety Gate: Severe Contradiction (tau >= 0.85 -> unsafe)│
└──────────────────────────────┬──────────────────────────────┘
                               │ [Claim Verification States]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Mitigation Pipeline & Safety Circuit Breaker                │
│  - Flag: Annotate metrics and claim metadata in headers     │
│  - Rewrite: Targeted local correction via Gemma 2B          │
│  - Redact: Surgical redaction of contradicted sentences     │
│  - Block: HTTP 422 failover for severe safety gate trips    │
│  - Secondary Verification: Re-verify rewrites against halluc│
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Characteristics
- **Split Compute Strategy**: CPU handles dense embeddings (`google/embeddinggemma-2` or `FastEmbed`) and NLI cross-encoder verification (`cross-encoder/nli-deberta-v3-base`). GPU is reserved for local generation and targeted rewrite execution (`google/gemma-4-E4B-it-qat-mobile-transformers` targeting $\le 2.5\text{ GB}$ VRAM).
- **Zero Token Leakage Under Streaming**: Upstream tokens are 100% buffered during verification; unverified or contradicted tokens are never released to the client before mitigation completes (`ADR-0008`).
- **Deterministic Segmenter**: Extracts atomic propositions using rule-based clause boundary splitting without introducing non-deterministic LLM extraction overhead.
- **Two-Tier Evaluation Suite**: Contains a 50-case local synthetic regression battery (`Tier A`, $\le 60\text{ s}$) and an empirical benchmark harness (`Tier B`) evaluating FEVER and HaluEval slices with threshold calibration and scikit-learn metrics.

---

## Prerequisites & System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version `3.11`
- **Memory**: Minimum 8 GB RAM (16 GB recommended for running full cross-encoders)
- **GPU (Optional)**: NVIDIA GPU with CUDA support for local quantized Gemma execution (CPU mock generator fallback is enabled by default for testing and headless environments)

---

## Installation & Setup

### 1. Clone Repository & Setup Environment

Using `conda` (recommended):
```bash
# Create isolated Python 3.11 environment
conda create -n dehallucination python=3.11 -y
conda activate dehallucination
```

Or using standard `venv`:
```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install runtime dependencies:
```bash
pip install -r requirements.txt
```

Install development and evaluation dependencies (pytest, mypy, flake8, black, scikit-learn, datasets, matplotlib, pandas):
```bash
pip install -r requirements-dev.txt
```

### 3. Initialize NLTK Tokenizer Models

Download required NLTK tokenization models for deterministic claim segmentation:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Configuration & Security Headers

The gateway accepts dynamic runtime overrides through HTTP request headers:

| Header | Allowed Values | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `X-Dehallucination-Mitigation` | `flag`, `rewrite`, `redact`, `block` | `flag` | Mitigation policy applied when contradicted claims are detected |
| `X-Dehallucination-Max-Tier` | `1`, `2`, `3` | `2` | Maximum allowable evidence retrieval tier (Tier 3 enables Web Grounding) |
| `X-Dehallucination-Strict-Grounding` | `true`, `false` | `false` | When true, marks claims missing ground-truth references as contradicted |
| `X-SSE-Pacing-Ms` | Integer $\ge 0$ | `15` | Simulated inter-token delay (in milliseconds) for synthetic SSE replay |

---

## Running the Application

### 1. Launch Gateway Server

Start the FastAPI application using `uvicorn`:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
Or directly execute the entry point module:
```bash
python -m src.main
```

### 2. Verify System Health

```bash
curl -X GET http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 3. Launch React Frontend (Awwwards-Tier Option B)

The project includes an Awwwards-tier, decoupled React 19 + TypeScript + Vite frontend located in `frontend/`, engineered under the `awwwards-hero-section` (Architecture B: Asymmetric Split) and `awwwards-sections` design pipelines:

```bash
# Option A: From repository root (recommended)
npm run frontend:dev

# Option B: From the frontend directory
cd frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser to access the interface.

#### Frontend Capabilities & Architecture
- **Floating Pill Navigation (`Navbar.tsx`)**: Dynamic heartbeat indicator displaying live FastAPI backend health, status ping, and real-time round-trip latency (`:8000`).
- **Hero Section (`HeroSection.tsx`)**: Viewport-scale architectural typography (`clamp(2.5rem, 6.2vw, 6.8rem)`, tight `-0.035em` tracking), live claim syntax highlighting, posterior probability meters, and instant rewrite/redact preview.
- **Interactive Playground (`Playground.tsx`)**:
  - **Mode 1 (`POST /verify`)**: Interactive proposition segmentation, 3-tier waterfall selector, mitigation mode switcher (`rewrite`, `redact`, `block`, `flag`), claim probability meters, and sanitized text output.
  - **Mode 2 (`POST /v1/chat/completions`)**: Buffered OpenAI reverse proxy chat completions with synthetic SSE token streaming playback (15ms pacing) and injected verification headers.
  - **Pre-loaded Presets**: Quick-load benchmark scenarios (Apollo 11 Moon landing vs Mars hallucination, Eiffel Tower height discrepancy, Berlin/Paris capital entity swap, Penicillin discovery).
- **Waterfall Grounding Engine (`WaterfallDiagram.tsx`)**: Interactive step-by-step resolution flow visualizing Tier 1 In-Context $\to$ Tier 2 Qdrant + BM25 RRF $\to$ Tier 3 External Web Grounding with latency budgets and architectural specifications.
- **Empirical Benchmark Telemetry (`BenchmarkStats.tsx`)**: High-contrast metric cards (60.71% Accuracy, 0.6076 Macro F1, 100% Contradiction Precision, 0% Token Leakage) and interactive Lightbox Modal viewing all 4 diagnostic evaluation figures from `reports/figures/`.
- **Developer Integration Tabs (`CodeSnippets.tsx`)**: Copy-pasteable integration code for Python OpenAI SDK, cURL CLI, and TypeScript Fetch.
- **Offline Simulator**: Seamless fallback when FastAPI is offline, enabling immediate interactive testing and demonstration without neural model loading.

#### Frontend Testing & Build Verification
```bash
# Check TypeScript typing and build production bundle
npm run frontend:build
# (or from root: npm run frontend:build)

# Run Oxlint / ESLint linter
npm run frontend:lint
# (or from root: npm run frontend:lint)

# Preview production build locally
npm run frontend:preview
```

---

## Knowledge Ingestion (Tier 2 Knowledge Base)

The hybrid knowledge store indexes documents into local file-backed Qdrant collections and serialized BM25 indices simultaneously using deterministic chunk hashing.

### CLI Batch Ingestion

Ingest a single document or directory of documents (`.pdf`, `.md`, `.txt`):
```bash
# Ingest all documents in a directory
python -m src.ingest --source ./docs

# Ingest specific document with custom chunking parameters
python -m src.ingest --source ./docs/sample.pdf --window-size 250 --overlap 50
```

### API Ingestion

Ingest raw text directly into the active knowledge store:
```bash
curl -X POST http://localhost:8000/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_space_01",
    "text": "Apollo 11 landed humans on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin formed the American crew."
  }'
```

Upload a document file via multipart form:
```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@./docs/sample.txt"
```

---

## API Usage & Endpoints

### 1. Verification Sidecar (`POST /verify`)

Submit pre-generated text to verify each constituent claim against tiered evidence:

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -H "X-Dehallucination-Mitigation: rewrite" \
  -d '{
    "text": "Apollo 11 landed on the Moon in July 1969. The mission was commanded by Buzz Lightyear.",
    "context_documents": [
      "Apollo 11 landed on the Moon on July 20, 1969. The mission was commanded by Neil Armstrong."
    ]
  }'
```

#### Verification Response Format
```json
{
  "claims": [
    {
      "claim_id": "claim_0",
      "text": "Apollo 11 landed on the Moon in July 1969.",
      "state": "SUPPORTED",
      "confidence": 0.985,
      "tier": 1,
      "source_evidence": "Apollo 11 landed on the Moon on July 20, 1969.",
      "is_severe_contradiction": false
    },
    {
      "claim_id": "claim_1",
      "text": "The mission was commanded by Buzz Lightyear.",
      "state": "CONTRADICTED",
      "confidence": 0.942,
      "tier": 1,
      "source_evidence": "The mission was commanded by Neil Armstrong.",
      "is_severe_contradiction": true
    }
  ],
  "verification_rate": 50.0,
  "hallucination_rate": 50.0,
  "unverified_rate": 0.0,
  "is_safe": false,
  "needs_review": false,
  "mitigated_text": "Apollo 11 landed on the Moon in July 1969. The mission was commanded by Neil Armstrong."
}
```

### 2. OpenAI-Compatible Reverse Proxy (`POST /v1/chat/completions`)

Drop-in replacement for OpenAI API endpoints with automatic factual interception and mitigation.

#### Standard JSON Response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Dehallucination-Mitigation: rewrite" \
  -d '{
    "model": "google/gemma-4-E4B-it-qat-mobile-transformers",
    "messages": [
      {"role": "user", "content": "Who commanded Apollo 11 and when did it land?"}
    ],
    "stream": false
  }'
```

#### Synthetic SSE Streaming Response
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Dehallucination-Mitigation: rewrite" \
  -H "X-SSE-Pacing-Ms: 15" \
  -d '{
    "model": "google/gemma-4-E4B-it-qat-mobile-transformers",
    "messages": [
      {"role": "user", "content": "Tell me about the Apollo 11 mission commander."}
    ],
    "stream": true
  }'
```

#### Integration via Official OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none-required-for-local-proxy",
)

# Standard call
response = client.chat.completions.create(
    model="google/gemma-4-E4B-it-qat-mobile-transformers",
    messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
    extra_headers={"X-Dehallucination-Mitigation": "rewrite"},
)
print(response.choices[0].message.content)

# Streaming call (Synthetic SSE Replay)
stream = client.chat.completions.create(
    model="google/gemma-4-E4B-it-qat-mobile-transformers",
    messages=[{"role": "user", "content": "Where is the Eiffel Tower located?"}],
    stream=True,
    extra_headers={
        "X-Dehallucination-Mitigation": "rewrite",
        "X-SSE-Pacing-Ms": "10",
    },
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

---

## Evaluation Harness & Empirical Benchmark Suite

The project includes a two-tier evaluation framework implementing academic validation requirements (`NFR-QUAL-02`, `NFR-QUAL-03`, `ADR-0009`).

### Tier A: CI/CD Synthetic Regression Test Suite (`tests/eval_local_50.py`)
Executes 50 edge-case test cases in `pytest` validating pipeline accuracy, mitigation modes, and latency:
- 10 True Facts (`TF-01` to `TF-10`)
- 10 Inverted Dates (`ID-01` to `ID-10`)
- 10 Swapped Entities (`SE-01` to `SE-10`)
- 10 Fabricated Citations (`FC-01` to `FC-10`)
- 10 Epistemic Hedges (`EH-01` to `EH-10`)

Run Tier A suite:
```bash
pytest tests/eval_local_50.py -v
```
*(Execution finishes in ~8.5 seconds on CPU, well below the 60-second CI ceiling).*

### Tier B: Empirical Benchmark & Calibration Runner (`scripts/run_benchmark.py`)
Evaluates FEVER and HaluEval validation slices, enforces strict train/calibration/test split discipline (pinned seed `42`), optimizes decision thresholds ($\tau_e$, $\tau_c$), and computes formal metrics.

Execute benchmark runner:
```bash
# Fast headless run using mock NLI verifier
python scripts/run_benchmark.py --use-mock

# Full neural inference run on GPU/CPU
python scripts/run_benchmark.py --fever-samples 60 --halueval-samples 60 --seed 42
```

#### Output Artifacts Generated
- **JSON Results**: `reports/benchmark_results.json` (complete serialized metrics and calibration results)
- **Markdown Summary**: `reports/benchmark_summary.md` (formatted tables for project documentation)
- **Diagnostic Plots (`reports/figures/`)**:
  - `confusion_matrix.png`: Held-out test split confusion matrix across claim states
  - `roc_curves.png`: One-vs-Rest ROC curves with per-class AUC
  - `calibration_grid.png`: Macro F1 optimization curve over candidate threshold grid
  - `retrieval_performance.png`: Qdrant + BM25 ranking metrics (Recall@1, Recall@5, Recall@10, MRR)

---

## Modular Architecture & Individual Student Contributions (UE24CS341A)

In compliance with PES University Department of Artificial Intelligence and Machine Learning curriculum standards (UE24CS341A), the system is structured into four independent modules with explicit typed boundaries, individual PR deliverables, dedicated unit test suites, and ADR governance (`ADR-0010`):

| Student | Functional Role | Owned Subsystem | Key Architecture Deliverables | Code Seams | Test Suites |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Student 1** | **API Gateway & Streaming Architect** | **Module 1**: Gateway & Streaming Proxy Engine | • OpenAI-compatible reverse proxy (`POST /v1/chat/completions`)<br>• Standalone verification sidecar (`POST /verify`)<br>• 100% full response buffering (`ADR-0008`) preventing hallucination leakage<br>• Synthetic SSE replay engine with 15ms pacing (`X-SSE-Pacing-Ms`)<br>• Decoupled React frontend UI scaffolding | `src/gateway/app.py`<br>`src/gateway/buffer.py`<br>`src/gateway/middleware.py`<br>`src/gateway/router.py`<br>`src/gateway/sse.py` | `test_gateway_scaffold.py`<br>`test_reverse_proxy.py`<br>`test_proxy_router.py`<br>`test_sse_replay.py` |
| **Student 2** | **NLP & Verification Engine Lead** | **Module 2**: Claim Segmentation & NLI Verification | • Deterministic syntactic proposition splitting (`ADR-0004`)<br>• Epistemic hedge detection (e.g. *"I think"*, *"arguably"*)<br>• CPU DeBERTa-v3 cross-encoder classification (`ADR-0005`)<br>• Severe Contradiction Circuit Breaker ($\tau_{severe} \ge 0.85$ trips `is_safe=False`)<br>• Document-level metrics (Hallucination/Verification/Unverified Rates) | `src/verifier/segmenter.py`<br>`src/verifier/nli.py`<br>`src/verifier/orchestrator.py`<br>`src/schemas/verification.py` | `test_claim_segmenter.py`<br>`test_nli_verifier.py`<br>`test_waterfall_e2e.py` |
| **Student 3** | **Knowledge Systems & Retrieval Lead** | **Module 3**: Hybrid Knowledge Ingestion & Retrieval | • Document parser & token chunker (250–300 tokens, 50-token overlap)<br>• Deterministic SHA-256 chunk identifier synchronization<br>• DualIndexStore (`ADR-0007`: Qdrant dense vector + BM25 keyword store)<br>• Reciprocal Rank Fusion ($k=60$, 100% Recall@10)<br>• 3-Tier Waterfall search (Context $\to$ Local KB $\to$ Web) | `src/retrieval/adapters.py`<br>`src/retrieval/chunker.py`<br>`src/retrieval/embedder.py`<br>`src/retrieval/fusion.py`<br>`src/retrieval/ingest.py`<br>`src/retrieval/store.py` | `test_ingestion.py`<br>`test_retrieval_adapters.py`<br>`test_web_retrieval.py` |
| **Student 4** | **Mitigation & Evaluation Lead** | **Module 4**: Mitigation & Benchmark Evaluation | • Local 4-bit Gemma 2B instruction model allocation (`ADR-0001`, `ADR-0002`)<br>• Prompt-guided surgical clause rewriter (`ADR-0006`)<br>• Mandatory secondary NLI verification loop & cascade to `redact`/`block`<br>• Tier A CI/CD regression test battery (50 cases in `tests/eval_local_50.py`)<br>• Tier B benchmark runner (`scripts/run_benchmark.py`), calibration grid & metrics | `src/mitigation/generator.py`<br>`src/mitigation/rewriter.py`<br>`src/mitigation/fallback.py`<br>`src/mitigation/pipeline.py`<br>`src/eval/benchmark.py`<br>`src/eval/calibration.py` | `test_targeted_rewriter.py`<br>`test_secondary_verification.py`<br>`test_eval_benchmark.py`<br>`eval_local_50.py` |

---

## Testing & Quality Verification

Run the full automated test suite (314 unit, integration, streaming, and evaluation tests) with statement and branch coverage:
```bash
pytest
```
*Configured in `pyproject.toml` with `--cov-fail-under=80`. Current test suite achieves 90.4% coverage.*

Run static type checking:
```bash
mypy src tests
```

Run code style and flake8 linting:
```bash
flake8 src tests scripts/run_benchmark.py
```

Verify Black formatting compliance:
```bash
black --check src tests scripts/run_benchmark.py
```
*(Run `black src tests scripts/run_benchmark.py` to automatically apply formatting).*

---

## Directory Structure

```
.
├── CONTEXT.md                    # Ubiquitous domain language, acronym definitions & ADRs
├── README.md                     # Technical overview, installation & operation guide
├── package.json                  # Root npm scripts for frontend coordination
├── pyproject.toml                # Pytest, coverage, mypy & tool configuration
├── requirements.txt              # Core runtime dependencies
├── requirements-dev.txt          # Test, lint, and benchmarking dependencies
├── frontend/                     # Decoupled Awwwards-tier React + Vite + Tailwind CSS frontend
│   ├── index.html                # Cybernetic dark viewport & typography configuration
│   ├── package.json              # React 19, Framer Motion, Lucide, Tailwind v4
│   ├── vite.config.ts            # Vite bundler configuration & backend proxy
│   ├── public/figures/           # Served diagnostic evaluation figures
│   └── src/
│       ├── App.tsx               # Root composition and live health check loop
│       ├── index.css             # Obsidian theme, ambient radial glows & film grain
│       ├── components/           # Navbar, HeroSection, Playground, Waterfall, Telemetry, CodeSnippets, Footer
│       ├── services/api.ts       # Unified API client & offline simulation fallback
│       └── types/index.ts        # TypeScript schemas matching FastAPI models
├── data/
│   └── eval/                     # Curated FEVER and HaluEval offline benchmark datasets
├── docs/                         # Technical documentation & project specifications
├── reports/                      # Generated benchmark outputs, metrics & figures
│   ├── benchmark_results.json
│   ├── benchmark_summary.md
│   └── figures/
├── scripts/
│   ├── run_benchmark.py          # Tier B empirical benchmark CLI runner
│   └── validate_models_smoke.py  # Model deployment sanity check
├── src/
│   ├── main.py                   # Uvicorn entry point
│   ├── ingest.py                 # Knowledge ingestion CLI entry point
│   ├── gateway/                  # FastAPI router, request buffer, middleware & SSE replay
│   ├── retrieval/                # Waterfall retriever, Qdrant/BM25 adapters, chunker, fusion
│   ├── verifier/                 # Claim segmenter, NLI verifiers, waterfall orchestrator
│   ├── mitigation/               # Gemma rewriter, surgical redactor, fallback pipeline
│   ├── eval/                     # Calibration, scikit-learn metrics, benchmark loaders
│   └── schemas/                  # Pydantic schemas (verification, OpenAI, ingestion)
└── tests/                        # Comprehensive unit, integration, and Tier A test suite
    ├── eval_local_50.py          # 50-case synthetic regression test battery (Tier A)
    ├── test_eval_benchmark.py    # Unit tests for metrics, calibration, and benchmark runner
    └── test_*.py                 # Unit & integration tests for all core modules
```
