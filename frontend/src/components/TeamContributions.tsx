import React from 'react';
import {
  Users,
  ShieldCheck,
  Cpu,
  Database,
  Sparkles,
  GitPullRequest,
  CheckCircle2,
  FileCode2,
  BookOpen,
} from 'lucide-react';

interface StudentContribution {
  student: string;
  role: string;
  featureTitle: string;
  module: string;
  icon: React.ReactNode;
  accentColor: string;
  borderColor: string;
  badgeBg: string;
  summary: string;
  adrs: string[];
  keyDeliverables: string[];
  codeSeams: string[];
  testSuites: string[];
}

export const TeamContributions: React.FC = () => {
  const contributions: StudentContribution[] = [
    {
      student: 'Student 1',
      role: 'API Gateway & Streaming Architect',
      featureTitle: 'Feature 1: Gateway & Streaming Proxy Engine',
      module: 'Module 1: src/gateway/',
      icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
      accentColor: 'text-emerald-400',
      borderColor: 'border-emerald-500/30 hover:border-emerald-500/60',
      badgeBg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      summary:
        'Architected the dual-mode FastAPI gateway exposing OpenAI-compatible endpoints, buffering 100% of upstream model output to guarantee 0% token leakage, and pacing synthetic SSE streams.',
      adrs: ['ADR-0001 (Dual-Mode Topology)', 'ADR-0008 (Buffered SSE Streaming)'],
      keyDeliverables: [
        'OpenAI-compatible reverse proxy (`POST /v1/chat/completions`) & sidecar (`POST /verify`).',
        'Request & response token buffering engine ensuring zero leakage of contradicted claims.',
        'Synthetic SSE replay engine with 15ms configurable inter-token pacing (`X-SSE-Pacing-Ms`).',
        'Dynamic security header parsing (`X-Dehallucination-Mitigation`, `X-Dehallucination-Max-Tier`).',
        'Decoupled modern React frontend interface scaffolding and live health heartbeat integration.',
      ],
      codeSeams: [
        'src/gateway/app.py',
        'src/gateway/buffer.py',
        'src/gateway/middleware.py',
        'src/gateway/router.py',
        'src/gateway/sse.py',
      ],
      testSuites: [
        'tests/test_gateway_scaffold.py',
        'tests/test_reverse_proxy.py',
        'tests/test_proxy_router.py',
        'tests/test_sse_replay.py',
      ],
    },
    {
      student: 'Student 2',
      role: 'NLP & Verification Engine Lead',
      featureTitle: 'Feature 2: Claim Segmentation & NLI Verification Engine',
      module: 'Module 2: src/verifier/',
      icon: <Cpu className="w-5 h-5 text-sky-400" />,
      accentColor: 'text-sky-400',
      borderColor: 'border-sky-500/30 hover:border-sky-500/60',
      badgeBg: 'bg-sky-500/10 text-sky-400 border-sky-500/20',
      summary:
        'Engineered deterministic syntactic proposition splitting without non-deterministic LLM calls, and built CPU-optimized DeBERTa-v3 cross-encoder verification with calibrated safety gates.',
      adrs: ['ADR-0003 (Deterministic Segmentation)', 'ADR-0005 (Safety Gating & Calibrated NLI)'],
      keyDeliverables: [
        'Deterministic sentence and subordinate clause boundary segmentation using NLTK and dependency parse heuristics.',
        'Epistemic hedge detection (e.g. "I think", "arguably") ensuring underlying propositions remain verifiable.',
        'CPU-optimized `cross-encoder/nli-deberta-v3-base` inference with label mapping [Contradiction, Entailment, Neutral].',
        'Document-level metrics: Hallucination Rate, Verification Rate, and Unverified Rate calculations.',
        'Severe Contradiction Circuit Breaker (tripping `is_safe = False` when τ_severe ≥ 0.85).',
      ],
      codeSeams: [
        'src/verifier/segmenter.py',
        'src/verifier/nli.py',
        'src/verifier/orchestrator.py',
        'src/schemas/verification.py',
      ],
      testSuites: [
        'tests/test_claim_segmenter.py',
        'tests/test_nli_verifier.py',
        'tests/test_waterfall_e2e.py',
      ],
    },
    {
      student: 'Student 3',
      role: 'Knowledge Systems & Retrieval Lead',
      featureTitle: 'Feature 3: Hybrid Knowledge Ingestion & Retrieval Engine',
      module: 'Module 3: src/retrieval/',
      icon: <Database className="w-5 h-5 text-amber-400" />,
      accentColor: 'text-amber-400',
      borderColor: 'border-amber-500/30 hover:border-amber-500/60',
      badgeBg: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      summary:
        'Developed synchronized hybrid storage combining Qdrant dense vector embeddings with BM25 lexical keyword matching fused via Reciprocal Rank Fusion, powering the 3-tier waterfall search.',
      adrs: ['ADR-0004 (Waterfall Retrieval Tiers)', 'ADR-0007 (Hybrid Ingestion & Persistence)'],
      keyDeliverables: [
        'Deterministic document chunking (250–300 tokens, 50-token sliding overlap) with SHA-256 chunk hashing.',
        'DualIndexStore coordinating local disk-backed Qdrant collections and serialized BM25 sparse indices.',
        'Reciprocal Rank Fusion (RRF, k=60) merging dense semantic and sparse keyword rankings (100% Recall@10).',
        '3-tier waterfall cascade: Tier 1 In-Context → Tier 2 Local KB → Tier 3 DuckDuckGo External Web.',
        'Multipart upload and JSON ingestion pipeline (`POST /knowledge/ingest` and CLI `python -m src.ingest`).',
      ],
      codeSeams: [
        'src/retrieval/adapters.py',
        'src/retrieval/chunker.py',
        'src/retrieval/embedder.py',
        'src/retrieval/fusion.py',
        'src/retrieval/ingest.py',
        'src/retrieval/parser.py',
        'src/retrieval/store.py',
      ],
      testSuites: [
        'tests/test_ingestion.py',
        'tests/test_retrieval_adapters.py',
        'tests/test_web_retrieval.py',
      ],
    },
    {
      student: 'Student 4',
      role: 'Mitigation & Benchmark Evaluation Lead',
      featureTitle: 'Feature 4: Targeted Mitigation & Secondary Verification',
      module: 'Module 4: src/mitigation/ & src/eval/',
      icon: <Sparkles className="w-5 h-5 text-purple-400" />,
      accentColor: 'text-purple-400',
      borderColor: 'border-purple-500/30 hover:border-purple-500/60',
      badgeBg: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
      summary:
        'Engineered targeted prompt-guided claim rewriting with Gemma 2B, mandatory secondary verification with automatic redaction fallback, and coordinated the empirical benchmark harness.',
      adrs: ['ADR-0002 (Split-Compute Architecture)', 'ADR-0006 (Mitigation & Secondary Loop)', 'ADR-0009 (Evaluation Architecture)'],
      keyDeliverables: [
        'Local Gemma 2B quantized instruction model execution targeting ≤ 2.5 GB VRAM with mock CPU fallback.',
        'Targeted sentence-level prompt rewriter amending contradicted claims using retrieved evidence premises.',
        'Mandatory secondary verification loop preventing newly introduced hallucinations in rewritten text.',
        'Automatic cascading fallback to surgical masking (`redact`) or HTTP 422 refusal (`block`).',
        'Two-tier evaluation battery: Tier A 50-case CI/CD suite (`tests/eval_local_50.py`) and Tier B empirical benchmark (`scripts/run_benchmark.py`).',
      ],
      codeSeams: [
        'src/mitigation/generator.py',
        'src/mitigation/rewriter.py',
        'src/mitigation/fallback.py',
        'src/mitigation/pipeline.py',
        'src/eval/benchmark.py',
        'src/eval/calibration.py',
        'src/eval/metrics.py',
      ],
      testSuites: [
        'tests/test_targeted_rewriter.py',
        'tests/test_secondary_verification.py',
        'tests/test_eval_benchmark.py',
        'tests/eval_local_50.py',
      ],
    },
  ];

  return (
    <section id="team" className="py-24 lg:py-32 px-6 max-w-7xl mx-auto border-t border-white/5">
      {/* Section Header */}
      <div className="flex flex-col items-center text-center mb-16">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-emerald-400 font-mono text-xs uppercase tracking-widest mb-4">
          <Users className="w-3.5 h-3.5" />
          PES University • Department of Artificial Intelligence and Machine Learning
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white max-w-3xl">
          Modular Team Architecture & Contributions
        </h2>
        <p className="text-slate-400 text-base max-w-2xl mt-4">
          In compliance with UE24CS341A individual contribution standards, each student owns one
          complete functional module with explicit typed contracts, dedicated unit tests, and ADR governance.
        </p>
      </div>

      {/* 4 Student Contribution Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
        {contributions.map((c) => (
          <div
            key={c.student}
            className={`p-6 sm:p-8 rounded-2xl bg-[#0b0e15] border ${c.borderColor} shadow-xl flex flex-col justify-between transition-all duration-300 hover:-translate-y-1`}
          >
            <div className="space-y-6">
              {/* Card Header */}
              <div className="flex items-start justify-between gap-4 pb-4 border-b border-white/10">
                <div className="flex items-center gap-3.5">
                  <div className="p-3 rounded-xl bg-white/[0.03] border border-white/8 shrink-0">
                    {c.icon}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-mono font-bold uppercase px-2 py-0.5 rounded border ${c.badgeBg}`}>
                        {c.student}
                      </span>
                      <span className="text-[11px] font-mono text-slate-400">{c.module}</span>
                    </div>
                    <h3 className="text-lg sm:text-xl font-bold text-white mt-1">
                      {c.role}
                    </h3>
                  </div>
                </div>

                <div className="hidden sm:block text-right">
                  <span className="text-[10px] font-mono uppercase text-slate-400">
                    UE24CS341A
                  </span>
                </div>
              </div>

              {/* Feature Title & Summary */}
              <div>
                <h4 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300">
                  {c.featureTitle}
                </h4>
                <p className="text-xs sm:text-sm text-slate-300 mt-2 leading-relaxed font-sans">
                  {c.summary}
                </p>
              </div>

              {/* Architectural ADRs */}
              <div className="space-y-1.5">
                <div className="text-[11px] font-mono uppercase text-slate-400 flex items-center gap-1.5">
                  <BookOpen className="w-3.5 h-3.5 text-slate-400" />
                  Governing Architecture Decisions:
                </div>
                <div className="flex flex-wrap gap-2">
                  {c.adrs.map((adr, idx) => (
                    <span
                      key={idx}
                      className="text-[10px] font-mono px-2.5 py-1 rounded-md bg-black/50 border border-white/10 text-slate-300"
                    >
                      {adr}
                    </span>
                  ))}
                </div>
              </div>

              {/* Key Deliverables */}
              <div className="space-y-2">
                <div className="text-[11px] font-mono uppercase text-slate-400 flex items-center gap-1.5">
                  <GitPullRequest className="w-3.5 h-3.5 text-slate-400" />
                  Key Functional Deliverables:
                </div>
                <ul className="space-y-1.5">
                  {c.keyDeliverables.map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                      <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 mt-0.5 ${c.accentColor}`} />
                      <span className="leading-relaxed">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Code Seams & Test Suites Bottom Strip */}
            <div className="pt-6 mt-6 border-t border-white/10 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[11px] font-mono">
                <div className="flex items-center gap-1.5 text-slate-400">
                  <FileCode2 className="w-3.5 h-3.5 text-slate-400" />
                  <span>Owned Files:</span>
                </div>
                <div className="flex flex-wrap gap-1 text-slate-300 text-[10px]">
                  {c.codeSeams.map((f, i) => (
                    <span key={i} className="px-1.5 py-0.5 rounded bg-white/[0.03] border border-white/5">
                      {f.split('/').pop()}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[11px] font-mono">
                <span className="text-slate-400">Automated Test Suites:</span>
                <span className={`${c.accentColor} font-semibold text-[10px]`}>
                  {c.testSuites.map((t) => t.split('/').pop()).join(' • ')}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
