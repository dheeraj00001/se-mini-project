import React, { useState } from 'react';
import {
  Layers,
  Database,
  Globe,
  Cpu,
  Sparkles,
  CheckCircle,
  FileSearch,
} from 'lucide-react';

interface WaterfallStage {
  id: string;
  name: string;
  subtitle: string;
  tier: string;
  latencyBudget: string;
  adr: string;
  icon: React.ReactNode;
  description: string;
  details: string[];
}

export const WaterfallDiagram: React.FC = () => {
  const [activeStageId, setActiveStageId] = useState<string>('tier2');

  const stages: WaterfallStage[] = [
    {
      id: 'segmentation',
      name: 'Deterministic Proposition Extraction',
      subtitle: 'Sentence & Clause Segmentation',
      tier: 'Module 2 Pre-filter',
      latencyBudget: '< 15ms',
      adr: 'ADR-0004',
      icon: <FileSearch className="w-5 h-5 text-emerald-400" />,
      description:
        'Splits raw generation into standalone atomic propositions using dependency parse trees and subordinate clause boundaries while flagging epistemic hedges.',
      details: [
        'Deterministic clause segmentation without non-deterministic LLM extractors.',
        'Preserves source sentence pointer for targeted surgical redaction.',
        'Epistemic hedge detection (e.g., "I think", "arguably") to prevent false alarms.',
      ],
    },
    {
      id: 'tier1',
      name: 'Tier 1: In-Context Grounding',
      subtitle: 'Fast Local Document Scrim',
      tier: 'Tier 1 Retrieval',
      latencyBudget: '< 5ms',
      adr: 'REQ-RET-05',
      icon: <Layers className="w-5 h-5 text-emerald-400" />,
      description:
        'Evaluates claims directly against provided reference passages (e.g. system instructions, uploaded document snippets) before dispatching index queries.',
      details: [
        'Zero network hops and immediate cross-encoder pairing.',
        'Short-circuits downstream retrieval if confidence threshold is met.',
        'Ideal for RAG architectures with active prompt context.',
      ],
    },
    {
      id: 'tier2',
      name: 'Tier 2: Hybrid Dense & Sparse KB',
      subtitle: 'Qdrant Vector + BM25 Keyword RRF',
      tier: 'Tier 2 Retrieval',
      latencyBudget: '< 85ms',
      adr: 'ADR-0007',
      icon: <Database className="w-5 h-5 text-emerald-400" />,
      description:
        'DualIndexStore combining fast 384-dimensional dense semantic embeddings with BM25 keyword matching fused via Reciprocal Rank Fusion (k=60).',
      details: [
        'Local Qdrant in-memory or on-disk vector store with Cosine distance.',
        'Rank-BM25 with tokenized lower-cased inverted index.',
        'Reciprocal Rank Fusion (RRF) delivers 100% Recall@10 on gold evaluation splits.',
      ],
    },
    {
      id: 'tier3',
      name: 'Tier 3: External Live Web',
      subtitle: 'DuckDuckGo Grounding Waterfall',
      tier: 'Tier 3 Retrieval',
      latencyBudget: '< 450ms',
      adr: 'REQ-RET-06',
      icon: <Globe className="w-5 h-5 text-emerald-400" />,
      description:
        'Fallback tier activated only if Tier 1 and Tier 2 fail to resolve the hypothesis, querying DuckDuckGo with automated passage text extraction.',
      details: [
        'Enforces server-side max-tier security ceiling to prevent quota exhaustion.',
        'Extracts relevant body snippets from top search hits.',
        'Concurrently feeds candidate premises into cross-encoder ranking.',
      ],
    },
    {
      id: 'nli',
      name: 'DeBERTa-v3 NLI Cross-Encoder',
      subtitle: 'CPU-Calibrated Natural Language Inference',
      tier: 'Module 4 Verification',
      latencyBudget: '< 120ms',
      adr: 'ADR-0009',
      icon: <Cpu className="w-5 h-5 text-emerald-400" />,
      description:
        'Lightweight CPU-optimized DeBERTa-v3 cross-encoder producing normalized posterior probabilities across Contradiction, Entailment, and Neutral states.',
      details: [
        'Calibrated thresholds: τ_entailment = 0.50, τ_contradiction = 0.50.',
        'Circuit-breaker safety gate trips at τ_severe = 0.85.',
        'Achieves 100% Contradiction Precision on held-out FEVER/HaluEval test splits.',
      ],
    },
    {
      id: 'mitigation',
      name: 'Targeted Mitigation & Secondary Loop',
      subtitle: 'Gemma 2B Rewriter + Surgical Redactor',
      tier: 'Module 5 Mitigation',
      latencyBudget: '< 300ms',
      adr: 'ADR-0001',
      icon: <Sparkles className="w-5 h-5 text-emerald-400" />,
      description:
        'Corrects contradicted assertions using local Gemma 2B instruction model, followed by secondary NLI verification and fallback to surgical masking.',
      details: [
        'Prompted with premise evidence to generate targeted clause corrections.',
        'Secondary NLI verification loop prevents introducing new hallucinations.',
        'Automatic fallback to surgical span masking if secondary verification fails.',
      ],
    },
  ];

  const activeStage = stages.find((s) => s.id === activeStageId) || stages[2];

  return (
    <section id="waterfall" className="py-24 lg:py-32 px-6 max-w-7xl mx-auto border-t border-white/5">
      {/* Eyebrow and Section Header */}
      <div className="flex flex-col items-center text-center mb-16">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-emerald-400 font-mono text-xs uppercase tracking-widest mb-4">
          <Layers className="w-3.5 h-3.5" />
          Multi-Tier Grounding Resolution
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white max-w-3xl">
          The 3-Tier Waterfall Architecture
        </h2>
        <p className="text-slate-400 text-base max-w-xl mt-4">
          A prioritized cascade of evidence retrieval minimizing external latency while guaranteeing
          exhaustive factual grounding.
        </p>
      </div>

      {/* Interactive Process Pipeline Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-8 items-start">
        {/* Left: Step Sequence */}
        <div className="space-y-3">
          {stages.map((stage, index) => {
            const isSelected = activeStageId === stage.id;
            return (
              <div
                key={stage.id}
                onClick={() => setActiveStageId(stage.id)}
                className={`p-4 sm:p-5 rounded-2xl border cursor-pointer transition-all ${
                  isSelected
                    ? 'bg-[#0f1420] border-emerald-500/60 shadow-[0_0_25px_rgba(16,185,129,0.15)] translate-x-1.5'
                    : 'bg-[#0a0d13]/80 border-white/8 hover:border-white/20 hover:bg-[#0c1018]'
                }`}
              >
                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3.5">
                    <div
                      className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border ${
                        isSelected
                          ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                          : 'bg-white/5 border-white/10 text-slate-400'
                      }`}
                    >
                      {stage.icon}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-white/5 text-slate-400">
                          Step 0{index + 1}
                        </span>
                        <span className="text-xs font-mono text-emerald-400">{stage.adr}</span>
                      </div>
                      <h4 className="text-sm sm:text-base font-bold text-white mt-0.5">
                        {stage.name}
                      </h4>
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <span className="text-xs font-mono font-semibold text-slate-300">
                      {stage.latencyBudget}
                    </span>
                    <span className="block text-[10px] font-mono text-slate-400">
                      Target Latency
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: Stage Deep-Dive Card */}
        <div className="sticky top-28 bg-[#0b0e15] border border-white/10 rounded-2xl p-6 sm:p-8 shadow-2xl space-y-6">
          <div className="flex items-center justify-between pb-4 border-b border-white/10">
            <div>
              <span className="text-[11px] font-mono uppercase text-emerald-400 tracking-wider">
                {activeStage.tier}
              </span>
              <h3 className="text-xl font-bold text-white mt-1">{activeStage.name}</h3>
              <p className="text-xs font-mono text-slate-400 mt-0.5">{activeStage.subtitle}</p>
            </div>
            <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              {activeStage.icon}
            </div>
          </div>

          <p className="text-sm text-slate-300 leading-relaxed font-sans">
            {activeStage.description}
          </p>

          <div className="space-y-3">
            <h5 className="text-xs font-mono uppercase text-slate-400 tracking-wider">
              Engineering Architecture Specifications:
            </h5>
            <ul className="space-y-2">
              {activeStage.details.map((detail, idx) => (
                <li key={idx} className="flex items-start gap-2.5 text-xs text-slate-300">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="leading-relaxed">{detail}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="pt-4 border-t border-white/10 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Latency Target: {activeStage.latencyBudget}</span>
            <span className="text-emerald-400 font-semibold">{activeStage.adr} Specification</span>
          </div>
        </div>
      </div>
    </section>
  );
};
