import React, { useState } from 'react';
import { motion, type Variants } from 'framer-motion';
import {
  AlertTriangle,
  ArrowRight,
  Sparkles,
  Layers,
  Cpu,
  CheckCircle2,
} from 'lucide-react';
import type { BackendHealth } from '../types';

interface HeroSectionProps {
  health: BackendHealth | null;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ health }) => {
  const [activeTab, setActiveTab] = useState<'annotated' | 'mitigated'>('annotated');
  const [selectedClaimIdx, setSelectedClaimIdx] = useState<number>(1);

  const heroClaims = [
    {
      id: 0,
      text: 'Apollo 11 landed on the lunar surface on July 20, 1969.',
      state: 'SUPPORTED',
      p_entailment: 0.984,
      p_contradiction: 0.012,
      p_neutral: 0.004,
      tier: 'Tier 1 (In-Context)',
      evidence: 'NASA Apollo 11 Mission Log: Lunar Module Eagle landed July 20, 1969.',
    },
    {
      id: 1,
      text: 'Michael Collins walked on Mars during the return flight in 1972.',
      state: 'CONTRADICTED',
      p_entailment: 0.015,
      p_contradiction: 0.942,
      p_neutral: 0.043,
      tier: 'Tier 2 (Qdrant + BM25 RRF)',
      evidence: 'Biographical Archive: Michael Collins remained in lunar orbit aboard Columbia.',
    },
    {
      id: 2,
      text: 'Neil Armstrong and Buzz Aldrin deployed the Apollo EASEP scientific package.',
      state: 'SUPPORTED',
      p_entailment: 0.976,
      p_contradiction: 0.008,
      p_neutral: 0.016,
      tier: 'Tier 1 (In-Context)',
      evidence: 'EASEP deployed on surface for seismic and solar wind experiments.',
    },
  ];

  const currentClaim = heroClaims[selectedClaimIdx];

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.05,
      },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 24, filter: 'blur(8px)' },
    visible: {
      opacity: 1,
      y: 0,
      filter: 'blur(0px)',
      transition: {
        duration: 0.7,
        ease: 'easeOut',
      },
    },
  };

  return (
    <section className="relative min-h-[100dvh] pt-28 pb-16 lg:pt-36 lg:pb-24 px-6 max-w-7xl mx-auto flex items-center justify-center overflow-hidden">
      {/* Radial ambient glow */}
      <div className="ambient-glow w-[600px] h-[500px] -top-20 left-1/4 -translate-x-1/2 opacity-70" />
      <div className="ambient-glow-rose w-[450px] h-[400px] top-1/3 right-10 opacity-50" />

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="w-full grid grid-cols-1 lg:grid-cols-[1.1fr_1fr] gap-12 lg:gap-16 items-center"
      >
        {/* Left Column: Viewport-scale Architectural Typography */}
        <div className="flex flex-col items-start z-10">
          {/* Eyebrow Status Pill */}
          <motion.div
            variants={itemVariants}
            className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/10 backdrop-blur-md mb-6"
          >
            <span className="flex h-2 w-2 relative">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  health?.isLive ? 'bg-emerald-400' : 'bg-emerald-500'
                }`}
              />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
            </span>
            <span className="mono-pill text-emerald-400 font-semibold tracking-wider">
              {health?.isLive ? 'SYSTEM NOMINAL // WIRE-SPEED ACTIVE' : '3-TIER GROUNDING ENGINE ACTIVE'}
            </span>
          </motion.div>

          {/* Massive Viewport-Scale Heading */}
          <motion.h1
            variants={itemVariants}
            className="hero-heading text-white tracking-tight mb-6"
          >
            Factual Grounding at <span className="text-emerald-400 italic font-semibold">Wire Speed</span>.
          </motion.h1>

          {/* Subtext (< 20 words, muted) */}
          <motion.p
            variants={itemVariants}
            className="text-base sm:text-lg text-slate-300/90 max-w-[44ch] leading-relaxed mb-8 font-normal"
          >
            Deterministic NLI claim verification and surgical hallucination mitigation with sub-15ms streaming fidelity.
          </motion.p>

          {/* Primary CTA and Secondary Link */}
          <motion.div variants={itemVariants} className="flex flex-wrap items-center gap-4">
            <a
              href="#playground"
              className="inline-flex items-center gap-2.5 px-8 py-3.5 rounded-full bg-emerald-500 text-slate-950 font-semibold text-sm tracking-wide hover:bg-emerald-400 active:scale-98 transition-all shadow-[0_0_30px_rgba(16,185,129,0.35)]"
            >
              <span>Open Live Playground</span>
              <ArrowRight className="w-4 h-4" />
            </a>

            <a
              href="#telemetry"
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 hover:text-white border border-white/10 text-sm font-medium transition-all"
            >
              <span>View Benchmark Telemetry</span>
            </a>
          </motion.div>

          {/* Monospace Architectural Badges Strip */}
          <motion.div
            variants={itemVariants}
            className="mt-12 pt-8 border-t border-white/10 w-full grid grid-cols-3 gap-4"
          >
            <div>
              <div className="text-[11px] font-mono uppercase text-slate-400">Zero Leakage</div>
              <div className="text-xl font-bold font-mono text-emerald-400 tracking-tight mt-0.5">0.00%</div>
            </div>
            <div>
              <div className="text-[11px] font-mono uppercase text-slate-400">Recall @ 10</div>
              <div className="text-xl font-bold font-mono text-white tracking-tight mt-0.5">100.0%</div>
            </div>
            <div>
              <div className="text-[11px] font-mono uppercase text-slate-400">Contradiction Prec.</div>
              <div className="text-xl font-bold font-mono text-emerald-400 tracking-tight mt-0.5">100.0%</div>
            </div>
          </motion.div>
        </div>

        {/* Right Column: Interactive Live Verification Terminal Card */}
        <motion.div variants={itemVariants} className="relative z-10 w-full">
          {/* Card Frame */}
          <div className="relative rounded-2xl bg-[#0b0e15]/90 border border-white/12 shadow-[0_20px_50px_rgba(0,0,0,0.7)] backdrop-blur-xl overflow-hidden">
            {/* Terminal Top Bar */}
            <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/10 bg-white/[0.02]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
                <span className="ml-2 font-mono text-[11px] text-slate-400 tracking-wide">
                  ORCHESTRATOR // NLI-INSPECT
                </span>
              </div>

              {/* View toggle */}
              <div className="flex items-center gap-1 bg-black/40 p-1 rounded-lg border border-white/10 text-[11px] font-mono">
                <button
                  onClick={() => setActiveTab('annotated')}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    activeTab === 'annotated'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Raw Claims
                </button>
                <button
                  onClick={() => setActiveTab('mitigated')}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    activeTab === 'mitigated'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Rewritten Output
                </button>
              </div>
            </div>

            {/* Terminal Body */}
            <div className="p-6 space-y-5">
              {activeTab === 'annotated' ? (
                <>
                  {/* Interactive Claim Proposition Strip */}
                  <div className="space-y-2.5">
                    <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider flex items-center justify-between">
                      <span>Parsed Propositions (Click to audit)</span>
                      <span className="text-emerald-400">3 CLAIMS IDENTIFIED</span>
                    </div>

                    <div className="space-y-2">
                      {heroClaims.map((claim, idx) => {
                        const isContradicted = claim.state === 'CONTRADICTED';
                        const isSelected = selectedClaimIdx === idx;

                        return (
                          <div
                            key={claim.id}
                            onClick={() => setSelectedClaimIdx(idx)}
                            className={`p-3 rounded-xl border text-xs cursor-pointer transition-all ${
                              isSelected
                                ? isContradicted
                                  ? 'bg-rose-950/30 border-rose-500/60 shadow-[0_0_15px_rgba(244,63,94,0.15)]'
                                  : 'bg-emerald-950/30 border-emerald-500/60 shadow-[0_0_15px_rgba(16,185,129,0.15)]'
                                : 'bg-white/[0.02] border-white/8 hover:border-white/20'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <span className="text-slate-200 font-medium leading-relaxed">
                                {claim.text}
                              </span>
                              <span
                                className={`shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                                  isContradicted
                                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                }`}
                              >
                                {isContradicted ? (
                                  <AlertTriangle className="w-3 h-3 text-rose-400" />
                                ) : (
                                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                                )}
                                {claim.state}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Focused Claim NLI Breakdown Meter */}
                  <div className="p-4 rounded-xl bg-black/40 border border-white/8 space-y-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-slate-400 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                        <Cpu className="w-3.5 h-3.5 text-slate-400" />
                        DeBERTa-v3 Cross-Encoder Posterior
                      </span>
                      <span className="font-mono text-[11px] text-slate-300">
                        Source: {currentClaim.tier}
                      </span>
                    </div>

                    {/* Probability bars */}
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-[11px] font-mono mb-1">
                          <span className="text-emerald-400 font-medium">Entailment</span>
                          <span className="text-slate-300">
                            {(currentClaim.p_entailment * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                            style={{ width: `${currentClaim.p_entailment * 100}%` }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-[11px] font-mono mb-1">
                          <span className="text-rose-400 font-medium">Contradiction</span>
                          <span className="text-slate-300">
                            {(currentClaim.p_contradiction * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-rose-500 rounded-full transition-all duration-500"
                            style={{ width: `${currentClaim.p_contradiction * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="pt-2 text-[11px] text-slate-400 font-mono flex items-center gap-1.5 border-t border-white/5">
                      <span className="text-slate-400">Premise:</span>
                      <span className="text-slate-300 italic truncate">{currentClaim.evidence}</span>
                    </div>
                  </div>
                </>
              ) : (
                /* Mitigated Text Preview */
                <div className="p-5 rounded-xl bg-black/50 border border-emerald-500/20 space-y-4">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-emerald-400 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                      Gemma 2B Targeted Rewrite Active
                    </span>
                    <span className="text-slate-400">0% Token Leakage</span>
                  </div>

                  <p className="text-sm text-slate-200 leading-relaxed font-sans">
                    Apollo 11 landed on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar
                    Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle,{' '}
                    <mark className="bg-emerald-500/20 text-emerald-300 px-1 py-0.5 rounded border border-emerald-500/30 font-medium">
                      while command module pilot Michael Collins remained in lunar orbit aboard Columbia
                    </mark>
                    . Neil Armstrong and Buzz Aldrin deployed the Apollo EASEP scientific package.
                  </p>

                  <div className="pt-3 border-t border-white/10 flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>Mitigation: REWRITE</span>
                    <span className="text-emerald-400">Safety Verdict: SAFE</span>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Terminal Footnote */}
            <div className="px-5 py-2.5 bg-white/[0.015] border-t border-white/8 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span className="flex items-center gap-1">
                <Layers className="w-3 h-3 text-slate-400" />
                Deterministic Sentence & Clause Segmentation
              </span>
              <span>τ_severe = 0.85</span>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
};
