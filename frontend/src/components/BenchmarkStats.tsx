import React, { useState } from 'react';
import {
  Activity,
  BarChart3,
  Maximize2,
  X,
  CheckCircle2,
} from 'lucide-react';

interface BenchmarkFigure {
  id: string;
  title: string;
  subtitle: string;
  src: string;
  caption: string;
}

export const BenchmarkStats: React.FC = () => {
  const [selectedFigure, setSelectedFigure] = useState<BenchmarkFigure | null>(null);
  const [activeTab, setActiveTab] = useState<string>('confusion');

  const figures: BenchmarkFigure[] = [
    {
      id: 'confusion',
      title: 'Confusion Matrix (Normalized)',
      subtitle: 'Evaluation on Held-Out Test Split (FEVER & HaluEval)',
      src: '/figures/confusion_matrix.png',
      caption:
        'Demonstrates 100% precision on CONTRADICTED claims with zero false accusations against ground-truth facts.',
    },
    {
      id: 'roc',
      title: 'Multiclass ROC Curves (OvR)',
      subtitle: 'One-vs-Rest Cross-Encoder Discriminative Capacity',
      src: '/figures/roc_curves.png',
      caption:
        'Cross-encoder ROC-AUC curves demonstrating robust separation across Supported, Contradicted, and Unverified claim states.',
    },
    {
      id: 'calibration',
      title: 'NLI Threshold Calibration Grid',
      subtitle: 'Grid Search on Held-Out Calibration Split (ADR-0009)',
      src: '/figures/calibration_grid.png',
      caption:
        'Threshold calibration curve yielding optimal decision boundaries: τ_entailment = 0.50, τ_contradiction = 0.50.',
    },
    {
      id: 'retrieval',
      title: 'Hybrid Retrieval Recall@k & MRR',
      subtitle: 'Qdrant Dense 384d + BM25 Lexical Keyword RRF',
      src: '/figures/retrieval_performance.png',
      caption:
        'Reciprocal Rank Fusion achieves 100% Recall@5 and Recall@10 with a Mean Reciprocal Rank (MRR) of 0.7000.',
    },
  ];

  const currentFigure = figures.find((f) => f.id === activeTab) || figures[0];

  return (
    <section id="telemetry" className="py-24 lg:py-32 px-6 max-w-7xl mx-auto border-t border-white/5">
      {/* Section Header */}
      <div className="flex flex-col items-center text-center mb-16">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-emerald-400 font-mono text-xs uppercase tracking-widest mb-4">
          <Activity className="w-3.5 h-3.5" />
          Academic Telemetry & Evaluation
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white max-w-2xl">
          Empirical Benchmark Results
        </h2>
        <p className="text-slate-400 text-base max-w-xl mt-4">
          Evaluated rigorously across 120 samples (FEVER + HaluEval) following strict 70/30
          train-calibration vs test split isolation (ADR-0009).
        </p>
      </div>

      {/* Key Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-16">
        <div className="p-5 rounded-2xl bg-[#0a0d14] border border-white/8 text-center glass-panel-hover">
          <div className="text-[11px] font-mono uppercase text-slate-400">Test Accuracy</div>
          <div className="text-3xl font-extrabold font-mono text-white mt-1">60.71%</div>
          <div className="text-[10px] font-mono text-emerald-400 mt-1">84 Test Samples</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0d14] border border-white/8 text-center glass-panel-hover">
          <div className="text-[11px] font-mono uppercase text-slate-400">Macro F1</div>
          <div className="text-3xl font-extrabold font-mono text-white mt-1">0.6076</div>
          <div className="text-[10px] font-mono text-emerald-400 mt-1">Balanced Classes</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0d14] border border-emerald-500/30 text-center glass-panel-hover bg-emerald-950/10">
          <div className="text-[11px] font-mono uppercase text-emerald-400 font-semibold">
            Contradiction Prec.
          </div>
          <div className="text-3xl font-extrabold font-mono text-emerald-400 mt-1">100.0%</div>
          <div className="text-[10px] font-mono text-emerald-400/80 mt-1">Zero False Alarms</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0d14] border border-white/8 text-center glass-panel-hover">
          <div className="text-[11px] font-mono uppercase text-slate-400">Recall @ 10</div>
          <div className="text-3xl font-extrabold font-mono text-white mt-1">100.0%</div>
          <div className="text-[10px] font-mono text-emerald-400 mt-1">Hybrid RRF Fusion</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0d14] border border-white/8 text-center glass-panel-hover">
          <div className="text-[11px] font-mono uppercase text-slate-400">Mean RR (MRR)</div>
          <div className="text-3xl font-extrabold font-mono text-white mt-1">0.7000</div>
          <div className="text-[10px] font-mono text-emerald-400 mt-1">Top-Rank Quality</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#0a0d14] border border-emerald-500/30 text-center glass-panel-hover bg-emerald-950/10">
          <div className="text-[11px] font-mono uppercase text-emerald-400 font-semibold">
            Token Leakage
          </div>
          <div className="text-3xl font-extrabold font-mono text-emerald-400 mt-1">0.00%</div>
          <div className="text-[10px] font-mono text-emerald-400/80 mt-1">100% SSE Buffering</div>
        </div>
      </div>

      {/* Diagnostic Figure Showcase Box */}
      <div className="bg-[#0b0e15] border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
        {/* Figure Tabs Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/[0.02] flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-mono uppercase tracking-wider text-slate-300 font-bold">
              Diagnostic Evaluation Plots:
            </span>
          </div>

          <div className="flex items-center gap-1.5 bg-black/50 p-1 rounded-xl border border-white/10">
            {figures.map((fig) => (
              <button
                key={fig.id}
                onClick={() => setActiveTab(fig.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                  activeTab === fig.id
                    ? 'bg-emerald-500 text-slate-950 font-bold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {fig.title.split(' ')[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Figure Display Area */}
        <div className="p-6 sm:p-8 grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-8 items-center">
          {/* Chart Image Container with Hover Zoom Button */}
          <div className="relative group rounded-xl overflow-hidden bg-black/60 border border-white/10 p-2">
            <img
              src={currentFigure.src}
              alt={currentFigure.title}
              className="w-full h-auto object-contain rounded-lg max-h-[380px] mx-auto transition-transform duration-300 group-hover:scale-[1.01]"
            />
            <button
              onClick={() => setSelectedFigure(currentFigure)}
              className="absolute top-4 right-4 p-2.5 rounded-xl bg-black/80 hover:bg-black text-white border border-white/20 backdrop-blur-md transition-all shadow-lg cursor-pointer"
              title="Click to view full resolution"
            >
              <Maximize2 className="w-4 h-4 text-emerald-400" />
            </button>
          </div>

          {/* Explanation Text */}
          <div className="space-y-4">
            <div>
              <span className="text-xs font-mono uppercase text-emerald-400 tracking-wider">
                Figure Inspection
              </span>
              <h3 className="text-xl font-bold text-white mt-1">{currentFigure.title}</h3>
              <p className="text-xs font-mono text-slate-400 mt-1">{currentFigure.subtitle}</p>
            </div>

            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              {currentFigure.caption}
            </p>

            <div className="p-4 rounded-xl bg-black/40 border border-white/8 space-y-2">
              <div className="text-[11px] font-mono uppercase text-slate-400">
                Evaluation Verification Check:
              </div>
              <ul className="text-xs text-slate-300 space-y-1.5 font-mono">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Random seed fixed to 42 (NFR-QUAL-03)</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Held-out calibration split isolation (ADR-0009)</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Full multiclass confusion matrix verified</span>
                </li>
              </ul>
            </div>

            <button
              onClick={() => setSelectedFigure(currentFigure)}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] text-white border border-white/10 text-xs font-mono font-medium transition-all cursor-pointer"
            >
              <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Inspect High-Resolution Artifact</span>
            </button>
          </div>
        </div>
      </div>

      {/* Fullscreen Lightbox Modal */}
      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-4 sm:p-8"
          onClick={() => setSelectedFigure(null)}
        >
          <div
            className="relative max-w-5xl w-full bg-[#0b0e15] border border-white/20 rounded-2xl overflow-hidden p-6 shadow-2xl space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div>
                <h4 className="text-lg font-bold text-white">{selectedFigure.title}</h4>
                <p className="text-xs font-mono text-slate-400">{selectedFigure.subtitle}</p>
              </div>
              <button
                onClick={() => setSelectedFigure(null)}
                className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex items-center justify-center bg-black/60 rounded-xl p-4">
              <img
                src={selectedFigure.src}
                alt={selectedFigure.title}
                className="max-h-[70vh] w-auto object-contain rounded-lg"
              />
            </div>

            <p className="text-xs font-mono text-slate-300 text-center">
              {selectedFigure.caption}
            </p>
          </div>
        </div>
      )}
    </section>
  );
};
