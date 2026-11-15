import React from 'react';
import { ShieldCheck, BookOpen, ArrowUp, GitBranch } from 'lucide-react';

export const Footer: React.FC = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="border-t border-white/8 py-16 px-6 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8 pb-12 border-b border-white/5">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <span className="font-bold text-base text-white tracking-tight">
              DEHALLUCINATION <span className="text-emerald-400 font-mono text-xs">GATEWAY</span>
            </span>
          </div>
          <p className="text-xs text-slate-400 max-w-md font-sans">
            Real-Time Dual-Mode Hallucination Mitigation & NLI Waterfall Grounding Sidecar for LLM
            Applications.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-6 text-xs font-mono text-slate-400">
          <a
            href="https://github.com/Dheeraj-2005"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-emerald-400 transition-colors flex items-center gap-1.5"
          >
            <GitBranch className="w-4 h-4" />
            GitHub Repo
          </a>
          <a
            href="#waterfall"
            className="hover:text-emerald-400 transition-colors flex items-center gap-1.5"
          >
            <BookOpen className="w-4 h-4" />
            Architecture
          </a>
          <a
            href="#telemetry"
            className="hover:text-emerald-400 transition-colors flex items-center gap-1.5"
          >
            Benchmarks
          </a>
          <button
            onClick={scrollToTop}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white transition-colors cursor-pointer"
            title="Scroll to top"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] font-mono text-slate-400">
        <div>
          PES University • Department of Computer Science & Engineering • UE24CS341A
        </div>
        <div className="flex items-center gap-4">
          <span>Sprint 1 & Sprint 2 Certified</span>
          <span className="text-emerald-400">314 Tests Passing (90.4% Coverage)</span>
        </div>
      </div>
    </footer>
  );
};
