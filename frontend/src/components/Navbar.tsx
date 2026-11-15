import React, { useEffect, useState } from 'react';
import { ShieldCheck, Activity, Terminal, ArrowUpRight, Cpu, GitBranch, Users } from 'lucide-react';
import type { BackendHealth } from '../types';

interface NavbarProps {
  health: BackendHealth | null;
  onRefreshHealth: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, onRefreshHealth }) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className="fixed top-5 left-1/2 -translate-x-1/2 z-50 w-[94%] max-w-6xl transition-all duration-300">
      <nav
        className={`flex items-center justify-between px-4 sm:px-6 py-3 rounded-full border transition-all duration-300 ${
          scrolled
            ? 'bg-[#090b10]/90 backdrop-blur-xl border-white/15 shadow-[0_8px_32px_rgba(0,0,0,0.6)]'
            : 'bg-[#0b0e14]/75 backdrop-blur-md border-white/10 shadow-[0_4px_24px_rgba(0,0,0,0.4)]'
        }`}
      >
        {/* Brand identity */}
        <a href="#" className="flex items-center gap-3 group">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-950/40 border border-emerald-500/30 group-hover:border-emerald-400 transition-colors">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            <span className="absolute inset-0 rounded-xl bg-emerald-400/10 blur-sm pointer-events-none group-hover:opacity-100 opacity-50 transition-opacity" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-sm tracking-tight text-white flex items-center gap-1.5">
              DEHALLUCINATION <span className="text-emerald-400 font-mono text-[11px] font-semibold">GATEWAY</span>
            </span>
            <span className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">
              PESU • AIML
            </span>
          </div>
        </a>

        {/* Center navigation links */}
        <div className="hidden md:flex items-center gap-1 lg:gap-2 text-xs font-medium text-slate-300">
          <a
            href="#playground"
            className="px-3 py-1.5 rounded-full hover:text-white hover:bg-white/5 transition-colors flex items-center gap-1.5"
          >
            <Terminal className="w-3.5 h-3.5 text-emerald-400" />
            Playground
          </a>
          <a
            href="#waterfall"
            className="px-3 py-1.5 rounded-full hover:text-white hover:bg-white/5 transition-colors flex items-center gap-1.5"
          >
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            Waterfall
          </a>
          <a
            href="#telemetry"
            className="px-3 py-1.5 rounded-full hover:text-white hover:bg-white/5 transition-colors flex items-center gap-1.5"
          >
            <Activity className="w-3.5 h-3.5 text-slate-400" />
            Benchmarks
          </a>
          <a
            href="#team"
            className="px-3 py-1.5 rounded-full hover:text-white hover:bg-white/5 transition-colors flex items-center gap-1.5"
          >
            <Users className="w-3.5 h-3.5 text-slate-400" />
            Team (4 Students)
          </a>
          <a
            href="#integration"
            className="px-3 py-1.5 rounded-full hover:text-white hover:bg-white/5 transition-colors"
          >
            API & SDK
          </a>
        </div>

        {/* Right action & backend status ping */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          <button
            onClick={onRefreshHealth}
            title="Click to re-ping FastAPI backend at http://localhost:8000"
            className={`hidden sm:flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-mono transition-all border ${
              health?.isLive
                ? 'bg-emerald-950/40 text-emerald-300 border-emerald-500/30 hover:border-emerald-400/60'
                : 'bg-amber-950/30 text-amber-300 border-amber-500/30 hover:border-amber-400/60'
            }`}
          >
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  health?.isLive ? 'bg-emerald-400' : 'bg-amber-400'
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  health?.isLive ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
              />
            </span>
            <span>
              {health?.isLive ? `Uvicorn :8000 (${health.latencyMs}ms)` : 'FastAPI Offline (Simulator)'}
            </span>
          </button>

          <a
            href="https://github.com/dheeraj00001/se-mini-project"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-full text-slate-400 hover:text-white hover:bg-white/5 transition-colors flex items-center justify-center border border-transparent hover:border-white/10"
            title="View on GitHub"
          >
            <GitBranch className="w-4 h-4 text-slate-300 hover:text-emerald-400 transition-colors" />
          </a>

          <a
            href="#playground"
            className="flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold bg-emerald-500 text-slate-950 hover:bg-emerald-400 active:scale-95 transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)]"
          >
            <span>Launch Demo</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </a>
        </div>
      </nav>
    </header>
  );
};
