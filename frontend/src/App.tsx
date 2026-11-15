import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { HeroSection } from './components/HeroSection';
import { Playground } from './components/Playground';
import { WaterfallDiagram } from './components/WaterfallDiagram';
import { BenchmarkStats } from './components/BenchmarkStats';
import { CodeSnippets } from './components/CodeSnippets';
import { Footer } from './components/Footer';
import type { BackendHealth } from './types';
import { checkBackendHealth } from './services/api';

export const App: React.FC = () => {
  const [health, setHealth] = useState<BackendHealth | null>(null);

  const refreshHealth = async () => {
    const res = await checkBackendHealth();
    setHealth(res);
  };

  useEffect(() => {
    let isMounted = true;
    checkBackendHealth().then((res) => {
      if (isMounted) setHealth(res);
    });

    const interval = setInterval(() => {
      checkBackendHealth().then((res) => {
        if (isMounted) setHealth(res);
      });
    }, 10000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="relative min-h-screen bg-[#07080a] text-slate-100 selection:bg-emerald-500/20 selection:text-emerald-300">
      {/* Floating Pill Nav */}
      <Navbar health={health} onRefreshHealth={refreshHealth} />

      {/* Hero Section (Awwwards Architecture B) */}
      <HeroSection health={health} />

      {/* Below-the-Fold Pipeline Sections */}
      <main className="space-y-12">
        {/* Core Live Verification & Streaming Playground */}
        <Playground />

        {/* 3-Tier Waterfall Grounding Architecture Flow */}
        <WaterfallDiagram />

        {/* Academic Benchmark Telemetry & Figure Lightbox */}
        <BenchmarkStats />

        {/* Developer Integration Tabs */}
        <CodeSnippets />
      </main>

      {/* Minimalist Editorial Footer */}
      <Footer />
    </div>
  );
};

export default App;
