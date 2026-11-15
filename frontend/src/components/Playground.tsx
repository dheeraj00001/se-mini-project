import React, { useState } from 'react';
import {
  Play,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Copy,
  Check,
  Zap,
  Sliders,
  Send,
  FileText,
  Layers,
} from 'lucide-react';
import type {
  ClaimReport,
  MitigationMode,
  VerificationRequest,
  VerificationResponse,
} from '../types';
import type { PresetDemo } from '../services/api';
import {
  PRESET_DEMOS,
  streamChatCompletions,
  verifyText,
} from '../services/api';

export const Playground: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'verify' | 'streaming'>('verify');
  const [selectedPresetId, setSelectedPresetId] = useState<string>(PRESET_DEMOS[0].id);

  // Verification State
  const [textInput, setTextInput] = useState<string>(PRESET_DEMOS[0].text);
  const [contextDocs, setContextDocs] = useState<string>(
    PRESET_DEMOS[0].context.join('\n\n')
  );
  const [mitigationMode, setMitigationMode] = useState<MitigationMode>(PRESET_DEMOS[0].mode);
  const [maxTier, setMaxTier] = useState<number>(PRESET_DEMOS[0].maxTier);
  const [isVerifying, setIsVerifying] = useState<boolean>(false);
  const [verifyResult, setVerifyResult] = useState<VerificationResponse | null>(null);
  const [selectedClaim, setSelectedClaim] = useState<ClaimReport | null>(null);
  const [verifyLatency, setVerifyLatency] = useState<number | null>(null);
  const [isLiveApi, setIsLiveApi] = useState<boolean>(false);

  // Streaming State
  const [streamPrompt, setStreamPrompt] = useState<string>(
    'Summarize the primary crew members of the Apollo 11 lunar mission and their roles.'
  );
  const [streamMitigation, setStreamMitigation] = useState<MitigationMode>('rewrite');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamedText, setStreamedText] = useState<string>('');
  const [streamHeaders, setStreamHeaders] = useState<Record<string, string> | null>(null);
  const [streamIsLive, setStreamIsLive] = useState<boolean>(false);

  // Copy feedback state
  const [copiedSanitized, setCopiedSanitized] = useState(false);

  // Apply preset
  const handleSelectPreset = (preset: PresetDemo) => {
    setSelectedPresetId(preset.id);
    setTextInput(preset.text);
    setContextDocs(preset.context.join('\n\n'));
    setMitigationMode(preset.mode);
    setMaxTier(preset.maxTier);
    setVerifyResult(null);
    setSelectedClaim(null);
  };

  // Run verification
  const handleRunVerify = async () => {
    if (!textInput.trim()) return;
    setIsVerifying(true);
    setSelectedClaim(null);

    const contextArray = contextDocs
      .split('\n\n')
      .map((c) => c.trim())
      .filter(Boolean);

    const req: VerificationRequest = {
      text: textInput,
      context_documents: contextArray,
      max_retrieval_tier: maxTier,
      mitigation_mode: mitigationMode,
    };

    const res = await verifyText(req);
    setVerifyResult(res.data);
    setIsLiveApi(res.isLiveBackend);
    setVerifyLatency(res.latencyMs);
    if (res.data.claims && res.data.claims.length > 0) {
      // Pick first contradicted or first claim
      const contradicted = res.data.claims.find((c) => c.state === 'CONTRADICTED');
      setSelectedClaim(contradicted || res.data.claims[0]);
    }
    setIsVerifying(false);
  };

  // Run streaming
  const handleRunStream = async () => {
    if (!streamPrompt.trim() || isStreaming) return;
    setIsStreaming(true);
    setStreamedText('');
    setStreamHeaders(null);

    const res = await streamChatCompletions(
      {
        model: 'google/gemma-2b-it',
        messages: [{ role: 'user', content: streamPrompt }],
        stream: true,
        mitigation_mode: streamMitigation,
      },
      (chunk) => {
        setStreamedText((prev) => prev + chunk);
      },
      (headers) => {
        setStreamHeaders(headers);
      }
    );

    setStreamIsLive(res.isLive);
    setIsStreaming(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSanitized(true);
    setTimeout(() => setCopiedSanitized(false), 2000);
  };

  return (
    <section id="playground" className="py-24 lg:py-32 px-6 max-w-7xl mx-auto">
      {/* Section Header */}
      <div className="flex flex-col items-center text-center mb-16">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-emerald-400 font-mono text-xs uppercase tracking-widest mb-4">
          <Sparkles className="w-3.5 h-3.5" />
          Live Interactive Verification Suite
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white max-w-2xl">
          Test Factual Verification in Real Time
        </h2>
        <p className="text-slate-400 text-base max-w-xl mt-4">
          Submit proposition claims to the NLI cross-encoder, toggle grounding tiers, and evaluate
          surgical mitigation modes without model hallucination leakage.
        </p>

        {/* Dual Mode Switcher Tabs */}
        <div className="mt-8 flex items-center p-1.5 rounded-2xl bg-[#0d1017] border border-white/10 shadow-inner">
          <button
            onClick={() => setActiveTab('verify')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-xs font-semibold font-mono tracking-wide transition-all ${
              activeTab === 'verify'
                ? 'bg-emerald-500 text-slate-950 shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            Mode 1: Sidecar Verification (/verify)
          </button>
          <button
            onClick={() => setActiveTab('streaming')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-xs font-semibold font-mono tracking-wide transition-all ${
              activeTab === 'streaming'
                ? 'bg-emerald-500 text-slate-950 shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Zap className="w-4 h-4" />
            Mode 2: OpenAI Proxy Stream (/v1/chat/completions)
          </button>
        </div>
      </div>

      {activeTab === 'verify' ? (
        /* ================= MODE 1: SIDECAR VERIFICATION ================= */
        <div className="space-y-8">
          {/* Preset Selector Strip */}
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-emerald-400" />
              Pre-loaded Factual Calibration Presets:
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {PRESET_DEMOS.map((preset) => {
                const isSelected = selectedPresetId === preset.id;
                return (
                  <button
                    key={preset.id}
                    onClick={() => handleSelectPreset(preset)}
                    className={`p-3.5 rounded-xl border text-left transition-all ${
                      isSelected
                        ? 'bg-emerald-950/30 border-emerald-500/60 shadow-[0_0_20px_rgba(16,185,129,0.15)]'
                        : 'bg-[#0d1017]/70 border-white/8 hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-white/5 text-slate-300">
                        {preset.category}
                      </span>
                      <span className="text-[10px] font-mono text-emerald-400">
                        Tier {preset.maxTier}
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-white">{preset.title}</div>
                    <p className="text-[11px] text-slate-400 line-clamp-2 mt-1 leading-snug">
                      {preset.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Interactive Playground Control Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            {/* Left: Input Configuration */}
            <div className="space-y-6 bg-[#0b0e15] border border-white/10 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono uppercase text-slate-300 tracking-wider flex items-center gap-2">
                  <FileText className="w-4 h-4 text-emerald-400" />
                  Generated Text to Verify
                </label>
                <span className="text-[11px] font-mono text-slate-400">
                  {textInput.length} chars
                </span>
              </div>

              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                rows={5}
                placeholder="Enter AI-generated text with factual assertions..."
                className="w-full rounded-xl bg-black/60 border border-white/10 p-4 text-xs sm:text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/30 font-sans leading-relaxed resize-y"
              />

              {/* In-Context Grounding Passages (Tier 1) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-mono uppercase text-slate-300 tracking-wider flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-slate-400" />
                    Tier 1 In-Context Passages (Separate with empty line)
                  </label>
                  <span className="text-[11px] font-mono text-emerald-400">0ms Fast Grounding</span>
                </div>
                <textarea
                  value={contextDocs}
                  onChange={(e) => setContextDocs(e.target.value)}
                  rows={4}
                  placeholder="Paste reference passages to serve as in-context truth..."
                  className="w-full rounded-xl bg-black/60 border border-white/10 p-3.5 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/30 font-mono leading-relaxed resize-y"
                />
              </div>

              {/* Configuration Controls: Max Tier & Mitigation Mode */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                {/* Max Retrieval Tier */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-mono uppercase text-slate-400">
                    Max Retrieval Waterfall:
                  </span>
                  <div className="grid grid-cols-3 gap-1 bg-black/50 p-1 rounded-xl border border-white/10">
                    {[
                      { tier: 1, label: 'T1: Context' },
                      { tier: 2, label: 'T2: Local KB' },
                      { tier: 3, label: 'T3: Web' },
                    ].map((t) => (
                      <button
                        key={t.tier}
                        type="button"
                        onClick={() => setMaxTier(t.tier)}
                        className={`py-1.5 rounded-lg text-[10px] font-mono font-medium transition-all ${
                          maxTier === t.tier
                            ? 'bg-emerald-500 text-slate-950 font-bold'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Mitigation Mode */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-mono uppercase text-slate-400">
                    Mitigation Action:
                  </span>
                  <div className="grid grid-cols-4 gap-1 bg-black/50 p-1 rounded-xl border border-white/10">
                    {(['rewrite', 'redact', 'block', 'flag'] as MitigationMode[]).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setMitigationMode(mode)}
                        className={`py-1.5 rounded-lg text-[10px] font-mono uppercase font-medium transition-all ${
                          mitigationMode === mode
                            ? 'bg-emerald-500 text-slate-950 font-bold'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                onClick={handleRunVerify}
                disabled={isVerifying}
                className="w-full py-3.5 px-6 rounded-xl bg-emerald-500 hover:bg-emerald-400 active:scale-98 disabled:opacity-50 text-slate-950 font-bold text-sm tracking-wide transition-all shadow-[0_0_25px_rgba(16,185,129,0.3)] flex items-center justify-center gap-2 cursor-pointer"
              >
                {isVerifying ? (
                  <>
                    <RotateCcw className="w-4 h-4 animate-spin text-slate-950" />
                    <span>Executing NLI DeBERTa Cross-Encoder...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Verify Claims (POST /verify)</span>
                  </>
                )}
              </button>
            </div>

            {/* Right: Real-time Telemetry & Output */}
            <div className="space-y-6">
              {verifyResult ? (
                <div className="space-y-6 bg-[#0b0e15] border border-white/10 rounded-2xl p-6 shadow-xl">
                  {/* Verdict Banner */}
                  <div className="flex items-center justify-between pb-4 border-b border-white/10">
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-9 h-9 rounded-xl flex items-center justify-center ${
                          verifyResult.is_safe
                            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                            : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                        }`}
                      >
                        {verifyResult.is_safe ? (
                          <CheckCircle2 className="w-5 h-5" />
                        ) : (
                          <AlertTriangle className="w-5 h-5" />
                        )}
                      </div>
                      <div>
                        <div className="text-xs font-mono uppercase text-slate-400">
                          Safety Verdict
                        </div>
                        <div
                          className={`text-base font-bold font-mono tracking-tight ${
                            verifyResult.is_safe ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          {verifyResult.is_safe
                            ? 'VERIFIED SAFE (τ_c < 0.85)'
                            : 'CONTRADICTION DETECTED'}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-[10px] font-mono uppercase px-2 py-1 rounded bg-white/5 text-slate-300">
                        {isLiveApi ? 'FastAPI Uvicorn' : 'Offline Simulator'}
                      </span>
                      {verifyLatency !== null && (
                        <div className="text-[11px] font-mono text-slate-400 mt-1">
                          {verifyLatency}ms latency
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Rate Meters */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 rounded-xl bg-black/40 border border-white/8 text-center">
                      <div className="text-[10px] font-mono uppercase text-slate-400">
                        Hallucination
                      </div>
                      <div className="text-lg font-bold font-mono text-rose-400 mt-0.5">
                        {verifyResult.hallucination_rate.toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-3 rounded-xl bg-black/40 border border-white/8 text-center">
                      <div className="text-[10px] font-mono uppercase text-slate-400">
                        Verified Rate
                      </div>
                      <div className="text-lg font-bold font-mono text-emerald-400 mt-0.5">
                        {verifyResult.verification_rate.toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-3 rounded-xl bg-black/40 border border-white/8 text-center">
                      <div className="text-[10px] font-mono uppercase text-slate-400">
                        Claims Extracted
                      </div>
                      <div className="text-lg font-bold font-mono text-white mt-0.5">
                        {verifyResult.claims.length}
                      </div>
                    </div>
                  </div>

                  {/* Claims Inspector Strip */}
                  <div className="space-y-2">
                    <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                      Segmented Propositions (Select to view evidence):
                    </div>
                    <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                      {verifyResult.claims.map((claim) => {
                        const isContradicted = claim.state === 'CONTRADICTED';
                        const isSelected = selectedClaim?.claim_id === claim.claim_id;

                        return (
                          <div
                            key={claim.claim_id}
                            onClick={() => setSelectedClaim(claim)}
                            className={`p-3 rounded-xl border text-xs cursor-pointer transition-all ${
                              isSelected
                                ? isContradicted
                                  ? 'bg-rose-950/30 border-rose-500/70'
                                  : 'bg-emerald-950/30 border-emerald-500/70'
                                : 'bg-black/40 border-white/8 hover:border-white/20'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <span className="text-slate-200">{claim.claim_text}</span>
                              <span
                                className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                                  isContradicted
                                    ? 'bg-rose-500/20 text-rose-300'
                                    : 'bg-emerald-500/20 text-emerald-300'
                                }`}
                              >
                                {claim.state}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Selected Claim Deep Inspection */}
                  {selectedClaim && (
                    <div className="p-4 rounded-xl bg-black/60 border border-white/10 space-y-2.5">
                      <div className="flex items-center justify-between text-[11px] font-mono">
                        <span className="text-slate-400">
                          Claim #{selectedClaim.claim_id} Posterior Probabilities:
                        </span>
                        <span className="text-emerald-400">
                          Tier {selectedClaim.evidence_tier || 1} Evidence
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-2 text-center text-[11px] font-mono">
                        <div className="p-2 rounded bg-white/[0.03] border border-white/5">
                          <span className="text-emerald-400 block font-semibold">
                            {(selectedClaim.probabilities.entailment * 100).toFixed(1)}%
                          </span>
                          <span className="text-slate-400 text-[10px]">Entailment</span>
                        </div>
                        <div className="p-2 rounded bg-white/[0.03] border border-white/5">
                          <span className="text-rose-400 block font-semibold">
                            {(selectedClaim.probabilities.contradiction * 100).toFixed(1)}%
                          </span>
                          <span className="text-slate-400 text-[10px]">Contradiction</span>
                        </div>
                        <div className="p-2 rounded bg-white/[0.03] border border-white/5">
                          <span className="text-slate-300 block font-semibold">
                            {(selectedClaim.probabilities.neutral * 100).toFixed(1)}%
                          </span>
                          <span className="text-slate-400 text-[10px]">Neutral</span>
                        </div>
                      </div>

                      {selectedClaim.evidence_text && (
                        <div className="text-[11px] font-mono text-slate-300 pt-2 border-t border-white/5 leading-relaxed">
                          <span className="text-slate-400">Grounding Evidence: </span>
                          <span className="italic">{selectedClaim.evidence_text}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Sanitized Text Result */}
                  <div className="space-y-2 pt-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-emerald-400 flex items-center gap-1.5 uppercase tracking-wider">
                        <Sparkles className="w-3.5 h-3.5" />
                        Finalized Sanitized Output:
                      </span>
                      <button
                        onClick={() => copyToClipboard(verifyResult.sanitized_text)}
                        className="text-slate-400 hover:text-white flex items-center gap-1 text-[11px]"
                      >
                        {copiedSanitized ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                        <span>{copiedSanitized ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>

                    <div className="p-4 rounded-xl bg-black/70 border border-emerald-500/30 text-xs sm:text-sm text-slate-100 font-sans leading-relaxed">
                      {verifyResult.sanitized_text}
                    </div>
                  </div>
                </div>
              ) : (
                /* Empty Prompt State */
                <div className="bg-[#0b0e15] border border-dashed border-white/10 rounded-2xl p-12 text-center flex flex-col items-center justify-center space-y-4">
                  <div className="w-12 h-12 rounded-2xl bg-white/[0.02] border border-white/8 flex items-center justify-center text-slate-400">
                    <ShieldCheck className="w-6 h-6" />
                  </div>
                  <h3 className="text-base font-semibold text-white">Awaiting Verification Task</h3>
                  <p className="text-xs text-slate-400 max-w-sm leading-relaxed">
                    Choose one of the presets above or type custom propositions into the input editor,
                    then click <strong className="text-emerald-400 font-mono">Verify Claims</strong> to
                    trigger the verification sidecar.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* ================= MODE 2: OPENAI PROXY STREAMING ================= */
        <div className="bg-[#0b0e15] border border-white/10 rounded-2xl p-6 sm:p-8 shadow-2xl max-w-4xl mx-auto space-y-8">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/10">
            <div>
              <div className="text-xs font-mono uppercase text-emerald-400 tracking-wider flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Buffered Reverse Proxy + Synthetic SSE Playback
              </div>
              <h3 className="text-lg font-bold text-white mt-1">
                OpenAI-Compatible Streaming Endpoint
              </h3>
            </div>

            <div className="flex items-center gap-2 text-xs font-mono">
              <span className="text-slate-400">Mitigation:</span>
              <select
                value={streamMitigation}
                onChange={(e) => setStreamMitigation(e.target.value as MitigationMode)}
                className="bg-black/60 border border-white/10 rounded-lg px-2.5 py-1 text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                <option value="rewrite">Rewrite (Gemma 2B)</option>
                <option value="redact">Redact (Surgical)</option>
                <option value="block">Block (Safety Gate)</option>
                <option value="flag">Flag (Audit)</option>
              </select>
            </div>
          </div>

          {/* Prompt Input */}
          <div className="space-y-2">
            <label className="text-xs font-mono uppercase text-slate-400 tracking-wider">
              Prompt Input (User Role):
            </label>
            <div className="flex gap-3">
              <input
                type="text"
                value={streamPrompt}
                onChange={(e) => setStreamPrompt(e.target.value)}
                placeholder="Ask a factual question..."
                className="flex-1 rounded-xl bg-black/60 border border-white/10 px-4 py-3 text-xs sm:text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/60"
              />
              <button
                onClick={handleRunStream}
                disabled={isStreaming}
                className="px-6 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 active:scale-98 disabled:opacity-50 text-slate-950 font-bold text-xs tracking-wide transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] flex items-center gap-2 shrink-0 cursor-pointer"
              >
                {isStreaming ? (
                  <RotateCcw className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                <span>Stream Response</span>
              </button>
            </div>
          </div>

          {/* Response Telemetry Headers Strip */}
          {streamHeaders && (
            <div className="p-4 rounded-xl bg-black/40 border border-white/8 space-y-2">
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider flex items-center justify-between">
                <span>Injected Gateway Headers (Zero Token Leakage)</span>
                <span className="text-emerald-400">
                  {streamIsLive ? 'FastAPI SSE Replay' : 'Simulated SSE Replay'}
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-slate-400 block text-[10px]">X-Dehallucination-Safe</span>
                  <span className="text-emerald-400 font-bold">
                    {streamHeaders['x-dehallucination-safe']}
                  </span>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-slate-400 block text-[10px]">X-Hallucination-Rate</span>
                  <span className="text-white font-bold">
                    {streamHeaders['x-hallucination-rate']}%
                  </span>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-slate-400 block text-[10px]">X-Verification-Rate</span>
                  <span className="text-emerald-400 font-bold">
                    {streamHeaders['x-verification-rate']}%
                  </span>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-slate-400 block text-[10px]">X-Needs-Review</span>
                  <span className="text-white font-bold">
                    {streamHeaders['x-needs-review']}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* SSE Terminal Stream Box */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Synthetic SSE Event Stream:</span>
              {isStreaming && (
                <span className="inline-flex items-center gap-1.5 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Streaming 15ms Paced Tokens...
                </span>
              )}
            </div>

            <div className="min-h-36 p-5 rounded-xl bg-black/80 border border-white/10 font-mono text-xs sm:text-sm text-slate-200 leading-relaxed overflow-x-auto">
              {streamedText ? (
                <span>
                  {streamedText}
                  {isStreaming && (
                    <span className="inline-block w-2 h-4 bg-emerald-400 ml-1 animate-pulse align-middle" />
                  )}
                </span>
              ) : (
                <span className="text-slate-600 italic">
                  Tokens will stream here in real-time after 100% full-response buffering and verification...
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
