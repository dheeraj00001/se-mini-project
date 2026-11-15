import React, { useState } from 'react';
import { Copy, Check, Code2 } from 'lucide-react';

export const CodeSnippets: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'python' | 'curl' | 'typescript'>('python');
  const [copied, setCopied] = useState<boolean>(false);

  const snippets = {
    python: `# 1. Use the official OpenAI Python SDK with Dehallucination Gateway
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-not-required",
    default_headers={
        # Enforce surgical rewrite or redaction mitigation on contradicted claims
        "X-Dehallucination-Mitigation": "rewrite",
        "X-Dehallucination-Max-Tier": "2",
    }
)

# 2. Issue streaming completion with zero token leakage guarantee
response = client.chat.completions.create(
    model="google/gemma-2b-it",
    messages=[
        {"role": "user", "content": "What were the primary events of the Apollo 11 moon landing?"}
    ],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
`,
    curl: `# 1. Sidecar Verification Endpoint (POST /verify)
curl -X POST http://localhost:8000/verify \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Apollo 11 landed on the Moon in July 1969. Michael Collins walked on Mars.",
    "context_documents": [
      "Apollo 11 landed on the Moon on July 20, 1969. Michael Collins stayed in lunar orbit."
    ],
    "max_retrieval_tier": 2,
    "mitigation_mode": "rewrite"
  }'

# 2. OpenAI-Compatible Chat Completions Reverse Proxy (POST /v1/chat/completions)
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "X-Dehallucination-Mitigation: redact" \\
  -H "X-Dehallucination-Max-Tier: 2" \\
  -d '{
    "model": "google/gemma-2b-it",
    "messages": [
      {"role": "user", "content": "Summarize the Apollo 11 mission facts."}
    ],
    "stream": true
  }'
`,
    typescript: `// TypeScript / Next.js client for Dehallucination Streaming Gateway
async function streamVerifiedChat(prompt: string) {
  const response = await fetch("http://localhost:8000/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Dehallucination-Mitigation": "rewrite",
      "X-Dehallucination-Max-Tier": "2",
    },
    body: JSON.stringify({
      model: "google/gemma-2b-it",
      messages: [{ role: "user", content: prompt }],
      stream: true,
    }),
  });

  // Inspect injected verification metadata headers
  console.log("Safe:", response.headers.get("x-dehallucination-safe"));
  console.log("Hallucination Rate:", response.headers.get("x-hallucination-rate"));

  // Read synthetic SSE chunks with 15ms pacing
  const reader = response.body?.getReader();
  const decoder = new TextDecoder("utf-8");

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;
    const chunk = decoder.decode(value);
    // Parse SSE 'data: {...}' payloads
  }
}
`,
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(snippets[activeTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="integration" className="py-24 lg:py-32 px-6 max-w-7xl mx-auto border-t border-white/5">
      <div className="flex flex-col items-center text-center mb-16">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-emerald-400 font-mono text-xs uppercase tracking-widest mb-4">
          <Code2 className="w-3.5 h-3.5" />
          Developer Quickstart
        </div>
        <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white max-w-2xl">
          Drop-in OpenAI Compatibility
        </h2>
        <p className="text-slate-400 text-base max-w-xl mt-4">
          Route standard requests through the gateway simply by swapping your API base URL. Add custom
          security headers to enforce mitigation policies dynamically.
        </p>
      </div>

      <div className="max-w-4xl mx-auto bg-[#0b0e15] border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
        {/* Tab Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/[0.02]">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
            <span className="ml-2 font-mono text-xs text-slate-400">gateway-client</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 bg-black/50 p-1 rounded-xl border border-white/10 text-xs font-mono">
              <button
                onClick={() => setActiveTab('python')}
                className={`px-3 py-1 rounded-lg transition-colors ${
                  activeTab === 'python'
                    ? 'bg-emerald-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Python SDK
              </button>
              <button
                onClick={() => setActiveTab('curl')}
                className={`px-3 py-1 rounded-lg transition-colors ${
                  activeTab === 'curl'
                    ? 'bg-emerald-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                cURL CLI
              </button>
              <button
                onClick={() => setActiveTab('typescript')}
                className={`px-3 py-1 rounded-lg transition-colors ${
                  activeTab === 'typescript'
                    ? 'bg-emerald-500 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                TypeScript Fetch
              </button>
            </div>

            <button
              onClick={handleCopy}
              className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white border border-white/10 transition-colors flex items-center gap-1.5 text-xs font-mono cursor-pointer"
              title="Copy snippet"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              <span className="hidden sm:inline">{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
        </div>

        {/* Code Block Container */}
        <div className="p-6 sm:p-8 bg-black/80 font-mono text-xs sm:text-sm text-slate-200 overflow-x-auto leading-relaxed">
          <pre>
            <code>{snippets[activeTab]}</code>
          </pre>
        </div>

        {/* Bottom Metadata */}
        <div className="px-6 py-3 bg-white/[0.015] border-t border-white/8 flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Port 8000 • CORS Enabled (*)</span>
          <span className="text-emerald-400">RFC SSE Streaming Compliant</span>
        </div>
      </div>
    </section>
  );
};
