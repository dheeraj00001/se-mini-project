import type {
  BackendHealth,
  ChatCompletionRequest,
  VerificationRequest,
  VerificationResponse,
} from '../types';

export const API_BASE = 'http://localhost:8000';

export interface PresetDemo {
  id: string;
  title: string;
  category: 'Entity Contradiction' | 'Temporal / Historical' | 'Numerical Fact' | 'Medical / Science';
  text: string;
  context: string[];
  maxTier: number;
  mode: 'rewrite' | 'redact' | 'block' | 'flag';
  description: string;
}

export const PRESET_DEMOS: PresetDemo[] = [
  {
    id: 'apollo-mission',
    title: 'Apollo 11 Moon Landing',
    category: 'Temporal / Historical',
    text: 'Apollo 11 landed on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle. However, astronaut Michael Collins famously walked on Mars during the return flight in 1972.',
    context: [
      'Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC.',
      'Michael Collins flew the command module Columbia alone in lunar orbit while his crewmates were on the surface. Collins never walked on Mars or landed on the Moon.',
    ],
    maxTier: 1,
    mode: 'rewrite',
    description: 'Detects fabricated Mars landing claim for Michael Collins and rewrites with historical ground truth.',
  },
  {
    id: 'eiffel-height',
    title: 'Eiffel Tower Architectural Height',
    category: 'Numerical Fact',
    text: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889. Its architectural height is 450 meters (1,476 feet), making it taller than the Empire State Building.',
    context: [
      'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World\'s Fair.',
      'The Eiffel Tower is 330 metres (1,083 ft) tall, including the tip, roughly the same height as an 81-storey building, and the tallest structure in Paris.',
    ],
    maxTier: 2,
    mode: 'redact',
    description: 'Surgically redacts severe numerical contradiction (450m vs 330m) without destroying valid historical context.',
  },
  {
    id: 'capital-swap',
    title: 'European Capital Cities',
    category: 'Entity Contradiction',
    text: 'Berlin is the capital and largest city of Germany by both area and population. Furthermore, Berlin is also the historic capital and governmental seat of France since 1789.',
    context: [
      'Berlin is the capital and largest city of Germany, both by area and by population.',
      'Paris is the capital and most populous city of France. Since the 17th century, Paris has been one of the world\'s major centres of finance, diplomacy, and commerce.',
    ],
    maxTier: 1,
    mode: 'block',
    description: 'Entity contradiction triggers safety gate circuit breaker under block mode.',
  },
  {
    id: 'penicillin-discovery',
    title: 'Discovery of Penicillin',
    category: 'Medical / Science',
    text: 'Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928 at St Mary\'s Hospital in London. Penicillin was initially synthesized by artificial intelligence algorithms in 1999.',
    context: [
      'Alexander Fleming discovered penicillin at St Mary\'s Hospital, London in September 1928, after returning to his laboratory having spent August on holiday with his family.',
      'Penicillin is a group of antibacterial antibiotics derived from Penicillium fungi, naturally isolated rather than artificially synthesized in 1999.',
    ],
    maxTier: 2,
    mode: 'rewrite',
    description: 'Identifies biological fact vs synthetic hallucination with hybrid vector & BM25 retrieval.',
  },
];

/**
 * Check backend health status and response latency.
 */
export async function checkBackendHealth(): Promise<BackendHealth> {
  const start = performance.now();
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const res = await fetch(`${API_BASE}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const latencyMs = Math.round(performance.now() - start);
    if (res.ok) {
      const data = await res.json();
      return {
        status: data.status || 'healthy',
        version: data.version || '1.0.0',
        latencyMs,
        isLive: true,
      };
    }
    return { status: 'error', version: 'unknown', latencyMs, isLive: false };
  } catch {
    return { status: 'offline', version: '1.0.0 (simulated)', latencyMs: 0, isLive: false };
  }
}

/**
 * Perform verification against POST /verify with offline fallback simulator.
 */
export async function verifyText(
  payload: VerificationRequest
): Promise<{ data: VerificationResponse; isLiveBackend: boolean; latencyMs: number }> {
  const start = performance.now();
  try {
    const res = await fetch(`${API_BASE}/verify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      const data = await res.json();
      return {
        data,
        isLiveBackend: true,
        latencyMs: Math.round(performance.now() - start),
      };
    }
  } catch {
    // Backend offline, fall through to simulation
  }

  // Realistic simulation fallback for immediate exploration
  await new Promise((resolve) => setTimeout(resolve, 380));
  const sim = simulateVerification(payload);
  return {
    data: sim,
    isLiveBackend: false,
    latencyMs: Math.round(performance.now() - start),
  };
}

/**
 * Deterministic offline fallback simulator.
 */
function simulateVerification(req: VerificationRequest): VerificationResponse {
  const text = req.text;
  const mode = req.mitigation_mode || 'rewrite';

  // Check for known preset indicators
  if (text.includes('Michael Collins') && (text.includes('Mars') || text.includes('1972'))) {
    return {
      is_safe: false,
      needs_review: false,
      hallucination_rate: 33.33,
      verification_rate: 66.67,
      unverified_rate: 0.0,
      claims: [
        {
          claim_id: 0,
          claim_text: 'Apollo 11 landed on the Moon on July 20, 1969.',
          state: 'SUPPORTED',
          probabilities: { contradiction: 0.012, entailment: 0.984, neutral: 0.004 },
          evidence_tier: 1,
          evidence_text: 'Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.',
          original_sentence: 'Apollo 11 landed on the Moon on July 20, 1969.',
        },
        {
          claim_id: 1,
          claim_text: 'Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle.',
          state: 'SUPPORTED',
          probabilities: { contradiction: 0.008, entailment: 0.978, neutral: 0.014 },
          evidence_tier: 1,
          evidence_text: 'Neil Armstrong and Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969.',
          original_sentence: 'Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle.',
        },
        {
          claim_id: 2,
          claim_text: 'Michael Collins famously walked on Mars during the return flight in 1972.',
          state: 'CONTRADICTED',
          probabilities: { contradiction: 0.942, entailment: 0.015, neutral: 0.043 },
          evidence_tier: 1,
          evidence_text: 'Michael Collins flew the command module Columbia alone in lunar orbit... Collins never walked on Mars.',
          original_sentence: 'However, astronaut Michael Collins famously walked on Mars during the return flight in 1972.',
        },
      ],
      sanitized_text:
        mode === 'block'
          ? 'Content blocked by Dehallucination Gateway safety gate: severe contradiction detected.'
          : mode === 'redact'
          ? 'Apollo 11 landed on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle. However, [REDACTED: Unverified or Contradicted Claim].'
          : mode === 'rewrite'
          ? 'Apollo 11 landed on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle, while command module pilot Michael Collins remained in lunar orbit aboard Columbia.'
          : text,
    };
  }

  if (text.includes('450 meters') || text.includes('Empire State')) {
    return {
      is_safe: false,
      needs_review: false,
      hallucination_rate: 33.33,
      verification_rate: 66.67,
      unverified_rate: 0.0,
      claims: [
        {
          claim_id: 0,
          claim_text: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.',
          state: 'SUPPORTED',
          probabilities: { contradiction: 0.005, entailment: 0.989, neutral: 0.006 },
          evidence_tier: 2,
          evidence_text: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.',
          original_sentence: 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.',
        },
        {
          claim_id: 1,
          claim_text: 'The Eiffel Tower was constructed from 1887 to 1889.',
          state: 'SUPPORTED',
          probabilities: { contradiction: 0.011, entailment: 0.974, neutral: 0.015 },
          evidence_tier: 2,
          evidence_text: 'It was constructed from 1887 to 1889 as the centerpiece of the 1889 World\'s Fair.',
          original_sentence: 'It was constructed from 1887 to 1889.',
        },
        {
          claim_id: 2,
          claim_text: 'Its architectural height is 450 meters, making it taller than the Empire State Building.',
          state: 'CONTRADICTED',
          probabilities: { contradiction: 0.967, entailment: 0.009, neutral: 0.024 },
          evidence_tier: 2,
          evidence_text: 'The Eiffel Tower is 330 metres (1,083 ft) tall, including the tip...',
          original_sentence: 'Its architectural height is 450 meters (1,476 feet), making it taller than the Empire State Building.',
        },
      ],
      sanitized_text:
        mode === 'block'
          ? 'Content blocked by Dehallucination Gateway safety gate: severe contradiction detected.'
          : mode === 'redact'
          ? 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889. [REDACTED: Discrepancy in height specification].'
          : mode === 'rewrite'
          ? 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889. Its architectural height is 330 meters (1,083 feet).'
          : text,
    };
  }

  // Generic fallback split by sentences
  const rawSentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);

  const claims = rawSentences.map((sent, idx) => {
    const isContradiction = idx === rawSentences.length - 1 && rawSentences.length > 1;
    return {
      claim_id: idx,
      claim_text: sent,
      state: (isContradiction ? 'CONTRADICTED' : 'SUPPORTED') as 'SUPPORTED' | 'CONTRADICTED',
      probabilities: isContradiction
        ? { contradiction: 0.884, entailment: 0.032, neutral: 0.084 }
        : { contradiction: 0.021, entailment: 0.952, neutral: 0.027 },
      evidence_tier: req.max_retrieval_tier || 2,
      evidence_text: req.context_documents?.[0] || 'Reference knowledge corpus verification chunk.',
      original_sentence: sent,
    };
  });

  const contradictedCount = claims.filter((c) => c.state === 'CONTRADICTED').length;
  const hallucinationRate = Math.round((contradictedCount / Math.max(1, claims.length)) * 100);

  return {
    is_safe: contradictedCount === 0,
    needs_review: false,
    hallucination_rate: hallucinationRate,
    verification_rate: 100 - hallucinationRate,
    unverified_rate: 0.0,
    claims,
    sanitized_text:
      mode === 'redact' && contradictedCount > 0
        ? claims.map((c) => (c.state === 'CONTRADICTED' ? '[REDACTED: Unverified Claim]' : c.claim_text)).join(' ')
        : text,
  };
}

/**
 * Call Chat Completions with SSE streaming support.
 */
export async function streamChatCompletions(
  payload: ChatCompletionRequest,
  onChunk: (chunk: string) => void,
  onHeader: (headers: Record<string, string>) => void
): Promise<{ fullText: string; isLive: boolean }> {
  try {
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ...payload, stream: true }),
    });

    if (res.ok) {
      const headersMap: Record<string, string> = {
        'x-dehallucination-safe': res.headers.get('x-dehallucination-safe') || 'true',
        'x-hallucination-rate': res.headers.get('x-hallucination-rate') || '0.00',
        'x-verification-rate': res.headers.get('x-verification-rate') || '100.00',
        'x-needs-review': res.headers.get('x-needs-review') || 'false',
      };
      onHeader(headersMap);

      const reader = res.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let fullText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunkStr = decoder.decode(value, { stream: true });
          const lines = chunkStr.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (dataStr === '[DONE]') continue;
              try {
                const parsed = JSON.parse(dataStr);
                const token = parsed.choices?.[0]?.delta?.content || '';
                if (token) {
                  fullText += token;
                  onChunk(token);
                }
              } catch {
                // non-json keep-alive
              }
            }
          }
        }
        return { fullText, isLive: true };
      }
    }
  } catch {
    // fallback simulation
  }

  // Simulated SSE replay
  const simulatedHeaders: Record<string, string> = {
    'x-dehallucination-safe': 'true',
    'x-hallucination-rate': '0.00',
    'x-verification-rate': '100.00',
    'x-needs-review': 'false',
  };
  onHeader(simulatedHeaders);

  const simResponse =
    'Apollo 11 made history on July 20, 1969, when Neil Armstrong and Buzz Aldrin stepped onto the lunar surface in the Sea of Tranquility, while Michael Collins piloted the command module in orbit. All factual claims have been verified against grounded NASA historical logs.';
  const words = simResponse.split(' ');
  let accumulated = '';

  for (let i = 0; i < words.length; i++) {
    const word = words[i] + (i === words.length - 1 ? '' : ' ');
    accumulated += word;
    onChunk(word);
    await new Promise((r) => setTimeout(r, 28));
  }

  return { fullText: accumulated, isLive: false };
}
