export type ClaimState = 'SUPPORTED' | 'CONTRADICTED' | 'UNVERIFIED';

export type MitigationMode = 'rewrite' | 'redact' | 'block' | 'flag';

export interface ClaimProbabilities {
  contradiction: number;
  entailment: number;
  neutral: number;
}

export interface ClaimReport {
  claim_id: number;
  claim_text: string;
  state: ClaimState;
  probabilities: ClaimProbabilities;
  evidence_tier?: number | null;
  evidence_text?: string | null;
  original_sentence?: string | null;
}

export interface VerificationRequest {
  text: string;
  prompt?: string;
  context_documents?: string[];
  max_retrieval_tier?: number;
  mitigation_mode?: MitigationMode;
}

export interface VerificationResponse {
  is_safe: boolean;
  needs_review: boolean;
  hallucination_rate: number;
  verification_rate: number;
  unverified_rate: number;
  claims: ClaimReport[];
  sanitized_text: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  context_documents?: string[];
  max_retrieval_tier?: number;
  mitigation_mode?: MitigationMode;
}

export interface BackendHealth {
  status: string;
  version: string;
  latencyMs?: number;
  isLive: boolean;
}
