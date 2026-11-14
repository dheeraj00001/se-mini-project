"""Unit tests for Quantized Local Gemma GPU Generator & Targeted Rewriter.

Implements REQ-MIT-01, REQ-MIT-02, REQ-MIT-05, NFR-PERF-03, ADR-0002, ADR-0006.

RTM Mapping:
- UT_MIT_01: 4-bit Gemma GPU generation within 2.5GB VRAM (ADR-0002, NFR-PERF-03).
- UT_MIT_02: Targeted prompt construction pairing contradicted claims with premises (ADR-0006).
- UT_MIT_05: Business Rule BR-01 preservation of UNVERIFIED claims without modification.
- ST_MIT_01: End-to-end targeted rewrite accuracy and identity retention.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.schemas.verification import (
    ClaimReport,
    ClaimState,
)
from src.mitigation.generator import (
    BaseGenerator,
    MockGenerator,
    QuantizedGemmaGenerator,
    get_generator,
)

# ==============================================================================
# Slice 1: Generator Abstraction, Quantization Config & Mock (UT_MIT_01)
# ==============================================================================


def test_ut_mit_01_mock_generator_properties_and_vram_budget() -> None:
    """Verify MockGenerator exposes correct properties and respects <= 2.5 GB VRAM budget."""
    gen = MockGenerator(model_name="mock-gemma-4-E4B", simulated_vram_gb=2.2)

    assert isinstance(gen, BaseGenerator)
    assert gen.model_name == "mock-gemma-4-E4B"
    assert gen.device == "cpu"
    assert gen.is_quantized is True
    assert gen.vram_allocated_gb == 2.2
    assert gen.vram_allocated_gb <= 2.5  # REQ-MIT-01: <= 2.5 GB VRAM budget


def test_ut_mit_01_mock_generator_custom_registration_and_fallback() -> None:
    """Verify MockGenerator returns registered responses and fallback text."""
    gen = MockGenerator()
    gen.register_response("capital of France", "The capital of France is Paris.")

    res1 = gen.generate("What is the capital of France?")
    assert res1 == "The capital of France is Paris."

    res2 = gen.generate("Unregistered prompt text")
    assert isinstance(res2, str)
    assert len(res2) > 0


def test_ut_mit_01_quantized_gemma_generator_initialization_and_config() -> None:
    """Verify QuantizedGemmaGenerator configuration adheres to ADR-0002 and REQ-MIT-01."""
    gen = QuantizedGemmaGenerator(
        model_name="google/gemma-4-E4B-it-qat-mobile-transformers",
        device="cuda:0",
        quantize_4bit=True,
        max_vram_gb=2.5,
    )

    assert gen.model_name == "google/gemma-4-E4B-it-qat-mobile-transformers"
    assert gen.device == "cuda:0"
    assert gen.is_quantized is True
    assert gen.max_vram_gb == 2.5

    # Check 4-bit BitsAndBytes quantization configuration
    bnb_config = gen.get_quantization_config()
    assert bnb_config is not None
    assert getattr(bnb_config, "load_in_4bit", None) is True
    assert getattr(bnb_config, "bnb_4bit_quant_type", None) == "nf4"
    assert getattr(bnb_config, "bnb_4bit_use_double_quant", None) is True

    # When quantize_4bit=False, config is None
    unquant = QuantizedGemmaGenerator(quantize_4bit=False)
    assert unquant.is_quantized is False
    assert unquant.get_quantization_config() is None


def test_ut_mit_01_quantized_gemma_generator_vram_guard() -> None:
    """Verify that QuantizedGemmaGenerator asserts VRAM ceiling <= 3.5 GB per NFR-PERF-03."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.memory_allocated", return_value=int(3.8 * (1024**3))):
            gen = QuantizedGemmaGenerator(device="cuda:0")
            # Over the 3.5 GB hard threshold should report violation
            assert gen.vram_allocated_gb == pytest.approx(3.8, rel=1e-2)
            assert gen.is_vram_within_budget(limit_gb=3.5) is False


def test_ut_mit_01_quantized_gemma_generator_cuda_unavailable_error() -> None:
    """Verify QuantizedGemmaGenerator raises RuntimeError if loaded on CUDA without driver."""
    with patch("torch.cuda.is_available", return_value=False):
        gen = QuantizedGemmaGenerator(device="cuda:0")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            gen._ensure_loaded()


def test_ut_mit_01_quantized_gemma_generator_vram_budget_guards_on_load() -> None:
    """Verify QuantizedGemmaGenerator enforces 2.5GB budget and 3.5GB ceiling on load."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
                with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
                    mock_tok.return_value = MagicMock()
                    mock_model.return_value = MagicMock()

                    # 1. Test exceeding 2.5 GB budget (REQ-MIT-01)
                    with patch("torch.cuda.memory_allocated", return_value=int(2.8 * (1024**3))):
                        gen_25 = QuantizedGemmaGenerator(device="cuda:0", max_vram_gb=2.5)
                        with pytest.raises(RuntimeError, match="VRAM Budget Exceeded"):
                            gen_25._ensure_loaded()

                    # 2. Test exceeding 3.5 GB hard ceiling (NFR-PERF-03)
                    with patch("torch.cuda.memory_allocated", return_value=int(3.9 * (1024**3))):
                        gen_35 = QuantizedGemmaGenerator(device="cuda:0", max_vram_gb=4.0)
                        with pytest.raises(RuntimeError, match="CUDA OOM Risk"):
                            gen_35._ensure_loaded()


def test_ut_mit_01_quantized_gemma_generator_generate_inference() -> None:
    """Verify QuantizedGemmaGenerator.generate correctly tokenizes and decodes output."""
    import torch

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls:
                with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_cls:
                    with patch("torch.cuda.memory_allocated", return_value=int(2.1 * (1024**3))):
                        mock_tokenizer = MagicMock()
                        mock_model = MagicMock()

                        # Setup tokenizer return tensors
                        mock_inputs = {
                            "input_ids": torch.tensor([[101, 102, 103]]),
                            "attention_mask": torch.tensor([[1, 1, 1]]),
                        }
                        mock_tokenizer.return_value = mock_inputs
                        mock_tokenizer.eos_token_id = 1
                        mock_tokenizer.decode.return_value = "Paris is the capital of France."

                        # Mock model generate returning prompt tokens + new tokens
                        mock_model.generate.return_value = torch.tensor(
                            [[101, 102, 103, 201, 202, 203]]
                        )

                        mock_tok_cls.return_value = mock_tokenizer
                        mock_model_cls.return_value = mock_model

                        gen = QuantizedGemmaGenerator(device="cuda:0")
                        output = gen.generate(
                            "What is the capital of France?",
                            max_new_tokens=64,
                            temperature=0.7,
                        )

                        assert output == "Paris is the capital of France."
                        # Call _ensure_loaded a second time to verify early return
                        gen._ensure_loaded()
                        mock_tok_cls.assert_called_once()


def test_ut_mit_01_quantized_gemma_generator_active_vram_spike_guard() -> None:
    """Verify QuantizedGemmaGenerator raises RuntimeError if decoding VRAM exceeds 3.5 GB."""
    import torch

    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls:
                with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_cls:
                    # Allocate under budget at load (2.1 GB), spike to 3.8 GB during decode
                    allocations = [int(2.1 * (1024**3)), int(3.8 * (1024**3))]
                    with patch("torch.cuda.memory_allocated", side_effect=allocations):
                        mock_tok = MagicMock()
                        mock_model = MagicMock()
                        mock_tok.return_value = {
                            "input_ids": torch.tensor([[101, 102]]),
                            "attention_mask": torch.tensor([[1, 1]]),
                        }
                        mock_tok.eos_token_id = 1
                        mock_model.generate.return_value = torch.tensor([[101, 102, 201]])
                        mock_tok_cls.return_value = mock_tok
                        mock_model_cls.return_value = mock_model

                        gen = QuantizedGemmaGenerator(device="cuda:0")
                        with pytest.raises(RuntimeError, match="CUDA OOM Violation"):
                            gen.generate("Test prompt", max_new_tokens=64, temperature=0.7)
                        mock_model.generate.assert_called_once()
                        call_kwargs = mock_model.generate.call_args[1]
                        assert call_kwargs["max_new_tokens"] == 64
                        assert call_kwargs["do_sample"] is True
                        assert call_kwargs["temperature"] == 0.7


def test_ut_mit_01_mock_generator_rewrite_heuristic() -> None:
    """Verify MockGenerator handles targeted rewrite prompts with heuristic substitution."""
    gen = MockGenerator()
    prompt = (
        "ORIGINAL TEXT: The capital of France is Berlin. It is beautiful.\n\n"
        "CONTRADICTED CLAIM: The capital of France is Berlin.\n\n"
        "EVIDENCE PREMISE: The capital of France is Paris.\n"
    )
    result = gen.generate(prompt)
    assert result == "The capital of France is Paris. It is beautiful."


def test_ut_mit_01_get_generator_factory_mock_and_cuda_fallback() -> None:
    """Verify get_generator returns MockGenerator when requested or CUDA unavailable."""
    # Explicit use_mock=True
    gen_mock = get_generator(use_mock=True)
    assert isinstance(gen_mock, MockGenerator)

    # When CUDA is unavailable, defaults gracefully to MockGenerator
    with patch("torch.cuda.is_available", return_value=False):
        gen_auto = get_generator(use_mock=False)
        assert isinstance(gen_auto, MockGenerator)

    # When CUDA is available, returns QuantizedGemmaGenerator
    with patch("torch.cuda.is_available", return_value=True):
        gen_gpu = get_generator(use_mock=False)
        assert isinstance(gen_gpu, QuantizedGemmaGenerator)
        assert gen_gpu.device == "cuda:0"

    # Environment variable override force mock
    with patch.dict("os.environ", {"DEHALLUCINATION_USE_MOCK_GENERATOR": "1"}):
        with patch("torch.cuda.is_available", return_value=True):
            gen_env = get_generator()
            assert isinstance(gen_env, MockGenerator)


# ==============================================================================
# Slice 2: Prompt Construction & BR-01 Claim Filtering (UT_MIT_02, UT_MIT_05)
# ==============================================================================


def test_ut_mit_02_filter_contradicted_claims_threshold_and_state() -> None:
    """Verify filter_contradicted_claims selects only CONTRADICTED with P >= 0.70."""
    from src.mitigation.rewriter import filter_contradicted_claims

    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Water boils at 100 degrees Celsius.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
            evidence_tier=1,
            evidence_text="Water boils at 100 C at standard pressure.",
        ),
        ClaimReport(
            claim_id=1,
            claim_text="The capital of France is Berlin.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.02, "neutral": 0.04, "contradiction": 0.94},
            evidence_tier=2,
            evidence_text="Paris is the capital and largest city of France.",
        ),
        ClaimReport(
            claim_id=2,
            claim_text="Quantum computers use flux qubits.",
            state=ClaimState.UNVERIFIED,
            probabilities={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05},
            evidence_tier=None,
            evidence_text=None,
        ),
        ClaimReport(
            claim_id=3,
            claim_text="Marginal claim with weak contradiction.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.30, "neutral": 0.10, "contradiction": 0.60},
            evidence_tier=2,
            evidence_text="Some weak contrary evidence.",
        ),
    ]

    filtered = filter_contradicted_claims(claims, contradiction_threshold=0.70)
    assert len(filtered) == 1
    assert filtered[0].claim_id == 1
    assert filtered[0].claim_text == "The capital of France is Berlin."
    assert filtered[0].state == ClaimState.CONTRADICTED


def test_ut_mit_05_br01_unverified_claims_never_flagged_for_rewriting() -> None:
    """Verify Business Rule BR-01: UNVERIFIED claims are strictly excluded from rewriting."""
    from src.mitigation.rewriter import filter_contradicted_claims

    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="A rare manuscript was written in 1432.",
            state=ClaimState.UNVERIFIED,
            probabilities={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
            evidence_tier=None,
            evidence_text=None,
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Another unverified claim with zero KB hits.",
            state=ClaimState.UNVERIFIED,
            probabilities={"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05},
            evidence_tier=None,
            evidence_text=None,
        ),
    ]

    # Must return empty list - no claims eligible for rewrite!
    filtered = filter_contradicted_claims(claims)
    assert filtered == []


def test_ut_mit_02_build_rewrite_prompt_structure_and_constraints() -> None:
    """Verify construct_rewrite_prompt formats instructions, premises, and constraints."""
    from src.mitigation.rewriter import build_rewrite_prompt

    original_text = "The capital of France is Berlin. It is known for its architecture."
    contradicted = [
        ClaimReport(
            claim_id=0,
            claim_text="The capital of France is Berlin.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95},
            evidence_tier=2,
            evidence_text="Paris is the capital and largest city of France.",
        )
    ]

    prompt = build_rewrite_prompt(original_text=original_text, contradicted_claims=contradicted)

    assert "ORIGINAL TEXT:" in prompt
    assert original_text in prompt
    assert "CONTRADICTED CLAIM:" in prompt or "CLAIM 1:" in prompt
    assert "The capital of France is Berlin." in prompt
    assert "EVIDENCE PREMISE:" in prompt
    assert "Paris is the capital and largest city of France." in prompt
    # Preserves non-contradicted instructions per REQ-MIT-02, REQ-MIT-05
    assert "DO NOT modify" in prompt or "preserve" in prompt.lower()


# ==============================================================================
# Slice 3: TargetedRewriter / GemmaRewriter End-to-End Mitigation (ST_MIT_01, DoD)
# ==============================================================================


def test_st_mit_01_rewriter_replaces_contradiction_with_premise() -> None:
    """Verify GemmaRewriter replaces contradicted statement with premise while retaining context."""
    from src.mitigation.rewriter import GemmaRewriter

    rewriter = GemmaRewriter(generator=MockGenerator())
    original = "The Apollo 11 mission landed on Mars in 1969. It was a historic feat."
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="The Apollo 11 mission landed on Mars in 1969.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=2,
            evidence_text="The Apollo 11 mission landed on the Moon in 1969.",
        ),
        ClaimReport(
            claim_id=1,
            claim_text="It was a historic feat.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.96, "neutral": 0.03, "contradiction": 0.01},
            evidence_tier=1,
            evidence_text="Apollo 11 was widely regarded as a historic feat of human spaceflight.",
        ),
    ]

    rewritten = rewriter.rewrite(original_text=original, claims=claims)

    assert "Moon" in rewritten
    assert "Mars" not in rewritten
    assert "It was a historic feat." in rewritten


def test_st_mit_01_rewriter_identity_preservation_when_zero_contradictions() -> None:
    """Verify that text with zero contradicted claims is returned completely unmodified."""
    from src.mitigation.rewriter import TargetedRewriter

    rewriter = TargetedRewriter(generator=MockGenerator())
    original = "Water freezes at 0 degrees Celsius. Some people claim ancient ice holds secrets."
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Water freezes at 0 degrees Celsius.",
            state=ClaimState.SUPPORTED,
            probabilities={"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01},
            evidence_tier=1,
            evidence_text="Water freezes at 0 C at standard atmospheric pressure.",
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Some people claim ancient ice holds secrets.",
            state=ClaimState.UNVERIFIED,
            probabilities={"entailment": 0.10, "neutral": 0.88, "contradiction": 0.02},
            evidence_tier=None,
            evidence_text=None,
        ),
    ]

    # Zero contradicted claims: generator should NOT even be invoked
    rewritten = rewriter.rewrite(original_text=original, claims=claims)
    assert rewritten == original


def test_st_mit_01_rewriter_preserves_unverified_claims_alongside_contradictions() -> None:
    """Verify REQ-MIT-05: UNVERIFIED statements remain intact during contradiction fixes."""
    from src.mitigation.rewriter import TargetedRewriter

    rewriter = TargetedRewriter(generator=MockGenerator())
    original = (
        "Berlin is the capital of France. "
        "A hidden 12th-century vault may exist beneath Notre-Dame."
    )
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Berlin is the capital of France.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=2,
            evidence_text="Paris is the capital of France.",
        ),
        ClaimReport(
            claim_id=1,
            claim_text="A hidden 12th-century vault may exist beneath Notre-Dame.",
            state=ClaimState.UNVERIFIED,
            probabilities={"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05},
            evidence_tier=None,
            evidence_text=None,
        ),
    ]

    rewritten = rewriter.rewrite(original_text=original, claims=claims)

    # Contradicted part corrected
    assert "Paris is the capital of France." in rewritten
    assert "Berlin is the capital of France." not in rewritten
    # Unverified part preserved unaltered
    assert "A hidden 12th-century vault may exist beneath Notre-Dame." in rewritten


def test_st_mit_01_rewriter_clean_output_formatting() -> None:
    """Verify TargetedRewriter removes code fences or prefixed headers from generator output."""
    from src.mitigation.rewriter import TargetedRewriter

    mock_gen = MockGenerator()
    mock_gen.register_response(
        "ORIGINAL TEXT:",
        "```markdown\nREWRITTEN TEXT: Paris is the capital of France.\n```",
    )
    rewriter = TargetedRewriter(generator=mock_gen)

    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Berlin is the capital of France.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=2,
            evidence_text="Paris is the capital of France.",
        )
    ]

    cleaned = rewriter.rewrite(original_text="Berlin is the capital of France.", claims=claims)
    assert cleaned == "Paris is the capital of France."


def test_st_mit_01_rewriter_edge_cases_empty_and_null() -> None:
    """Verify TargetedRewriter handles empty text, whitespace, and empty claim lists safely."""
    from src.mitigation.rewriter import TargetedRewriter

    rewriter = TargetedRewriter(generator=MockGenerator())

    assert rewriter.rewrite("", []) == ""
    assert rewriter.rewrite("   \n\t  ", []) == "   \n\t  "
    assert rewriter.rewrite("Sample text.", []) == "Sample text."


def test_nfr_perf_03_rewriting_latency_under_budget() -> None:
    """Verify rewriting pass executes within latency expectations."""
    import time
    from src.mitigation.rewriter import TargetedRewriter

    rewriter = TargetedRewriter(generator=MockGenerator())
    original = "Berlin is the capital of France. Water freezes at 50 degrees Celsius."
    claims = [
        ClaimReport(
            claim_id=0,
            claim_text="Berlin is the capital of France.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=2,
            evidence_text="Paris is the capital of France.",
        ),
        ClaimReport(
            claim_id=1,
            claim_text="Water freezes at 50 degrees Celsius.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=2,
            evidence_text="Water freezes at 0 degrees Celsius.",
        ),
    ]

    t0 = time.perf_counter()
    res = rewriter.rewrite(original_text=original, claims=claims)
    elapsed_s = time.perf_counter() - t0

    assert "Paris" in res
    assert "0 degrees" in res
    assert elapsed_s < 2.5  # Well within the 2.5s budget


def test_st_mit_01_rewriter_dod_signature_and_keyword_args() -> None:
    """Verify GemmaRewriter.rewrite supports contradicted_claims keyword argument (DoD)."""
    from src.mitigation.rewriter import GemmaRewriter

    rewriter = GemmaRewriter(generator=MockGenerator())
    original = "The capital of France is Berlin. It is a lovely place."
    contradicted = [
        ClaimReport(
            claim_id=0,
            claim_text="The capital of France is Berlin.",
            state=ClaimState.CONTRADICTED,
            probabilities={"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97},
            evidence_tier=2,
            evidence_text="Paris is the capital and largest city of France.",
        )
    ]

    # Call with keyword argument matching Definition of Done specification
    result = rewriter.rewrite(original_text=original, contradicted_claims=contradicted)

    assert "Paris" in result
    assert "Berlin" not in result
    assert "It is a lovely place." in result
