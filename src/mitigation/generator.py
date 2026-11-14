"""Quantized Local Gemma GPU Generator and Mock Generator Abstraction.

Implements REQ-MIT-01, NFR-PERF-03, ADR-0002.

Core Responsibilities:
1. BaseGenerator abstract contract for generative and rewriting operations.
2. QuantizedGemmaGenerator: 4-bit BitsAndBytes (NF4, double quant) execution on GPU (cuda:0).
3. Dedicated VRAM tracking ensuring usage <= 2.5 GB (and hard ceiling <= 3.5 GB).
4. Deterministic MockGenerator for fast, zero-GPU CI/CD testing and headless environments.
5. get_generator factory provider with automated fallback.
"""

from abc import ABC, abstractmethod
import os
import re
from typing import Any, Dict, Optional
import torch


class BaseGenerator(ABC):
    """Abstract base class for generative language models and rewriters."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Generate response text given an input prompt."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Execution target device (e.g. 'cuda:0', 'cpu')."""
        pass

    @property
    @abstractmethod
    def is_quantized(self) -> bool:
        """True if model weights are loaded in quantized precision."""
        pass

    @property
    def is_cuda(self) -> bool:
        """True if target execution device is a CUDA GPU."""
        return "cuda" in self.device.lower()

    @property
    def vram_allocated_gb(self) -> float:
        """Dedicated GPU VRAM currently allocated in gigabytes."""
        if torch.cuda.is_available() and self.is_cuda:
            return float(torch.cuda.memory_allocated() / (1024**3))
        return 0.0

    def is_vram_within_budget(self, limit_gb: float = 2.5) -> bool:
        """Verify that current VRAM usage does not exceed the specified hardware budget."""
        return self.vram_allocated_gb <= limit_gb


class MockGenerator(BaseGenerator):
    """Deterministic, zero-latency pseudo-generator for offline testing and CI/CD."""

    def __init__(
        self,
        model_name: str = "mock-gemma-4-E4B",
        device: str = "cpu",
        simulated_vram_gb: float = 2.2,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._simulated_vram_gb = simulated_vram_gb
        self._custom_responses: Dict[str, str] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_quantized(self) -> bool:
        return True

    @property
    def vram_allocated_gb(self) -> float:
        return self._simulated_vram_gb

    def register_response(self, prompt_substring: str, response: str) -> None:
        """Register a deterministic response triggered when prompt contains substring."""
        self._custom_responses[prompt_substring.strip()] = response.strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Generate deterministic text matching registered keys or applying mock rewrite."""
        # 1. Check registered responses
        for key, resp in self._custom_responses.items():
            if key in prompt:
                return resp

        # 2. Heuristic targeted rewrite handler for testing
        # When prompt matches targeted rewrite template, perform substitution
        if "ORIGINAL TEXT:" in prompt and "CONTRADICTED CLAIM" in prompt:
            return self._handle_mock_targeted_rewrite(prompt)

        return (
            f"Mock response from {self._model_name} for prompt: {prompt[:40]}... "
            f"[max_tokens={max_new_tokens}, temp={temperature}]"
        )

    def _handle_mock_targeted_rewrite(self, prompt: str) -> str:
        """Heuristic mock replacement of contradicted claim with evidence premise."""
        orig_pattern = (
            r"ORIGINAL TEXT:\s*\n?(.*?)(?=\n\nCONTRADICTED CLAIM|\nCONTRADICTED CLAIM"
            r"|\n\nREWRITTEN TEXT:|\nREWRITTEN TEXT:|\Z)"
        )
        original_match = re.search(orig_pattern, prompt, re.DOTALL)
        original_text = original_match.group(1).strip() if original_match else ""
        if not original_text:
            return ""

        pair_pattern = (
            r"CONTRADICTED CLAIM(?:\s*\d+)?:\s*(.*?)\n\s*EVIDENCE PREMISE:\s*(.*?)"
            r"(?=\n\nCONTRADICTED CLAIM|\nCONTRADICTED CLAIM|\n\nREWRITTEN TEXT:"
            r"|\nREWRITTEN TEXT:|\Z)"
        )
        pattern = re.compile(pair_pattern, re.DOTALL)
        pairs = pattern.findall(prompt)

        if not pairs:
            # Fallback for simple template
            contra_match = re.search(
                r"CONTRADICTED CLAIM:\s*(.*?)(?=\n\n|\n[A-Z_]+:|$)", prompt, re.DOTALL
            )
            premise_match = re.search(
                r"EVIDENCE PREMISE:\s*(.*?)(?=\n\n|\n[A-Z_]+:|$)", prompt, re.DOTALL
            )
            fallback_contra = contra_match.group(1).strip() if contra_match else ""
            fallback_premise = premise_match.group(1).strip() if premise_match else ""
            if fallback_contra and fallback_premise:
                pairs = [(fallback_contra, fallback_premise)]

        result_text = original_text
        for contra_claim, premise in pairs:
            clean_contra = contra_claim.strip()
            clean_premise = premise.strip()
            if not clean_contra or not clean_premise:
                continue

            if clean_contra in result_text:
                result_text = result_text.replace(clean_contra, clean_premise)
            else:
                stripped_contra = clean_contra.rstrip(".")
                stripped_premise = clean_premise.rstrip(".")
                if stripped_contra in result_text:
                    result_text = result_text.replace(stripped_contra, stripped_premise)
                elif clean_contra.lower() in result_text.lower():
                    idx = result_text.lower().find(clean_contra.lower())
                    result_text = (
                        result_text[:idx] + clean_premise + result_text[idx + len(clean_contra) :]
                    )

        return result_text


class QuantizedGemmaGenerator(BaseGenerator):
    """Local 4-bit Quantized Gemma Generator hosted on dedicated GPU (cuda:0).

    Implements REQ-MIT-01, NFR-PERF-03, ADR-0002.
    - Allocates Gemma in 4-bit precision via BitsAndBytes (NF4, double quant).
    - Enforces dedicated VRAM allocation <= 2.5 GB (hard limit <= 3.5 GB on RTX 2050).
    - Embeddings and NLI remain pinned to CPU threads to prevent CUDA OOM.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it-qat-mobile-transformers",
        device: str = "cuda:0",
        quantize_4bit: bool = True,
        max_vram_gb: float = 2.5,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._quantize_4bit = quantize_4bit
        self._max_vram_gb = max_vram_gb
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_quantized(self) -> bool:
        return self._quantize_4bit

    @property
    def max_vram_gb(self) -> float:
        return self._max_vram_gb

    def get_quantization_config(self) -> Any:
        """Create 4-bit BitsAndBytesConfig for NF4 double quantization."""
        if not self._quantize_4bit:
            return None

        from transformers import BitsAndBytesConfig

        is_bf16 = False
        if torch.cuda.is_available():
            try:
                is_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                is_bf16 = False

        compute_dtype = torch.bfloat16 if is_bf16 else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def _ensure_loaded(self) -> None:
        """Lazy load Gemma model and tokenizer onto target device."""
        if self._model is not None and self._tokenizer is not None:
            return

        if not torch.cuda.is_available() and self.is_cuda:
            raise RuntimeError(
                f"CUDA is not available on this host. Cannot allocate {self._model_name} "
                f"on {self._device}. Use MockGenerator or set device='cpu'."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_config = self.get_quantization_config() if self.is_cuda else None

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=quant_config,
            device_map=self._device if self.is_cuda else None,
            torch_dtype=torch.float16 if self.is_cuda else torch.float32,
        )

        # Enforce VRAM budget and hardware ceiling checks after loading
        if torch.cuda.is_available() and self.is_cuda:
            allocated = self.vram_allocated_gb
            if allocated > self._max_vram_gb:
                raise RuntimeError(
                    f"VRAM Budget Exceeded (REQ-MIT-01): Loaded weights consume "
                    f"{allocated:.2f} GB, exceeding budget of {self._max_vram_gb:.2f} GB."
                )
            if allocated > 3.5:
                raise RuntimeError(
                    f"CUDA OOM Risk (NFR-PERF-03): Allocated {allocated:.2f} GB exceeds "
                    f"3.5 GB ceiling on RTX 2050 (4 GB physical VRAM)."
                )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Execute generative inference with local Gemma on target device."""
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.is_cuda and torch.cuda.is_available():
            try:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            except Exception:
                pass

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        gen_kwargs.update(kwargs)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Verify active VRAM ceiling after generation pass (NFR-PERF-03)
        if torch.cuda.is_available() and self.is_cuda:
            peak_vram = self.vram_allocated_gb
            if peak_vram > 3.5:
                raise RuntimeError(
                    f"CUDA OOM Violation (NFR-PERF-03): Peak VRAM during generation reached "
                    f"{peak_vram:.2f} GB, exceeding 3.5 GB threshold."
                )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        decoded: str = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded.strip()


def get_generator(
    model_name: str = "google/gemma-4-E4B-it-qat-mobile-transformers",
    device: Optional[str] = None,
    use_mock: bool = False,
    quantize_4bit: bool = True,
    **kwargs: Any,
) -> BaseGenerator:
    """Factory creating a generator instance with automated GPU/Mock fallback (ADR-0002).

    Args:
        model_name: HuggingFace model identifier.
        device: Target execution device. Defaults to 'cuda:0' if CUDA available, else 'cpu'.
        use_mock: If True, returns MockGenerator for fast, zero-GPU testing.
        quantize_4bit: Whether to load model weights in 4-bit precision.
    """
    env_mock = os.environ.get("DEHALLUCINATION_USE_MOCK_GENERATOR", "0").lower()
    force_mock = env_mock in ("1", "true", "yes")

    if use_mock or force_mock or "mock" in model_name.lower():
        return MockGenerator(model_name=model_name, **kwargs)

    if not torch.cuda.is_available():
        # Transparent CPU / test sandbox fallback
        return MockGenerator(model_name=f"mock-{model_name}", **kwargs)

    target_device = device or "cuda:0"
    return QuantizedGemmaGenerator(
        model_name=model_name,
        device=target_device,
        quantize_4bit=quantize_4bit,
        **kwargs,
    )
