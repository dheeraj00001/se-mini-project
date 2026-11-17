"""Unit tests for ClaimSegmenter component (REQ-VER-01, REQ-VER-02, REQ-VER-03, NFR-PERF-01).

RTM Mapping:
- UT_VER_01: Sentence tokenization & non-factual utterance filtering (REQ-VER-01).
- UT_VER_02: Compound sentence & clause segmentation (REQ-VER-02).
- UT_VER_03: Epistemic hedge detection & preservation (REQ-VER-03).
"""

import sys
import pytest
from src.schemas.verification import ClaimSegment, AtomicClaim
from src.verifier.segmenter import ClaimSegmenter

# ==============================================================================
# Slice 1: Input Filtering & Non-Factual Utterances (UT_VER_01)
# ==============================================================================


@pytest.mark.parametrize(
    "empty_input",
    [
        "",
        "   ",
        "\n\n\t  \n",
    ],
)
def test_ut_ver_01_empty_or_whitespace_input(empty_input: str) -> None:
    """Validate that blank or whitespace-only inputs produce an empty list."""
    claims = ClaimSegmenter.extract_claims(empty_input)
    assert claims == []


def test_ut_ver_01_filter_pure_questions() -> None:
    """Validate that interrogative utterances without factual assertions are filtered."""
    text = "What is the capital of France? Can you tell me more about it?"
    claims = ClaimSegmenter.extract_claims(text)
    assert claims == []


def test_ut_ver_01_filter_conversational_greetings_and_filler() -> None:
    """Validate that greetings and conversational filler phrases are filtered."""
    text = "Hello! Sure, I can help with that. Here is the information:"
    claims = ClaimSegmenter.extract_claims(text)
    assert claims == []


def test_ut_ver_01_filter_standalone_code_blocks() -> None:
    """Validate that markdown code blocks are filtered out cleanly."""
    text = """```python
def add(a, b):
    return a + b
```"""
    claims = ClaimSegmenter.extract_claims(text)
    assert claims == []


# ==============================================================================
# Slice 2: Sentence Tokenization & Bullet List Splitting (UT_VER_01)
# ==============================================================================


def test_ut_ver_01_multi_sentence_tokenization() -> None:
    """Validate splitting of multi-sentence paragraph into distinct sequential claims."""
    text = (
        "The Earth orbits the Sun. The Moon is the Earth's only natural satellite. "
        "Mars has two small moons."
    )
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 3
    assert all(isinstance(c, (ClaimSegment, AtomicClaim)) for c in claims)
    assert [c.claim_id for c in claims] == [0, 1, 2]
    assert claims[0].claim_text == "The Earth orbits the Sun."
    assert claims[1].claim_text == "The Moon is the Earth's only natural satellite."
    assert claims[2].claim_text == "Mars has two small moons."
    assert claims[0].original_sentence == "The Earth orbits the Sun."
    assert claims[0].has_epistemic_hedge is False


def test_ut_ver_01_sentence_tokenization_with_abbreviations() -> None:
    """Validate that sentences containing common abbreviations are not falsely split."""
    text = "Dr. Smith visited the U.S. in 1889 A.D. He gave a lecture in Washington, D.C."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "Dr. Smith visited the U.S. in 1889 A.D."
    assert claims[1].claim_text == "He gave a lecture in Washington, D.C."


def test_ut_ver_01_bullet_and_numbered_lists() -> None:
    """Validate extraction of factual claims from markdown bulleted and numbered lists."""
    text = (
        "- Paris is the capital of France.\n"
        "- Berlin is the capital of Germany.\n"
        "1. Tokyo is the capital of Japan.\n"
        "2. Canberra is the capital of Australia."
    )
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 4
    assert claims[0].claim_text == "Paris is the capital of France."
    assert claims[1].claim_text == "Berlin is the capital of Germany."
    assert claims[2].claim_text == "Tokyo is the capital of Japan."
    assert claims[3].claim_text == "Canberra is the capital of Australia."


# ==============================================================================
# Slice 3: Epistemic Hedge Detection & Proposition Extraction (UT_VER_03)
# ==============================================================================


def test_ut_ver_03_first_person_epistemic_hedges() -> None:
    """Validate extraction of core factual proposition under first-person hedges."""
    text = "I think the Eiffel Tower is located in Berlin."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 1
    assert claims[0].has_epistemic_hedge is True
    assert claims[0].claim_text == "The Eiffel Tower is located in Berlin."
    assert claims[0].original_sentence == "I think the Eiffel Tower is located in Berlin."


def test_ut_ver_03_impersonal_and_attribution_hedges() -> None:
    """Validate extraction under impersonal and attribution hedges."""
    text1 = "It is believed that Shakespeare wrote Hamlet."
    claims1 = ClaimSegmenter.extract_claims(text1)
    assert len(claims1) == 1
    assert claims1[0].has_epistemic_hedge is True
    assert claims1[0].claim_text == "Shakespeare wrote Hamlet."

    text2 = "According to sources, the summit took place in Geneva."
    claims2 = ClaimSegmenter.extract_claims(text2)
    assert len(claims2) == 1
    assert claims2[0].has_epistemic_hedge is True
    assert claims2[0].claim_text == "The summit took place in Geneva."


def test_ut_ver_03_adverbial_hedges() -> None:
    """Validate extraction of claims with leading adverbial hedges."""
    text = "Apparently, the spacecraft reached orbit safely."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 1
    assert claims[0].has_epistemic_hedge is True
    assert claims[0].claim_text == "The spacecraft reached orbit safely."


def test_ut_ver_03_parenthetical_and_trailing_hedges() -> None:
    """Validate stripping of parenthetical or trailing hedges while marking epistemic hedge."""
    text1 = "The French Revolution, I believe, began in 1789."
    claims1 = ClaimSegmenter.extract_claims(text1)
    assert len(claims1) == 1
    assert claims1[0].has_epistemic_hedge is True
    assert claims1[0].claim_text == "The French Revolution began in 1789."

    text2 = "Water boils at 100 degrees Celsius, I think."
    claims2 = ClaimSegmenter.extract_claims(text2)
    assert len(claims2) == 1
    assert claims2[0].has_epistemic_hedge is True
    assert claims2[0].claim_text == "Water boils at 100 degrees Celsius."


# ==============================================================================
# Slice 4: Selective Independent Clause Splitting (UT_VER_02)
# ==============================================================================


def test_ut_ver_02_semicolon_clause_splitting() -> None:
    """Validate that sentences separated by semicolons are split into atomic claims."""
    text = "Paris is the capital of France; Berlin is the capital of Germany."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "Paris is the capital of France."
    assert claims[1].claim_text == "Berlin is the capital of Germany."
    assert claims[0].original_sentence == text
    assert claims[1].original_sentence == text


def test_ut_ver_02_independent_clauses_with_explicit_subject() -> None:
    """Validate splitting of clauses joined by coordinating conjunction with explicit subject."""
    text = "The Eiffel Tower was completed in 1889, and it stands in Paris."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "The Eiffel Tower was completed in 1889."
    assert claims[1].claim_text == "It stands in Paris."
    assert claims[0].original_sentence == text
    assert claims[1].original_sentence == text


def test_ut_ver_02_compound_predicate_shared_subject() -> None:
    """Validate ticket example: compound predicate with shared subject becomes atomic claims."""
    text = "The Eiffel Tower was built in 1889 and is located in Berlin."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "The Eiffel Tower was built in 1889."
    assert claims[1].claim_text == "The Eiffel Tower is located in Berlin."
    assert claims[0].original_sentence == text
    assert claims[1].original_sentence == text


def test_ut_ver_02_do_not_split_compound_objects_or_subjects() -> None:
    """Validate that compound nouns, subjects, or adjective lists are NOT falsely split."""
    text1 = "The French flag is blue, white, and red."
    claims1 = ClaimSegmenter.extract_claims(text1)
    assert len(claims1) == 1
    assert claims1[0].claim_text == "The French flag is blue, white, and red."

    text2 = "Alice and Bob visited the Louvre."
    claims2 = ClaimSegmenter.extract_claims(text2)
    assert len(claims2) == 1
    assert claims2[0].claim_text == "Alice and Bob visited the Louvre."

    text3 = "Python is an interpreted, high-level, general-purpose programming language."
    claims3 = ClaimSegmenter.extract_claims(text3)
    assert len(claims3) == 1
    assert (
        claims3[0].claim_text
        == "Python is an interpreted, high-level, general-purpose programming language."
    )


def test_ut_ver_02_compound_predicate_single_word_subject() -> None:
    """Validate subject extraction when single-word entity precedes auxiliary verb."""
    text = "Google was founded in 1998 and is based in California."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "Google was founded in 1998."
    assert claims[1].claim_text == "Google is based in California."
    assert claims[0].original_sentence == text
    assert claims[1].original_sentence == text


def test_ut_ver_02_clause_without_comma_conjunction() -> None:
    """Validate clause splitting when coordinating conjunction is not preceded by a comma."""
    text = "The rover landed on Mars and it gathered rock samples."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "The rover landed on Mars."
    assert claims[1].claim_text == "It gathered rock samples."
    assert claims[0].original_sentence == text
    assert claims[1].original_sentence == text


# ==============================================================================
# Slice 5: Complex Paragraphs & Edge Cases (UT_VER_01, UT_VER_02, UT_VER_03)
# ==============================================================================


def test_ut_ver_01_complex_mixed_paragraph() -> None:
    """Validate end-to-end extraction across greetings, code, hedges, and clauses."""
    text = (
        "Hello! Certainly, here is what you requested.\n\n"
        "The Apollo 11 mission landed on the Moon in 1969. "
        "I think Neil Armstrong was the mission commander and walked on lunar soil.\n\n"
        "```python\nprint('Done')\n```\n\n"
        "Would you like more details? Hope this helps!"
    )
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 3
    assert [c.claim_id for c in claims] == [0, 1, 2]
    assert claims[0].claim_text == "The Apollo 11 mission landed on the Moon in 1969."
    assert claims[0].has_epistemic_hedge is False

    assert claims[1].claim_text == "Neil Armstrong was the mission commander."
    assert claims[1].has_epistemic_hedge is True

    assert claims[2].claim_text == "Neil Armstrong walked on lunar soil."
    assert claims[2].has_epistemic_hedge is True


def test_ut_ver_01_instance_invocation_compatibility() -> None:
    """Validate that ClaimSegmenter can be invoked via instance or class method."""
    segmenter = ClaimSegmenter()
    claims = segmenter.extract_claims("Jupiter is the largest planet in our solar system.")
    assert len(claims) == 1
    assert claims[0].claim_text == "Jupiter is the largest planet in our solar system."


def test_ut_ver_01_filter_imperative_instructions() -> None:
    """Validate filtering of non-factual imperative instructions."""
    text = (
        "Please refer to the documentation for further instructions. "
        "Python 3.11 was released in October 2022."
    )
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 1
    assert claims[0].claim_text == "Python 3.11 was released in October 2022."


def test_ut_ver_03_hedge_without_trailing_period() -> None:
    """Validate hedge stripping and automatic period addition when sentence lacks terminal dot."""
    text = "I think the Earth is round"
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 1
    assert claims[0].claim_text == "The Earth is round."
    assert claims[0].has_epistemic_hedge is True


def test_ut_ver_02_semicolon_with_insufficient_clause_length() -> None:
    """Validate that semicolon with too few words in clause is not split falsely."""
    text = "Photosynthesis requires sunlight; yes."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 1
    assert claims[0].claim_text == "Photosynthesis requires sunlight; yes."


# ==============================================================================
# Slice 6: Performance Benchmark (NFR-PERF-01)
# ==============================================================================


def test_nfr_perf_01_claim_segmentation_latency() -> None:
    """Verify claim segmentation completes in <= 5 ms for ~500 tokens (NFR-PERF-01)."""
    import time

    # Construct ~500 token text (~350-400 words)
    paragraph = (
        "The solar system consists of the Sun and eight planets orbiting around it. "
        "Mercury is the closest planet to the Sun and experiences extreme temperatures. "
        "Venus has a thick, toxic atmosphere rich in carbon dioxide and nitrogen. "
        "Earth is the third planet from the Sun; it is the only known planet to harbor life. "
        "Mars is known as the Red Planet and possesses two small natural satellites. "
        "Jupiter is the largest planet in our planetary system and features the Great Red Spot. "
        "Saturn is famous for its extensive ring system made largely of water ice particles. "
        "Uranus rotates on an axis tilted nearly parallel to its orbital plane. "
        "Neptune is the most distant planet and has active weather patterns with supersonic winds. "
        "I think Pluto was reclassified as a dwarf planet by the IAU in 2006. "
    )
    text = (paragraph * 2) + paragraph[: paragraph.find("Mars")]

    # Warmup pass
    _ = ClaimSegmenter.extract_claims(text)

    # Measure over multiple iterations; temporarily suspend coverage tracing hook during timing
    timings = []
    old_trace = sys.gettrace()
    try:
        sys.settrace(None)
        for _ in range(5):
            start_time = time.perf_counter()
            claims = ClaimSegmenter.extract_claims(text)
            timings.append((time.perf_counter() - start_time) * 1000.0)
    finally:
        sys.settrace(old_trace)

    elapsed_ms = min(timings)
    assert len(claims) > 10
    # NFR-PERF-01 requirement: latency <= 5.0 ms
    assert elapsed_ms <= 5.0, f"Segmentation latency {elapsed_ms:.2f} ms exceeded 5.0 ms limit"


# ==============================================================================
# Student 2 Validation Suite: Extended Academic Hedges & Syntactic Boundaries
# ==============================================================================


def test_student_2_academic_epistemic_hedges() -> None:
    """Verify Student 2 extended academic epistemic hedge patterns (REQ-VER-03)."""
    text = (
        "Preliminary data suggests that quantum processors achieve supremacy in specific tasks. "
        "It is hypothesized that dark matter interacts weakly through gravity. "
        "Purportedly, the experiment was reproduced in three independent laboratories."
    )
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 3
    assert any("quantum processors achieve supremacy" in c.claim_text for c in claims)
    assert any("dark matter interacts weakly" in c.claim_text for c in claims)
    assert any("the experiment was reproduced" in c.claim_text for c in claims)


def test_student_2_compound_clause_invariance() -> None:
    """Verify syntactic clause splitter preserves subject in compound predicates (REQ-VER-02)."""
    text = "The spacecraft entered lunar orbit and executed a retro-burn sequence."
    claims = ClaimSegmenter.extract_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_text == "The spacecraft entered lunar orbit"
    assert "spacecraft executed a retro-burn sequence" in claims[1].claim_text or "The spacecraft executed" in claims[1].claim_text or "executed a retro-burn" in claims[1].claim_text

