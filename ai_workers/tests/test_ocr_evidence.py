from ai_workers.modules.summarization.ocr_evidence import (
    MAX_OCR_SHARE,
    merge_speech_and_ocr,
    split_ocr_lines,
)


def _utts(n=10, words=20):
    return [
        {"start": float(i * 10), "end": float(i * 10 + 9), "text": " ".join(f"w{i}x{j}" for j in range(words))}
        for i in range(n)
    ]


def _plain(utterances):
    return " ".join((u.get("text") or "").strip() for u in utterances if (u.get("text") or "").strip())


def _slide(start, ocr):
    return {"start_seconds": start, "ocr_text": ocr}


def test_no_ocr_is_byte_identical_to_transcript():
    utts = _utts()
    for slides in ([], [_slide(0, "")], [_slide(0, None)], [None, "junk"], [_slide(0, "12 | -- | a")]):
        source, stats = merge_speech_and_ocr(utts, slides)
        assert source == _plain(utts)
        assert stats["ocr_words"] == 0 and stats["slides_used"] == 0


def test_blank_utterances_are_skipped_like_the_plain_transcript():
    utts = [{"start": 0, "text": "  "}, {"start": 1, "text": " hello world "}, {"start": 2, "text": None}]
    source, _ = merge_speech_and_ocr(utts, [])
    assert source == "hello world"


def test_slide_in_the_middle_precedes_first_utterance_at_or_after_its_start():
    utts = _utts()
    source, stats = merge_speech_and_ocr(utts, [_slide(35.0, "Gradient descent basics")])
    block = "Slide: Gradient descent basics."
    assert stats["slides_used"] == 1
    assert source.index(block) < source.index("w4x0")
    assert source.index(block) > source.index("w3x19")


def test_slide_starting_exactly_at_utterance_start_goes_before_it():
    source, _ = merge_speech_and_ocr(_utts(), [_slide(20.0, "Learning rate")])
    assert source.index("Slide: Learning rate.") < source.index("w2x0")


def test_slide_after_last_utterance_goes_last():
    source, _ = merge_speech_and_ocr(_utts(), [_slide(999.0, "Final remarks")])
    assert source.endswith("Slide: Final remarks.")


def test_slides_out_of_order_are_placed_by_time():
    slides = [_slide(60.0, "Second topic"), _slide(10.0, "First topic")]
    source, _ = merge_speech_and_ocr(_utts(), slides)
    assert source.index("First topic") < source.index("Second topic")


def test_footer_repeated_on_every_slide_appears_once():
    slides = [_slide(i * 20.0, f"Topic number {i} | Course Footer 2024") for i in range(4)]
    source, _ = merge_speech_and_ocr(_utts(), slides)
    assert source.count("Course Footer 2024") == 1
    assert source.count("Slide:") == 4


def test_progressive_slide_only_adds_new_lines():
    slides = [
        _slide(0.0, "Agenda | First point"),
        _slide(30.0, "Agenda | First point | Second point"),
    ]
    source, _ = merge_speech_and_ocr(_utts(), slides)
    assert source.count("First point") == 1
    assert source.count("Agenda") == 1
    assert source.count("Second point") == 1


def test_noise_lines_are_dropped():
    source, stats = merge_speech_and_ocr(_utts(), [_slide(0.0, "12 | -- | a | Real content")])
    assert "Slide: Real content." in source
    assert stats["ocr_words"] == 3  # two content words plus the Slide: prefix


def test_long_ocr_stays_within_share_and_last_slide_survives():
    utts = _utts(n=10, words=20)  # 200 transcript words
    slides = [
        _slide(i * 10.0, " | ".join(f"slide{i} line{j} alpha" for j in range(30)))
        for i in range(10)
    ]
    source, stats = merge_speech_and_ocr(utts, slides)
    assert stats["transcript_words"] == 200
    assert 0 < stats["ocr_words"] <= int(200 * MAX_OCR_SHARE)
    assert "slide9" in source


def test_zero_share_returns_plain_transcript():
    utts = [{"start": 0.0, "text": "tiny transcript"}]
    source, stats = merge_speech_and_ocr(utts, [_slide(0.0, "Some slide text here")])
    assert source == "tiny transcript"
    assert stats["ocr_words"] == 0


def test_removing_slide_blocks_restores_transcript():
    utts = _utts()
    slides = [_slide(0.0, "Intro heading"), _slide(45.0, "Middle heading | Detail text")]
    source, _ = merge_speech_and_ocr(utts, slides)
    stripped = source.replace("Slide: Intro heading.", "").replace("Slide: Middle heading. Detail text.", "")
    assert " ".join(stripped.split()) == _plain(utts)


def test_split_ocr_lines_handles_pipe_and_newline():
    assert split_ocr_lines("A | B\nC") == ["A", "B", "C"]
    assert split_ocr_lines(None) == []
    assert split_ocr_lines("  spaced   out  |  ") == ["spaced out"]
