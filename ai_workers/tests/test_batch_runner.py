import csv
import io

from ai_workers import batch_runner as br

TIMING_LINE = (
    "[TIMING] total=322.9s, video=830.6s, realtime_factor=0.39x | stage spans: preflight=0.0s, "
    "audio=213.2s, visual=95.8s, object_tracking=191.6s, speaker=0.2s, semantic=17.8s, timeline=0.3s, "
    "quality=0.1s, storage=2.2s, intermediate_ready=0.5s, summarizing=5.7s, completed=0.0s"
)


def test_parse_timing_reads_totals_and_stage_spans():
    timing = br.parse_timing("noise\n" + TIMING_LINE + "\nmore noise")
    assert timing["total"] == 322.9
    assert timing["video"] == 830.6
    assert timing["realtime_factor"] == 0.39
    assert timing["spans"]["audio"] == 213.2
    assert timing["spans"]["intermediate_ready"] == 0.5


def test_parse_timing_without_realtime_factor_and_without_line():
    line = "[TIMING] total=10.0s, video=0.0s | stage spans: audio=4.0s"
    assert br.parse_timing(line)["realtime_factor"] is None
    assert br.parse_timing("nothing here") is None


def test_youtube_id_handles_watch_short_and_share_links():
    assert br.youtube_id("https://www.youtube.com/watch?v=FLWvcdfGPxs") == "FLWvcdfGPxs"
    assert br.youtube_id("https://youtu.be/AuNfxVLdo0A?si=abc") == "AuNfxVLdo0A"
    assert br.youtube_id("/content/drive/video.mp4") == ""


def test_count_slide_elements_ignores_unknown_labels():
    scenes = [
        {"detected_objects": {"slide_elements": [{"label": "illustration"}, {"label": "table"}]}},
        {"detected_objects": {"slide_elements": [{"label": "illustration"}, {"label": "logo"}]}},
        {"detected_objects": {}},
        {},
    ]
    assert br.count_slide_elements(scenes) == {"table": 1, "chart": 0, "photo": 0, "illustration": 2}


def test_build_row_for_a_finished_job():
    result = {
        "video_title": "Talk",
        "duration": 830.6,
        "processing_time": 322.9,
        "scenes": [{"detected_objects": {"slide_elements": [{"label": "chart"}]}}, {}],
        "keyframes": [1, 2],
        "chapters": [1],
        "word_count": 93,
        "summary_method": "led",
        "asr_model": "Qwen/Qwen3-ASR-0.6B",
        "detected_language": "en",
        "fallback_used": False,
    }
    entry = {"url": "https://www.youtube.com/watch?v=FLWvcdfGPxs", "group": "baseline"}
    row = br.build_row(
        entry, 1, status="done", download_sec=12.34, wall_sec=330.0, result=result,
        timing=br.parse_timing(TIMING_LINE),
    )
    assert set(row) == set(br.COLUMNS)
    assert row["video_id"] == "FLWvcdfGPxs"
    assert row["processing_sec"] == 322.9
    assert row["realtime_factor"] == 0.39
    assert row["stage_audio_sec"] == 213.2
    assert row["stage_object_tracking_sec"] == 191.6
    assert (row["n_scenes"], row["n_keyframes"], row["n_chapters"], row["n_chart"]) == (2, 2, 1, 1)
    assert row["download_sec"] == 12.3


def test_build_row_for_a_failed_download_leaves_timings_empty():
    row = br.build_row({"url": "https://youtu.be/AuNfxVLdo0A"}, 5, status="download_failed", error="bot check")
    assert row["status"] == "download_failed"
    assert row["processing_sec"] is None and row["video_sec"] is None
    assert row["stage_audio_sec"] == ""


def test_csv_rows_append_and_only_done_urls_are_skipped(tmp_path):
    path = tmp_path / "nested" / "results.csv"
    done = br.build_row({"url": "https://youtu.be/AAAAAAAAAAA"}, 1, status="done")
    failed = br.build_row({"url": "https://youtu.be/BBBBBBBBBBB"}, 2, status="error", error="boom")
    br.append_row(path, done)
    br.append_row(path, failed)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert path.read_text(encoding="utf-8-sig").count("index,group,url") == 1  # one header only
    assert br.load_done_urls(path) == {"https://youtu.be/AAAAAAAAAAA"}
    assert br.load_done_urls(tmp_path / "missing.csv") == set()


def test_tee_keeps_everything_but_echoes_only_progress_lines():
    echo = io.StringIO()
    tee = br._Tee(echo)
    tee.write("Fetching 10 files: 50%\r")
    tee.write("[AUDIO +12.3s] Bắt đầu\nplain print\n[TIMING] total=1.0s, video=2.0s | stage spans: a=1.0s\n")
    assert "Fetching" in tee.getvalue() and "plain print" in tee.getvalue()
    assert echo.getvalue().splitlines() == [
        "[AUDIO +12.3s] Bắt đầu",
        "[TIMING] total=1.0s, video=2.0s | stage spans: a=1.0s",
    ]


def test_acquire_prefers_a_predownloaded_copy_and_falls_back_to_youtube(tmp_path, monkeypatch):
    import ai_workers.core.youtube_ingest as ingest

    asked = []

    def refuse(url, dest, max_duration=0):
        asked.append(url)
        raise RuntimeError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(ingest, "download_youtube", refuse)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "FLWvcdfGPxs.mp4").write_bytes(b"x")
    work = tmp_path / "work"
    work.mkdir()

    path, is_temp = br.acquire({"url": "https://www.youtube.com/watch?v=FLWvcdfGPxs"}, work, videos, 3600)
    assert is_temp and path == str(work / "FLWvcdfGPxs.mp4")
    assert (videos / "FLWvcdfGPxs.mp4").exists()  # the Drive copy stays untouched
    assert asked == []  # YouTube is not contacted when the file is there

    try:
        br.acquire({"url": "https://www.youtube.com/watch?v=ZZZZZZZZZZZ"}, work, videos, 3600)
    except RuntimeError as err:
        assert "bot" in str(err)
    else:
        raise AssertionError("expected the download error to propagate")
    assert len(asked) == 1
