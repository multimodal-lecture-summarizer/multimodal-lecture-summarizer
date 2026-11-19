import zipfile

from ai_workers import batch_download_videos as dl


def test_notebook_list_gives_the_thirty_unique_youtube_links():
    urls = dl.read_notebook_urls()
    assert len(urls) == 30
    assert all(dl.youtube_id(url) for url in urls)
    assert len({dl.youtube_id(url) for url in urls}) == 30


def test_read_urls_file_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "urls.txt"
    path.write_text(
        "# my list\nhttps://youtu.be/AAAAAAAAAAA  # first\n\nhttps://youtu.be/AAAAAAAAAAA\nhttps://youtu.be/BBBBBBBBBBB\n",
        encoding="utf-8",
    )
    assert dl.read_urls_file(path) == ["https://youtu.be/AAAAAAAAAAA", "https://youtu.be/BBBBBBBBBBB"]


def test_download_all_uses_the_pipeline_downloader_and_reports_failures(tmp_path, monkeypatch):
    import ai_workers.core.youtube_ingest as ingest

    calls = []

    def fake_download(url, dest, max_duration=0):
        calls.append((url, dest, max_duration))
        if "BBBBBBBBBBB" in url:
            raise RuntimeError("HTTP Error 403")
        if "CCCCCCCCCCC" in url:
            return dest  # yt-dlp returned without writing a file
        open(dest, "wb").write(b"x")
        return dest

    monkeypatch.setattr(ingest, "download_youtube", fake_download)
    out = tmp_path / "out"
    (out).mkdir()
    (out / "DDDDDDDDDDD.mp4").write_bytes(b"old")
    urls = [f"https://youtu.be/{c * 11}" for c in "ABCD"]

    failed = dl.download_all(urls, out, 3600)

    assert failed == [urls[1], urls[2]]
    assert (out / "AAAAAAAAAAA.mp4").exists()
    assert [c[0] for c in calls] == urls[:3]  # D was already there, so it is not requested again
    assert all(c[2] == 3600 for c in calls)


def test_zip_is_flat_and_only_holds_downloaded_videos(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "AAAAAAAAAAA.mp4").write_bytes(b"a")
    (videos / "BBBBBBBBBBB.webm").write_bytes(b"b")
    urls = ["https://youtu.be/AAAAAAAAAAA", "https://youtu.be/BBBBBBBBBBB", "https://youtu.be/CCCCCCCCCCC"]

    zip_path = tmp_path / "out.zip"
    assert dl.make_zip(urls, videos, zip_path) == 2
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["AAAAAAAAAAA.mp4", "BBBBBBBBBBB.webm"]
