import io
import urllib.error
from unittest.mock import patch

from app.services.chromadb import _short_error, is_chromadb_responsive

CLOUDFLARE_PAGE = (
    "<!doctype html>\n<html class=\"no-js\"><head>\n"
    "<title>Cloudflare Tunnel error | chroma.example.com | Cloudflare</title>\n"
    "</head><body>" + "<p>filler</p>" * 400 + "</body></html>"
)


class _Response:
    def __init__(self, code=200, content_type="application/json"):
        self._code = code
        self.headers = {"Content-Type": content_type}

    def getcode(self):
        return self._code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, content_type):
    return urllib.error.HTTPError("http://x", code, "err", {"Content-Type": content_type}, io.BytesIO(b""))


def _responsive(side_effect):
    with patch("urllib.request.urlopen", side_effect=side_effect):
        return is_chromadb_responsive("chroma.example.com", 443, ssl=True)


def test_json_heartbeat_is_responsive():
    assert _responsive(lambda *a, **k: _Response()) is True


def test_html_page_with_200_is_not_a_chroma_server():
    assert _responsive(lambda *a, **k: _Response(content_type="text/html; charset=UTF-8")) is False


def test_cloudflare_error_status_is_not_responsive():
    def tunnel_down(*a, **k):
        raise _http_error(530, "text/html; charset=UTF-8")

    assert _responsive(tunnel_down) is False


def test_auth_error_from_a_real_server_still_counts_as_responsive():
    def unauthorized(*a, **k):
        raise _http_error(401, "application/json")

    assert _responsive(unauthorized) is True


def test_auth_status_with_an_html_body_is_a_proxy_not_chroma():
    def forbidden_page(*a, **k):
        raise _http_error(403, "text/html")

    assert _responsive(forbidden_page) is False


def test_html_error_page_is_reduced_to_one_short_line():
    text = _short_error(RuntimeError(CLOUDFLARE_PAGE))
    assert "\n" not in text and len(text) < 200
    assert "Cloudflare Tunnel error" in text


def test_long_plain_errors_are_truncated():
    text = _short_error(RuntimeError("boom " * 500))
    assert len(text) <= 203 and text.endswith("...")
