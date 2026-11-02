"""The Hub opener waits out a rate limit (HTTP 429) instead of failing (E8 lost
models to 429s); other client errors fail at once."""
import email.message
import urllib.error

import pytest

from stowaway import hub


def _error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://example.invalid", code, "x", headers, None)


def test_rate_limit_is_retried(monkeypatch):
    calls, waits = [], []
    def fake(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise _error(429, retry_after=7)
        return "ok"
    monkeypatch.setattr(hub.urllib.request, "urlopen", fake)
    monkeypatch.setattr(hub.time, "sleep", waits.append)
    assert hub._open("https://example.invalid") == "ok"
    assert len(calls) == 3 and all(w >= 7 for w in waits)


def test_not_found_fails_at_once(monkeypatch):
    def fake(req, timeout):
        raise _error(404)
    monkeypatch.setattr(hub.urllib.request, "urlopen", fake)
    monkeypatch.setattr(hub.time, "sleep", lambda s: pytest.fail("must not wait on a 404"))
    with pytest.raises(urllib.error.HTTPError):
        hub._open("https://example.invalid")
