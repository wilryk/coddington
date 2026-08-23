"""Frozen-build canary for the Phase 3a workspace at ``/static/next/``.

Every asset the shell references (stylesheet, module script, importmap
entries) has to actually exist under ``src/heliostat/web/static`` -- an ES
module app with no build step has no bundler to catch a typo'd path, so
this is the only thing standing between a broken reference and a blank
page. Split out of ``test_web.py`` per that module's own one-file-per-
surface-area convention (see ``test_web_geometry.py``, ``test_web_library.py``).
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.web.app import STATIC_DIR, create_app  # noqa: E402

NEXT_INDEX = "/static/next/index.html"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def index_html(client):
    resp = client.get(NEXT_INDEX)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    return resp.text


def _referenced_static_urls(html: str) -> set[str]:
    """Every ``/static/...`` URL the page names: href=, src=, and every
    string value inside its ``<script type="importmap">`` block.
    """
    urls: set[str] = set()

    for attr in ("href", "src"):
        for m in re.finditer(rf'{attr}\s*=\s*"(/static/[^"]+)"', html):
            urls.add(m.group(1))
        for m in re.finditer(rf"{attr}\s*=\s*'(/static/[^']+)'", html):
            urls.add(m.group(1))

    importmap_match = re.search(
        r'<script[^>]+type="importmap"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert importmap_match, "index.html must declare an importmap for the vendored three.js"
    for m in re.finditer(r'"(/static/[^"]+)"', importmap_match.group(1)):
        urls.add(m.group(1))

    return urls


def test_next_index_is_html(index_html):
    assert "<title>Coddington</title>" in index_html


def test_next_index_declares_importmap_for_three(index_html):
    assert 'type="importmap"' in index_html
    assert "three.module.min.js" in index_html
    assert "OrbitControls.js" in index_html


def test_every_referenced_static_asset_exists_on_disk(index_html):
    urls = _referenced_static_urls(index_html)
    # Sanity: the extraction itself found something real, so a regression
    # that empties the importmap or drops the script tag fails loudly
    # rather than this test silently passing on zero URLs.
    assert len(urls) >= 3

    missing = []
    for url in sorted(urls):
        assert url.startswith("/static/"), url
        rel = url[len("/static/") :]
        path = STATIC_DIR / rel
        if not path.is_file():
            missing.append(url)
    assert not missing, f"referenced but missing on disk: {missing}"


def test_every_referenced_static_asset_serves_200(client, index_html):
    urls = _referenced_static_urls(index_html)
    failures = {}
    for url in sorted(urls):
        resp = client.get(url)
        if resp.status_code != 200:
            failures[url] = resp.status_code
    assert not failures, f"referenced static assets that did not serve 200: {failures}"


def test_old_frontend_is_untouched(client):
    """Phase 3a lives entirely under /static/next/ -- the legacy GUI at
    /static/index.html (served at the app root) must be unaffected."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
