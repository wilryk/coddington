"""Store-level tests for :mod:`heliostat.web.library`'s name-identity rules
-- built-ins that cannot be shadowed or deleted, and case-insensitive
uniqueness among user entries -- exercised directly against the store, no
FastAPI or ``app.py`` involved.

See ``tests/test_web_library.py`` for the same rules exercised through the
HTTP layer (status codes, the endpoint's own built-in check).
"""

from __future__ import annotations

import json

import pytest

from heliostat.web.builtin_library import BUILTIN_DESIGNS
from heliostat.web.library import (
    LibraryError,
    delete_entry,
    list_entries,
    load_entry,
    save_entry,
)

DESIGN_DOC = {"type": "rect", "width_mm": 1000.0, "height_mm": 800.0}
OTHER_DOC = {"type": "rect", "width_mm": 2000.0, "height_mm": 900.0}
_BUILTIN_NAME = next(iter(BUILTIN_DESIGNS))


@pytest.fixture
def library_dir(tmp_path, monkeypatch):
    """Point the library store at a temp dir, never the real home directory."""
    monkeypatch.setenv("HELIOSTAT_LIBRARY_DIR", str(tmp_path / "library"))
    return tmp_path / "library"


# ---------------------------------------------------------------------------
# built-ins: unshadowable regardless of whitespace or case


@pytest.mark.parametrize(
    "mangled",
    [
        f"{_BUILTIN_NAME} ",
        f" {_BUILTIN_NAME}",
        _BUILTIN_NAME.upper(),
        _BUILTIN_NAME.lower(),
    ],
)
def test_padded_or_recased_name_cannot_save_over_a_builtin(library_dir, mangled):
    with pytest.raises(LibraryError):
        save_entry("designs", mangled, OTHER_DOC)
    assert list_entries("designs") == []  # no shadow file was created


@pytest.mark.parametrize("mangled", [f"{_BUILTIN_NAME} ", _BUILTIN_NAME.upper()])
def test_padded_or_recased_name_cannot_delete_a_builtin(library_dir, mangled):
    with pytest.raises(LibraryError):
        delete_entry("designs", mangled)


def test_padded_name_query_never_returns_a_stray_shadow_file(library_dir):
    """Even if a shadow file already sits on disk (left over from a version
    that let one through), asking for the built-in's name with different
    padding must refuse, not silently resolve to the stray file."""
    collection_dir = library_dir / "designs"
    collection_dir.mkdir(parents=True)
    (collection_dir / f"{_BUILTIN_NAME}.json").write_text(
        json.dumps({"name": _BUILTIN_NAME, "saved_at": "x", "document": OTHER_DOC}),
        encoding="utf-8",
    )
    with pytest.raises(LibraryError):
        load_entry("designs", f"{_BUILTIN_NAME} ")
    assert list_entries("designs") == []  # the stray file is never listed either


def test_runs_and_projects_have_no_builtins_to_collide_with(library_dir):
    # Neither collection has built-ins, so an ordinary name always saves.
    save_entry("runs", _BUILTIN_NAME, {"kind": "day"})
    save_entry("projects", _BUILTIN_NAME, {"a": 1})


# ---------------------------------------------------------------------------
# case-insensitive uniqueness among user entries


def test_name_differing_only_by_case_is_rejected(library_dir):
    save_entry("designs", "My design", DESIGN_DOC)
    with pytest.raises(LibraryError):
        save_entry("designs", "my design", OTHER_DOC)
    # The original survives with its content intact -- not just "a file
    # exists": the bug this guards against is a silent overwrite.
    assert load_entry("designs", "My design")["document"] == DESIGN_DOC


def test_resaving_the_identical_name_still_overwrites(library_dir):
    save_entry("designs", "My design", DESIGN_DOC)
    save_entry("designs", "My design", OTHER_DOC)
    assert load_entry("designs", "My design")["document"] == OTHER_DOC


def test_case_collision_is_caught_even_if_both_files_already_coexist(library_dir):
    """Simulate what only a case-sensitive filesystem (Linux) could ever
    produce on its own: two files coexisting that differ only by case. The
    rejection must come from the directory listing in Python, not from the
    host's own case folding -- so this must catch it on any platform,
    including the case-insensitive one these files were just written on."""
    collection_dir = library_dir / "designs"
    collection_dir.mkdir(parents=True)
    (collection_dir / "My design.json").write_text(
        json.dumps({"name": "My design", "saved_at": "a", "document": DESIGN_DOC}),
        encoding="utf-8",
    )
    (collection_dir / "my design.json").write_text(
        json.dumps({"name": "my design", "saved_at": "b", "document": OTHER_DOC}),
        encoding="utf-8",
    )
    # Both pre-existing entries still load individually.
    assert load_entry("designs", "My design")["document"] == DESIGN_DOC
    assert load_entry("designs", "my design")["document"] == OTHER_DOC
    # A *new* save adding a third colliding spelling is refused.
    with pytest.raises(LibraryError):
        save_entry("designs", "MY DESIGN", DESIGN_DOC)


def test_runs_collection_also_enforces_case_uniqueness(library_dir):
    run_doc = {"kind": "day"}
    save_entry("runs", "Day One", run_doc)
    with pytest.raises(LibraryError):
        save_entry("runs", "day one", {"kind": "different"})
    assert load_entry("runs", "Day One")["document"] == run_doc


def test_projects_collection_also_enforces_case_uniqueness(library_dir):
    save_entry("projects", "Proj", {"a": 1})
    with pytest.raises(LibraryError):
        save_entry("projects", "PROJ", {"a": 2})


# ---------------------------------------------------------------------------
# entries saved before this rule existed keep working


def test_preexisting_entry_still_lists_and_loads(library_dir):
    collection_dir = library_dir / "receivers"
    collection_dir.mkdir(parents=True)
    doc = {"optics": "prime_focus", "params": {}}
    (collection_dir / "Old tower.json").write_text(
        json.dumps({"name": "Old tower", "saved_at": "2020-01-01T00:00:00+00:00", "document": doc}),
        encoding="utf-8",
    )
    assert [e["name"] for e in list_entries("receivers")] == ["Old tower"]
    assert load_entry("receivers", "Old tower")["document"] == doc
