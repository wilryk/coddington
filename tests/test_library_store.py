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
    # Neither collection has built-ins, so an ordinary name always saves --
    # note this deliberately does not reuse _BUILTIN_NAME, since some
    # built-in names (e.g. "Axicon 27 m / 20 deg / 14 m") use characters no
    # store name is ever allowed to use.
    save_entry("runs", "whatever i like", {"kind": "day"})
    save_entry("projects", "whatever i like", {"a": 1})


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


class _FakeExisting:
    """Stand-in for a ``Path`` result from a directory scan -- only
    ``.stem`` is read by :func:`_find_by_identity`, so a real file is not
    needed to exercise its selection logic."""

    def __init__(self, stem):
        self.stem = stem

    def __repr__(self):
        return f"_FakeExisting({self.stem!r})"


class _FakeDir:
    """Duck-typed stand-in for the directory :func:`_find_by_identity`
    scans -- lets a test hand it two colliding stems directly, which is the
    only way to exercise "two entries already collide" at all: on the
    case-insensitive filesystem this suite runs its other tests on
    (Windows), writing "My design.json" then "my design.json" for real
    does not produce two files, it silently overwrites the first -- the
    very bug this module exists to prevent going forward. This is what
    "simulate rather than depend on the host" means in practice."""

    def __init__(self, stems):
        self._stems = stems

    def is_dir(self):
        return True

    def glob(self, pattern):
        return (_FakeExisting(s) for s in self._stems)


def test_reject_case_collision_does_not_depend_on_host_case_folding():
    """The collision check is pure string comparison over a supplied stem
    list -- proof it cannot be relying on the host filesystem to fold
    names together, since no filesystem is involved here at all."""
    from heliostat.web.library import _reject_case_collision as reject

    with pytest.raises(LibraryError):
        reject(["My design"], "my design", LibraryError, "design entry")
    with pytest.raises(LibraryError):
        reject(["MY DESIGN"], "My Design", LibraryError, "design entry")
    reject(["My design"], "My design", LibraryError, "design entry")  # same name: no conflict
    reject(["Other design"], "My design", LibraryError, "design entry")  # unrelated: no conflict


def test_find_by_identity_prefers_the_exact_match_when_a_collision_already_exists():
    """If two on-disk entries already collide by identity (only reachable
    from data predating this rule), a lookup by either one's *exact* saved
    name still finds that one, deterministically -- not whichever the
    directory scan happens to reach first."""
    from heliostat.web.library import _find_by_identity

    fake_dir = _FakeDir(["My design", "my design"])
    assert _find_by_identity(fake_dir, "My design").stem == "My design"
    assert _find_by_identity(fake_dir, "my design").stem == "my design"
    # A third spelling that only fold-matches is inherently ambiguous
    # between the two -- it resolves to one of them, not to neither.
    assert _find_by_identity(fake_dir, "MY DESIGN").stem in {"My design", "my design"}


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
