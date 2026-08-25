"""Store-level tests for :mod:`heliostat.web.setups`'s case-insensitive
name-uniqueness rule, exercised directly against the store -- no FastAPI or
``app.py`` involved (the HTTP-level setup tests live in ``tests/test_web.py``,
which this module deliberately does not touch).

Setups have no built-in entries (unlike the library's designs/receivers), so
only the case-collision side of the identity rule applies here.
"""

from __future__ import annotations

import json

import pytest

from heliostat.web.setups import (
    SetupError,
    _find_by_identity,
    _reject_case_collision,
    delete_setup,
    list_setups,
    load_setup,
    save_setup,
)

DOC_A = {"n": 1}
DOC_B = {"n": 2}


@pytest.fixture
def setups_dir(tmp_path, monkeypatch):
    """Point the setups store at a temp dir, never the real home directory."""
    monkeypatch.setenv("HELIOSTAT_SETUPS_DIR", str(tmp_path / "setups"))
    return tmp_path / "setups"


def test_name_differing_only_by_case_is_rejected(setups_dir):
    save_setup("My Tower", DOC_A)
    with pytest.raises(SetupError):
        save_setup("my tower", DOC_B)
    # The original survives with its content intact.
    assert load_setup("My Tower")["document"] == DOC_A


def test_resaving_the_identical_name_still_overwrites(setups_dir):
    save_setup("My Tower", DOC_A)
    save_setup("My Tower", DOC_B)
    assert load_setup("My Tower")["document"] == DOC_B


class _FakeExisting:
    """Stand-in for a ``Path`` result from a directory scan -- only
    ``.stem`` is read by :func:`_find_by_identity`, so a real file is not
    needed to exercise its selection logic."""

    def __init__(self, stem):
        self.stem = stem


class _FakeDir:
    """Duck-typed stand-in for the directory :func:`_find_by_identity`
    scans -- lets a test hand it two colliding stems directly, which is the
    only way to exercise "two entries already collide" at all: on the
    case-insensitive filesystem this suite's other tests run on (Windows),
    writing "My Tower.json" then "my tower.json" for real does not produce
    two files, it silently overwrites the first -- the very bug this module
    exists to prevent going forward. This is what "simulate rather than
    depend on the host" means in practice."""

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
    with pytest.raises(SetupError):
        _reject_case_collision(["My Tower"], "my tower", SetupError, "setup")
    with pytest.raises(SetupError):
        _reject_case_collision(["MY TOWER"], "My Tower", SetupError, "setup")
    _reject_case_collision(["My Tower"], "My Tower", SetupError, "setup")  # same name: no conflict
    _reject_case_collision(["Other"], "My Tower", SetupError, "setup")  # unrelated: no conflict


def test_find_by_identity_prefers_the_exact_match_when_a_collision_already_exists():
    """If two on-disk entries already collide by identity (only reachable
    from data predating this rule), a lookup by either one's *exact* saved
    name still finds that one, deterministically -- not whichever the
    directory scan happens to reach first."""
    fake_dir = _FakeDir(["My Tower", "my tower"])
    assert _find_by_identity(fake_dir, "My Tower").stem == "My Tower"
    assert _find_by_identity(fake_dir, "my tower").stem == "my tower"
    # A third spelling that only fold-matches is inherently ambiguous
    # between the two -- it resolves to one of them, not to neither.
    assert _find_by_identity(fake_dir, "MY TOWER").stem in {"My Tower", "my tower"}


def test_preexisting_setup_still_lists_and_loads(setups_dir):
    setups_dir.mkdir(parents=True)
    (setups_dir / "Old setup.json").write_text(
        json.dumps({"name": "Old setup", "saved_at": "2020-01-01T00:00:00+00:00", "document": DOC_A}),
        encoding="utf-8",
    )
    assert [e["name"] for e in list_setups()] == ["Old setup"]
    assert load_setup("Old setup")["document"] == DOC_A


def test_deleting_after_a_rejected_collision_leaves_original_intact(setups_dir):
    save_setup("My Tower", DOC_A)
    with pytest.raises(SetupError):
        save_setup("MY TOWER", DOC_B)
    delete_setup("My Tower")
    assert list_setups() == []
