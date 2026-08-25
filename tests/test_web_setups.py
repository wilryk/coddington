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

from heliostat.web.setups import SetupError, delete_setup, list_setups, load_setup, save_setup

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


def test_case_collision_is_caught_even_if_both_files_already_coexist(setups_dir):
    """Simulate what only a case-sensitive filesystem (Linux) could ever
    produce on its own: two files coexisting that differ only by case. The
    rejection is computed from the directory listing in Python, not from
    the host's own case folding, so it must hold everywhere."""
    setups_dir.mkdir(parents=True)
    (setups_dir / "My Tower.json").write_text(
        json.dumps({"name": "My Tower", "saved_at": "a", "document": DOC_A}), encoding="utf-8"
    )
    (setups_dir / "my tower.json").write_text(
        json.dumps({"name": "my tower", "saved_at": "b", "document": DOC_B}), encoding="utf-8"
    )
    # Both pre-existing setups still load individually.
    assert load_setup("My Tower")["document"] == DOC_A
    assert load_setup("my tower")["document"] == DOC_B
    # A new save adding a third colliding spelling is refused.
    with pytest.raises(SetupError):
        save_setup("MY TOWER", DOC_A)


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
