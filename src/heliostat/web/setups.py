"""Named setups: save what is on screen, load it back later.

A "setup" is whatever the GUI needs to restore a session -- the design and
its surface, the optical layout and its geometry, the sun (or the site and
moment that produced it), the fidelity mode, and the field layout. This
module only stores and returns that document; it does not interpret it, so
adding a control to the GUI does not need a change here.

Setups live in files under a per-user directory rather than in a database:
the app is a local tool, and a folder of readable JSON is something the
user can back up, diff, hand to a colleague, or delete without this
program's help.

**Names are not paths.** A setup name is used to build a filename, so it is
validated against a strict pattern and the resolved path is checked to be
inside the setups directory before anything is written or read. Both
checks, not one: the pattern is the rule, and the containment check is what
catches a mistake in the pattern.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

#: Setup names: letters, digits, spaces, dot, dash, underscore. No path
#: separators, no leading dot, bounded length.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")

#: Windows reserves these regardless of extension.
_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class SetupError(ValueError):
    """A setup could not be saved or loaded (reason is the message)."""


def setups_dir() -> Path:
    """Where setups are kept.

    ``HELIOSTAT_SETUPS_DIR`` overrides it, which is what the tests use so
    they never touch a real user's directory.
    """
    override = os.environ.get("HELIOSTAT_SETUPS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".heliostat" / "setups"


def _validate_name(name: str) -> str:
    """Return ``name`` if it is safe to turn into a filename."""
    # Normalise first: without this, a name that looks like ASCII can carry
    # combining characters that the pattern would accept and the filesystem
    # would fold into something else.
    normalised = unicodedata.normalize("NFC", name).strip()
    if not _NAME_RE.match(normalised):
        raise SetupError(
            "a setup name may use letters, digits, spaces, dots, dashes and "
            "underscores, must start with a letter or digit, and must be at "
            "most 64 characters"
        )
    if normalised.split(".")[0].lower() in _RESERVED:
        raise SetupError(f"{normalised!r} is a reserved device name on Windows")
    return normalised


def _path_for(name: str) -> Path:
    root = setups_dir().resolve()
    path = (root / f"{_validate_name(name)}.json").resolve()
    # Belt and braces: the pattern above should make this impossible, so if
    # it ever fires the pattern is what is wrong.
    if path.parent != root:
        raise SetupError("that setup name does not resolve inside the setups directory")
    return path


def save_setup(name: str, document: dict) -> dict:
    """Write one setup, overwriting any setup of the same name."""
    path = _path_for(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": _validate_name(name),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": document,
    }
    # Write beside, then replace: a crash mid-write leaves the previous
    # setup intact rather than a truncated file that will not parse.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return {"name": payload["name"], "saved_at": payload["saved_at"]}


def load_setup(name: str) -> dict:
    """Read one setup back."""
    path = _path_for(name)
    if not path.is_file():
        raise SetupError(f"no setup named {name!r}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SetupError(f"setup {name!r} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or "document" not in payload:
        raise SetupError(f"setup {name!r} does not look like a saved setup")
    return payload


def list_setups() -> list[dict]:
    """Every saved setup, newest first, without loading its document."""
    root = setups_dir()
    if not root.is_dir():
        return []
    entries = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "name": str(payload.get("name", path.stem)),
                    "saved_at": str(payload.get("saved_at", "")),
                }
            )
        except (OSError, json.JSONDecodeError):
            # A file someone hand-edited into invalid JSON should not make
            # the whole list unreadable; it simply does not appear.
            continue
    entries.sort(key=lambda e: e["saved_at"], reverse=True)
    return entries


def delete_setup(name: str) -> None:
    path = _path_for(name)
    if not path.is_file():
        raise SetupError(f"no setup named {name!r}")
    path.unlink()
