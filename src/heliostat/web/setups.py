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

**Names are case-insensitively unique.** ``"My Tower"`` and ``"my tower"``
are the same entry, not two -- on the case-insensitive filesystem this ships
on they would already be the same *file*, so treating them as the same
*name* everywhere (Windows or not) is what stops a second save from
silently destroying the first instead of failing loudly. A save that would
introduce a second name differing only by case (or by a cheaply-normalised
Unicode confusable) from an existing one is refused as a conflict; saving
the exact same name again is an ordinary overwrite. Entries already on disk
before this rule existed are unaffected -- they keep listing and loading
under whatever name they were given; only a new save that would collide is
turned away.
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


def _fold(name: str) -> str:
    """The identity a name collides under, independent of the host
    filesystem's own case-folding rules.

    NFKC first, so cheap Unicode look-alikes (full-width letters, compat
    ligatures) fold together the same way a real font would render them;
    casefold second, for case. Deliberately *not* what ``_validate_name``
    stores as the entry's name -- that keeps the user's own spelling; this
    is only ever used to decide whether two spellings are "the same" name.
    """
    return unicodedata.normalize("NFKC", name).casefold()


def _reject_case_collision(existing_stems, validated_name: str, error_cls, kind: str) -> None:
    """Refuse ``validated_name`` if it collides, by :func:`_fold` identity,
    with a *different* name already on disk -- resaving the identical name
    is an ordinary overwrite and is not a collision.

    Enforced here in Python, before any filesystem call, rather than left to
    however the host happens to fold names: that is what makes the rule
    behave the same on a case-preserving-insensitive filesystem (Windows,
    default macOS) and a case-sensitive one (Linux) alike. It is also the
    reason ``validated_name`` must come straight from :func:`_validate_name`
    and never from an already-resolved ``Path.stem``: on Windows,
    ``Path.resolve()`` silently rewrites a path's casing to match an
    existing directory entry, so a resolved path for ``"my tower"`` is
    already ``"My Tower"`` the moment that file exists -- exactly the
    collision this function exists to catch, self-erased before it could be
    seen.
    """
    key = _fold(validated_name)
    for stem in existing_stems:
        if stem == validated_name:
            continue
        if _fold(stem) == key:
            raise error_cls(
                f"{validated_name!r} conflicts with the existing {kind} {stem!r} -- "
                "names that differ only by case or accents are treated as the same entry"
            )


def _find_by_identity(root: Path, validated_name: str) -> Path | None:
    """The on-disk ``*.json`` file for ``validated_name``.

    An explicit directory scan, not ``Path.resolve()``'s own on-disk
    case-correction (see :func:`_reject_case_collision`'s docstring for why
    that would give a different answer on Windows than on Linux) -- so a
    lookup finds the same entry, by any spelling that shares its identity,
    on either platform.

    Exact stem match first, so the common case (querying the name exactly
    as saved) is deterministic; :func:`_fold` identity second, so a
    differing-case spelling still finds the one entry that identity is
    allowed to have. Two on-disk files that already collide by identity
    (only possible from data that predates this rule, e.g. carried over
    from a case-sensitive filesystem) are not something this can
    disambiguate further -- both still exist and still list, but a fold-only
    lookup between them returns whichever this scan reaches first.
    """
    if not root.is_dir():
        return None
    candidates = list(root.glob("*.json"))
    for p in candidates:
        if p.stem == validated_name:
            return p
    key = _fold(validated_name)
    for p in candidates:
        if _fold(p.stem) == key:
            return p
    return None


def save_setup(name: str, document: dict) -> dict:
    """Write one setup, overwriting any setup of the same name.

    A different name that collides case-insensitively with an existing one
    is refused rather than silently overwriting it -- see the module
    docstring.
    """
    path = _path_for(name)
    validated_name = _validate_name(name)
    existing_stems = (p.stem for p in path.parent.glob("*.json")) if path.parent.is_dir() else ()
    _reject_case_collision(existing_stems, validated_name, SetupError, "setup")
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
    """Read one setup back, by any spelling that shares its saved name's
    case/confusable identity, not only the exact one it was saved under."""
    path = _find_by_identity(setups_dir(), _validate_name(name))
    if path is None:
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
    path = _find_by_identity(setups_dir(), _validate_name(name))
    if path is None:
        raise SetupError(f"no setup named {name!r}")
    path.unlink()
