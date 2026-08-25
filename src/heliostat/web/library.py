"""The Library: named heliostat designs, receiver configs, projects and runs.

Four collections (``designs``, ``receivers``, ``projects``, ``runs``), each a
folder of named JSON documents, generalising :mod:`heliostat.web.setups`'s
single-collection store rather than duplicating its file-safety machinery
four times over. This module is deliberately as opinion-free as setups.py
is: it stores and returns whatever document it is given and does not
interpret it -- what a receiver, a project or a run document must actually
contain is :mod:`heliostat.web.app`'s business (see ``ReceiverDocument``,
``ProjectDocument``, ``SavedRunDocument`` and ``_validate_library_document``
there), not this one's.

``setups.py`` itself is untouched: it is a separate, older store (the GUI's
free-form "save what's on screen" snapshots, honouring its own
``HELIOSTAT_SETUPS_DIR``) that this module does not replace or migrate --
see docs/ui-spec.md 5, "Migration", for where that is headed. What *is*
shared, by import rather than by copy, are the private name-safety rules
(the character pattern, the Windows reserved-name list, and the
case/confusable-folding identity below): the two stores must never disagree
on what a safe filename looks like or what makes two names "the same", and
importing settles that automatically instead of trusting two copies to be
kept equal.

**Identity, same rule as setups.py:** two names that fold together (case,
and cheaply-normalised Unicode confusables) are the same entry, because on
the case-insensitive filesystem this ships on they would already be the
same *file* -- a second save under such a name is refused as a conflict
rather than silently destroying the first, in every collection, on every
platform. Entries saved before this rule existed keep listing and loading
under their existing name; only a new save that would collide is turned
away.

**Built-ins are unshadowable by the same identity rule, enforced here.**
Saving or deleting a name that folds to a built-in's name -- whitespace,
case, or a cheap Unicode confusable apart -- is refused by this module
itself, not left to the endpoint's exact-string check: a store that only
*sometimes* agrees with the endpoint about which names are protected is
the bug this fixes, so the store now knows built-in names too and is the
one place that can never be talked past.
"""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .builtin_library import BUILTIN_DESIGNS, BUILTIN_RECEIVERS
from .setups import _NAME_RE, _RESERVED, _fold, _reject_case_collision

#: The four collections a library entry can belong to. ``projects`` and
#: ``runs`` have no built-ins -- a project or a finished run is always
#: something a user made, never a manuscript default.
COLLECTIONS: tuple[str, ...] = ("designs", "receivers", "projects", "runs")

#: Built-in names per collection, for :func:`_builtin_collision` -- the
#: same designs/receivers constants :mod:`heliostat.web.app` serves built-in
#: documents from (as ``_BUILTIN_LIBRARY`` there), kept here too so this
#: module can protect them without trusting the endpoint to have done it.
_BUILTIN_NAMES: dict[str, dict[str, dict]] = {
    "designs": BUILTIN_DESIGNS,
    "receivers": BUILTIN_RECEIVERS,
    "projects": {},
    "runs": {},
}


class LibraryError(ValueError):
    """A library entry could not be saved, loaded or deleted (reason is the message)."""


def library_dir() -> Path:
    """Where the library is kept.

    ``HELIOSTAT_LIBRARY_DIR`` overrides it, which is what the tests use so
    they never touch a real user's directory -- the same convention
    :func:`heliostat.web.setups.setups_dir` uses for its own store.
    """
    override = os.environ.get("HELIOSTAT_LIBRARY_DIR")
    if override:
        return Path(override)
    return Path.home() / ".heliostat" / "library"


def _validate_collection(collection: str) -> str:
    if collection not in COLLECTIONS:
        raise LibraryError(
            f"unknown library collection {collection!r} -- must be one of " + ", ".join(COLLECTIONS)
        )
    return collection


def _validate_name(name: str) -> str:
    """Return ``name`` if it is safe to turn into a filename.

    Identical rule to :func:`heliostat.web.setups._validate_name` (NFC
    normalisation, then the same character pattern and reserved-name check),
    reused via the shared pattern/set imported above rather than re-typed, so
    the two stores can never quietly diverge on what counts as a safe name.
    """
    normalised = unicodedata.normalize("NFC", name).strip()
    if not _NAME_RE.match(normalised):
        raise LibraryError(
            "a library entry name may use letters, digits, spaces, dots, dashes and "
            "underscores, must start with a letter or digit, and must be at "
            "most 64 characters"
        )
    if normalised.split(".")[0].lower() in _RESERVED:
        raise LibraryError(f"{normalised!r} is a reserved device name on Windows")
    return normalised


def _collection_dir(collection: str) -> Path:
    return library_dir() / _validate_collection(collection)


def _path_for(collection: str, name: str) -> Path:
    root = _collection_dir(collection).resolve()
    path = (root / f"{_validate_name(name)}.json").resolve()
    # Belt and braces, matching setups._path_for: the pattern above should
    # make this impossible, so if it ever fires the pattern is what is wrong.
    if path.parent != root:
        raise LibraryError("that name does not resolve inside its library collection")
    return path


def _builtin_collision(collection: str, validated_name: str) -> str | None:
    """The built-in name ``validated_name`` folds to in ``collection``, or
    ``None`` -- whitespace is already gone by the time a name is validated,
    so this only has to fold case and cheap Unicode confusables to catch
    what an exact string comparison would miss."""
    key = _fold(validated_name)
    for builtin_name in _BUILTIN_NAMES.get(collection, ()):
        if _fold(builtin_name) == key:
            return builtin_name
    return None


def save_entry(collection: str, name: str, document: dict) -> dict:
    """Write one entry, overwriting any entry of the same name in ``collection``.

    Refuses a name that folds to a built-in's name, or that folds to a
    *different* existing entry's name -- see the module docstring.
    """
    path = _path_for(collection, name)
    builtin = _builtin_collision(collection, path.stem)
    if builtin is not None:
        raise LibraryError(
            f"{path.stem!r} is a built-in {collection[:-1]} ({builtin!r}) and cannot be "
            "saved over"
        )
    existing_stems = (p.stem for p in path.parent.glob("*.json")) if path.parent.is_dir() else ()
    _reject_case_collision(existing_stems, path.stem, LibraryError, f"{collection[:-1]} entry")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": _validate_name(name),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": document,
    }
    # Write beside, then replace: a crash mid-write leaves the previous
    # entry intact rather than a truncated file that will not parse.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return {"name": payload["name"], "saved_at": payload["saved_at"]}


def load_entry(collection: str, name: str) -> dict:
    """Read one user-saved entry back. Built-ins are not stored here -- the
    endpoint serves those from :mod:`heliostat.web.builtin_library` directly."""
    path = _path_for(collection, name)
    builtin = _builtin_collision(collection, path.stem)
    if builtin is not None:
        raise LibraryError(f"{path.stem!r} is a built-in {collection[:-1]} ({builtin!r})")
    if not path.is_file():
        raise LibraryError(f"no {collection} entry named {name!r}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LibraryError(f"{collection} entry {name!r} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or "document" not in payload:
        raise LibraryError(f"{collection} entry {name!r} does not look like a saved entry")
    return payload


def list_entries(collection: str) -> list[dict]:
    """Every user-saved entry in ``collection``, newest first, without
    loading its document. Built-ins are not listed here -- the endpoint
    prepends those from the constants before this.

    ``size_bytes`` is the entry's file size on disk -- exactly what a
    ``runs`` entry's "Manage saved runs" footprint reports, since the whole
    document (including any embedded flux PNGs) lives in that one file.
    """
    root = _collection_dir(collection)
    if not root.is_dir():
        return []
    entries = []
    for path in sorted(root.glob("*.json")):
        if _builtin_collision(collection, path.stem) is not None:
            # A stray file from before built-ins were protected here --
            # never created by this version, but if one exists it must not
            # be listed as if it were a real user entry.
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "name": str(payload.get("name", path.stem)),
                    "saved_at": str(payload.get("saved_at", "")),
                    "size_bytes": path.stat().st_size,
                }
            )
        except (OSError, json.JSONDecodeError):
            # A file someone hand-edited into invalid JSON should not make
            # the whole list unreadable; it simply does not appear.
            continue
    entries.sort(key=lambda e: e["saved_at"], reverse=True)
    return entries


def delete_entry(collection: str, name: str) -> None:
    path = _path_for(collection, name)
    builtin = _builtin_collision(collection, path.stem)
    if builtin is not None:
        raise LibraryError(
            f"{path.stem!r} is a built-in {collection[:-1]} ({builtin!r}) and cannot be deleted"
        )
    if not path.is_file():
        raise LibraryError(f"no {collection} entry named {name!r}")
    path.unlink()
