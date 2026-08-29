"""Word budgets for user-facing documentation.

These pages reached 12,000 words once — roughly forty pages for a tool whose
usage is "type ``heliostat``" — because rationale kept being written into
documents whose readers only wanted instructions. The rule lives in
CONTRIBUTING.md; this is what makes it fail rather than be forgotten.

Budgets are deliberately generous. They are here to catch a doubling, not to
argue about a paragraph. If a document genuinely outgrows its budget, raise
the number here and say why in the commit message — the point is that the
growth is a decision someone made, not something that happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: path -> maximum words. Roughly 1.6x what each is today.
BUDGETS = {
    "README.md": 900,
    "REFERENCES.md": 1900,
    "CONTRIBUTING.md": 600,
    "CHANGELOG.md": 1800,
    "docs/index.md": 600,
    # The concepts page is where long-form detail is *supposed* to live.
    "docs/guide.md": 2200,
    "docs/paper.md": 700,
    # Mostly mkdocstrings directives rather than prose.
    "docs/api.md": 800,
    # Not user documentation: the signed-off build contract for the workspace
    # UI restructure (screens, fields, validation behavior). It is long
    # because it is the spec — implementation checks against it — and it
    # freezes at sign-off rather than growing. Delete the entry when the
    # restructure ships and the spec's content moves into guide.md.
    "docs/ui-spec.md": 4000,
    # Not user documentation: the plan for the stress harness -- what counts
    # as a failure and what gets varied. Delete the entry once the harness
    # exists and its own docstrings carry the detail.
    "docs/stress-test-plan.md": 1100,
    # Not user documentation: the rules the repo's own agent sessions run
    # under. Terse by design -- every word is an instruction someone must
    # follow, so the budget is the point.
    "CLAUDE.md": 400,
    # Not user documentation: the signed-off v0.2 build contract (same
    # reasoning and lifecycle as ui-spec.md above -- freezes at sign-off,
    # delete when v0.2 ships and the content moves into guide.md). Raised
    # 2026-08-29 (1st time): added draft riders §O (sunshape CSR) and §P
    # (pre-built reference fields), awaiting their own sign-off alongside
    # §M/§N. Raised again 2026-08-29 (2nd time): added draft rider §Q
    # ("Measure performance" calibrated duration estimates), awaiting its
    # own sign-off alongside §M/§N/§O/§P.
    "docs/ui-spec-v0.2.md": 6100,
    # Not user documentation: the formula-level build plan spec §C was
    # implemented from. Historical once §C shipped (2026-08-28); kept as
    # the derivation record for the secondary parameterization. Frozen.
    "docs/secondary-irradiance-plan.md": 700,
    # User-facing notes for the published release.
    "RELEASE_NOTES.md": 900,
    "examples/paper/README.md": 1600,
    "packaging/desktop/README.md": 400,
}


def _words(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("relative_path", sorted(BUDGETS))
def test_document_stays_within_its_budget(relative_path):
    path = ROOT / relative_path
    assert path.is_file(), f"{relative_path} is missing; update BUDGETS if it moved"
    count = _words(path)
    budget = BUDGETS[relative_path]
    assert count <= budget, (
        f"{relative_path} is {count} words, over its {budget}-word budget. "
        "Rationale belongs in code comments and commit messages; docs say how "
        "to use the thing. If the growth is deliberate, raise the budget in "
        "tests/test_docs_style.py and say why."
    )


def test_every_user_facing_document_has_a_budget():
    """A new page with no budget is how the last one crept up unnoticed."""
    tracked = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")]
    }
    # references.md is a one-line include of REFERENCES.md, budgeted above.
    tracked.discard("docs/references.md")
    missing = tracked - set(BUDGETS)
    assert not missing, (
        f"no word budget for {sorted(missing)} — add one to BUDGETS so it cannot grow unnoticed"
    )
