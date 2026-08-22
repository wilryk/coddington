# Contributing

## Writing documentation

**Docs say how to use the thing. Code comments and commit messages say why
it is the way it is.** Mixing those is how a README becomes forty pages: a
reader who wants to trace a mirror should not have to walk through the
reasoning behind every default to find the command.

Rules of thumb:

- Lead with the command, not the justification.
- One sentence of rationale is usually enough; if it needs a paragraph, it
  belongs in a code comment where the maintainer will meet it.
- Tables beat prose for anything enumerable — files, options, results.
- Do not repeat a section across README, `docs/` and an example README.
  Pick a home and link to it.

**Never cut**, however long it makes the page: limitations, caveats and
anything that stops a reader trusting a number they should not trust.
Brevity is not a licence to imply more confidence than the code has earned.

### Budgets

`tests/test_docs_style.py` enforces a word budget per document, so a slide
back is a test failure rather than something noticed a year later. Budgets
are generous — they exist to catch a doubling, not to police a paragraph.
If a document genuinely outgrows its budget, raise it in that file and say
why in the commit message.

## Code

- `ruff check` and `ruff format` (pinned in the `dev` extra so CI cannot
  disagree with your machine).
- `pytest` must be green. Physics changes need a test that would fail
  without them.
- Comments explain *why*, not *what* — the code already says what.

```
pip install -e .[dev,web,docs]
pytest -q && ruff check src tests examples && mkdocs build --strict
```
