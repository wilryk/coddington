# Coddington — rules for Claude Code sessions and agents

## Testing (hard rule)
- Run ONLY the tests focused on what you changed (e.g. `pytest tests/test_solar.py -k floor`). The full suite takes ~10 minutes — do NOT run it per change or per agent.
- One full-suite run happens at integration time, before merging a branch to `main` — nowhere else.
- Use the project venv: `.venv\Scripts\python.exe -m pytest ...`.

## Machine resources (hard rule — stacked worker pools froze the machine on 2026-08-25)
- Each dev server lazily spawns a ~7-process trace worker pool. At most ONE server with a pool alive at a time; check `Get-CimInstance Win32_Process -Filter "Name = 'python.exe'"` before starting another, and stop your server when done.
- Never run a field trace, a day/year sweep, and the pool-spawning tests (`test_web.py` field-trace tests) concurrently — sequence heavy operations.
- Do not trace the full 643-heliostat field for UI verification — single heliostats or small fields suffice.

## Gotchas that have cost real time
- The dev server does NOT hot-reload Python — restart it after any backend change. Frontend JS/CSS needs a hard reload (`fetch(url, {cache: "reload"})` then reload).
- `git add -A` while agents work sweeps their in-progress files into your commit — always commit explicit paths. Agents themselves must never stage or commit.
- `textContent` includes hidden DOM elements — check computed visibility before declaring a UI bug found or fixed.
- Windows: process-pool scripts need an `if __name__ == "__main__":` guard; `Path.resolve()` silently rewrites path casing.

## Process
- Feature work requires Ryker's sign-off on the spec (docs/ui-spec.md, docs/ui-spec-v0.2.md) AND mockups (docs/mockups/) BEFORE app code. Bug fixes to shipped behavior don't wait.
- Keep user-facing updates design-level (screens and workflows, not implementation internals).
- Trust Ryker's physics corrections — they are a professional optics researcher.
