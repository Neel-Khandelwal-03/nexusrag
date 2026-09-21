## Summary

<!-- What does this PR do, and why? One or two sentences. -->

## Changes

<!-- Bullet list of the notable changes. -->

-

## How to test

<!-- Exact commands and manual steps a reviewer can follow. -->

```bash

```

## Screenshots

<!-- UI changes: add screenshots or a GIF. Otherwise write "n/a". -->

## Checklist

- [ ] Branch is based on `staging` (or this is a `staging` → `main` release PR)
- [ ] `ruff check .` passes
- [ ] `pytest` passes (Gemini mocked, no real API calls)
- [ ] No secrets, `.env`, `storage/`, SQLite files or model caches are committed
- [ ] `CHANGELOG.md` updated under **Unreleased**
- [ ] Docs / README updated if behaviour or configuration changed
