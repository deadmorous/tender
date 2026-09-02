---
name: release-notes
description: Write or update RELEASE_NOTES.md by summarizing the commits between git tags. Use when the user asks for release notes, a changelog, "what changed in v0.x", or after creating a new tag.
---

# Release notes from tags

A *release* is a tagged revision.  The notes live in `RELEASE_NOTES.md` at the
repo root, newest release first.  They are a **summary, not a log**: a reader
should get the shape of a release in under a minute.

## Procedure

1. **List the releases in order.**

   ```bash
   git tag --sort=creatordate
   ```

   Consecutive tags define the ranges: `<prev>..<tag>`, plus the range before
   the first tag (`<first-tag>` alone) and, if `HEAD` is ahead of the last tag,
   an `Unreleased` section for `<last-tag>..HEAD`.

2. **Skip what is already written.**  Read `RELEASE_NOTES.md` if it exists and
   only generate sections for tags missing from it.  Always regenerate
   `Unreleased`.  Never rewrite an existing release section unless asked —
   published notes are stable.

3. **Read the range.**

   ```bash
   git log --format='%s' <prev>..<tag>          # subjects, newest first
   git log --format='%s%n%b' <prev>..<tag>      # add bodies if subjects are thin
   git log --date=short --format='%ad' -1 <tag>^{commit}   # release date
   git rev-list --count <prev>..<tag>           # commit count
   git diff --stat <prev>..<tag> | tail -1      # rough size, optional
   ```

   For a large range, read subjects only; drop to bodies for the handful of
   commits whose subject is unclear.

4. **Summarize aggressively.**  Cluster commits into themes and write one
   bullet per theme, not per commit.

   - **3–7 bullets per release.**  Ten commits may be one bullet; a hundred
     commits are still at most seven.  If a range resists, the release had
     several arcs — name the arcs, not the steps.
   - Each bullet is **one line**: what a user of the library can now do, or
     what changed for them.  Capability over implementation.
   - Lead with the most consequential item.
   - Fold pure churn — formatting, doc typos, CI fiddling, notebook
     stripping, refactors with no visible effect — into at most one trailing
     bullet, or omit it.
   - Name a breaking change explicitly, first, marked **Breaking:**.

5. **Voice.**  Match the repo's commit style: lower-case, plain, declarative,
   present tense, no marketing.  Use the project's own vocabulary (see
   `vibes/glossary.md`), and spell out an abbreviation at first use in the
   file.  Do not credit tools or agents.

## Format

````markdown
# Release notes

## Unreleased

- ...

## v0.4 — 2026-09-01

*21 commits since v0.3.*

- an interactive derivation surface: a derivation is a list of steps, ...
- ...

## v0.3 — 2026-08-30
...
````

Vibe numbers (`vibe 000108`) are internal; mention a vibe file only when it is
the deliverable of the release, and then by subject rather than number.

## After writing

Show the user the new sections in the reply, not just the file path.  Do not
commit unless asked; if the user does ask, follow the repo's committing rules
in `CLAUDE.md` (clang-format is irrelevant here — this touches Markdown only).
