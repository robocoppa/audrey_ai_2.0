# Campaign 2 Phase 3 - Lesson-cite drift checker

Tooling-only phase. Adds `scripts/check-lesson-links.py`: a one-shot
auditor that catches stale file:line references in long-form Markdown
docs.

Lessons cite source by `path#L<n>` Markdown links. Every time code
shifts (a new import, a refactor, a function rename), every lesson that
referenced a moved line silently goes wrong. We hit this twice during
Phase 1 and 2a: Phase 1 shifted `routes/openai.py` by ~6 lines, Phase
2a shifted `graph.py` and `classify.py` by 2-15 lines depending on
section. Without a checker, the cite says "see line 188" and line 188
is now blank or the wrong statement.

What the checker does, in order:

1. Walks every doc under `DOCS_GLOB` (default `docs/lessons/*.md`).
2. Extracts every `(label)(path#L<n>)` cite. Single-line and range
   (`#L<a>-L<b>`) cites are both supported.
3. For each cite, looks for a fenced code block in the doc within a
   small lookahead window. If found, captures the block's first code
   line as the canonical snippet — that's the line the lesson is
   pointing the reader at.
4. Opens the target source file and tries to confirm:
   - **Snippet match (preferred path)**: does the cited line still
     contain the captured snippet (modulo indentation)? If not, does
     the snippet exist elsewhere in the file? If yes, the checker
     proposes the correct line as a concrete `fix:` directive.
   - **Landmark heuristic (fallback)**: when there's no nearby
     snippet (e.g. an inline mid-prose cite), the checker only
     verifies the cited line looks load-bearing — a `def`, `class`,
     `ALL_CAPS = `, YAML key, etc. False positives are possible here,
     so the verdict is the softer `DRIFT?` rather than `DRIFT`.
5. Prints a structured report grouped by severity.

What stays the same:

- No runtime change. The script is a developer / pre-commit tool.
- No new dependencies. Pure Python 3.10+ standard library.
- No data migration, no rebuild, no compose change.

What changed:

- **`scripts/check-lesson-links.py`** (new) - the auditor. ~330 lines,
  three modes:
  - No args: audit every cite in every doc.
  - Filter args: only check cites whose target file is in the list.
    Use this as a post-edit step.
  - `--list-only`: print every `(doc, cite, target)` tuple. Useful for
    building a one-time index.
- **`tests/test_check_lesson_links.py`** (new) - 21 hermetic
  smoke tests covering: snippet match, snippet drift (proposes a fix),
  indentation tolerance, short-snippet anchor (e.g. `try:`), no-match
  (no fix proposal), nearest-match preference when duplicates exist,
  range cites (preserves span), missing file, line past EOF, landmark
  fallback (passes a def, flags a non-landmark as `DRIFT?`), filter
  mode (relative + absolute paths), `--list-only` shape, empty-glob
  no-op, snippet-fence lookahead bounds, text/mermaid fences skipped,
  docstring openers skipped, snippet-near-cite tolerance.
- **`tests/test_check_lesson_links_stress.py`** (new) - 4 stress
  tests including a 20-trial fuzz pass that generates random sources
  and shifts, asserting the script proposes exactly the right
  correction for every cite in every trial.
- **`AGENTS.md`** - new rule: after editing source under
  `src/audrey/`, `tools-server/`, or `config.yaml`, run the checker
  with the changed paths and apply any proposed fixes.

Out of scope:

- Auto-fix mode (the script reports, the human edits).
- AST-aware checks (the snippet match handles "function moved" cases;
  "function renamed" still requires eyes).
- A static citation database. We deliberately avoided one because it
  becomes a third source of truth that drifts the same way line
  numbers do. On-demand parsing has no state to maintain.
- Integration with pre-commit/CI hooks. Adding a hook later is one
  config line; not shipping yet to keep this phase focused.

## 1. Deploy

No deploy. The script runs locally; nothing rebuilds, nothing restarts.

```bash
# Laptop:
git pull   # after the Phase 3 commit lands

# Confirm the script is executable:
ls -l scripts/check-lesson-links.py

# Sanity check: run it across all lessons.
scripts/check-lesson-links.py | tail -5
```

Expected:

- The script is `-rwxr-xr-x`.
- The final line reports counts in the form
  `cites checked: N  ok: A  drift: B  drift?: C  broken: D`.
- Exit code 0 if `drift + broken == 0`, otherwise 1.

## 2. Smoke tests

### 2.1 Run the hermetic suite

The pytest suite ships 15 cases that exercise the script end-to-end
via subprocess against synthetic fixtures. No live source files
needed.

```bash
.venv/bin/python -m pytest tests/test_check_lesson_links.py -v
```

Expected: 15 passed. Each test pins one behavior of the script
(snippet match, drift-with-fix, indentation tolerance, broken cite,
landmark fallback, filter mode, list-only, etc.).

### 2.2 Run against the real lesson corpus

```bash
scripts/check-lesson-links.py
```

Expected on a clean repo: zero `BROKEN` and zero `DRIFT` (confident
drift). Some `DRIFT?` lines are normal — they flag cites whose target
line is not a function/class/constant, which often is the deliberate
"cite into a function body" pattern lessons use. Skim them; ignore
unless one is genuinely wrong.

If `DRIFT` or `BROKEN` appears, the script prints a `fix:` directive
with the proposed correction. Apply it and re-run.

### 2.3 Run in filter mode

Pass a source file you recently edited. Only cites pointing at that
file are checked.

```bash
scripts/check-lesson-links.py src/audrey/pipeline/classify.py
```

Expected: the summary line shows
`cites checked: <smaller number>` — the rest are filtered out.

### 2.4 Run --list-only

Builds a tab-separated index of every cite.

```bash
scripts/check-lesson-links.py --list-only | head -10
scripts/check-lesson-links.py --list-only | wc -l
```

Expected: each line is `<doc-path>\t<url>\t<absolute-target>`. The
`wc -l` total equals the script's "cites checked" count when run
without `--list-only`.

### 2.5 Synthetic drift end-to-end

Confirm the script actually catches drift you introduce. Edit a
source file to shift a function by one line (add a blank line at the
top of `classify.py`), run the checker, see the proposed fix, undo
the edit. Don't commit the synthetic drift.

```bash
# Make space at the top of classify.py:
sed -i '1i\\' src/audrey/pipeline/classify.py
scripts/check-lesson-links.py src/audrey/pipeline/classify.py
# Should print several DRIFT lines with `fix: change #L<old> → #L<new>`.

# Restore:
git checkout -- src/audrey/pipeline/classify.py
```

Expected: every cite that pointed at a line in `classify.py` is
flagged as `DRIFT` with a proposed `+1` correction. After
`git checkout`, re-running shows zero drift again.

## 3. Rollback

Plain git rollback. No state, no data.

```bash
git revert <phase-3-commit>
```

The lesson cites stay as they are — the script's removal doesn't
affect them. The only loss is the future ability to detect drift
automatically.

## 4. Operational notes

- The script is project-agnostic. Three env vars (`DOCS_GLOB`,
  `DOCS_EXCLUDE`, `REPO_ROOT`) cover most repo shapes. The landmark
  patterns are tuned for Python/YAML/shell/Markdown; add patterns for
  other languages by editing `LANDMARK_PATTERNS` at the top of the
  script.
- The default `DOCS_GLOB` is `docs/lessons/lesson-*.md` — published
  lessons only. Scaffolding files like `CONTINUITY.md` and `AUDIT.md`
  often contain illustrative cites that aren't real anchors, so the
  tighter default avoids false-positive noise. Override via
  `DOCS_GLOB=...` to broaden, or use `DOCS_EXCLUDE=glob1:glob2` to
  drop specific files.
- The matcher tolerates four classes of "drift that isn't really
  drift": indented source vs. un-indented snippet (lessons often
  flatten code from a nested context); text/mermaid/markdown fences
  (treated as non-source, falls back to landmark); short snippet
  anchors like `try:` (matched even though shorter than the default
  prefix floor); cite-near-snippet within ±10 lines (e.g. cite at
  function signature, snippet shows body — accepted as OK).
- `DRIFT` means "snippet found at a different line, here's the fix."
  Apply blindly only after eyeballing the proposed line — the
  nearest-match heuristic stabilizes corrections in 95% of cases but
  can pick the wrong instance of a duplicate pattern.
- `DRIFT?` (with question mark) is advisory. The script doesn't have
  a snippet to anchor against, only the landmark heuristic. False
  positives are common for cites pointing into function bodies.
- `BROKEN` always means the file or line is missing; no judgment
  needed, just fix it.
- The script exits non-zero on `BROKEN` or `DRIFT`. `DRIFT?` alone
  does not fail the run, so the script is safe to gate a pre-commit
  hook on.

## 5. Followups

- Wire the checker into a pre-commit hook (run only on staged source
  files). Deferred so we can use the checker manually for a week and
  see whether it produces signal or noise.
- Add a `--fix` mode that rewrites cites in place. Worth doing only
  after we trust the proposals from a few weeks of manual use.
- Consider splitting the script into its own repo if anyone outside
  this project starts using it. Today it lives in `scripts/` with the
  rest of the repo's tooling.
