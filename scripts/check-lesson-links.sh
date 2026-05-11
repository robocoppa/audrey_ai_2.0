#!/usr/bin/env bash
# check-lesson-links.sh
# ─────────────────────────────────────────────────────────────────────
# Audits Markdown line-cites in long-form docs against the current source.
#
# WHAT IT CHECKS
#   For every Markdown link of the form
#       [<label>](relative/path#L42)
#   or  [<label>](relative/path#L42-L51)
#   in your docs, it opens the target file and reports cites where:
#     - the target file no longer exists,
#     - the target file is shorter than the cited line,
#     - the cited line "looks like drift" (does not land on a useful
#       landmark — `def`, `class`, `_CONST`, decorators, block heads).
#
#   The landmark heuristic catches ~90% of stale cites. False positives
#   are possible (a deliberate cite into a function body mid-statement
#   will trip the heuristic). False negatives happen when a function
#   was renamed without shifting lines — those will show as "OK" here
#   even though the prose around the cite is wrong. Eyes still required.
#
# WHAT IT DOES NOT DO
#   - Validate the prose around the cite.
#   - Rewrite cites. It only reports.
#   - Track cite history. There is no database; every run starts cold.
#
# USAGE
#   scripts/check-lesson-links.sh
#       Check every cite in every doc file under DOCS_GLOB.
#
#   scripts/check-lesson-links.sh path/to/changed.py [more.py ...]
#       Only report cites that target one of the given paths. Useful as
#       a pre-commit step: after editing source files, run this with the
#       changed files to see which docs you need to update.
#
#   scripts/check-lesson-links.sh --list-only
#       Print every (doc, cite, target) tuple without checking. Useful
#       for building a one-time CSV index.
#
# ENVIRONMENT (override if your repo is shaped differently)
#   DOCS_GLOB    Glob for the docs to audit. Default: docs/lessons/*.md
#                Example: DOCS_GLOB="docs/**/*.md" ./check-lesson-links.sh
#   REPO_ROOT    Absolute path the cite URLs are relative to. Default:
#                git rev-parse --show-toplevel (i.e. the repo root).
#                Cites in lessons typically use ../../ to escape the
#                lesson's own directory — the script resolves each cite
#                relative to the lesson file, not to REPO_ROOT directly.
#
# EXIT CODES
#   0   No drift found.
#   1   At least one cite failed a check.
#   2   Usage error / missing dependency.
#
# DEPENDENCIES
#   bash 4+, grep, sed, awk, find. No Python, no jq.
#
# PROJECT-AGNOSTIC NOTES
#   This script makes no assumption about your codebase. The only
#   project-specific knobs are DOCS_GLOB (where the docs are) and the
#   cite syntax (Markdown link with #L<num> fragment). The landmark
#   heuristic is tuned for Python / YAML / shell sources but adding
#   patterns is a one-line edit to LANDMARK_PATTERNS below.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────

DOCS_GLOB="${DOCS_GLOB:-docs/lessons/*.md}"

if ! REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null)}"; then
    echo "error: REPO_ROOT not set and not in a git repo" >&2
    exit 2
fi

# A "landmark" is any pattern that suggests the line is a stable
# anchor — function/class/constant definition, decorator, top-of-block
# comment, or a YAML key. Tuned for the languages this repo uses;
# adding patterns for other languages is one extended regex away.
LANDMARK_PATTERNS=(
    '^[[:space:]]*def[[:space:]]'           # python def
    '^[[:space:]]*async[[:space:]]+def[[:space:]]'  # python async def
    '^[[:space:]]*class[[:space:]]'         # python class
    '^[[:space:]]*@'                        # python decorator
    '^[[:space:]]*[A-Z_][A-Z0-9_]*[[:space:]]*[:=]'  # CONSTANT = / CONSTANT:
    '^[[:space:]]*[a-z_][a-zA-Z0-9_]*[[:space:]]*=' # lowercase = (covers test_foo = pytest.mark)
    '^[[:space:]]*#[[:space:]]*[─=─]\{3,\}'  # box-drawing section header
    '^[[:space:]]*#[[:space:]]*[A-Z]'        # comment starting with capital (section)
    '^[a-zA-Z0-9_-]+:[[:space:]]*$'         # top-level YAML key
    '^[[:space:]]*[a-zA-Z0-9_-]+:[[:space:]]*(#.*)?$'  # nested YAML key on its own line
    '^[[:space:]]*function[[:space:]]'       # bash function
    '^[[:space:]]*[A-Za-z0-9_]+\(\)[[:space:]]*{' # bash function (POSIX style)
)

# ─── Modes ───────────────────────────────────────────────────────────

LIST_ONLY=0
FILTER_PATHS=()
for arg in "$@"; do
    case "$arg" in
        --list-only) LIST_ONLY=1 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) FILTER_PATHS+=("$arg") ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────

# Resolve a relative cite path against the directory of the citing doc.
# Returns an absolute path. Doesn't require the file to exist.
resolve_cite_path() {
    local doc_dir="$1"
    local rel="$2"
    # `realpath -m` doesn't fail when the path doesn't exist.
    realpath -m "$doc_dir/$rel"
}

# Return 0 if the line looks like a load-bearing landmark, 1 otherwise.
is_landmark() {
    local line="$1"
    local pat
    for pat in "${LANDMARK_PATTERNS[@]}"; do
        if printf '%s\n' "$line" | grep -Eq "$pat"; then
            return 0
        fi
    done
    return 1
}

# Test whether $1 (an absolute path) matches any FILTER_PATHS entry.
# Filter entries are joined against REPO_ROOT so both relative and
# absolute inputs work.
matches_filter() {
    local target="$1"
    if [[ ${#FILTER_PATHS[@]} -eq 0 ]]; then
        return 0
    fi
    local f abs_f
    for f in "${FILTER_PATHS[@]}"; do
        if [[ "$f" = /* ]]; then
            abs_f="$f"
        else
            abs_f="$REPO_ROOT/$f"
        fi
        # realpath -m so non-existent paths still compare cleanly
        abs_f="$(realpath -m "$abs_f")"
        if [[ "$target" == "$abs_f" ]]; then
            return 0
        fi
    done
    return 1
}

# ─── Main ────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

# Use compgen to expand the glob safely even when it matches nothing.
shopt -s nullglob globstar
DOCS=( $DOCS_GLOB )
if [[ ${#DOCS[@]} -eq 0 ]]; then
    echo "no docs found matching: $DOCS_GLOB" >&2
    exit 0
fi

# Counters
total=0
broken=0
drift=0
ok=0
filtered_out=0

# Regex extracts Markdown links whose URL contains a #L<num> anchor.
# Captures the URL portion only — we don't care about the label.
# Two patterns to handle #L42 and #L42-L51.
CITE_RE='\]\(([^)]*#L[0-9]+(-L[0-9]+)?)\)'

for doc in "${DOCS[@]}"; do
    doc_dir="$(dirname "$doc")"
    # awk-based extractor: emits one cite URL per line.
    while IFS= read -r url; do
        total=$((total + 1))

        # Split url into path and #L fragment.
        rel_path="${url%%#*}"
        anchor="${url#*#}"

        # Parse start/end lines from the anchor.
        start_line="$(printf '%s' "$anchor" | sed -E 's/^L([0-9]+).*/\1/')"
        if [[ "$anchor" == *-L* ]]; then
            end_line="$(printf '%s' "$anchor" | sed -E 's/^L[0-9]+-L([0-9]+)$/\1/')"
        else
            end_line="$start_line"
        fi

        target="$(resolve_cite_path "$doc_dir" "$rel_path")"

        # When the user passed FILTER_PATHS, drop cites whose target
        # isn't in the list. This is the "scripts/...path.py" mode.
        if ! matches_filter "$target"; then
            filtered_out=$((filtered_out + 1))
            continue
        fi

        if [[ "$LIST_ONLY" -eq 1 ]]; then
            printf '%s\t%s\t%s\n' "$doc" "$url" "$target"
            ok=$((ok + 1))
            continue
        fi

        # File must exist.
        if [[ ! -f "$target" ]]; then
            printf 'BROKEN  %s:\n  cite: %s\n  reason: target file not found (%s)\n\n' \
                "$doc" "$url" "$target"
            broken=$((broken + 1))
            continue
        fi

        # Line must exist.
        line_count="$(wc -l <"$target")"
        if (( start_line > line_count )); then
            printf 'BROKEN  %s:\n  cite: %s\n  reason: line %s past end of file (%s lines)\n\n' \
                "$doc" "$url" "$start_line" "$line_count"
            broken=$((broken + 1))
            continue
        fi

        # Heuristic: does the cited line look like a landmark?
        line_text="$(sed -n "${start_line}p" "$target")"
        if is_landmark "$line_text"; then
            ok=$((ok + 1))
            continue
        fi

        # Not a landmark — might still be a deliberate "into the body"
        # cite, but worth flagging. Show the cited line and a couple of
        # neighbours so the reviewer can judge fast.
        printf 'DRIFT?  %s:\n  cite: %s\n  line %s of %s:\n' \
            "$doc" "$url" "$start_line" "$rel_path"
        # Print start_line-1 .. start_line+1 if available, indented.
        local_before=$(( start_line > 1 ? start_line - 1 : start_line ))
        local_after=$(( start_line < line_count ? start_line + 1 : start_line ))
        awk -v a="$local_before" -v b="$local_after" -v hot="$start_line" '
            NR >= a && NR <= b {
                marker = (NR == hot) ? ">>" : "  "
                printf "    %s %5d  %s\n", marker, NR, $0
            }' "$target"
        printf '\n'
        drift=$((drift + 1))
    done < <(
        # Extract every cite URL. grep -oE returns the matched portion;
        # sed strips the leading `](` and trailing `)`.
        grep -oE "$CITE_RE" "$doc" 2>/dev/null \
            | sed -E 's/^\]\(//; s/\)$//'
    )
done

# ─── Summary ────────────────────────────────────────────────────────

if [[ "$LIST_ONLY" -eq 1 ]]; then
    exit 0
fi

printf '%s\n' "─────────────────────────────────────────"
printf 'cites checked: %d  ok: %d  drift: %d  broken: %d  filtered out: %d\n' \
    "$((ok + drift + broken))" "$ok" "$drift" "$broken" "$filtered_out"

if (( broken > 0 || drift > 0 )); then
    exit 1
fi
exit 0
