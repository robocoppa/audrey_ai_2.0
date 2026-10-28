# Campaign 3 Phase 25 — Research fact checking

**Closed 2026-10-27.** Historical contract; see the [campaign index](README.md).

The opt-in research pipeline sequences research, verification, bounded
tool-assisted fact checking, then writing. Actionable corrections override the
research findings in the writer input. Missing configuration, unavailable tools
or stage failure degrade to the earlier verify/write path.

The progress sequence includes fact checking only when that stage runs. It is
advisory and fallible; hermetic plumbing checks do not establish answer quality.
The [claim ledger](phase-26-research-claim-ledger.md) adds structured handling.
