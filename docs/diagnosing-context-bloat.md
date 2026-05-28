# Diagnosing context bloat — `headroom xray`

`headroom xray` is a multi-CLI diagnostic command that answers
*"where did my tokens go?"* across 25+ coding agents (Claude Code, Codex,
Gemini CLI, Cursor, Cline, Forge, Goose, Antigravity, Warp, …).

It is **NOT** the same as the `/xray` Claude Code skill (which X-rays the
current conversation). `headroom xray` is a CLI subcommand that scans your
historical session transcripts on disk.

## What it shows

Under the hood, `headroom xray` wraps [**CodeBurn**](https://github.com/getagentseal/codeburn)
(MIT, the leading multi-agent usage analyzer) and adds a Headroom-specific
footer:

```
─────────────────────────────────────────────────────────────
Headroom: top compression opportunities in this session
  ▸ Bash outputs       53k tokens (24%)  — likely CCR-compressible
  ▸ Read tool results  28k tokens (13%)  — ContentRouter target
  ▸ MCP tool schemas    8k tokens  (4%)  — re-injected every turn
  → Run `headroom xray replay` (coming soon) for exact savings.
─────────────────────────────────────────────────────────────
```

The footer ranks the top-3 tool types by token usage and labels each with
the Headroom compressor that would typically handle it.

## Prerequisites

- **Node 20+** on `PATH` (CodeBurn runs via `npx`). The first invocation
  downloads `codeburn` into the npx cache; subsequent runs are fast.
- A session transcript on disk (e.g., `~/.claude/projects/<slug>/*.jsonl`).

## Common workflows

```bash
# 30-day usage report
headroom xray report

# Today's activity
headroom xray today

# Find waste patterns with paste-ready fixes (CodeBurn's `optimize`)
headroom xray optimize

# Compare two sessions
headroom xray compare <session-a> <session-b>

# CodeBurn's own help (forwards through the wrapper)
headroom xray --help-codeburn

# Suppress the Headroom footer
headroom xray --no-footer report
# Or via env var
HEADROOM_XRAY_NO_FOOTER=1 headroom xray report
```

## Installation

`headroom xray` is shipped as a separate Rust binary (`headroom-xray`)
alongside the Python wheel. Phase 1.0 supports source-installed users:

```bash
# Build the binary
cargo build --release -p headroom-xray

# Or install to ~/.headroom/bin/ (auto-discovered)
cargo install --path crates/headroom-xray --root ~/.headroom
```

Pre-built binaries (cross-platform downloads à la RTK) arrive in Phase 1.1.

## Phase 1 caveats

- The Headroom footer parses **Claude Code** session JSONLs only. CodeBurn
  itself handles all 25+ agents for the dashboard; full-coverage footer
  arrives in Phase 2.
- Footer hints are **labels**, not predictions. Phase 2 (`headroom xray
  replay`) actually runs Headroom's compressors over the transcript and
  reports counterfactual savings.

## Acknowledgments

`headroom xray` is powered by [CodeBurn](https://github.com/getagentseal/codeburn)
(MIT, © 2026 AgentSeal). Huge thanks to the CodeBurn maintainers; their
tool is the dashboard layer Headroom builds on.

## See also

- `/xray` Claude Code skill — X-rays your *live* conversation (different
  command, different surface)
- `/context` — built-in Claude Code single-session snapshot
- [RTK](https://github.com/rtk-ai/rtk) — the existing bundled CLI-output
  rewriter that Headroom uses for runtime compression
