---
name: debug-profile-open-darts
description: Debug and profile open-DARTS with Debug builds, Valgrind, VTune, and built-in timers. Use when diagnosing crashes, leaks, regressions, or runtime hotspots.
---

# Debug and Profile

Use this skill for root-cause analysis and performance investigation.
Use the session conda environment. If the prompt defines one, use it. Otherwise create and activate one session-level conda environment once, then reuse it across the rest of the session.

## Steps

1. Confirm the session conda environment is active; create it once if needed.
2. Build with appropriate instrumentation (Debug or Valgrind mode).
3. Reproduce the issue on a targeted model.
4. Capture and summarize diagnostics (logs, leak reports, hotspot reports).

## Primary commands

- Debug build: `./helper_scripts/build_darts_cmake.sh -c -d Debug -j 8`
- Valgrind build: `./helper_scripts/build_darts_cmake.sh -c -v -j 8`
- Valgrind run set: `python helper_scripts/valgrind_check.py`
- VTune run set: `python helper_scripts/vtune_profiling.py`

## Reference

Read `references/debugging-and-profiling.md` for workflows and troubleshooting.
