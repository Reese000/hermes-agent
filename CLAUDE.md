# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Configuration and system prompt repository for Hermes AI agent. No build process or runtime code — this is a pure configuration/documentation project.

## Key Files

- `SOUL.md` — Agent personality definition
- `system_prompt.md` — Full system prompt for the Hermes agent
- `system_prompt_dump.md` — Exported/dumped version of the system prompt
- `lessons_learned.md` — Accumulated knowledge and corrections

## Running

```bash
run-hermes.bat    # Launch Hermes agent (Windows batch script)
```

## Notes

When modifying agent behavior, edit `SOUL.md` for personality changes and `system_prompt.md` for instruction changes. Keep `lessons_learned.md` updated with new findings.
