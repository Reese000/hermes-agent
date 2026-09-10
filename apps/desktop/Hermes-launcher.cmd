@echo off
rem ─────────────────────────────────────────────────────────────────────────
rem Hermes Desktop launcher — protects the packaged app from environment
rem footguns that break or sabotage a normal double-click launch.
rem
rem Copy this file NEXT TO Hermes.exe in the unpacked release directory
rem (apps/desktop/release/win-unpacked/) and launch via it.
rem
rem It does three things BEFORE the binary starts:
rem
rem  1. Clears ELECTRON_RUN_AS_NODE. If that variable leaks into the
rem     environment (Kilo Code agent sessions, CI runners, dev shells),
rem     Electron's NATIVE launcher switches Hermes.exe into plain-Node mode
rem     before any of main.cjs runs — the window never appears and the app
rem     exits silently or prints Node help. Clearing it inside main.cjs is
rem     too late; it must be gone before the process starts (verified
rem     empirically against Electron 40: the JS delete runs, but the process
rem     is already node-mode).
rem
rem  2. Sets HERMES_DESKTOP_NO_AUTO_UPDATE=1 (DEVELOPER override). Stops the
rem     app from updating/reinstalling itself: a backend crash would
rem     otherwise hand off to Hermes-Setup.exe --update and overwrite custom
rem     code in the AppData install. This is an INTERNAL bridge only — the
rem     user-facing switch is Settings → About → Automatic updates. Omit this
rem     line if you are shipping the launcher to end users.
rem
rem  3. Clears a stale .hermes-update-in-progress marker. If the previous
rem     update was interrupted, the Tauri updater refuses to launch with the
rem     "UPDATE DIDN'T FINISH" dialog even though no Hermes process is
rem     running.
rem ─────────────────────────────────────────────────────────────────────────
setlocal
set "ELECTRON_RUN_AS_NODE="
set "HERMES_DESKTOP_NO_AUTO_UPDATE=1"
if exist "%LOCALAPPDATA%\hermes\.hermes-update-in-progress" del /q "%LOCALAPPDATA%\hermes\.hermes-update-in-progress"
start "" "%~dp0Hermes.exe" %*
