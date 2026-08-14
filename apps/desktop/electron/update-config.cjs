/**
 * Desktop self-update configuration — branch pin + auto-update toggle.
 *
 * Lives in `<userData>/updates.json`. Two writers:
 *   - the branch picker (IPC hermes:updates:branch:set)
 *   - the "Automatic updates" switch in Settings → About
 *     (IPC hermes:updates:auto-update:set)
 *
 * The auto-update flag is the user-facing switch. `HERMES_DESKTOP_NO_AUTO_UPDATE=1`
 * is an INTERNAL override bridge for developer launchers (see
 * apps/desktop/Hermes-launcher.cmd) — it is not a supported user-facing setting.
 *
 * Run with: node --test electron/update-config.test.cjs
 */

const fs = require('node:fs')
const path = require('node:path')

const DEFAULT_UPDATE_BRANCH = 'main'

// Atomic file write: temp + rename (atomic on all platforms). Prevents
// partial writes on crash/power loss that corrupt JSON config files.
function writeFileAtomic(targetPath, data, encoding) {
  const tmp = targetPath + '.tmp'
  fs.writeFileSync(tmp, data, encoding)
  fs.renameSync(tmp, targetPath)
}

/**
 * Read the desktop update config. Absent/corrupt files fall back to the
 * defaults: `main` branch, auto-update enabled (the app's long-standing
 * behavior). `autoUpdate` is strictly boolean — an explicit `false` in the
 * file is the only way to disable; anything else keeps the default so a
 * hand-edited or legacy file can never silently turn updates off.
 */
function readUpdateConfigFile(configPath) {
  try {
    const parsed = JSON.parse(fs.readFileSync(configPath, 'utf8'))
    const branch = typeof parsed?.branch === 'string' ? parsed.branch.trim() : ''
    const autoUpdate = parsed?.autoUpdate === false ? false : true
    return { branch: branch || DEFAULT_UPDATE_BRANCH, autoUpdate }
  } catch {
    return { branch: DEFAULT_UPDATE_BRANCH, autoUpdate: true }
  }
}

function writeUpdateConfigFile(configPath, config) {
  fs.mkdirSync(path.dirname(configPath), { recursive: true })
  writeFileAtomic(configPath, JSON.stringify(config, null, 2))
}

/**
 * Effective "auto-update disabled" state. True when either:
 *   1. the developer launcher override HERMES_DESKTOP_NO_AUTO_UPDATE=1 is set
 *      (internal bridge only — see file header), or
 *   2. the user disabled updates in Settings → About (config `autoUpdate: false`).
 * Only the exact value '1' counts for the env override: the variable's
 * *presence* is not the signal (unlike ELECTRON_RUN_AS_NODE), so a stray
 * empty/`0` value must not disable updates by accident.
 */
function isAutoUpdateDisabled(configPath, env = process.env) {
  if (env.HERMES_DESKTOP_NO_AUTO_UPDATE === '1') return true
  return readUpdateConfigFile(configPath).autoUpdate === false
}

module.exports = {
  DEFAULT_UPDATE_BRANCH,
  readUpdateConfigFile,
  writeUpdateConfigFile,
  isAutoUpdateDisabled
}
