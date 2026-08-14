/**
 * Tests for electron/update-config.cjs — the desktop update config
 * (branch pin + auto-update toggle) that Settings → About reads and writes.
 *
 * Run with: node --test electron/update-config.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the auto-update guards in main.cjs (applyUpdates,
 * handOffWindowsBootstrapRecovery, ensureRuntime) all key off this module.
 * The user-facing switch is the `autoUpdate` key in updates.json; the
 * HERMES_DESKTOP_NO_AUTO_UPDATE env var is an INTERNAL developer-launcher
 * bridge. Regressions here silently re-enable (or permanently lock out)
 * in-app updates.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')

const {
  DEFAULT_UPDATE_BRANCH,
  readUpdateConfigFile,
  writeUpdateConfigFile,
  isAutoUpdateDisabled
} = require('./update-config.cjs')

function tmpConfig(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-update-config-${tag}-`))
  return path.join(dir, 'updates.json')
}

// ─── readUpdateConfigFile ─────────────────────────────────────────────────

test('absent config file => defaults (main branch, auto-update ON)', () => {
  const cfg = readUpdateConfigFile(tmpConfig('absent'))
  assert.equal(cfg.branch, DEFAULT_UPDATE_BRANCH)
  assert.equal(cfg.autoUpdate, true)
})

test('corrupt JSON => defaults, never throws', () => {
  const p = tmpConfig('corrupt')
  fs.writeFileSync(p, '{not json!!')
  const cfg = readUpdateConfigFile(p)
  assert.equal(cfg.branch, DEFAULT_UPDATE_BRANCH)
  assert.equal(cfg.autoUpdate, true)
})

test('branch is trimmed and falls back to default when empty', () => {
  const p = tmpConfig('branch')
  writeUpdateConfigFile(p, { branch: '  bb/gui  ', autoUpdate: true })
  assert.equal(readUpdateConfigFile(p).branch, 'bb/gui')

  writeUpdateConfigFile(p, { branch: '   ' })
  assert.equal(readUpdateConfigFile(p).branch, DEFAULT_UPDATE_BRANCH)
})

test('autoUpdate is strict-boolean: only explicit false disables', () => {
  const p = tmpConfig('auto')
  // explicit false → off
  writeUpdateConfigFile(p, { branch: 'main', autoUpdate: false })
  assert.equal(readUpdateConfigFile(p).autoUpdate, false)
  // missing key → on (default)
  writeUpdateConfigFile(p, { branch: 'main' })
  assert.equal(readUpdateConfigFile(p).autoUpdate, true)
  // truthy values → on
  for (const v of [true, 'false', '0', 0, 1]) {
    writeUpdateConfigFile(p, { branch: 'main', autoUpdate: v })
    assert.equal(readUpdateConfigFile(p).autoUpdate, true, `autoUpdate=${JSON.stringify(v)} must stay on`)
  }
})

test('write → read roundtrip preserves both fields', () => {
  const p = tmpConfig('roundtrip')
  writeUpdateConfigFile(p, { branch: 'dev/experiment', autoUpdate: false })
  const cfg = readUpdateConfigFile(p)
  assert.equal(cfg.branch, 'dev/experiment')
  assert.equal(cfg.autoUpdate, false)
  assert.ok(!fs.existsSync(p + '.tmp'), 'atomic write must not leave a .tmp behind')
})

test('branch merge preserves autoUpdate (branch:set must never re-enable updates)', () => {
  const p = tmpConfig('merge')
  writeUpdateConfigFile(p, { branch: 'main', autoUpdate: false })
  // Mirrors hermes:updates:branch:set's read-modify-write:
  const merged = { ...readUpdateConfigFile(p), branch: 'release/2.x' }
  writeUpdateConfigFile(p, merged)
  const cfg = readUpdateConfigFile(p)
  assert.equal(cfg.branch, 'release/2.x')
  assert.equal(cfg.autoUpdate, false, 'changing branch must not flip auto-update back on')
})

// ─── isAutoUpdateDisabled ─────────────────────────────────────────────────

test('config autoUpdate:false disables; default config does not', () => {
  const off = tmpConfig('off')
  writeUpdateConfigFile(off, { branch: 'main', autoUpdate: false })
  assert.equal(isAutoUpdateDisabled(off, {}), true)

  const on = tmpConfig('on')
  assert.equal(isAutoUpdateDisabled(on, {}), false)
})

test('env override is internal-only: exact value 1 disables', () => {
  const on = tmpConfig('env')
  writeUpdateConfigFile(on, { branch: 'main', autoUpdate: true })
  assert.equal(isAutoUpdateDisabled(on, { HERMES_DESKTOP_NO_AUTO_UPDATE: '1' }), true)
  // Presence alone is NOT the signal (unlike ELECTRON_RUN_AS_NODE) — empty,
  // "0", or unset must not disable updates by accident.
  assert.equal(isAutoUpdateDisabled(on, { HERMES_DESKTOP_NO_AUTO_UPDATE: '' }), false)
  assert.equal(isAutoUpdateDisabled(on, { HERMES_DESKTOP_NO_AUTO_UPDATE: '0' }), false)
  assert.equal(isAutoUpdateDisabled(on, {}), false)
})

test('env override wins over config', () => {
  const off = tmpConfig('envwin')
  writeUpdateConfigFile(off, { branch: 'main', autoUpdate: false })
  // Still disabled with the override; and the override alone disables even a
  // config that says on (covered above).
  assert.equal(isAutoUpdateDisabled(off, { HERMES_DESKTOP_NO_AUTO_UPDATE: '1' }), true)
})
