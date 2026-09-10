#!/usr/bin/env node
/**
 * build-test-asar.mjs — Build a test app-TEST.asar from current dist/.
 *
 * Workflow:
 *   1. Verify production ASAR exists (read-only — never modified)
 *   2. Verify dist/ is built (checkDistBuilt)
 *   3. Extract production ASAR to temp dir
 *   4. Replace dist/ in extracted tree with current dist/
 *   5. Repack to test-build/app-TEST.asar
 *   6. Verify the new ASAR has correct structure
 *   7. Verify native-deps simple-git fallback path is loadable
 *
 * SAFETY:
 *   - Production ASAR at release/win-unpacked/resources/app.asar is NEVER modified
 *   - Output goes to test-build/ only
 *   - Never copies to production location
 *
 * Usage:
 *   node scripts/build-test-asar.mjs
 */

import { execSync } from 'node:child_process'
import { existsSync, rmSync, cpSync, mkdirSync, statSync, readFileSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { checkDistBuilt } from './assert-dist-built.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const desktopRoot = resolve(__dirname, '..')
const distDir = join(desktopRoot, 'dist')
const prodAsar = join(desktopRoot, 'release', 'win-unpacked', 'resources', 'app.asar')
const testBuildDir = join(desktopRoot, 'test-build')
const extractDir = join(testBuildDir, 'asar-extracted')
const testAsar = join(testBuildDir, 'app-TEST.asar')

function run(cmd, opts = {}) {
  console.log(`  $ ${cmd}`)
  return execSync(cmd, { cwd: desktopRoot, stdio: 'pipe', ...opts }).toString().trim()
}

function asar(cmd) {
  return run(`npx asar ${cmd}`)
}

function die(msg) {
  console.error(`\n✗ ${msg}`)
  process.exit(1)
}

// --- Step 1: Verify production ASAR exists ---
if (!existsSync(prodAsar)) {
  die(`Production ASAR not found at ${prodAsar}`)
}
console.log(`✓ Production ASAR: ${prodAsar} (${(statSync(prodAsar).size / 1e6).toFixed(1)} MB)`)

// --- Step 2: Verify dist/ is built ---
const distCheck = checkDistBuilt(distDir)
if (!distCheck.ok) {
  die(`dist/ not built: ${distCheck.error}`)
}
console.log('✓ dist/ verified (index.html + assets present)')

// --- Step 3: Extract production ASAR ---
console.log('Extracting production ASAR...')
if (existsSync(extractDir)) {
  rmSync(extractDir, { recursive: true, force: true })
}
mkdirSync(extractDir, { recursive: true })
asar(`extract "${prodAsar}" "${extractDir}"`)

// Verify extraction has required structure
const required = ['package.json', 'electron', 'assets', 'public']
for (const entry of required) {
  if (!existsSync(join(extractDir, entry))) {
    die(`Extraction missing required entry: ${entry}`)
  }
}
console.log('✓ Production ASAR extracted with correct structure')

// --- Step 4: Replace dist/ ---
console.log('Replacing dist/ with current build...')
const extractedDist = join(extractDir, 'dist')
if (existsSync(extractedDist)) {
  rmSync(extractedDist, { recursive: true, force: true })
}
cpSync(distDir, extractedDist, { recursive: true })
console.log('✓ dist/ replaced')

// --- Step 5: Repack ---
console.log('Packing test ASAR...')
mkdirSync(testBuildDir, { recursive: true })
asar(`pack "${extractDir}" "${testAsar}"`)

const sizeMB = (statSync(testAsar).size / 1e6).toFixed(1)
console.log(`✓ Test ASAR packed: ${testAsar} (${sizeMB} MB)`)

// --- Step 6: Verify structure ---
console.log('Verifying test ASAR structure...')
const topLevel = asar(`list "${testAsar}"`)
  .split('\n')
  .map(l => l.trim())
  .filter(l => l.match(/^\\[^\\]+$/))

const expectedTop = ['\\assets', '\\dist', '\\electron', '\\package.json', '\\public']
const missing = expectedTop.filter(e => !topLevel.includes(e))
const extra = topLevel.filter(e => !expectedTop.includes(e))

if (missing.length > 0) {
  die(`Test ASAR missing root entries: ${missing.join(', ')}`)
}
if (extra.length > 0) {
  console.log(`  ⚠ Extra root entries (non-fatal): ${extra.join(', ')}`)
}

// Verify package.json has main field (read from extraction dir — extract-file
// mixes npm warnings into stdout on some npm/config combinations)
const extractedPkgPath = join(extractDir, 'package.json')
if (!existsSync(extractedPkgPath)) {
  die('package.json not found in extraction directory')
}
try {
  const pkgRaw = readFileSync(extractedPkgPath, 'utf8')
  const pkg = JSON.parse(pkgRaw)
  if (!pkg.main) {
    die('package.json missing "main" field')
  }
  console.log(`✓ package.json main: "${pkg.main}"`)
} catch {
  die('package.json is not valid JSON')
}

// --- Step 7: Verify native-deps vendor fallback path ---
// git-review-ops.cjs falls back to require()-ing simple-git from
// resources/native-deps/vendor/node_modules/ when the hoisted workspace
// copy is not reachable.  This check is NON-TAUTOLOGICAL: it loads the
// module through the EXACT path the runtime uses, verifying the
// extraResources staging was applied correctly.  A missing
// extraResources entry or a build that skipped stage-native-deps will
// fail here before an isolated test instance ever launches.
console.log('Verifying native-deps simple-git fallback path...')
const nativeDepsSimpleGit = join(
  desktopRoot,
  'release', 'win-unpacked', 'resources',
  'native-deps', 'vendor', 'node_modules', 'simple-git'
)
if (!existsSync(nativeDepsSimpleGit)) {
  die(
    `native-deps simple-git not found at: ${nativeDepsSimpleGit}\n` +
    '  Run: npm run build (which calls stage-native-deps.mjs)\n' +
    '  Ensure package.json extraResources includes build/native-deps -> native-deps'
  )
}
// Non-tautological: actually require the module from the fallback path
// in a subprocess to prove it is loadable, not just a stale directory.
try {
  run(`node -e "require('${nativeDepsSimpleGit.replace(/\\/g, '\\\\')}')"`)
  console.log('✓ native-deps simple-git fallback path is loadable')
} catch {
  die(
    `native-deps simple-git exists but failed to load from: ${nativeDepsSimpleGit}\n` +
    '  The directory may be incomplete — re-run: npm run build'
  )
}

// --- Cleanup ---
console.log('Cleaning up extraction directory...')
rmSync(extractDir, { recursive: true, force: true })

console.log('\n✓ Build complete — test ASAR ready at:')
console.log(`  ${testAsar}`)
