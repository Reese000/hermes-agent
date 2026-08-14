"use strict";

// electron/preload.ts
var import_electron = require("electron");
import_electron.contextBridge.exposeInMainWorld("hermesDesktop", {
  getConnection: (profile) => import_electron.ipcRenderer.invoke("hermes:connection", profile),
  revalidateConnection: () => import_electron.ipcRenderer.invoke("hermes:connection:revalidate"),
  touchBackend: (profile) => import_electron.ipcRenderer.invoke("hermes:backend:touch", profile),
  getGatewayWsUrl: (profile) => import_electron.ipcRenderer.invoke("hermes:gateway:ws-url", profile),
  openSessionWindow: (sessionId, opts) => import_electron.ipcRenderer.invoke("hermes:window:openSession", sessionId, opts),
  openWindow: () => import_electron.ipcRenderer.invoke("hermes:window:openInstance"),
  claimAmbientCue: (key) => import_electron.ipcRenderer.invoke("hermes:ambient:claim", key),
  wakeIndicator: {
    getState: () => import_electron.ipcRenderer.invoke("hermes:wake-indicator:get"),
    setState: (state) => import_electron.ipcRenderer.send("hermes:wake-indicator:set", state),
    onState: (callback) => {
      const listener = (_event, state) => callback(state);
      import_electron.ipcRenderer.on("hermes:wake-indicator:state", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:wake-indicator:state", listener);
    }
  },
  petOverlay: {
    // Main renderer → main process: window lifecycle + drag. `request` is
    // `{ bounds, screen }`; resolves with the screen bounds it actually used.
    open: (request) => import_electron.ipcRenderer.invoke("hermes:pet-overlay:open", request),
    close: () => import_electron.ipcRenderer.invoke("hermes:pet-overlay:close"),
    setBounds: (bounds) => import_electron.ipcRenderer.send("hermes:pet-overlay:set-bounds", bounds),
    setIgnoreMouse: (ignore) => import_electron.ipcRenderer.send("hermes:pet-overlay:ignore-mouse", ignore),
    // Flip the overlay focusable (and focus it) while the composer needs keys.
    setFocusable: (focusable) => import_electron.ipcRenderer.send("hermes:pet-overlay:set-focusable", focusable),
    // Main renderer → overlay (forwarded by main): push the latest pet state.
    pushState: (payload) => import_electron.ipcRenderer.send("hermes:pet-overlay:state", payload),
    // Overlay → main renderer (forwarded by main): pop back in / composer submit.
    control: (payload) => import_electron.ipcRenderer.send("hermes:pet-overlay:control", payload),
    // Overlay subscribes to state pushes.
    onState: (callback) => {
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on("hermes:pet-overlay:state", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:pet-overlay:state", listener);
    },
    // Main renderer subscribes to overlay control messages.
    onControl: (callback) => {
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on("hermes:pet-overlay:control", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:pet-overlay:control", listener);
    }
  },
  // HUD mode: the chrome-free floating chat. A full app renderer (own gateway)
  // sized as a floating bar, so it mounts the real composer. Main owns the
  // window; `onChanged` keeps every window's toggle truthful.
  hud: {
    open: (request) => import_electron.ipcRenderer.invoke("hermes:hud:open", request),
    close: () => import_electron.ipcRenderer.invoke("hermes:hud:close"),
    setIgnoreMouse: (ignore) => import_electron.ipcRenderer.send("hermes:hud:ignore-mouse", ignore),
    moveBy: (delta) => import_electron.ipcRenderer.send("hermes:hud:move-by", delta),
    setBounds: (bounds) => import_electron.ipcRenderer.send("hermes:hud:set-bounds", bounds),
    setVibrancy: (on) => import_electron.ipcRenderer.invoke("hermes:hud:vibrancy", on),
    // The HUD tells main which session it is on; main hands that back to the
    // app window when the HUD closes, so the app can re-home onto it.
    setSession: (sessionId) => import_electron.ipcRenderer.send("hermes:hud:session", sessionId),
    onGoto: (callback) => {
      const listener = (_event, sessionId) => callback(sessionId);
      import_electron.ipcRenderer.on("hermes:hud:goto", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:hud:goto", listener);
    },
    onChanged: (callback) => {
      const listener = (_event, state) => callback(state);
      import_electron.ipcRenderer.on("hermes:hud:changed", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:hud:changed", listener);
    },
    // Linux only, and silent elsewhere: where the cursor is, in page
    // coordinates, or null when it has left the window. Stands in for the
    // mousemove that `setIgnoreMouseEvents(true, { forward: true })` delivers on
    // macOS and Windows but not here.
    onCursor: (callback) => {
      const listener = (_event, point) => callback(point);
      import_electron.ipcRenderer.on("hermes:hud:cursor", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:hud:cursor", listener);
    }
  },
  // Quick Entry: the global-hotkey mini composer window. Main owns the OS
  // shortcut + the persisted preference; the quick window only captures text
  // and hands it back, and the primary renderer submits it through the normal
  // prompt path.
  quickEntry: {
    getSettings: () => import_electron.ipcRenderer.invoke("hermes:quick-entry:settings:get"),
    setSettings: (patch) => import_electron.ipcRenderer.invoke("hermes:quick-entry:settings:set", patch),
    submit: (payload) => import_electron.ipcRenderer.send("hermes:quick-entry:submit", payload),
    dismiss: () => import_electron.ipcRenderer.send("hermes:quick-entry:dismiss"),
    // Primary renderer → main → quick window: gateway connection state + the
    // recent-session options the target picker offers. Main caches the latest
    // payload so a freshly spawned quick window starts from truth.
    pushState: (payload) => import_electron.ipcRenderer.send("hermes:quick-entry:state", payload),
    // Quick window subscribes to those pushes.
    onState: (callback) => {
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on("hermes:quick-entry:state", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:quick-entry:state", listener);
    },
    // Main → primary renderer: a submit captured by the quick window.
    onSubmit: (callback) => {
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on("hermes:quick-entry:submit", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:quick-entry:submit", listener);
    },
    // Main → quick window: you were just summoned (reset draft + refocus).
    onShown: (callback) => {
      const listener = () => callback();
      import_electron.ipcRenderer.on("hermes:quick-entry:shown", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:quick-entry:shown", listener);
    }
  },
  getBootProgress: () => import_electron.ipcRenderer.invoke("hermes:boot-progress:get"),
  getConnectionConfig: (profile) => import_electron.ipcRenderer.invoke("hermes:connection-config:get", profile),
  saveConnectionConfig: (payload) => import_electron.ipcRenderer.invoke("hermes:connection-config:save", payload),
  applyConnectionConfig: (payload) => import_electron.ipcRenderer.invoke("hermes:connection-config:apply", payload),
  testConnectionConfig: (payload) => import_electron.ipcRenderer.invoke("hermes:connection-config:test", payload),
  sshConfigHosts: () => import_electron.ipcRenderer.invoke("hermes:ssh-config:hosts"),
  sshResolveHost: (host) => import_electron.ipcRenderer.invoke("hermes:ssh-config:resolve", host),
  probeConnectionConfig: (remoteUrl) => import_electron.ipcRenderer.invoke("hermes:connection-config:probe", remoteUrl),
  oauthLoginConnectionConfig: (remoteUrl) => import_electron.ipcRenderer.invoke("hermes:connection-config:oauth-login", remoteUrl),
  oauthLogoutConnectionConfig: (remoteUrl) => import_electron.ipcRenderer.invoke("hermes:connection-config:oauth-logout", remoteUrl),
  // Hermes Cloud: one portal login powers discovery + silent per-agent sign-in
  // (cloud-auto-discovery Phase 3).
  cloud: {
    status: () => import_electron.ipcRenderer.invoke("hermes:cloud:status"),
    login: () => import_electron.ipcRenderer.invoke("hermes:cloud:login"),
    logout: () => import_electron.ipcRenderer.invoke("hermes:cloud:logout"),
    discover: (org) => import_electron.ipcRenderer.invoke("hermes:cloud:discover", org),
    agentSignIn: (dashboardUrl) => import_electron.ipcRenderer.invoke("hermes:cloud:agent-sign-in", dashboardUrl)
  },
  profile: {
    get: () => import_electron.ipcRenderer.invoke("hermes:profile:get"),
    set: (name) => import_electron.ipcRenderer.invoke("hermes:profile:set", name)
  },
  api: (request) => import_electron.ipcRenderer.invoke("hermes:api", request),
  // Streaming API — sends chunks via event listeners instead of buffering.
  // Returns { onChunk, onDone, dispose } for the caller to consume events.
  apiStream: (request, callbacks) => {
    const { onChunk, onDone } = callbacks;
    const chunkHandler = (_event, payload) => onChunk(payload);
    const doneHandler = (_event, payload) => {
      cleanup();
      onDone(payload);
    };
    const cleanup = () => {
      import_electron.ipcRenderer.removeListener("hermes:api-stream:chunk", chunkHandler);
      import_electron.ipcRenderer.removeListener("hermes:api-stream:done", doneHandler);
    };
    import_electron.ipcRenderer.on("hermes:api-stream:chunk", chunkHandler);
    import_electron.ipcRenderer.on("hermes:api-stream:done", doneHandler);
    import_electron.ipcRenderer.send("hermes:api-stream", request);
    return { dispose: cleanup };
  },
  notify: (payload) => import_electron.ipcRenderer.invoke("hermes:notify", payload),
  requestMicrophoneAccess: () => import_electron.ipcRenderer.invoke("hermes:requestMicrophoneAccess"),
  readWindowBelow: () => import_electron.ipcRenderer.invoke("hermes:window:readBelow"),
  readFileDataUrl: (filePath) => import_electron.ipcRenderer.invoke("hermes:readFileDataUrl", filePath),
  readFileDataUrlForAttach: (filePath) => import_electron.ipcRenderer.invoke("hermes:readFileDataUrlForAttach", filePath),
  dataUrlReadMax: {
    get: () => import_electron.ipcRenderer.invoke("hermes:data-url-read-max:get"),
    set: (maxMb) => import_electron.ipcRenderer.invoke("hermes:data-url-read-max:set", maxMb)
  },
  readFileText: (filePath) => import_electron.ipcRenderer.invoke("hermes:readFileText", filePath),
  selectPaths: (options) => import_electron.ipcRenderer.invoke("hermes:selectPaths", options),
  selectSavePath: (options) => import_electron.ipcRenderer.invoke("hermes:selectSavePath", options),
  writeClipboard: (text) => import_electron.ipcRenderer.invoke("hermes:writeClipboard", text),
  readClipboard: () => import_electron.ipcRenderer.invoke("hermes:readClipboard"),
  saveImageFromUrl: (url) => import_electron.ipcRenderer.invoke("hermes:saveImageFromUrl", url),
  saveImageBuffer: (data, ext) => import_electron.ipcRenderer.invoke("hermes:saveImageBuffer", { data, ext }),
  saveClipboardImage: () => import_electron.ipcRenderer.invoke("hermes:saveClipboardImage"),
  getPathForFile: (file) => {
    try {
      return import_electron.webUtils.getPathForFile(file) || "";
    } catch {
      return "";
    }
  },
  normalizePreviewTarget: (target, baseDir) => import_electron.ipcRenderer.invoke("hermes:normalizePreviewTarget", target, baseDir),
  watchPreviewFile: (url) => import_electron.ipcRenderer.invoke("hermes:watchPreviewFile", url),
  watchDirectory: (dir) => import_electron.ipcRenderer.invoke("hermes:watchDirectory", dir),
  stopPreviewFileWatch: (id) => import_electron.ipcRenderer.invoke("hermes:stopPreviewFileWatch", id),
  setActiveWork: (payload) => import_electron.ipcRenderer.send("hermes:active-work", payload),
  setTitleBarTheme: (payload) => import_electron.ipcRenderer.send("hermes:titlebar-theme", payload),
  setNativeTheme: (mode) => import_electron.ipcRenderer.send("hermes:native-theme", mode),
  setTranslucency: (payload) => import_electron.ipcRenderer.send("hermes:translucency", payload),
  setKeepAwake: (on) => import_electron.ipcRenderer.send("hermes:keep-awake", on),
  setPreviewShortcutActive: (active) => import_electron.ipcRenderer.send("hermes:previewShortcutActive", Boolean(active)),
  openExternal: (url) => import_electron.ipcRenderer.invoke("hermes:openExternal", url),
  openPreviewInBrowser: (url) => import_electron.ipcRenderer.invoke("hermes:openPreviewInBrowser", url),
  fetchLinkTitle: (url) => import_electron.ipcRenderer.invoke("hermes:fetchLinkTitle", url),
  sanitizeWorkspaceCwd: (cwd) => import_electron.ipcRenderer.invoke("hermes:workspace:sanitize", cwd),
  settings: {
    getDefaultProjectDir: () => import_electron.ipcRenderer.invoke("hermes:setting:defaultProjectDir:get"),
    setDefaultProjectDir: (dir) => import_electron.ipcRenderer.invoke("hermes:setting:defaultProjectDir:set", dir),
    pickDefaultProjectDir: () => import_electron.ipcRenderer.invoke("hermes:setting:defaultProjectDir:pick")
  },
  zoom: {
    // Current zoom of this window, as { level, percent }.
    get: () => import_electron.ipcRenderer.invoke("hermes:zoom:get"),
    setPercent: (percent) => import_electron.ipcRenderer.send("hermes:zoom:set-percent", percent),
    // Fires on every zoom change, including the Ctrl/Cmd +/-/0 shortcuts,
    // so the settings UI can stay in sync with the keyboard.
    onChanged: (callback) => {
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on("hermes:zoom:changed", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:zoom:changed", listener);
    }
  },
  revealLogs: () => import_electron.ipcRenderer.invoke("hermes:logs:reveal"),
  getRecentLogs: () => import_electron.ipcRenderer.invoke("hermes:logs:recent"),
  // Fire-and-forget: persists a renderer error-boundary catch (with component
  // stack) to desktop.log so crashes survive the window (#79428).
  reportRendererError: (report) => import_electron.ipcRenderer.send("hermes:logs:renderer-error", report),
  readDir: (dirPath) => import_electron.ipcRenderer.invoke("hermes:fs:readDir", dirPath),
  gitRoot: (startPath) => import_electron.ipcRenderer.invoke("hermes:fs:gitRoot", startPath),
  revealPath: (targetPath) => import_electron.ipcRenderer.invoke("hermes:fs:reveal", targetPath),
  openDir: (dirPath) => import_electron.ipcRenderer.invoke("hermes:fs:openDir", dirPath),
  desktopPluginsRoot: () => import_electron.ipcRenderer.invoke("hermes:fs:desktopPluginsRoot"),
  agentPluginsRoot: () => import_electron.ipcRenderer.invoke("hermes:fs:agentPluginsRoot"),
  renamePath: (targetPath, newName) => import_electron.ipcRenderer.invoke("hermes:fs:rename", targetPath, newName),
  writeTextFile: (filePath, content) => import_electron.ipcRenderer.invoke("hermes:fs:writeText", filePath, content),
  trashPath: (targetPath) => import_electron.ipcRenderer.invoke("hermes:fs:trash", targetPath),
  git: {
    worktreeList: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:worktreeList", repoPath),
    worktreeAdd: (repoPath, options) => import_electron.ipcRenderer.invoke("hermes:git:worktreeAdd", repoPath, options),
    worktreeRemove: (repoPath, worktreePath, options) => import_electron.ipcRenderer.invoke("hermes:git:worktreeRemove", repoPath, worktreePath, options),
    branchSwitch: (repoPath, branch) => import_electron.ipcRenderer.invoke("hermes:git:branchSwitch", repoPath, branch),
    branchList: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:branchList", repoPath),
    baseBranchList: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:baseBranchList", repoPath),
    repoStatus: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:repoStatus", repoPath),
    fileDiff: (repoPath, filePath) => import_electron.ipcRenderer.invoke("hermes:git:fileDiff", repoPath, filePath),
    scanRepos: (roots, options) => import_electron.ipcRenderer.invoke("hermes:git:scanRepos", roots, options),
    review: {
      list: (repoPath, scope, baseRef) => import_electron.ipcRenderer.invoke("hermes:git:review:list", repoPath, scope, baseRef),
      diff: (repoPath, filePath, scope, baseRef, staged) => import_electron.ipcRenderer.invoke("hermes:git:review:diff", repoPath, filePath, scope, baseRef, staged),
      stage: (repoPath, filePath) => import_electron.ipcRenderer.invoke("hermes:git:review:stage", repoPath, filePath),
      unstage: (repoPath, filePath) => import_electron.ipcRenderer.invoke("hermes:git:review:unstage", repoPath, filePath),
      revert: (repoPath, filePath) => import_electron.ipcRenderer.invoke("hermes:git:review:revert", repoPath, filePath),
      revParse: (repoPath, ref) => import_electron.ipcRenderer.invoke("hermes:git:review:revParse", repoPath, ref),
      commit: (repoPath, message, push) => import_electron.ipcRenderer.invoke("hermes:git:review:commit", repoPath, message, push),
      commitContext: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:review:commitContext", repoPath),
      push: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:review:push", repoPath),
      shipInfo: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:review:shipInfo", repoPath),
      prList: (repoPath, branches, numbers) => import_electron.ipcRenderer.invoke("hermes:git:review:prList", repoPath, branches, numbers),
      fetchPrComment: (repoPath, url) => import_electron.ipcRenderer.invoke("hermes:git:review:fetchPrComment", repoPath, url),
      createPr: (repoPath) => import_electron.ipcRenderer.invoke("hermes:git:review:createPr", repoPath)
    }
  },
  terminal: {
    cwd: (id) => import_electron.ipcRenderer.invoke("hermes:terminal:cwd", id),
    dispose: (id) => import_electron.ipcRenderer.invoke("hermes:terminal:dispose", id),
    resize: (id, size) => import_electron.ipcRenderer.invoke("hermes:terminal:resize", id, size),
    start: (options) => import_electron.ipcRenderer.invoke("hermes:terminal:start", options),
    write: (id, data) => import_electron.ipcRenderer.invoke("hermes:terminal:write", id, data),
    onData: (id, callback) => {
      const channel = `hermes:terminal:${id}:data`;
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on(channel, listener);
      return () => import_electron.ipcRenderer.removeListener(channel, listener);
    },
    onExit: (id, callback) => {
      const channel = `hermes:terminal:${id}:exit`;
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on(channel, listener);
      return () => import_electron.ipcRenderer.removeListener(channel, listener);
    }
  },
  onClosePreviewRequested: (callback) => {
    const listener = () => callback();
    import_electron.ipcRenderer.on("hermes:close-preview-requested", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:close-preview-requested", listener);
  },
  onOpenFolderRequested: (callback) => {
    const listener = () => callback();
    import_electron.ipcRenderer.on("hermes:open-folder-requested", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:open-folder-requested", listener);
  },
  onOpenUpdatesRequested: (callback) => {
    const listener = () => callback();
    import_electron.ipcRenderer.on("hermes:open-updates", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:open-updates", listener);
  },
  onDeepLink: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:deep-link", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:deep-link", listener);
  },
  signalDeepLinkReady: () => import_electron.ipcRenderer.invoke("hermes:deep-link-ready"),
  onWindowStateChanged: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:window-state-changed", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:window-state-changed", listener);
  },
  onFocusSession: (callback) => {
    const listener = (_event, sessionId) => callback(sessionId);
    import_electron.ipcRenderer.on("hermes:focus-session", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:focus-session", listener);
  },
  onNotificationAction: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:notification-action", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:notification-action", listener);
  },
  onPreviewFileChanged: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:preview-file-changed", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:preview-file-changed", listener);
  },
  onBackendExit: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:backend-exit", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:backend-exit", listener);
  },
  // Soft gateway-mode apply finished tearing down the primary backend. Renderer
  // should wipe session lists + re-dial without a window reload.
  onConnectionApplied: (callback) => {
    const listener = () => callback();
    import_electron.ipcRenderer.on("hermes:connection:applied", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:connection:applied", listener);
  },
  onPowerResume: (callback) => {
    const listener = () => callback();
    import_electron.ipcRenderer.on("hermes:power-resume", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:power-resume", listener);
  },
  // AC ↔ battery transitions; renderers slow their backstop polls on battery.
  getOnBattery: () => import_electron.ipcRenderer.invoke("hermes:power-battery:get"),
  onBatteryChanged: (callback) => {
    const listener = (_event, onBattery) => callback(Boolean(onBattery));
    import_electron.ipcRenderer.on("hermes:power-battery", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:power-battery", listener);
  },
  onBootProgress: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:boot-progress", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:boot-progress", listener);
  },
  // First-launch bootstrap progress -- emitted by the install.ps1 stage
  // runner in main.ts (apps/desktop/electron/bootstrap-runner.ts).
  // Renderer's install overlay subscribes to live events and queries the
  // current snapshot via getBootstrapState() to recover after a devtools
  // reload mid-bootstrap.
  getBootstrapState: () => import_electron.ipcRenderer.invoke("hermes:bootstrap:get"),
  continueBootstrapLocal: () => import_electron.ipcRenderer.invoke("hermes:bootstrap:continue-local"),
  resetBootstrap: () => import_electron.ipcRenderer.invoke("hermes:bootstrap:reset"),
  repairBootstrap: () => import_electron.ipcRenderer.invoke("hermes:bootstrap:repair"),
  cancelBootstrap: () => import_electron.ipcRenderer.invoke("hermes:bootstrap:cancel"),
  onBootstrapEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    import_electron.ipcRenderer.on("hermes:bootstrap:event", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:bootstrap:event", listener);
  },
  getVersion: () => import_electron.ipcRenderer.invoke("hermes:version"),
  getRemoteDisplayReason: () => import_electron.ipcRenderer.invoke("hermes:get-remote-display-reason"),
  uninstall: {
    summary: () => import_electron.ipcRenderer.invoke("hermes:uninstall:summary"),
    run: (mode) => import_electron.ipcRenderer.invoke("hermes:uninstall:run", { mode })
  },
  updates: {
    check: () => import_electron.ipcRenderer.invoke("hermes:updates:check"),
    apply: (opts) => import_electron.ipcRenderer.invoke("hermes:updates:apply", opts),
    getBranch: () => import_electron.ipcRenderer.invoke("hermes:updates:branch:get"),
    setBranch: (name) => import_electron.ipcRenderer.invoke("hermes:updates:branch:set", name),
    onProgress: (callback) => {
      const listener = (_event, payload) => callback(payload);
      import_electron.ipcRenderer.on("hermes:updates:progress", listener);
      return () => import_electron.ipcRenderer.removeListener("hermes:updates:progress", listener);
    }
  },
  themes: {
    fetchMarketplace: (id) => import_electron.ipcRenderer.invoke("hermes:vscode-theme:fetch", id),
    searchMarketplace: (query) => import_electron.ipcRenderer.invoke("hermes:vscode-theme:search", query)
  },
  // Find-in-page (Ctrl/Cmd+F): delegates to Electron's
  // webContents.findInPage on the IPC sender's window so a Cmd+F pressed
  // in a secondary session window searches THAT window, not the primary.
  // `onFoundInPage` returns the unsubscribe fn; the renderer wires it via
  // `initFindInPageListener` in store/find-in-page.ts and tears it down
  // when the FindBar unmounts.
  findInPage: (query, options) => import_electron.ipcRenderer.invoke("hermes:find-in-page", query, options),
  stopFindInPage: () => import_electron.ipcRenderer.invoke("hermes:stop-find-in-page"),
  onFoundInPage: (callback) => {
    const listener = (_event, result) => callback(result);
    import_electron.ipcRenderer.on("hermes:found-in-page", listener);
    return () => import_electron.ipcRenderer.removeListener("hermes:found-in-page", listener);
  }
});
