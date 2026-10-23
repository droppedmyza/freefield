#!/usr/bin/env node
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdtempSync, rmSync, readdirSync } from "node:fs";
import { tmpdir, constants } from "node:os";
import { join } from "node:path";

const [apiPath, stagesPath, outJson, outPng, url = "http://127.0.0.1:18188", notePath] = process.argv.slice(2);
if (!outPng) { console.error("usage: export_ui_workflow.mjs api.json stages.json out.json out.png [url] [note.md]"); process.exit(2); }
const noteText = notePath ? readFileSync(notePath, "utf8") : "";
const api = JSON.parse(readFileSync(apiPath, "utf8"));
const stages = JSON.parse(readFileSync(stagesPath, "utf8"));

const ids = Object.keys(api);
const num = Object.fromEntries(ids.map((id, i) => [id, i + 1]));
const stageOf = (id) => {
  const exact = stages.findIndex((s) => s.match.includes(id));
  return exact >= 0 ? exact : stages.findIndex((s) => s.match.some((m) => id.startsWith(m)));
};
const numbered = {}, stageByNum = {};
for (const id of ids) {
  const n = structuredClone(api[id]);
  for (const [k, v] of Object.entries(n.inputs)) if (Array.isArray(v) && typeof v[0] === "string" && v[0] in num) n.inputs[k] = [String(num[v[0]]), v[1]];
  numbered[num[id]] = n;
  const s = stageOf(id);
  if (s < 0) { console.error(`no stage for node ${id}`); process.exit(1); }
  stageByNum[num[id]] = s;
}

const chrome = ["chromium", "google-chrome-stable", "google-chrome"].find((c) => { try { return spawn(c, ["--version"]) && true; } catch { return false; } });
const port = 9333;
const profile = mkdtempSync(join(tmpdir(), "ffchrome-"));
const proc = spawn(chrome, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, "--window-size=3800,2200",
  `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore", detached: true });
proc.once("error", () => {});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const signalGroup = (sig) => { try { process.kill(-proc.pid, sig); return true; } catch { return false; } };
const groupAlive = (pgid) => {
  try {
    return readdirSync("/proc").some((d) => {
      try { const st = readFileSync(`/proc/${d}/stat`, "utf8"); const f = st.slice(st.lastIndexOf(")") + 2).split(" "); return f[2] === String(pgid) && f[0] !== "Z"; } catch { return false; }
    });
  } catch { try { process.kill(-pgid, 0); return true; } catch { return false; } }
};
let ws, stopping;
const shutdown = () => stopping ??= (async () => {
  ws?.close();
  if (proc.pid) {
    for (const sig of ["SIGTERM", "SIGKILL"]) {
      signalGroup(sig);
      for (let i = 0; i < 50 && groupAlive(proc.pid); i++) await sleep(100);
    }
    if (groupAlive(proc.pid)) throw new Error(`chrome process group ${proc.pid} survived SIGKILL; leaving ${profile} in place`);
  }
  rmSync(profile, { recursive: true, force: true, maxRetries: 3 });
})();
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, () => shutdown().finally(() => process.exit(128 + constants.signals[sig])));
process.once("exit", () => {
  if (proc.pid && groupAlive(proc.pid)) { signalGroup("SIGKILL"); const until = Date.now() + 2000; while (groupAlive(proc.pid) && Date.now() < until) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 20); }
  if (proc.pid && groupAlive(proc.pid)) { console.error(`chrome process group ${proc.pid} is still running; leaving ${profile} in place`); return; }
  rmSync(profile, { recursive: true, force: true });
});
let seq = 0; const pending = new Map();
const cdp = (method, params = {}) => new Promise((r) => { const id = ++seq; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expr) => {
  const r = await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 800));
  return r.result?.result?.value;
};

try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page"); } catch {}
  }
  ws = new WebSocket(target.webSocketDebuggerUrl);
  ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  await new Promise((r) => ws.addEventListener("open", r));
  await cdp("Page.enable");
  await cdp("Page.navigate", { url });
  for (let i = 0; i < 120; i++) { await sleep(500); if (await evaluate("!!(window.app && window.app.graph && window.app.canvas)")) break; }
  await sleep(3000);
  const result = await evaluate(`(async () => {
    const api = ${JSON.stringify(numbered)}, stageByNum = ${JSON.stringify(stageByNum)}, stages = ${JSON.stringify(stages)}, noteText = ${JSON.stringify(noteText)};
    await app.loadApiJson(api, "freefield");
    await new Promise((r) => setTimeout(r, 1500));
    const cols = stages.map(() => []);
    for (const n of app.graph._nodes) if (n.id in stageByNum) cols[stageByNum[n.id]].push(n);
    const groups = []; let x = 0;
    const PAD = 40, TITLE = 50, GAP = 30;
    if (noteText) {
      const note = LiteGraph.createNode("MarkdownNote") || LiteGraph.createNode("Note");
      note.title = "About this workflow";
      note.widgets[0].value = noteText;
      note.size = [560, Math.max(520, 22 * noteText.split("\\n").length)];
      note.pos = [PAD, TITLE + PAD];
      app.graph.add(note);
      x += note.size[0] + 2 * PAD + 60;
    }
    cols.forEach((nodes, s) => {
      if (!nodes.length) return;
      nodes.sort((a, b) => a.id - b.id);
      const w = Math.max(...nodes.map((n) => n.size[0]));
      let y = TITLE + PAD;
      for (const n of nodes) { n.pos = [x + PAD, y]; y += n.size[1] + LiteGraph.NODE_TITLE_HEIGHT + GAP; }
      groups.push({ title: stages[s].title, bounding: [x, 0, w + 2 * PAD, y + PAD], color: stages[s].color || "#3f789e", font_size: 28, flags: {} });
      x += w + 2 * PAD + 60;
    });
    // Seeds stay as authored (the preset's), not re-randomized on every queue.
    for (const n of app.graph._nodes) for (const w of n.widgets || []) if (w.name === "control_after_generate") w.value = "fixed";
    const wf = app.graph.serialize();
    wf.groups = groups;
    await app.loadGraphData(wf);
    await new Promise((r) => setTimeout(r, 1500));
    // Hide first-run dialogs, toasts and side panels so the screenshot shows only the graph.
    const st = document.createElement("style");
    st.textContent = ".p-dialog-mask,.p-dialog,.p-toast,.p-message,[role=alert],[class*=toast],[class*=Toast],.side-tool-bar-container,.comfyui-menu,.p-panel,.comfy-menu,.graph-canvas-menu{display:none!important}";
    document.head.appendChild(st);
    // Fit everything in view for the screenshot.
    const W = x, H = Math.max(...groups.map((g) => g.bounding[3]));
    const ds = app.canvas.ds; const cw = app.canvas.canvas.width, ch = app.canvas.canvas.height;
    ds.scale = Math.min(cw / W, ch / H) * 0.98; ds.offset = [10 / ds.scale, 10 / ds.scale];
    app.canvas.setDirty(true, true); app.canvas.draw(true, true);
    return JSON.stringify(app.graph.serialize());
  })()`);
  const wf = JSON.parse(result);
  writeFileSync(outJson, JSON.stringify(wf, null, 1) + "\n");
  await sleep(1500);
  await evaluate(`document.querySelectorAll('[role=dialog],[role=alert],.p-dialog-mask,.p-toast,.p-message').forEach((e) => e.remove()); app.canvas.draw(true, true); true`);
  await sleep(500);
  const shot = await cdp("Page.captureScreenshot", { format: "png" });
  writeFileSync(outPng, Buffer.from(shot.result.data, "base64"));
  const zoomed = await evaluate(`(() => {
    const gs = app.graph._groups || app.graph.groups || [];
    const pick = gs.filter((g) => /^(3|4)\\./.test(g.title));
    if (!pick.length) return false;
    const x0 = Math.min(...pick.map((g) => g._bounding ? g._bounding[0] : g.bounding[0]));
    const y0 = Math.min(...pick.map((g) => g._bounding ? g._bounding[1] : g.bounding[1]));
    const ds = app.canvas.ds; ds.scale = 1.0; ds.offset = [-x0 + 20, -y0 + 20];
    app.canvas.setDirty(true, true); app.canvas.draw(true, true); return true;
  })()`);
  if (zoomed) {
    await sleep(800);
    await evaluate(`document.querySelectorAll('[role=dialog],[role=alert],.p-dialog-mask,.p-toast,.p-message,.p-component-overlay').forEach((e) => e.remove()); app.canvas.draw(true, true); true`);
    await sleep(300);
    const z = await cdp("Page.captureScreenshot", { format: "png" });
    writeFileSync(outPng.replace(/\.png$/, "-stages-3-4.png"), Buffer.from(z.result.data, "base64"));
  }
  console.log(`wrote ${outJson} (${wf.nodes.length} nodes, ${wf.links.length} links, ${wf.groups.length} groups) and ${outPng}`);
} finally {
  await shutdown();
}
