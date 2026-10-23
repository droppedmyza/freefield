import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir, constants } from "node:os";
import { join, basename } from "node:path";

const [dir, url = "http://127.0.0.1:8188"] = process.argv.slice(2);
const chrome = ["chromium", "google-chrome-stable", "google-chrome"].find((c) => { try { return spawn(c, ["--version"]) && true; } catch { return false; } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const files = readdirSync(dir).filter((f) => f.endsWith(".json") && !f.endsWith(".api.json") && f !== "export.json" && f !== "load-proof.json").sort();
const results = [];
let port = 9400;
let live = null;
const signalGroup = (c, sig) => { try { process.kill(-c.proc.pid, sig); return true; } catch { return false; } };
const groupAlive = (pgid) => {
  try {
    return readdirSync("/proc").some((d) => {
      try { const st = readFileSync(`/proc/${d}/stat`, "utf8"); const f = st.slice(st.lastIndexOf(")") + 2).split(" "); return f[2] === String(pgid) && f[0] !== "Z"; } catch { return false; }
    });
  } catch { try { process.kill(-pgid, 0); return true; } catch { return false; } }
};
const stopChrome = (c) => c.stopping ??= (async () => {
  if (c.proc.pid) {
    for (const sig of ["SIGTERM", "SIGKILL"]) {
      signalGroup(c, sig);
      for (let i = 0; i < 50 && groupAlive(c.proc.pid); i++) await sleep(100);
    }
    if (groupAlive(c.proc.pid)) throw new Error(`chrome process group ${c.proc.pid} survived SIGKILL; leaving ${c.profile} in place`);
  }
  rmSync(c.profile, { recursive: true, force: true, maxRetries: 3 });
})();
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, () => (live ? stopChrome(live) : Promise.resolve()).finally(() => process.exit(128 + constants.signals[sig])));
process.once("exit", () => {
  if (!live) return;
  if (live.proc.pid && groupAlive(live.proc.pid)) { signalGroup(live, "SIGKILL"); const until = Date.now() + 2000; while (groupAlive(live.proc.pid) && Date.now() < until) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 20); }
  if (live.proc.pid && groupAlive(live.proc.pid)) { console.error(`chrome process group ${live.proc.pid} is still running; leaving ${live.profile} in place`); return; }
  rmSync(live.profile, { recursive: true, force: true });
});
for (const f of files) {
  port += 1;
  const profile = mkdtempSync(join(tmpdir(), "wf-"));
  const proc = spawn(chrome, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--window-size=1600,1000", "about:blank"], { stdio: "ignore", detached: true });
  proc.once("error", () => {});
  live = { proc, profile, stopping: null };
  try {
    let tabs;
    for (let i = 0; i < 50; i++) { try { tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); } }
    const ws = new WebSocket(tabs.find((t) => t.type === "page").webSocketDebuggerUrl);
    await new Promise((r) => ws.addEventListener("open", r));
    let id = 0; const pending = new Map();
    ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
    const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
    const ev = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.result.exceptionDetails) return { error: JSON.stringify(r.result.exceptionDetails).slice(0, 400) }; return r.result.result.value; };
    await send("Page.enable");
    await send("Page.navigate", { url });
    for (let i = 0; i < 80; i++) { if (await ev("!!(window.app && window.app.graph && window.LiteGraph)")) break; await sleep(500); }
    await sleep(1500);
    const wf = readFileSync(join(dir, f), "utf8");
    const res = await ev(`(async () => {
      const wf = ${wf};
      await window.app.loadGraphData(wf, true, false);
      const g = window.app.graph;
      const reg = window.LiteGraph.registered_node_types;
      const nodes = g._nodes || g.nodes;
      const missing = [...new Set(wf.nodes.map((n) => n.type).filter((t) => !reg[t]))];
      const absentModels = [];
      for (const n of nodes) for (const w of (n.widgets || [])) {
        const opts = w.options && Array.isArray(w.options.values) ? w.options.values : null;
        if (opts && typeof w.value === "string" && /\\.(safetensors|gguf|onnx|pt|pth|ckpt)$/.test(w.value) && !opts.includes(w.value)) absentModels.push(w.value);
      }
      const p = await window.app.graphToPrompt();
      const notes = wf.nodes.filter((n) => n.type === "MarkdownNote" || n.type === "Note").length;
      return { file_nodes: wf.nodes.length, note_nodes: notes, loaded_nodes: nodes.length, links: (wf.links || []).length,
               missing_node_types: missing, prompt_nodes: Object.keys(p.output).length,
               converts_back: Object.keys(p.output).length === wf.nodes.length - notes,
               models_not_on_this_machine: [...new Set(absentModels)] };
    })()`);
    const shot = await send("Page.captureScreenshot", { format: "jpeg", quality: 60 });
    writeFileSync(join(dir, "proof", basename(f, ".json") + ".jpg"), Buffer.from(shot.result.data, "base64"));
    results.push({ workflow: f, ...res });
    console.log(JSON.stringify(results.at(-1)));
    ws.close();
  } finally {
    await stopChrome(live);
    live = null;
  }
  await sleep(300);
}
writeFileSync(join(dir, "load-proof.json"), JSON.stringify({ comfyui: url, checked: new Date().toISOString(), results }, null, 2) + "\n");
process.exit(0);
