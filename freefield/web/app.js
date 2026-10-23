const $ = (id) => document.getElementById(id);
const FAMILY_EMOJI = { museum: "🖼️", toy: "🧸", spotlight: "🎤", built: "🛠️", other: "✨" };
let presets = [];
let current = null;
let pollTimer = null;
let file = null;

async function health() {
  try {
    const r = await (await fetch("/api/health")).json();
    const ok = r.upstream && r.upstream.status === "healthy";
    $("health").textContent = ok ? "Comfy API connected" : "Comfy API down: run scripts/start_comfy.sh";
    $("health").className = "health " + (ok ? "ok" : "down");
  } catch { $("health").textContent = "server offline"; $("health").className = "health down"; }
}

function card(p) {
  const el = document.createElement("div");
  el.className = "card"; el.tabIndex = 0; el.dataset.family = p.family;
  const media = p.preview
    ? (p.preview.endsWith(".mp4")
        ? `<video src="${p.preview}"${p.poster ? ` poster="${p.poster}"` : ""} muted loop playsinline autoplay preload="metadata"></video>`
        : `<img src="${p.preview}" alt="">`)
    : `<div class="ph">${FAMILY_EMOJI[p.family] || "✨"}</div>`;
  el.innerHTML = `${media}<div class="label"><b></b><span></span></div>`;
  el.querySelector("b").textContent = p.name;
  el.querySelector("span").textContent = p.description;
  el.addEventListener("click", () => open(p));
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") open(p); });
  return el;
}

function render(filter = "all") {
  $("grid").replaceChildren(...presets.filter((p) => filter === "all" || p.family === filter).map(card));
}

function filters() {
  const fams = ["all", ...new Set(presets.map((p) => p.family))];
  $("filters").replaceChildren(...fams.map((f, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === 0 ? " on" : ""); b.textContent = f === "all" ? "All" : f;
    b.onclick = () => { document.querySelectorAll(".chip").forEach((c) => c.classList.remove("on")); b.classList.add("on"); render(f); };
    return b;
  }));
}

function setMedia(html) { $("sheet-media").innerHTML = html; }

function open(p) {
  current = p;
  $("sheet-title").textContent = p.name;
  $("sheet-desc").textContent = p.description;
  setMedia(p.preview ? (p.preview.endsWith(".mp4") ? `<video src="${p.preview}" muted loop playsinline autoplay></video>` : `<img src="${p.preview}" alt="">`) : `<div class="ph" style="font-size:120px">${FAMILY_EMOJI[p.family] || "✨"}</div>`);
  ["progress", "result", "error"].forEach((id) => ($(id).hidden = true));
  $("texts").replaceChildren(...Object.entries(p.inputs || {}).filter(([, i]) => i.type === "text").map(([k, i]) => {
    const l = document.createElement("label");
    l.className = "text-input";
    l.textContent = i.label || k;
    const inp = document.createElement("input");
    inp.type = "text"; inp.name = k; inp.maxLength = i.max_length || 80; inp.dataset.text = k;
    l.appendChild(inp);
    return l;
  }));
  const vin = Object.values(p.inputs || {}).find((i) => i.type === "video");
  const video = !!vin;
  $("photo").accept = video && vin.photo_ok ? "video/*,image/*" : video ? "video/*" : "image/*";
  $("drop-text").textContent = (vin && vin.drop) || (video ? "Drop a 6-10 s selfie clip here, or tap to choose one" : "Drop a selfie here, or tap to choose one");
  $("fine").textContent = (vin && vin.fine) || (video
    ? "Prop the phone about 2 m away, sit still for a second, then lean in toward it smiling. Your own clip, or one from someone who said yes. Nothing leaves this machine."
    : "Use a photo of yourself or someone who said yes. Nothing leaves this machine.");
  file = null; $("go").disabled = true;
  $("photo-preview").hidden = true; $("clip-preview").hidden = true; $("drop-text").hidden = false;
  $("form").hidden = false;
  $("sheet").hidden = false;
}

function close() { $("sheet").hidden = true; clearInterval(pollTimer); }

function pickFile(f) {
  const kinds = $("photo").accept.split(",").map((a) => a.replace("*", ""));
  if (!f || !kinds.some((k) => f.type.startsWith(k))) return;
  file = f;
  const url = URL.createObjectURL(f);
  const v = f.type.startsWith("video/");
  $("photo-preview").hidden = v; $("clip-preview").hidden = !v;
  if (v) $("clip-preview").src = url; else $("photo-preview").src = url;
  $("drop-text").hidden = true;
  $("go").disabled = false;
}

async function submit(e, seedOverride) {
  if (e) e.preventDefault();
  if (!file || !current) return;
  const fd = new FormData();
  fd.append("preset", current.id);
  fd.append("photo", file);
  document.querySelectorAll("#texts input[data-text]").forEach((i) => { if (i.value.trim()) fd.append(i.dataset.text, i.value.trim()); });
  const seed = seedOverride ?? $("seed").value;
  if (seed !== "" && seed !== undefined) fd.append("seed", seed);
  $("error").hidden = true; $("result").hidden = true; $("form").hidden = true; $("progress").hidden = false;
  $("bar-fill").style.width = "2%"; $("stage").textContent = "Queued…";
  const r = await fetch("/api/runs", { method: "POST", body: fd });
  const body = await r.json();
  if (!r.ok) return fail(body.error || "could not start the run");
  poll(body.id);
}

function fail(msg) {
  $("progress").hidden = true; $("form").hidden = false;
  $("error").textContent = msg; $("error").hidden = false;
}

function poll(id) {
  const t0 = Date.now();
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let run;
    try { run = await (await fetch(`/api/runs/${id}`)).json(); } catch { return; }
    const secs = Math.round((Date.now() - t0) / 1000);
    $("elapsed").textContent = `${secs}s elapsed · seed ${run.seed}`;
    if (run.status === "queued") { $("stage").textContent = run.queue_position > 1 ? `In line: #${run.queue_position}` : "Starting…"; return; }
    if (run.status === "running") {
      const stepTxt = run.steps ? ` (step ${run.step}/${run.steps})` : "";
      $("stage").textContent = (run.stage || "Working") + stepTxt + "…";
      $("bar-fill").style.width = Math.max(4, Math.round(run.progress * 100)) + "%";
      return;
    }
    clearInterval(pollTimer);
    if (run.status !== "succeeded") return fail(run.error || "the run failed");
    const clip = (run.files.clip_save || [])[0];
    const still = (run.files.keyframe_save || [])[0];
    setMedia(clip ? `<video src="${clip}" controls autoplay loop playsinline></video>` : `<img src="${still}" alt="">`);
    $("progress").hidden = true; $("result").hidden = false;
    $("download").href = clip || still;
    $("result-meta").textContent = `Done in ${run.seconds}s on your GPU · seed ${run.seed}`;
    $("result-warning").hidden = !(run.warnings && run.warnings.length);
    $("result-warning").textContent = (run.warnings || []).join(" ");
    $("share").onclick = () => share(clip || still);
    $("again").onclick = () => submit(null, String(Math.floor(Math.random() * 2147483647)));
  }, 1000);
}

async function share(url) {
  try {
    const blob = await (await fetch(url)).blob();
    const f = new File([blob], url.split("/").pop(), { type: blob.type });
    if (navigator.canShare && navigator.canShare({ files: [f] })) {
      await navigator.share({ files: [f], title: "Made with FreeField", text: "#OpenSourceHiggsfield" });
      return;
    }
  } catch {  }
  $("download").click();
}

function wireDrop() {
  const d = $("drop");
  $("photo").addEventListener("change", (e) => pickFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => d.addEventListener(t, (e) => { e.preventDefault(); d.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => d.addEventListener(t, (e) => { e.preventDefault(); d.classList.remove("over"); }));
  d.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
}

async function main() {
  wireDrop();
  $("form").addEventListener("submit", submit);
  $("close").onclick = close;
  $("sheet").addEventListener("click", (e) => { if (e.target === $("sheet")) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  health(); setInterval(health, 15000);
  const data = await (await fetch("/api/presets")).json();
  presets = data.presets;
  filters(); render();
  const m = location.hash.match(/preset=([a-z0-9-]+)/);
  const linked = m && presets.find((p) => p.id === m[1]);
  if (linked) open(linked);
}
main();
