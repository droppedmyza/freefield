const B = { id: null, photo: null, timer: null };

function bShow(id, on) { $(id).hidden = !on; }

function bOpen() {
  ["b-recipe", "b-errors", "b-render", "b-progress", "b-approve", "b-done"].forEach((id) => bShow(id, false));
  $("b-media").innerHTML = '<div class="ph">🛠️</div>';
  $("builder").hidden = false;
  $("b-desc").focus();
}

function bClose() { $("builder").hidden = true; clearInterval(B.timer); }

function bRecipe(info) {
  const r = info.recipe;
  const f = info.files || {};
  const link = (name, label) => (f[name] ? `<a href="${f[name]}" target="_blank" rel="noopener">${label}</a>` : "");
  $("b-recipe").innerHTML = `<div><b>You become:</b> <span class="v-becomes"></span></div>
    <div><b>Then:</b> <span class="v-motion"></span></div>
    <div class="stages">${r.stages.map((s) => `<span>${s}</span>`).join("")}</div>
    <div class="muted small">Checked against ${info.schemas === "live" ? "your ComfyUI's" : "the bundled"} node list ·
      ${[link("graph.json", "graph (API)"), link("workflow.json", "workflow (open in ComfyUI)"), link("graph.png", "picture of the graph"), link("preset.json", "preset.json")].filter(Boolean).join(" · ")}</div>`;
  $("b-recipe").querySelector(".v-becomes").textContent = r.becomes;
  $("b-recipe").querySelector(".v-motion").textContent = r.motion;
  bShow("b-recipe", true);
  if (f["graph.png"]) $("b-media").innerHTML = `<img src="${f["graph.png"]}" alt="the generated ComfyUI graph">`;
}

async function bDraft(e) {
  e.preventDefault();
  const fd = new FormData();
  fd.append("description", $("b-desc").value);
  $("b-draft").disabled = true; $("b-draft").textContent = "Drafting…";
  try {
    const r = await fetch("/api/build", { method: "POST", body: fd });
    const info = await r.json();
    if (r.status === 400) { $("b-errors").textContent = info.error; bShow("b-errors", true); return; }
    B.id = info.id;
    bRecipe(info);
    if (info.errors && info.errors.length) {
      $("b-errors").textContent = "Rejected, nothing will render: " + info.errors.join(" · ");
      bShow("b-errors", true); bShow("b-render", false);
    } else { bShow("b-errors", false); bShow("b-render", true); }
  } finally { $("b-draft").disabled = false; $("b-draft").textContent = "Draft it"; }
}

function bPick(f) {
  if (!f || !f.type.startsWith("image/")) return;
  B.photo = f;
  $("b-drop-text").textContent = f.name;
  $("b-preview").disabled = false;
}

async function bPreview() {
  const fd = new FormData();
  fd.append("photo", B.photo);
  bShow("b-render", false); bShow("b-progress", true);
  $("b-stage").textContent = "Queued…"; $("b-bar").style.width = "3%";
  const r = await fetch(`/api/build/${B.id}/preview`, { method: "POST", body: fd });
  const body = await r.json();
  if (!r.ok) { $("b-errors").textContent = body.error; bShow("b-errors", true); bShow("b-progress", false); return; }
  const t0 = Date.now();
  clearInterval(B.timer);
  B.timer = setInterval(async () => {
    const run = await (await fetch(`/api/runs/${body.id}`)).json();
    $("b-stage").textContent = `${run.stage || "Waiting for the GPU"}… ${Math.round((Date.now() - t0) / 1000)}s`;
    $("b-bar").style.width = Math.min(95, 3 + (Date.now() - t0) / 1500) + "%";
    if (run.status !== "succeeded" && run.status !== "failed") return;
    clearInterval(B.timer); bShow("b-progress", false);
    const info = await (await fetch(`/api/build/${B.id}`)).json();
    const f = info.files || {};
    if (f["preview.mp4"]) $("b-media").innerHTML = `<video src="${f["preview.mp4"]}?t=${Date.now()}" controls autoplay loop playsinline></video>`;
    const c = info.preview || {};
    $("b-checks").textContent = `Rendered: ${c.render_ok ? "yes" : "no"} · face still matches your photo: ${c.face_match ?? "not found"}` +
      (c.passed ? " · checks passed" : " · checks failed, this stays a draft");
    bShow("b-approve", true);
    $("b-publish").disabled = !c.passed;
  }, 1500);
}

async function bPublish() {
  const fd = new FormData();
  fd.append("approve", "yes");
  const r = await fetch(`/api/build/${B.id}/publish`, { method: "POST", body: fd });
  const body = await r.json();
  if (!r.ok) { $("b-errors").textContent = body.error; bShow("b-errors", true); return; }
  bShow("b-approve", false);
  $("b-done").textContent = `Published as "${body.preset}". It is in the gallery now: try it with another photo.`;
  bShow("b-done", true);
  presets = (await (await fetch("/api/presets")).json()).presets;
  filters(); render();
}

$("make").onclick = bOpen;
$("b-close").onclick = bClose;
$("builder").addEventListener("click", (e) => { if (e.target === $("builder")) bClose(); });
$("b-form").addEventListener("submit", bDraft);
$("b-photo").addEventListener("change", (e) => bPick(e.target.files[0]));
$("b-preview").onclick = bPreview;
$("b-publish").onclick = bPublish;
async function bLoad(id) {
  bOpen();
  const r = await fetch(`/api/build/${id}`);
  if (!r.ok) return;
  const info = await r.json();
  B.id = id;
  $("b-desc").value = info.recipe.description || "";
  bRecipe(info);
  if (info.errors && info.errors.length) {
    $("b-errors").textContent = "Rejected, nothing will render: " + info.errors.join(" · ");
    bShow("b-errors", true);
    return;
  }
  const f = info.files || {};
  if (f["preview.mp4"] && info.preview) {
    $("b-media").innerHTML = `<video src="${f["preview.mp4"]}" poster="${f["poster.jpg"] || ""}" controls autoplay loop playsinline></video>`;
    const c = info.preview;
    $("b-checks").textContent = `Rendered: ${c.render_ok ? "yes" : "no"} · face still matches your photo: ${c.face_match ?? "not found"}` +
      (c.passed ? " · checks passed" : " · checks failed, this stays a draft");
    bShow("b-approve", info.status !== "published");
    $("b-publish").disabled = !c.passed;
    if (info.status === "published") { $("b-done").textContent = "Published: it is in the gallery."; bShow("b-done", true); }
  } else {
    if (info.preview && !info.preview.passed) {
      $("b-errors").textContent = "The last preview failed, so this is still a draft: " + (info.preview.error || "its checks did not pass");
      bShow("b-errors", true);
    }
    bShow("b-render", true);
  }
}

if (location.hash === "#build") bOpen();
const bm = location.hash.match(/^#build=([a-z0-9-]+)$/);
if (bm) bLoad(bm[1]);
