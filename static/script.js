"use strict";

const state = { reference: null, target: null, jobId: null, pollTimer: null };

const ALLOWED = ["image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff", "image/x-tiff"];

const zones = document.querySelectorAll(".drop-zone");

function fileFor(role) {
  return role === "reference" ? state.reference : state.target;
}

function isAllowed(file) {
  return ALLOWED.includes(file.type) || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name);
}

function renderZone(role, file) {
  const preview = document.getElementById(`preview-${role === "reference" ? "ref" : "tgt"}`);
  const placeholder = document.getElementById(`placeholder-${role === "reference" ? "ref" : "tgt"}`);
  const meta = document.getElementById(`meta-${role === "reference" ? "ref" : "tgt"}`);
  const removeBtn = document.getElementById(`remove-${role === "reference" ? "ref" : "tgt"}`);
  const zone = document.getElementById(`zone-${role === "reference" ? "ref" : "tgt"}`);

  if (!file) {
    preview.hidden = true;
    placeholder.hidden = false;
    meta.hidden = true;
    removeBtn.hidden = true;
    zone.classList.remove("has-file");
    return;
  }

  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  placeholder.hidden = true;
  meta.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  meta.hidden = false;
  removeBtn.hidden = false;
  zone.classList.add("has-file");
}

function setZoneFile(role, file) {
  if (!isAllowed(file)) {
    showToast("Unsupported file type. Use PNG, JPG, WEBP, BMP or TIFF.");
    return;
  }
  if (role === "reference") state.reference = file;
  else state.target = file;
  renderZone(role, file);
  updateGradeBtn();
}

function clearZone(role) {
  if (role === "reference") state.reference = null;
  else state.target = null;
  renderZone(role, null);
  updateGradeBtn();
}

zones.forEach((zone) => {
  const role = zone.dataset.role;
  const input = zone.querySelector('input[type="file"]');

  zone.addEventListener("click", (e) => {
    if (e.target.closest(".remove-btn")) return;
    input.click();
  });

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) setZoneFile(role, input.files[0]);
    input.value = "";
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setZoneFile(role, file);
  });

  document.getElementById(`remove-${role === "reference" ? "ref" : "tgt"}`)
    .addEventListener("click", () => clearZone(role));
});

const fgSlider = document.getElementById("fg-opacity");
const bgSlider = document.getElementById("bg-opacity");
const fgOut = document.getElementById("fg-out");
const bgOut = document.getElementById("bg-out");
const lutToggle = document.getElementById("generate-lut");
const lutSize = document.getElementById("lut-size");
const gradeBtn = document.getElementById("grade-btn");

fgSlider.addEventListener("input", () => (fgOut.value = Number(fgSlider.value).toFixed(2)));
bgSlider.addEventListener("input", () => (bgOut.value = Number(bgSlider.value).toFixed(2)));

function updateGradeBtn() {
  gradeBtn.disabled = !(state.reference && state.target);
}

function setUiLoading(loading) {
  gradeBtn.disabled = loading || !(state.reference && state.target);
  gradeBtn.textContent = loading ? "Processing..." : "\u25B6 Grade Images";
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => (toast.hidden = true), 5000);
}

function updateProgressBar(message, pct) {
  document.getElementById("progress-wrap").hidden = false;
  document.getElementById("progress-message").textContent = message;
  if (typeof pct === "number" && pct >= 0) {
    document.getElementById("progress-bar").style.width = `${pct}%`;
    document.getElementById("progress-pct").textContent = `${pct}%`;
  }
}

function showResults(result) {
  document.getElementById("results").hidden = false;
  const before = document.getElementById("before-img");
  before.src = URL.createObjectURL(state.target);

  const after = document.getElementById("after-img");
  after.src = result.image_url;

  const dlImage = document.getElementById("dl-image");
  dlImage.href = result.image_url;
  dlImage.download = "graded_image.jpg";

  const dlLut = document.getElementById("dl-lut");
  if (result.lut_url) {
    dlLut.href = result.lut_url;
    dlLut.download = "colorspace_grade.cube";
    dlLut.hidden = false;
  } else {
    dlLut.hidden = true;
  }

  document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function startPolling() {
  updateProgressBar("Starting...", 0);
  state.pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/progress/${state.jobId}`);
      const data = await resp.json();
      if (data.status === "done") {
        clearInterval(state.pollTimer);
        const resultResp = await fetch(`/api/result/${state.jobId}`);
        const result = await resultResp.json();
        showResults(result.result);
        setUiLoading(false);
      } else if (data.status === "error") {
        clearInterval(state.pollTimer);
        setUiLoading(false);
        updateProgressBar(data.message || "Processing failed", 0);
        showToast(data.error || "Processing failed. Check the console for details.");
      } else {
        updateProgressBar(data.message || "Working...", data.progress || 0);
      }
    } catch (err) {
      clearInterval(state.pollTimer);
      setUiLoading(false);
      showToast("Lost connection to the server. Is Flask still running?");
    }
  }, 600);
}

async function startProcessing() {
  if (!state.reference || !state.target) {
    showToast("Please upload both a reference and a target image first.");
    return;
  }

  setUiLoading(true);
  document.getElementById("results").hidden = true;

  const form = new FormData();
  form.append("reference", state.reference);
  form.append("target", state.target);
  form.append("generate_lut", lutToggle.checked ? "true" : "false");
  form.append("fg_color_opacity", fgSlider.value);
  form.append("bg_color_opacity", bgSlider.value);
  form.append("lut_size", lutSize.value);

  try {
    const resp = await fetch("/api/process", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Failed to start job.");
    state.jobId = data.job_id;
    startPolling();
  } catch (err) {
    setUiLoading(false);
    showToast(err.message);
  }
}

gradeBtn.addEventListener("click", startProcessing);
updateGradeBtn();
