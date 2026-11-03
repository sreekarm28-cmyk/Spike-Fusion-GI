"use strict";

const state = {
  modelReady: false,
  adminAuthenticated: false,
  adminConfigured: false,
  settings: { change_threshold: 12, min_gap: 0.8, max_gap: 3 },
  photo: null,
  video: null,
  photoUrl: null,
  videoUrl: null,
};

const $ = (selector) => document.querySelector(selector);

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function setMessage(text, kind = "error", element = $("#message")) {
  element.textContent = text;
  element.classList.remove("is-hidden", "is-success");
  if (kind === "success") element.classList.add("is-success");
}

function clearMessage(element = $("#message")) {
  element.textContent = "";
  element.classList.add("is-hidden");
  element.classList.remove("is-success");
}

function updateModelBadge() {
  const badge = $("#modelStatus");
  badge.classList.remove("status-loading", "status-ready", "status-missing");
  if (state.modelReady) {
    badge.classList.add("status-ready");
    badge.innerHTML = '<span class="status-dot"></span>Model ready';
  } else {
    badge.classList.add("status-missing");
    badge.innerHTML = '<span class="status-dot"></span>Model not set';
  }
  $("#adminModelStatus").textContent = state.modelReady
    ? "A saved model is ready. A new upload will replace it."
    : "No trained model is active yet.";
}

function updateAdminView() {
  $("#adminLoginView").classList.toggle("is-hidden", state.adminAuthenticated);
  $("#adminPanelView").classList.toggle("is-hidden", !state.adminAuthenticated);
  if (!state.adminConfigured) {
    setMessage("Admin access is not configured yet. Add APP_PASSWORD to .streamlit/secrets.toml.", "error", $("#adminMessage"));
  } else {
    clearMessage($("#adminMessage"));
  }
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    state.modelReady = data.model_ready;
    state.adminAuthenticated = data.admin_authenticated;
    state.adminConfigured = data.admin_configured;
    state.settings = data.video_settings || state.settings;
    updateModelBadge();
    updateAdminView();
    updateFrameSettings();
    if (data.model_error) setMessage(`Saved model could not be loaded: ${data.model_error}`);
  } catch (error) {
    setMessage(`Could not connect to the local app: ${error.message}`);
  }
}

function switchMode(mode) {
  const isPhoto = mode === "photo";
  $("#photoPanel").classList.toggle("is-hidden", !isPhoto);
  $("#videoPanel").classList.toggle("is-hidden", isPhoto);
  $("#photoTab").classList.toggle("is-active", isPhoto);
  $("#videoTab").classList.toggle("is-active", !isPhoto);
  $("#photoTab").setAttribute("aria-selected", String(isPhoto));
  $("#videoTab").setAttribute("aria-selected", String(!isPhoto));
  clearMessage();
}

function setPhoto(file) {
  if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
  state.photo = file || null;
  $("#photoResult").classList.add("is-hidden");
  clearMessage();
  if (!file) {
    $("#photoPreviewWrap").classList.add("is-hidden");
    $("#photoFileName").textContent = "";
    $("#analyzePhoto").disabled = true;
    $("#photoInput").value = "";
    return;
  }
  state.photoUrl = URL.createObjectURL(file);
  $("#photoPreview").src = state.photoUrl;
  $("#photoPreviewWrap").classList.remove("is-hidden");
  $("#photoFileName").textContent = file.name;
  $("#analyzePhoto").disabled = false;
}

function setVideo(file) {
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.video = file || null;
  $("#videoResult").classList.add("is-hidden");
  clearMessage();
  if (!file) {
    $("#videoPreviewWrap").classList.add("is-hidden");
    $("#videoFileName").textContent = "";
    $("#analyzeVideo").disabled = true;
    $("#videoInput").value = "";
    return;
  }
  state.videoUrl = URL.createObjectURL(file);
  $("#videoPreview").src = state.videoUrl;
  $("#videoPreviewWrap").classList.remove("is-hidden");
  $("#videoFileName").textContent = file.name;
  $("#analyzeVideo").disabled = false;
}

function connectDropzone(zoneId, inputId, acceptFile) {
  const zone = $(zoneId);
  const input = $(inputId);
  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) acceptFile(file);
  });
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
}

function renderProbabilities(probabilities, predictedLabel) {
  const sorted = [...probabilities].sort((a, b) => b.value - a.value);
  $("#probabilityList").replaceChildren();
  sorted.forEach((item) => {
    const row = document.createElement("div");
    row.className = `probability-row${item.label === predictedLabel ? " is-top" : ""}`;
    const name = document.createElement("span");
    name.textContent = item.label.replaceAll("-", " ");
    const track = document.createElement("span");
    track.className = "probability-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(100, item.value * 100))}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = `${(item.value * 100).toFixed(1)}%`;
    row.append(name, track, value);
    $("#probabilityList").append(row);
  });
}

async function analyzePhoto() {
  if (!state.photo) return;
  clearMessage();
  const button = $("#analyzePhoto");
  button.disabled = true;
  button.textContent = "Analyzing…";
  const body = new FormData();
  body.append("file", state.photo);
  try {
    const result = await api("/api/predict/image", { method: "POST", body });
    $("#photoPrediction").textContent = result.label.replaceAll("-", " ");
    $("#photoConfidence").textContent = `${(result.confidence * 100).toFixed(1)}%`;
    $("#photoConfidenceBar").style.width = `${Math.max(0, Math.min(100, result.confidence * 100))}%`;
    $("#resultOriginal").src = result.original;
    $("#resultHeatmap").src = result.heatmap;
    $("#resultOverlay").src = result.overlay;
    renderProbabilities(result.probabilities, result.label);
    $("#photoResult").classList.remove("is-hidden");
  } catch (error) {
    setMessage(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze image <span aria-hidden="true">→</span>';
  }
}

function updateFrameSettings() {
  const settings = state.settings;
  $("#changeThreshold").value = settings.change_threshold;
  $("#minGap").value = settings.min_gap;
  $("#maxGap").value = settings.max_gap;
  updateFrameSettingLabels();
}

function updateFrameSettingLabels() {
  $("#changeThresholdValue").textContent = $("#changeThreshold").value;
  $("#minGapValue").textContent = `${Number($("#minGap").value).toFixed(1)} sec`;
  $("#maxGapValue").textContent = `${Number($("#maxGap").value).toFixed(1)} sec`;
}

function readFrameSettings() {
  return {
    change_threshold: Number($("#changeThreshold").value),
    min_gap: Number($("#minGap").value),
    max_gap: Number($("#maxGap").value),
  };
}

function renderVideoResults(data) {
  $("#framesChecked").textContent = data.selected_count;
  $("#videoCounts").replaceChildren();
  Object.entries(data.counts).sort((a, b) => b[1] - a[1]).forEach(([label, count]) => {
    const pill = document.createElement("span");
    pill.className = "count-pill";
    pill.append(document.createTextNode(label.replaceAll("-", " ")));
    const total = document.createElement("strong");
    total.textContent = count;
    pill.append(total);
    $("#videoCounts").append(pill);
  });
  const frames = $("#videoFrames");
  frames.replaceChildren();
  data.results.forEach((result) => {
    const card = document.createElement("article");
    card.className = "frame-card";
    const image = document.createElement("img");
    image.src = result.preview;
    image.alt = `Selected video frame at ${result.time}`;
    const content = document.createElement("div");
    content.className = "frame-card-content";
    const title = document.createElement("div");
    title.className = "frame-card-title";
    const label = document.createElement("strong");
    label.textContent = result.label.replaceAll("-", " ");
    const time = document.createElement("span");
    time.textContent = result.time;
    title.append(label, time);
    const confidence = document.createElement("p");
    confidence.textContent = `Model confidence ${(result.confidence * 100).toFixed(1)}%`;
    content.append(title, confidence);
    card.append(image, content);
    frames.append(card);
  });
  $("#videoResult").classList.remove("is-hidden");
}

async function analyzeVideo() {
  if (!state.video) return;
  clearMessage();
  const button = $("#analyzeVideo");
  button.disabled = true;
  button.textContent = "Analyzing…";
  $("#videoProgress").classList.remove("is-hidden");
  const body = new FormData();
  body.append("file", state.video);
  Object.entries(state.settings).forEach(([key, value]) => body.append(key, value));
  try {
    const result = await api("/api/predict/video", { method: "POST", body });
    renderVideoResults(result);
  } catch (error) {
    setMessage(error.message);
  } finally {
    $("#videoProgress").classList.add("is-hidden");
    button.disabled = false;
    button.innerHTML = 'Analyze video <span aria-hidden="true">→</span>';
  }
}

async function handleLogin(event) {
  event.preventDefault();
  clearMessage($("#adminMessage"));
  const button = event.submitter;
  button.disabled = true;
  try {
    await api("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: $("#adminPassword").value }),
    });
    $("#adminPassword").value = "";
    await refreshStatus();
  } catch (error) {
    setMessage(error.message, "error", $("#adminMessage"));
  } finally {
    button.disabled = false;
  }
}

async function handleCheckpointUpload(event) {
  event.preventDefault();
  clearMessage($("#adminMessage"));
  const file = $("#checkpointInput").files[0];
  if (!file) return;
  const button = event.submitter;
  button.disabled = true;
  button.textContent = "Checking and saving…";
  const body = new FormData();
  body.append("checkpoint", file);
  try {
    const result = await api("/api/admin/checkpoint", { method: "POST", body });
    setMessage(result.message, "success", $("#adminMessage"));
    await refreshStatus();
  } catch (error) {
    setMessage(error.message, "error", $("#adminMessage"));
  } finally {
    button.disabled = false;
    button.textContent = "Save and activate model";
  }
}

async function saveFrameSettings() {
  clearMessage($("#adminMessage"));
  try {
    const result = await api("/api/admin/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readFrameSettings()),
    });
    state.settings = result.video_settings;
    updateFrameSettings();
    setMessage("Frame selection settings saved.", "success", $("#adminMessage"));
  } catch (error) {
    setMessage(error.message, "error", $("#adminMessage"));
  }
}

async function logoutAdmin() {
  try {
    await api("/api/admin/logout", { method: "POST" });
    await refreshStatus();
  } catch (error) {
    setMessage(error.message, "error", $("#adminMessage"));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#photoTab").addEventListener("click", () => switchMode("photo"));
  $("#videoTab").addEventListener("click", () => switchMode("video"));
  $("#photoInput").addEventListener("change", (event) => setPhoto(event.target.files[0]));
  $("#videoInput").addEventListener("change", (event) => setVideo(event.target.files[0]));
  $("#removePhoto").addEventListener("click", () => setPhoto(null));
  $("#removeVideo").addEventListener("click", () => setVideo(null));
  connectDropzone("#photoDropzone", "#photoInput", setPhoto);
  connectDropzone("#videoDropzone", "#videoInput", setVideo);
  $("#analyzePhoto").addEventListener("click", analyzePhoto);
  $("#analyzeVideo").addEventListener("click", analyzeVideo);

  $("#openAdmin").addEventListener("click", () => $("#adminDialog").showModal());
  $("#closeAdmin").addEventListener("click", () => $("#adminDialog").close());
  $("#adminLoginForm").addEventListener("submit", handleLogin);
  $("#checkpointForm").addEventListener("submit", handleCheckpointUpload);
  $("#saveSettings").addEventListener("click", saveFrameSettings);
  $("#adminLogout").addEventListener("click", logoutAdmin);

  $("#changeThreshold").addEventListener("input", updateFrameSettingLabels);
  $("#minGap").addEventListener("input", updateFrameSettingLabels);
  $("#maxGap").addEventListener("input", updateFrameSettingLabels);
  refreshStatus();
});
