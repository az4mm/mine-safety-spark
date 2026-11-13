"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { data: null, view: "overview", action: null, fetching: false, lastSuccess: null, failures: 0, assets: [], priorRunning: false };
  const views = { overview: "Overview", incidents: "Incident log", vision: "Vision evidence", performance: "Spark performance", method: "Method & sources", guide: "Run guide" };
  const numberFormatter = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
  const timeFormatter = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const shortTimeFormatter = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false });
  const dayFormatter = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short" });
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  const icon = (name, classes = "") => `<svg class="icon ${classes}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const finite = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const numeric = (value) => finite(value) ? Number(value) : null;
  const first = (...values) => values.find((value) => value !== null && value !== undefined && value !== "");
  const fmt = (value) => finite(value) ? numberFormatter.format(Number(value)) : "—";
  const pct = (value) => finite(value) ? `${Number(value).toFixed(1)}%` : "—";
  const setText = (id, value) => { if ($(id)) $(id).textContent = value ?? "—"; };
  const asArray = (value) => Array.isArray(value) ? value : [];
  const empty = (title, description, symbol = "data") => `<div class="empty-state compact">${icon(symbol)}<strong>${escape(title)}</strong><p>${escape(description)}</p></div>`;
  function safeUrl(value) {
    if (!value || typeof value !== "string") return null;
    try { const url = new URL(value, location.href); return ["http:", "https:"].includes(url.protocol) ? url.href : null; } catch { return null; }
  }
  function safeMediaUrl(value) {
    if (typeof value === "string" && /^data:(image\/(png|jpeg|webp|gif)|video\/(mp4|webm));base64,[A-Za-z0-9+/=\s]+$/.test(value)) return value;
    return safeUrl(value);
  }
  function dateValue(value) { if (!value) return null; const date = new Date(value); return Number.isNaN(date.getTime()) ? null : date; }
  function time(value, short = false) { const date = dateValue(value); return date ? (short ? shortTimeFormatter : timeFormatter).format(date) : "—"; }
  function duration(value) { const num = numeric(value); if (num === null) return "—"; if (num < 1) return `${Math.round(num * 1000)} ms`; if (num < 60) return `${num.toFixed(num % 1 ? 1 : 0)} s`; return `${Math.floor(num / 60)}m ${Math.round(num % 60)}s`; }
  function sourceName(source) { return ({ synthetic: "Synthetic", public_image: "Public image", video: "Video" })[source] || source || "Unspecified"; }
  function sourcePill(source) { return `<span class="pill ${source === "synthetic" ? "pill-amber" : source === "public_image" || source === "video" ? "pill-blue" : "pill-neutral"}">${escape(sourceName(source))}</span>`; }
  function incidentType(incident) { const value = String(incident.type || incident.violation_type || "").toLowerCase(); return /zone|intrus|restricted/.test(value) ? "zone" : /helmet|head/.test(value) ? "helmet" : "other"; }
  function incidentName(incident) { const type = incidentType(incident); return type === "zone" ? "Restricted-zone entry" : type === "helmet" ? "Missing helmet" : (incident.type || "Safety incident"); }
  function priority(incident) { const type = incidentType(incident); const value = String(incident.severity || (type === "zone" ? "high" : type === "helmet" ? "medium" : "unknown")).toLowerCase(); return { name: value.replace(/_/g, " "), className: /high|critical/.test(value) ? "pill-rose" : /medium|warning/.test(value) ? "pill-amber" : /low|normal/.test(value) ? "pill-teal" : "pill-neutral" }; }
  function normalizedIncidents(data) { return asArray(data?.incidents).slice().sort((a, b) => (dateValue(b.start_time)?.getTime() || 0) - (dateValue(a.start_time)?.getTime() || 0)); }
  function metricsOf(data) { return data && typeof data.metrics === "object" && data.metrics !== null ? data.metrics : {}; }
  function rateOf(metrics) { return first(metrics.processed_rows_per_second, metrics.processedRowsPerSecond, metrics.throughput_eps, metrics.records_per_second, metrics.throughput_rows_per_second); }
  function runDurationOf(metrics) { return first(metrics.duration_s, metrics.run_seconds, metrics.elapsed_seconds, metrics.elapsed_s, metrics.runtime_seconds, metrics.wall_time_seconds); }

  function showView(name, updateHash = true) {
    if (!Object.hasOwn(views, name)) name = "overview";
    state.view = name;
    document.querySelectorAll(".view").forEach((section) => { const active = section.id === `view-${name}`; section.hidden = !active; section.classList.toggle("active", active); });
    document.querySelectorAll(".nav-item[data-view]").forEach((button) => { const active = button.dataset.view === name; button.classList.toggle("active", active); if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current"); });
    setText("breadcrumbView", views[name]);
    document.title = `${views[name]} · MineGuard`;
    if (updateHash && location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    closeMenu();
    if (name === "incidents") renderIncidentLog();
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function closeMenu() { $("sidebar").classList.remove("open"); $("sidebarScrim").hidden = true; $("menuButton").setAttribute("aria-expanded", "false"); }

  function toast(message, error = false) {
    const element = document.createElement("div");
    element.className = `toast${error ? " error" : ""}`;
    element.innerHTML = `${icon(error ? "alert" : "check")}<span>${escape(message)}</span><button aria-label="Dismiss notification">${icon("close")}</button>`;
    element.querySelector("button").addEventListener("click", () => element.remove());
    $("toastContainer").append(element);
    setTimeout(() => element.remove(), error ? 10000 : 6500);
  }

  async function fetchJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) } });
      const type = response.headers.get("content-type") || "";
      const body = type.includes("application/json") ? await response.json() : { detail: (await response.text()).slice(0, 250) };
      if (!response.ok) { const detail = body.detail || body.error || body.message || `Request failed (${response.status})`; throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail)); }
      return body;
    } finally { clearTimeout(timeout); }
  }

  async function refresh(manual = false) {
    if (state.fetching) return;
    state.fetching = true;
    try {
      const data = await fetchJson("/api/status");
      if (!data || typeof data !== "object") throw new Error("The server returned an invalid status response.");
      state.data = data;
      state.lastSuccess = new Date();
      state.failures = 0;
      $("connectionError").hidden = true;
      $("connectionBadge").className = "connection-badge connected";
      $("connectionBadge").innerHTML = '<span class="status-dot"></span>Backend connected';
      render(data);
      if (manual) toast("Dashboard refreshed from the latest project outputs.");
    } catch (error) {
      state.failures++;
      $("connectionBadge").className = "connection-badge failed";
      $("connectionBadge").innerHTML = '<span class="status-dot"></span>Connection unavailable';
      $("connectionError").hidden = false;
      setText("connectionErrorText", `Cannot reach the analytics backend. ${state.data ? "The last received results remain visible. " : ""}${error.name === "AbortError" ? "The request timed out." : error.message}`);
      if (manual) toast("Could not refresh the dashboard. Check that the project server is running.", true);
    } finally { state.fetching = false; }
  }

  async function action(kind) {
    if (state.action) return;
    if (kind === "start" && state.data?.running) return;
    state.action = kind;
    updateButtons();
    const config = { start: { path: "/api/demo/start", body: {}, message: "Synthetic replay requested. Spark may take a moment to initialise." }, stop: { path: "/api/demo/stop", body: {}, message: "Stop requested. The current outputs remain available." }, batch: { path: "/api/analyze", body: { input: "demo" }, message: "Batch analytics requested. Results will update after Spark completes." } }[kind];
    if (!config) { state.action = null; updateButtons(); return; }
    try {
      const result = await fetchJson(config.path, { method: "POST", body: JSON.stringify(config.body) });
      if (result.error || result.status === "error") throw new Error(result.error || result.message || "The action could not be started.");
      toast(result.message || config.message);
      await refresh();
    } catch (error) { toast(error.name === "AbortError" ? "The action request timed out. Check status before retrying." : error.message, true); }
    finally { state.action = null; updateButtons(); }
  }

  function updateButtons() {
    const running = Boolean(state.data?.running);
    const busy = Boolean(state.action);
    document.querySelectorAll("[data-action]").forEach((button) => {
      const kind = button.dataset.action;
      button.disabled = busy || (running && ["start", "batch"].includes(kind));
      if (kind === "stop") { button.hidden = !running; button.disabled = busy; }
      if (kind === "start" && button.closest(".heading-actions")) button.hidden = running;
      button.setAttribute("aria-busy", state.action === kind ? "true" : "false");
    });
  }

  function render(data) {
    const summary = data.summary || {};
    const metrics = metricsOf(data);
    const incidents = normalizedIncidents(data);
    const cameras = asArray(data.cameras);
    const hasResults = finite(summary.observations);
    setText("metricObservations", fmt(summary.observations));
    setText("metricCompliance", pct(summary.compliance_pct));
    setText("metricHelmet", fmt(summary.helmet_incidents));
    setText("metricZones", fmt(summary.zone_incidents));
    setText("observationDetail", hasResults ? `${fmt(first(summary.cameras, cameras.length))} camera sources · ${fmt(summary.workers)} observed tracks` : "Waiting for Spark output");
    setText("complianceDetail", finite(summary.assessable_seconds) ? `${duration(summary.assessable_seconds)} assessable worker-time` : "Among assessable worker-time");
    $("complianceBar").style.width = `${Math.max(0, Math.min(100, numeric(summary.compliance_pct) || 0))}%`;
    const sourceTypes = getSourceTypes(data);
    const synthetic = sourceTypes.size === 1 && sourceTypes.has("synthetic");
    const actual = sourceTypes.size > 0 && !sourceTypes.has("synthetic");
    setText("sourceLabel", synthetic ? "Synthetic event replay" : actual ? "Observed vision records" : sourceTypes.size > 1 ? "Mixed sources · inspect provenance" : hasResults ? "Analytics results available" : "Waiting for data");
    setText("sourceDescription", synthetic ? "Controlled scenarios for analytics testing. Actual Spark processing." : actual ? "Inspect the evidence and source details for the scope of these results." : "Run the demo to process labelled synthetic observations with Spark.");
    $("sourceDot").className = `source-dot${synthetic ? " synthetic" : actual ? " actual" : ""}${data.running ? " running-pulse" : ""}`;
    setText("modeLabel", String(first(data.mode, metrics.mode, hasResults ? "batch" : "not run")).replace(/_/g, " ").toUpperCase());
    const updated = dateValue(data.last_updated || metrics.completed_at || metrics.finished_at);
    setText("updateCaption", updated ? `Updated ${time(updated)}` : hasResults ? "Results loaded" : "No results yet");
    const engine = typeof data.engine === "string" ? data.engine : first(data.engine?.name, metrics.engine, "Apache Spark");
    setText("engineName", engine);
    setText("engineVersion", metrics.spark_version ? `Version ${metrics.spark_version}` : hasResults ? "See execution metrics" : "Awaiting execution");
    setText("engineState", data.running ? "Running" : hasResults ? "Completed" : "Idle");
    $("engineState").className = `engine-state${data.running ? " running" : ""}`;
    setText("engineMaster", first(metrics.master, data.engine?.master, "—"));
    const rate = rateOf(metrics);
    setText("engineRate", finite(rate) ? `${Number(rate).toFixed(1)} rows/s` : "Not measured");
    const batchMs = first(metrics.batch_duration_ms, metrics.batchDuration, metrics.durationMs?.triggerExecution);
    setText("engineBatch", finite(batchMs) ? duration(Number(batchMs) / 1000) : finite(runDurationOf(metrics)) ? duration(runDurationOf(metrics)) : "Not measured");
    setText("engineUnknown", pct(summary.unknown_pct));
    setText("incidentNavCount", hasResults ? fmt(incidents.length) : "—");
    setText("cameraCount", cameras.length);
    setText("cameraScope", synthetic ? "SYNTHETIC CAMERA STREAMS" : "OBSERVED DATA");
    renderCameras(cameras, data);
    renderTrend(asArray(data.timeline));
    $("recentIncidents").innerHTML = incidents.length ? incidents.slice(0, 5).map((incident) => incidentRow(incident, false)).join("") : '<tr><td colspan="6" class="empty-table">No incident records available.</td></tr>';
    const previousCamera = $("incidentCamera").value;
    const cameraIds = Array.from(new Set([...cameras.map((camera) => camera.camera_id), ...incidents.map((incident) => incident.camera_id)].filter(Boolean))).sort();
    const optionKey = cameraIds.join("|");
    if ($("incidentCamera").dataset.key !== optionKey) { $("incidentCamera").innerHTML = '<option value="all">All cameras</option>' + cameraIds.map((camera) => `<option value="${escape(camera)}">${escape(camera)}</option>`).join(""); $("incidentCamera").dataset.key = optionKey; if (cameraIds.includes(previousCamera)) $("incidentCamera").value = previousCamera; }
    renderIncidentLog();
    renderAssets(asArray(data.assets));
    renderPerformance(data);
    renderSources(data.sources);
    if (asArray(data.limitations).length) $("limitationsList").innerHTML = data.limitations.map((item) => `<li>${escape(typeof item === "string" ? item : item.description || item.text || JSON.stringify(item))}</li>`).join("");
    const jobMessage = first(data.error, data.job?.error, data.message, data.job?.message);
    const jobActive = data.running || /running|starting|queued/.test(String(data.job?.status || ""));
    $("jobNotice").hidden = !jobActive && !data.error && !data.job?.error;
    $("jobNotice").classList.toggle("error-notice", Boolean(data.error || data.job?.error));
    $("jobNotice").classList.toggle("job-notice", !data.error && !data.job?.error);
    setText("jobNoticeText", jobMessage || (jobActive ? "Spark is processing incoming observations. Completed batches will update this dashboard automatically." : ""));
    $("jobNotice").querySelector(".job-spinner").hidden = !jobActive;
    updateButtons();
    document.querySelectorAll(".events-link").forEach((link) => { link.setAttribute("aria-disabled", String(!hasResults)); link.classList.toggle("disabled", !hasResults); link.title = hasResults ? "Export source observations" : "Run analytics before exporting observations."; });
    state.priorRunning = Boolean(data.running);
  }

  function getSourceTypes(data) {
    const set = new Set();
    const counts = first(data.source_counts, data.summary?.source_counts, data.metrics?.source_type_counts, data.metrics?.source_counts, data.metrics?.sources);
    if (counts && typeof counts === "object" && !Array.isArray(counts)) Object.entries(counts).forEach(([key, value]) => { if (Number(value) > 0) set.add(key); });
    asArray(data.incidents).forEach((incident) => { if (incident.source_type) set.add(incident.source_type); });
    if (data.source_type) set.add(data.source_type);
    if (!set.size && /demo|synthetic|replay/.test(String(data.mode || "").toLowerCase())) set.add("synthetic");
    return set;
  }

  function renderCameras(cameras, data) {
    if (!cameras.length) { $("cameraGrid").innerHTML = `<div class="panel empty-state camera-empty">${icon("camera")}<strong>No camera records yet</strong><p>Camera summaries appear after the first completed Spark batch.</p></div>`; return; }
    $("cameraGrid").innerHTML = cameras.map((camera) => {
      const comp = numeric(camera.compliance_pct);
      const incidents = finite(camera.helmet_incidents) && finite(camera.zone_incidents) ? Number(camera.helmet_incidents) + Number(camera.zone_incidents) : null;
      const cameraStatus = String(camera.status || "").toLowerCase();
      const active = data.running && !/offline|unavailable|error|high|warning/.test(cameraStatus);
      const statusClass = /high|critical/.test(cameraStatus) ? " status-high" : /warning|medium/.test(cameraStatus) ? " status-warning" : active ? " active" : "";
      return `<article class="camera-card"><div class="camera-card-head">${icon("camera")}<h3>${escape(camera.camera_id || "Camera")}</h3><span class="camera-status${statusClass}">${escape(camera.status || (active ? "Processing" : "Recorded"))}</span></div><div class="camera-zone">${escape(camera.zone_id || "Unassigned zone")}</div><div class="camera-numbers"><div><strong>${fmt(camera.observations)}</strong><span>observations</span></div><div><strong>${fmt(incidents)}</strong><span>incident episodes</span></div></div><div class="camera-compliance"><div class="camera-compliance-head"><span>Helmet compliance</span><strong>${comp === null ? "Not assessable" : pct(comp)}</strong></div><div class="camera-bar" role="img" aria-label="Helmet compliance ${escape(comp === null ? "not assessable" : pct(comp))}"><span style="width:${Math.max(0, Math.min(100, comp || 0))}%"></span></div></div></article>`;
    }).join("");
  }

  function renderTrend(items) {
    if (!items.length) { $("trendChart").innerHTML = empty("Your timeline will appear here", "Run analytics to see incident activity over time.", "activity"); setText("timelineCount", "No windows available"); return; }
    const points = items.slice().sort((a, b) => (dateValue(a.time)?.getTime() || 0) - (dateValue(b.time)?.getTime() || 0)).slice(-60);
    const width = 640, height = 188, left = 35, right = 17, top = 15, bottom = 31;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const max = Math.max(1, ...points.map((item) => Math.max(Number(item.helmet_incidents) || 0, Number(item.zone_incidents) || 0)));
    const maxAxis = max <= 4 ? Math.ceil(max) : Math.ceil(max / 4) * 4;
    const x = (index) => left + (points.length > 1 ? index / (points.length - 1) : .5) * plotWidth;
    const y = (value) => top + plotHeight - (Number(value) || 0) / maxAxis * plotHeight;
    const path = (key) => points.map((item, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(2)},${y(item[key]).toFixed(2)}`).join(" ");
    const area = (key) => `${path(key)} L${x(points.length - 1)},${top + plotHeight} L${x(0)},${top + plotHeight} Z`;
    let grid = "";
    for (let index = 0; index <= 4; index++) { const value = maxAxis * index / 4; const pos = y(value); grid += `<line x1="${left}" y1="${pos}" x2="${width - right}" y2="${pos}" stroke="#263447" stroke-dasharray="3 5" stroke-width=".8"/><text x="${left - 12}" y="${pos + 3}" text-anchor="end" fill="#627b98" font-size="9">${Number.isInteger(value) ? value : value.toFixed(1)}</text>`; }
    const labelIndices = [...new Set([0, Math.floor((points.length - 1) / 3), Math.floor((points.length - 1) * 2 / 3), points.length - 1])];
    const labels = labelIndices.map((index) => `<text x="${x(index)}" y="${height - 7}" text-anchor="middle" fill="#7188a2" font-size="9">${escape(time(points[index].time, true))}</text>`).join("");
    const dots = points.map((item, index) => `<circle cx="${x(index)}" cy="${y(item.helmet_incidents)}" r="${points.length > 20 ? 1.5 : 2.4}" fill="#f3b85b"><title>${escape(time(item.time))}: ${Number(item.helmet_incidents) || 0} helmet incidents</title></circle><circle cx="${x(index)}" cy="${y(item.zone_incidents)}" r="${points.length > 20 ? 1.5 : 2.4}" fill="#4ed6b6"><title>${escape(time(item.time))}: ${Number(item.zone_incidents) || 0} restricted-zone incidents</title></circle>`).join("");
    $("trendChart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Incident timeline. ${points.length} time points, showing helmet incidents and restricted-zone incidents."><defs><linearGradient id="helmetFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#f3b85b" stop-opacity=".09"/><stop offset="100%" stop-color="#f3b85b" stop-opacity="0"/></linearGradient><linearGradient id="zoneFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#4ed6b6" stop-opacity=".1"/><stop offset="100%" stop-color="#4ed6b6" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${area("helmet_incidents")}" fill="url(#helmetFill)"/><path d="${area("zone_incidents")}" fill="url(#zoneFill)"/><path d="${path("helmet_incidents")}" fill="none" stroke="#f3b85b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="${path("zone_incidents")}" fill="none" stroke="#4ed6b6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>${dots}${labels}</svg>`;
    setText("timelineCount", `${points.length} time ${points.length === 1 ? "point" : "points"}${items.length > 60 ? " · latest 60 shown" : ""}`);
  }

  function incidentRow(incident, full = false) {
    const kind = incidentType(incident), severity = priority(incident), started = dateValue(incident.start_time);
    const evidence = safeMediaUrl(incident.evidence_url || incident.evidence_path);
    return `<tr><td><span class="incident-label ${kind}">${icon(kind === "zone" ? "zone" : kind === "helmet" ? "helmet" : "alert")}<span><strong>${escape(incidentName(incident))}</strong>${full ? `<small class="monospace">${escape(incident.incident_id || "—")}</small>` : ""}</span></span></td><td><strong>${escape(incident.camera_id || "—")}</strong><small>${escape(incident.zone_id || "—")}</small></td><td class="monospace">${escape(time(incident.start_time))}${full && started ? `<small>${escape(dayFormatter.format(started))}</small>` : ""}</td><td class="monospace">${escape(duration(incident.duration_s))}</td><td><span class="pill ${severity.className}">${escape(severity.name.charAt(0).toUpperCase() + severity.name.slice(1))}</span></td><td>${sourcePill(incident.source_type)}</td>${full ? `<td>${evidence ? `<a class="text-button" href="${escape(evidence)}" target="_blank" rel="noopener noreferrer">View ${icon("external")}</a>` : '<span class="subtle-label">—</span>'}</td>` : ""}</tr>`;
  }

  function renderIncidentLog() {
    const query = $("incidentSearch").value.trim().toLowerCase(), type = $("incidentType").value, camera = $("incidentCamera").value;
    const incidents = normalizedIncidents(state.data).filter((item) => (type === "all" || incidentType(item) === type) && (camera === "all" || item.camera_id === camera) && (!query || [item.incident_id, item.camera_id, item.zone_id, item.track_id, incidentName(item)].some((value) => String(value || "").toLowerCase().includes(query))));
    setText("filteredCount", incidents.length);
    $("allIncidents").innerHTML = incidents.length ? incidents.slice(0, 500).map((incident) => incidentRow(incident, true)).join("") : `<tr><td colspan="7" class="empty-table">${query || type !== "all" || camera !== "all" ? "No incidents match these filters." : "No incident records available."}</td></tr>`;
    if (incidents.length > 500) $("allIncidents").insertAdjacentHTML("beforeend", '<tr><td colspan="7" class="empty-table">Showing the latest 500 matching incidents. Export observations for further analysis.</td></tr>');
  }

  function renderAssets(assets) {
    const key = JSON.stringify(assets);
    if ($("assetGrid").dataset.key === key) return;
    $("assetGrid").dataset.key = key;
    state.assets = assets;
    if (!assets.length) { $("assetGrid").innerHTML = `<div class="panel empty-state">${icon("camera")}<strong>No inference evidence is available yet</strong><p>Annotated image or video outputs will appear here when the vision pipeline produces them.</p></div>`; return; }
    $("assetGrid").innerHTML = assets.map((asset, index) => {
      const url = safeMediaUrl(asset.url || asset.path), isVideo = asset.kind === "video" || /^data:video\//.test(url || "") || /\.(mp4|webm|mov)(\?|$)/i.test(url || "");
      return `<article class="asset-card"><button class="asset-visual" data-asset="${index}" aria-label="Enlarge ${escape(asset.title || "evidence")}"${url ? "" : " disabled"}><span class="asset-source">${escape(sourceName(asset.source_type))}</span>${url ? isVideo ? `<video src="${escape(url)}" preload="metadata" muted playsinline aria-label="${escape(asset.title || "Video evidence")}"></video>` : `<img src="${escape(url)}" alt="${escape(asset.title || "Annotated detection evidence")}" loading="lazy">` : '<span class="image-error">No accessible evidence URL.</span>'}</button><div class="asset-details"><h3>${escape(asset.title || "Detection output")}</h3><p>${escape(asset.description || "Inspect the source manifest for details about this output.")}</p>${url ? `<a class="text-button asset-link" href="${escape(url)}" target="_blank" rel="noopener noreferrer">Open original output ${icon("external")}</a>` : ""}</div></article>`;
    }).join("");
    $("assetGrid").querySelectorAll("img").forEach((image) => { image.addEventListener("error", () => { const message = document.createElement("span"); message.className = "image-error"; message.textContent = "This evidence image could not be loaded."; image.replaceWith(message); }); });
  }

  function openAsset(index) {
    const asset = state.assets[index]; if (!asset) return;
    const url = safeMediaUrl(asset.url || asset.path); if (!url) return;
    const isVideo = asset.kind === "video" || /^data:video\//.test(url) || /\.(mp4|webm|mov)(\?|$)/i.test(url);
    setText("evidenceTitle", asset.title || "Evidence");
    setText("evidenceDescription", `${sourceName(asset.source_type)} · ${asset.description || "See the project source manifest for provenance."}`);
    $("evidenceContent").innerHTML = isVideo ? `<video src="${escape(url)}" controls playsinline></video>` : `<img src="${escape(url)}" alt="${escape(asset.title || "Detection evidence")}">`;
    $("evidenceDialog").showModal();
  }

  function renderPerformance(data) {
    const metrics = metricsOf(data), rate = rateOf(metrics), runDuration = runDurationOf(metrics);
    setText("perfRate", finite(rate) ? Number(rate).toFixed(1) : "—");
    setText("perfDuration", duration(runDuration));
    setText("perfInput", fmt(first(metrics.input_rows, metrics.input_records, metrics.numInputRows, metrics.total_input_rows)));
    setText("perfValid", fmt(first(metrics.valid_unique_rows, metrics.unique_observations, metrics.deduplicated_rows, data.summary?.observations, metrics.valid_rows, metrics.output_rows)));
    const facts = [ ["Engine", typeof data.engine === "string" ? data.engine : first(data.engine?.name, metrics.engine, "Not yet reported")], ["Spark version", first(metrics.spark_version, "Not reported")], ["Master", first(metrics.master, data.engine?.master, "Not reported")], ["Mode", first(data.mode, metrics.mode, "Not reported")], ["Partitions", first(metrics.partitions, metrics.shuffle_partitions, "Not reported")], ["Executors", first(metrics.executors, metrics.executor_count, "Not reported")], ["Malformed records", first(metrics.invalid_rows, metrics.rejected_rows, "Not reported")], ["Duplicate deliveries", first(metrics.duplicate_rows, metrics.duplicates_removed, "Not reported")] ];
    $("performanceFacts").innerHTML = facts.map(([key, value]) => `<div><dt>${escape(key)}</dt><dd>${escape(typeof value === "object" ? JSON.stringify(value) : value)}</dd></div>`).join("");
    $("rawMetrics").textContent = JSON.stringify(metrics, null, 2);
    const benchmarkObject = first(data.benchmark, metrics.benchmark, metrics.benchmarks);
    const runs = Array.isArray(benchmarkObject) ? benchmarkObject : asArray(first(benchmarkObject?.runs, benchmarkObject?.results));
    if (!runs.length) { $("benchmarkResults").innerHTML = empty("No benchmark results yet", "Run the project benchmark to compare execution configurations.", "activity"); return; }
    $("benchmarkResults").innerHTML = `<table><thead><tr><th>Configuration</th><th>Records</th><th>Duration</th><th>Rows/s</th></tr></thead><tbody>${runs.map((run) => `<tr><td>${escape(first(run.master, run.configuration, run.name, run.workers ? `${run.workers} workers` : "Run"))}</td><td>${fmt(first(run.rows, run.records, run.input_rows, run.n))}</td><td>${escape(duration(first(run.duration_s, run.elapsed_seconds, run.elapsed_s, run.runtime_seconds, run.seconds)))}</td><td>${finite(rateOf(run)) ? Number(rateOf(run)).toFixed(1) : "—"}</td></tr>`).join("")}</tbody></table><p class="benchmark-note">${escape(benchmarkObject?.note || "Compare measurements with their recorded CPU, memory, data size, and execution settings. Synthetic records test analytics throughput, not detection accuracy.")}</p>`;
  }

  function renderSources(sources) {
    let items = asArray(sources);
    if (!items.length && sources && typeof sources === "object") items = Object.entries(sources).map(([key, value]) => typeof value === "object" && value ? { name: key, ...value } : { name: key, description: String(value) });
    if (!items.length) { $("sourceList").innerHTML = `<div class="panel empty-state compact"><strong>No source manifest loaded</strong><p>Provenance will appear here when provided by the backend.</p></div>`; return; }
    $("sourceList").innerHTML = items.map((source) => {
      if (typeof source === "string") source = { name: source };
      const url = safeUrl(first(source.url, source.source_url, source.homepage, source.repository));
      return `<article class="panel source-card"><span class="source-category">${escape(first(source.type, source.kind, source.source_type, "Project source"))}</span><h3>${escape(first(source.title, source.name, source.id, "Source"))}</h3><p>${escape(first(source.description, source.notes, source.purpose, "No additional description supplied."))}</p>${source.license ? `<span class="source-license">License: ${escape(source.license)}</span>` : ""}${url ? `<a href="${escape(url)}" class="text-button" target="_blank" rel="noopener noreferrer">View source ${icon("external")}</a>` : ""}</article>`;
    }).join("");
  }

  document.addEventListener("click", (event) => {
    const disabledLink = event.target.closest('a[aria-disabled="true"]'); if (disabledLink) { event.preventDefault(); toast(disabledLink.title || "This download is not available yet."); return; }
    const viewButton = event.target.closest("[data-view]"); if (viewButton) { showView(viewButton.dataset.view); return; }
    const actionButton = event.target.closest("[data-action]"); if (actionButton && !actionButton.disabled) { action(actionButton.dataset.action); return; }
    const assetButton = event.target.closest("[data-asset]"); if (assetButton && !assetButton.disabled) openAsset(Number(assetButton.dataset.asset));
  });
  $("menuButton").addEventListener("click", () => { const open = !$("sidebar").classList.contains("open"); $("sidebar").classList.toggle("open", open); $("sidebarScrim").hidden = !open; $("menuButton").setAttribute("aria-expanded", String(open)); });
  $("sidebarScrim").addEventListener("click", closeMenu);
  $("retryButton").addEventListener("click", () => refresh(true));
  $("refreshEvidence").addEventListener("click", () => refresh(true));
  $("incidentSearch").addEventListener("input", renderIncidentLog);
  $("incidentType").addEventListener("change", renderIncidentLog);
  $("incidentCamera").addEventListener("change", renderIncidentLog);
  $("clearFilters").addEventListener("click", () => { $("incidentSearch").value = ""; $("incidentType").value = "all"; $("incidentCamera").value = "all"; renderIncidentLog(); });
  $("closeEvidence").addEventListener("click", () => $("evidenceDialog").close());
  $("evidenceDialog").addEventListener("close", () => { $("evidenceContent").innerHTML = ""; });
  $("evidenceDialog").addEventListener("click", (event) => { if (event.target === $("evidenceDialog")) { const bounds = $("evidenceDialog").getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $("evidenceDialog").close(); } });
  document.addEventListener("keydown", (event) => { if (event.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) { event.preventDefault(); showView("incidents"); $("incidentSearch").focus(); } if (event.key === "Escape") closeMenu(); });
  window.addEventListener("hashchange", () => showView(location.hash.slice(1), false));
  const tick = () => setText("clock", `${time(new Date())} IST`);
  tick(); setInterval(tick, 1000);
  showView(location.hash.slice(1) || "overview", false);
  refresh();
  setInterval(() => { if (!document.hidden) refresh(); }, 3000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
})();
