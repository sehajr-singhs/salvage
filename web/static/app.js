/* SALVAGE demo surface (brief 06). Render-only: scan upload, cached sample
   device, reverse search. Hard cuts, no fades. All model work happens on
   the server; the sample button renders from cache alone. */

"use strict";

const $ = (id) => document.getElementById(id);
const viewer = $("viewer");
const hint = $("viewer-hint");
const img = $("overlay-img");
const svg = $("label-svg");

let currentParts = [];

function setHint(text) {
  hint.textContent = text;
  hint.style.display = "flex";
  img.hidden = true;
  clearCallouts();
  svg.innerHTML = "";
}

function clearCallouts() {
  document.querySelectorAll(".callout").forEach((el) => el.remove());
}

function resolveUrl(url, data) {
  if (!url) return null;
  if (url.startsWith("/")) return url;
  const base = data.overlay && data.photo ? "/static/samples/" : "/data/";
  return base + url;
}

function pct(value, size) {
  return size ? (value / size) * 100 : 0;
}

img.onload = () => {
  hint.style.display = "none";
  img.hidden = false;
  svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
  for (const p of currentParts) drawCallout(p);
};
img.onerror = () => setHint("OVERLAY IMAGE MISSING FROM CACHE");

function render(data) {
  const overlayUrl = resolveUrl(data.overlay_url || data.overlay, data);
  if (!overlayUrl) {
    setHint((data.message || "NOTHING TO SHOW").toUpperCase());
    return;
  }
  currentParts = data.parts || [];
  renderParts(currentParts);
  const deviceName = data.device && data.device.name ? data.device.name : null;
  $("mode-line").textContent = deviceName
    ? [deviceName, data.device.source, data.note].filter(Boolean).join(" · ")
    : data.readings_mode === "offline mock reader"
      ? "OFFLINE MOCK READER: NO MODEL KEYS ON THIS MACHINE — EVERY VERDICT STAYS CAUTIOUS"
      : (data.message || "");
  img.src = "";
  img.src = overlayUrl;
}

function renderParts(parts) {
  const list = $("part-list");
  list.innerHTML = "";
  $("part-count").textContent = parts.length ? `${parts.length} PARTS` : "";

  for (const p of parts) {
    const li = document.createElement("li");
    li.className = "part";

    const id = document.createElement("span");
    id.className = "part-id";
    id.textContent = `M${p.n || p.id}`;
    li.appendChild(id);

    const name = document.createElement("span");
    name.className = "part-name";
    name.textContent = p.name || p.label || p.id;
    li.appendChild(name);

    const chip = document.createElement("span");
    chip.className = `chip ${p.state}`;
    chip.textContent = p.state;
    li.appendChild(chip);

    if (p.state === "NEEDS_MEASUREMENT" && p.requested_measurement) {
      const ask = document.createElement("span");
      ask.className = "ask";
      ask.textContent = p.requested_measurement;
      li.appendChild(ask);
    }

    const reasons = document.createElement("span");
    reasons.className = "reasons";
    reasons.textContent = (p.reasons || []).slice(0, 3).join(" · ");
    li.appendChild(reasons);

    list.appendChild(li);
  }
}

function drawCallout(p) {
  // Engineering-style callout: a 1 px leader line from the mask edge to a
  // `M1 · BLDC 2205 · VERIFIED` label. The overlay PNG already carries the
  // 1 px white mask outlines; the SVG only adds the leader and text.
  const bbox = p.bbox;
  if (!bbox || !img.naturalWidth) return;
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const [x1, y1, x2, y2] = bbox;
  const ym = (y1 + y2) / 2;
  const rightSide = pct(x2, w) < 62;

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x2.toFixed(1));
  line.setAttribute("y1", ym.toFixed(1));
  line.setAttribute("x2", (rightSide ? x2 + 26 : x1 - 26).toFixed(1));
  line.setAttribute("y2", ym.toFixed(1));
  line.setAttribute("stroke", p.state === "HAZARD" ? "var(--hazard)" : "#FFFFFF");
  line.setAttribute("stroke-width", "1");
  svg.appendChild(line);

  const label = document.createElement("span");
  label.className = "callout" + (p.state === "HAZARD" ? " hazard" : "");
  label.textContent = `M${p.n || ""} · ${p.name || p.label || p.id} · ${p.state}`;
  if (rightSide) {
    label.style.left = `calc(${pct(x2, w)}% + 28px)`;
  } else {
    label.style.right = `calc(${100 - pct(x1, w)}% + 28px)`;
  }
  label.style.top = `${pct(ym, h)}%`;
  label.style.transform = "translateY(-50%)";
  svg.insertAdjacentElement("afterend", label);
}

/* ---- actions: hard cuts only -------------------------------------------- */

$("sample-btn").addEventListener("click", async () => {
  setHint("LOADING SAMPLE…");
  try {
    const response = await fetch("/api/sample");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (err) {
    setHint(`SAMPLE FAILED: ${err.message}`);
  }
});

$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  setHint("SCANNING…");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const response = await fetch("/api/scan", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      setHint(`SCAN REJECTED: ${data.detail || response.status}`);
      return;
    }
    if (!data.parts || data.parts.length === 0) {
      setHint((data.message || "NO PARTS FOUND").toUpperCase());
      return;
    }
    render(data);
  } catch (err) {
    setHint(`SCAN FAILED: ${err.message}`);
  } finally {
    event.target.value = "";
  }
});

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("search-input").value.trim();
  const list = $("results");
  list.innerHTML = "";
  if (!text) return;
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const matches = await response.json();
    if (!Array.isArray(matches) || matches.length === 0) {
      addEmpty(list, "NOTHING MATCHED — NAME THE PART KIND (MOTOR, CELL, FAN…)");
      return;
    }
    for (const m of matches.slice(0, 12)) {
      const li = document.createElement("li");
      li.className = "result";

      const type = document.createElement("span");
      type.className = "result-topic";
      type.textContent = (m.part_type || m.label || "").toUpperCase();
      li.appendChild(type);

      const device = document.createElement("span");
      device.className = "result-device";
      device.textContent = (m.device || "").toUpperCase();
      li.appendChild(device);

      if (m.state) {
        const chip = document.createElement("span");
        chip.className = `chip ${m.state}`;
        chip.textContent = m.state;
        li.appendChild(chip);
      }

      const where = document.createElement("span");
      where.className = "where";
      const bits = [
        m.source === "inventory"
          ? `OPEN ${m.device} · PULL PART ${m.part_id}`
          : `LOOK INSIDE: ${m.device}`,
      ];
      const specs = [];
      if (m.voltage) specs.push(`${m.voltage} V`);
      if (m.current) specs.push(`${m.current} A`);
      if (m.interface) specs.push(m.interface);
      if (specs.length) bits.push(specs.join(" / "));
      if (m.requested_measurement) bits.push(`METER: ${m.requested_measurement}`);
      where.textContent = bits.join(" · ");
      li.appendChild(where);

      list.appendChild(li);
    }
  } catch (err) {
    addEmpty(list, `SEARCH FAILED: ${err.message}`);
  }
});

function addEmpty(list, text) {
  const empty = document.createElement("li");
  empty.className = "empty";
  empty.textContent = text;
  list.appendChild(empty);
}
