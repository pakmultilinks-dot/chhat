/* Chhat live demo - any address, real AI in the browser.
 * Parser: BiLSTM (parser.js, local). Segmenter: TinyUNet ONNX.
 * Geocode: Nominatim. Imagery: Esri World Imagery. Quote: prices.yaml math.
 */
"use strict";

// ---------- prices (from chhat/quote/prices.yaml, Oct 2026) ----------
const PRICES = {
  panel_550w_pkr: 42000, inverter_per_kw_pkr: 45000,
  mounting_per_kw_pkr: 12000, labor_per_kw_pkr: 8000,
  net_metering_fee_pkr: 25000, retail_rs_per_unit: 47.69,
  export_rs_per_unit: 9.5, annual_kwh_per_kw: 1550,
  self_consumption_ratio: 0.70,
};
const PANEL_W_M = 2.0, PANEL_H_M = 1.0, PANEL_WP = 550, SETBACK_M = 0.3;

// ---------- state ----------
let segSession = null;
let currentResult = null;

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
function setStatus(msg, spin = true) {
  $("status").style.display = "flex";
  $("statusText").textContent = msg;
  $("spinner").style.display = spin ? "block" : "none";
}
function hideStatus() { $("status").style.display = "none"; }
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3200);
}

// ---------- init ----------
async function init() {
  setStatus("Loading AI models...");
  try {
    await loadParser("./");
    $("parseHint").textContent = "AI parser ready. Type any address.";
  } catch (e) {
    console.error("parser load failed", e);
    toast("Parser failed to load: " + e.message);
  }
  try {
    segSession = await ort.InferenceSession.create("./rooftop_seg_v3.onnx");
    console.log("segmenter ready");
  } catch (e) {
    console.error("segmenter load failed", e);
    toast("Roof AI failed to load: " + e.message);
  }
  hideStatus();
}

// ---------- geocode ----------
async function geocode(query) {
  const url = "https://nominatim.openstreetmap.org/search?format=json&limit=1&q="
    + encodeURIComponent(query);
  const r = await fetch(url, { headers: { "Accept": "application/json" } });
  if (!r.ok) throw new Error("geocode http " + r.status);
  const j = await r.json();
  if (!j.length) return null;
  return { lat: parseFloat(j[0].lat), lng: parseFloat(j[0].lon), name: j[0].display_name };
}

// ---------- tiles ----------
function latLngToTile(lat, lng, z) {
  const n = Math.pow(2, z);
  const x = Math.floor((lng + 180) / 360 * n);
  const latR = lat * Math.PI / 180;
  const y = Math.floor((1 - Math.log(Math.tan(latR) + 1 / Math.cos(latR)) / Math.PI) / 2 * n);
  return { x, y };
}
function tileUrl(x, y, z) {
  return `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;
}
function loadImage(url) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = url;
  });
}
// meters per pixel at z19
function mpp(lat, z) { return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, z); }

// ---------- segmenter ----------
async function segmentTile(img) {
  // draw to 256x256 canvas, get RGB bytes
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, 256, 256);
  const d = ctx.getImageData(0, 0, 256, 256).data;
  const inp = new Float32Array(1 * 3 * 256 * 256);
  for (let i = 0; i < 256 * 256; i++) {
    inp[i] = d[i * 4] / 255;
    inp[256 * 256 + i] = d[i * 4 + 1] / 255;
    inp[2 * 256 * 256 + i] = d[i * 4 + 2] / 255;
  }
  const feeds = { image: new ort.Tensor("float32", inp, [1, 3, 256, 256]) };
  const out = await segSession.run(feeds);
  const mask = out.mask.data; // 1x1x256x256 logits
  const prob = new Float32Array(256 * 256);
  let fg = 0, confSum = 0;
  for (let i = 0; i < 256 * 256; i++) {
    const p = 1 / (1 + Math.exp(-mask[i]));
    prob[i] = p;
    if (p >= 0.5) { fg++; confSum += p; }
  }
  const frac = fg / (256 * 256);
  const conf = fg ? confSum / fg : 0;
  return { prob, bin: null, frac, conf, ok: conf >= 0.55 && fg > 0 && frac <= 0.75 };
}

// ---------- quote ----------
function computeQuote(panelCount) {
  const kw = panelCount * PANEL_WP / 1000;
  const annual_kwh = kw * PRICES.annual_kwh_per_kw;
  const self_kwh = annual_kwh * PRICES.self_consumption_ratio;
  const export_kwh = annual_kwh * (1 - PRICES.self_consumption_ratio);
  const annual_savings = self_kwh * PRICES.retail_rs_per_unit + export_kwh * PRICES.export_rs_per_unit;
  const cost = panelCount * PRICES.panel_550w_pkr
    + kw * (PRICES.inverter_per_kw_pkr + PRICES.mounting_per_kw_pkr + PRICES.labor_per_kw_pkr)
    + PRICES.net_metering_fee_pkr;
  const payback = annual_savings > 0 ? cost / annual_savings : Infinity;
  return {
    kw: +kw.toFixed(2), panels: panelCount,
    annual_kwh: Math.round(annual_kwh),
    annual_savings: Math.round(annual_savings),
    cost: Math.round(cost),
    payback: +payback.toFixed(1),
  };
}
function chhatScore(usableM2, kw, payback) {
  const area = Math.min(usableM2 / 120, 1) * 100;
  const fit = Math.min(kw / 5, 1) * 100;
  const pay = Math.max(0, Math.min(1, (8 - payback) / 6)) * 100;
  // shade unknown from single mask: neutral 70
  return Math.round(area * 0.35 + 70 * 0.2 + fit * 0.25 + pay * 0.2);
}

// ---------- panel layout (port of optimizer) ----------
function layoutPanels(usableM2) {
  const w = Math.sqrt(usableM2 * 4 / 3), h = usableM2 / w;
  const cols = Math.max(Math.floor((w - SETBACK_M) / (PANEL_W_M + 0.05)), 0);
  const rows = Math.max(Math.floor((h - SETBACK_M) / (PANEL_H_M + 0.05)), 0);
  const maxPanels = Math.floor(10 * 1000 / PANEL_WP);
  const panels = [];
  for (let r = 0; r < rows && panels.length < maxPanels; r++)
    for (let c = 0; c < cols && panels.length < maxPanels; c++)
      panels.push({ x: c * (PANEL_W_M + 0.05) + SETBACK_M / 2, y: r * (PANEL_H_M + 0.05) + SETBACK_M / 2 });
  return { panels, w, h };
}

// ---------- main flow ----------
async function analyze() {
  const text = $("addr").value.trim();
  if (!text) { toast("Type an address first."); return; }
  if (!segSession) { toast("Roof AI still loading, one moment."); return; }

  try {
    // 1. parse (local BiLSTM)
    setStatus("AI is reading your address...");
    const p = parseAddress(text);
    const bits = [p.house, p.street, p.block, p.colony, p.city].filter(Boolean);
    const query = bits.length ? bits.join(", ") : text;
    renderParsed(p);

    // 2. geocode
    setStatus("Finding it on the map...");
    const g = await geocode(query + ", Pakistan");
    if (!g) {
      // fallback: try colony + city
      const g2 = p.colony ? await geocode(p.colony + ", " + (p.city || "Lahore") + ", Pakistan") : null;
      if (!g2) { hideStatus(); toast("Could not find that place. Try a nearby known area."); return; }
      Object.assign(g || {}, g2);
    }

    // 3. satellite tile
    setStatus("Fetching satellite view...");
    const Z = 19;
    const t = latLngToTile(g.lat, g.lng, Z);
    const img = await loadImage(tileUrl(t.x, t.y, Z));
    const ppm = mpp(g.lat, Z); // meters per pixel
    drawSatellite(img);

    // 4. roof AI
    setStatus("AI is finding the roof...");
    const seg = await segmentTile(img);
    if (!seg.ok) {
      hideStatus();
      drawMask(img, seg.prob, false);
      $("roofMsg").style.display = "block";
      $("roofMsg").innerHTML = "<b>AI is not confident on this view</b> (confidence "
        + Math.round(seg.conf * 100) + "%). It may be a non-residential area, heavy shade, or an unclear tile. Try a nearby house address.";
      $("quoteCard").style.display = "none";
      return;
    }
    $("roofMsg").style.display = "none";

    // 5. area -> layout -> quote
    const tileM = 256 * ppm;
    const roofM2 = seg.frac * tileM * tileM;
    const usable = Math.min(roofM2 * 0.8, 250); // 80% usable, cap 250
    const lay = layoutPanels(usable);
    const q = computeQuote(lay.panels.length);
    const score = chhatScore(usable, q.kw, q.payback);
    currentResult = { p, g, q, score, seg, lay, usable, conf: seg.conf };

    drawMask(img, seg.prob, true);
    drawPanels(lay);
    renderQuote(q, score, usable, seg.conf);
    hideStatus();
  } catch (e) {
    console.error(e);
    hideStatus();
    toast("Something failed: " + e.message);
  }
}

function renderParsed(p) {
  const rows = [["House", p.house], ["Street", p.street], ["Block", p.block], ["Area", p.colony], ["City", p.city]]
    .filter(([, v]) => v);
  $("parsedFields").innerHTML = rows.map(([k, v]) =>
    `<div class="frow"><span class="fk">${k}</span><span class="fv">${escapeHtml(v)}</span></div>`).join("");
  $("parsedCard").style.display = "block";
}
function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

function drawSatellite(img) {
  const c = $("satCanvas"), ctx = c.getContext("2d");
  c.width = c.height = 512;
  ctx.drawImage(img, 0, 0, 512, 512);
  $("mapCard").style.display = "block";
}
function drawMask(img, prob, ok) {
  const c = $("satCanvas"), ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, 512, 512);
  if (ok) {
    // upscale 256 mask to 512 overlay
    const mc = document.createElement("canvas");
    mc.width = mc.height = 256;
    const mctx = mc.getContext("2d");
    const id = mctx.createImageData(256, 256);
    for (let i = 0; i < 256 * 256; i++) {
      if (prob[i] >= 0.5) {
        id.data[i * 4] = 16; id.data[i * 4 + 1] = 185; id.data[i * 4 + 2] = 129;
        id.data[i * 4 + 3] = Math.round(120 + prob[i] * 80);
      }
    }
    mctx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(mc, 0, 0, 512, 512);
  }
  // toggle button
  $("maskToggle").style.display = "inline-block";
  $("maskToggle").onclick = () => {
    const showing = $("maskToggle").dataset.on === "1";
    if (showing) { drawSatellite(img); $("maskToggle").dataset.on = "0"; $("maskToggle").textContent = "Show AI mask"; }
    else { drawMask(img, prob, ok); $("maskToggle").dataset.on = "1"; $("maskToggle").textContent = "Hide AI mask"; }
  };
  $("maskToggle").dataset.on = "1";
  $("maskToggle").textContent = "Hide AI mask";
}
function drawPanels(lay) {
  // draw panel rects scaled into the roof bbox area (center of tile)
  const c = $("satCanvas"), ctx = c.getContext("2d");
  const scale = 512 / Math.max(lay.w, lay.h, 1) * 0.55;
  const ox = 256 - (lay.w * scale) / 2, oy = 256 - (lay.h * scale) / 2;
  ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(251,191,36,0.25)";
  for (const pn of lay.panels) {
    const x = ox + pn.x * scale, y = oy + pn.y * scale;
    const w = PANEL_W_M * scale, h = PANEL_H_M * scale;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }
}
function renderQuote(q, score, usable, conf) {
  $("quoteCard").style.display = "block";
  $("qPanels").textContent = q.panels;
  $("qKw").textContent = q.kw + " kW";
  $("qCost").textContent = "Rs " + q.cost.toLocaleString("en-PK");
  $("qSave").textContent = "Rs " + q.annual_savings.toLocaleString("en-PK") + "/yr";
  $("qPayback").textContent = q.payback + " yrs";
  $("qArea").textContent = Math.round(usable) + " m²";
  $("qConf").textContent = Math.round(conf * 100) + "%";
  const ring = $("scoreRing");
  ring.textContent = score;
  ring.style.background = `conic-gradient(${score >= 70 ? "#10b981" : score >= 40 ? "#fbbf24" : "#ef4444"} ${score * 3.6}deg, #1e293b 0deg)`;
  // panel slider
  const sl = $("panelSlider");
  sl.max = Math.min(q.panels + 6, 18);
  sl.value = q.panels;
  $("panelCountLbl").textContent = q.panels;
  sl.oninput = () => {
    const n = +sl.value;
    $("panelCountLbl").textContent = n;
    const qq = computeQuote(n);
    $("qPanels").textContent = qq.panels;
    $("qKw").textContent = qq.kw + " kW";
    $("qCost").textContent = "Rs " + qq.cost.toLocaleString("en-PK");
    $("qSave").textContent = "Rs " + qq.annual_savings.toLocaleString("en-PK") + "/yr";
    $("qPayback").textContent = qq.payback + " yrs";
  };
}

document.addEventListener("DOMContentLoaded", () => {
  $("analyzeBtn").addEventListener("click", analyze);
  $("addr").addEventListener("keydown", (e) => { if (e.key === "Enter") analyze(); });
  init();
});
