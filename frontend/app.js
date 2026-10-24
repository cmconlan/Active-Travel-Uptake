// Central Brighton cycle-uptake scenario map. All data comes from data/scenario.json
// (pipeline/export_frontend.py); the feature + prediction logic lives in model.js.
import { buildGraph, computeFeatures, predictUplift } from "./model.js";

const ZONE_COLOURS = ["#f3f1e8", "#cfe8c6", "#7cc4a4", "#2f9a9a", "#1f5f86"];
const LTS_COLOURS = { 1: "#1a9850", 2: "#91cf60", 3: "#fc8d59", 4: "#d73027" };
const PROPOSED = "#5b2a86";
const MIN_SCALE_MAX = 5;

const state = { scenario: null, graph: null, lts0: null, base: null, current: null, uplift: [], proposed: new Set() };
const zoneLayers = [];
const streetLayers = new Map(); // edge index -> polyline

// --- colour helpers -------------------------------------------------------------------
function lerpHex(a, b, t) {
  const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16));
  const pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
  return "#" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, "0")).join("");
}
function zoneColour(t) {
  const x = Math.min(Math.max(t, 0), 1) * (ZONE_COLOURS.length - 1);
  const i = Math.min(Math.floor(x), ZONE_COLOURS.length - 2);
  return lerpHex(ZONE_COLOURS[i], ZONE_COLOURS[i + 1], x - i);
}

// --- model ----------------------------------------------------------------------------
function scenarioLts() {
  const up = Int8Array.from(state.lts0);
  const target = state.scenario.params.upgrade_lts;
  for (const e of state.proposed) up[e] = Math.min(up[e], target);
  return up;
}

function recompute() {
  const s = state.scenario;
  state.current = computeFeatures(s, state.graph, scenarioLts());
  state.uplift = s.lsoas.map((z, i) => predictUplift(s.model, state.base[i], state.current[i]));
  render();
}

// --- rendering ------------------------------------------------------------------------
const fmt = (x, d = 0) => (Number.isFinite(x) ? x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }) : "–");

function render() {
  const s = state.scenario;
  const extra = s.lsoas.map((z, i) => (state.uplift[i] / 100) * z.workers);
  const max = Math.max(MIN_SCALE_MAX, ...extra);
  zoneLayers.forEach((layer, i) => layer.setStyle({ fillColor: zoneColour(extra[i] / max) }));

  for (const [e, layer] of streetLayers) {
    const on = state.proposed.has(e);
    layer.setStyle({ color: on ? PROPOSED : LTS_COLOURS[state.lts0[e]], weight: on ? 6 : state.lts0[e] >= 3 ? 3 : 2 });
  }

  let km = 0;
  for (const e of state.proposed) km += state.graph.len[e] / 1000;
  document.getElementById("stat-km").textContent = fmt(km, 1);
  document.getElementById("stat-cyclists").textContent = "+" + fmt(extra.reduce((a, b) => a + b, 0));
  document.getElementById("legend-max").textContent = fmt(max);
  document.querySelectorAll("#preset-buttons button").forEach((b) => {
    const edges = JSON.parse(b.dataset.edges);
    b.setAttribute("aria-pressed", String(edges.every((e) => state.proposed.has(e))));
  });
  if (state.hovered != null) showZone(state.hovered);
}

// Ray-casting point-in-polygon over each zone's rings (geometry is [[ring, ...holes], ...] in lat/lon)
function inRing(lat, lng, ring) {
  let inside = false;
  for (let a = 0, b = ring.length - 1; a < ring.length; b = a++) {
    const [ya, xa] = ring[a], [yb, xb] = ring[b];
    if ((ya > lat) !== (yb > lat) && lng < ((xb - xa) * (lat - ya)) / (yb - ya) + xa) inside = !inside;
  }
  return inside;
}
function zoneAt(lat, lng) {
  return state.scenario.lsoas.findIndex((z) =>
    z.geometry.some(([outer, ...holes]) => inRing(lat, lng, outer) && !holes.some((h) => inRing(lat, lng, h))));
}

function showZone(i) {
  if (state.hovered != null && state.hovered !== i) zoneLayers[state.hovered].setStyle({ color: "#ffffff", weight: 1.2 });
  zoneLayers[i].setStyle({ color: "#1d2321", weight: 2.5 });
  state.hovered = i;
  const z = state.scenario.lsoas[i], b = state.base[i], c = state.current[i];
  const share = (z.cyclists / z.workers) * 100;
  const up = state.uplift[i];
  const row = (label, f, d, cls = "") => `<tr class="${cls}"><td>${label}</td><td>${fmt(b[f], d)}</td><td>${fmt(c[f], d)}</td></tr>`;
  document.getElementById("zone-info").innerHTML = `
    <strong>${z.name}</strong> · ${z.code}<br>
    ${fmt(z.workers)} working residents, ${fmt(share, 1)}% cycle to work <em>(synthetic, ${state.scenario.meta.census_year})</em>
    <table>
      <tr><td></td><td>now</td><td>with lanes</td></tr>
      ${row("Mean route stress (LTS)", "mean_route_lts", 2)}
      <tr><td>Routes with no LTS 3–4</td><td>${fmt(b.share_routes_low_stress * 100)}%</td><td>${fmt(c.share_routes_low_stress * 100)}%</td></tr>
      ${row("Places reachable on calm streets", "ls_reachable_pois", 0)}
      ${row("Calm-street km within 1 km", "ls_local_km", 1)}
      <tr class="headline"><td>Cycle-to-work share</td><td>${fmt(share, 1)}%</td><td>${fmt(share + up, 1)}%</td></tr>
      <tr class="headline"><td>Extra cycle commuters</td><td></td><td>+${fmt((up / 100) * z.workers, 1)}</td></tr>
    </table>`;
}

function toggle(edges) {
  const allOn = edges.every((e) => state.proposed.has(e));
  for (const e of edges) allOn ? state.proposed.delete(e) : state.proposed.add(e);
  recompute();
}

// --- setup ----------------------------------------------------------------------------
function buildMap(s) {
  const renderer = L.canvas({ tolerance: 8 });
  const map = L.map("map", { renderer, preferCanvas: true, zoomSnap: 0.25 });
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19, subdomains: "abcd",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
  }).addTo(map);

  // Zones are not interactive: dense street lines sit on top and would swallow their hover
  // events, so the hovered zone is found by point-in-polygon on mousemove instead.
  s.lsoas.forEach((z) => {
    const layer = L.polygon(z.geometry, { renderer, color: "#ffffff", weight: 1.2, fillColor: ZONE_COLOURS[0], fillOpacity: 0.6, interactive: false }).addTo(map);
    zoneLayers.push(layer);
  });
  map.on("mousemove", (ev) => {
    const i = zoneAt(ev.latlng.lat, ev.latlng.lng);
    if (i !== -1 && i !== state.hovered) showZone(i);
  });

  const target = s.params.upgrade_lts;
  for (const st of s.streets) {
    const lts = state.lts0[st.e];
    const clickable = lts > target;
    const layer = L.polyline(st.coords, {
      renderer, color: LTS_COLOURS[lts], weight: lts >= 3 ? 3 : 2, opacity: 0.95, lineCap: "round", interactive: clickable,
    }).addTo(map);
    if (clickable) {
      layer.bindTooltip(() => `${st.name || "Unnamed street"} · LTS ${lts}` +
        (state.proposed.has(st.e) ? " · click to remove lane" : " · click to add protected lane"), { sticky: true });
      layer.on("click", () => toggle([st.e]));
      streetLayers.set(st.e, layer);
    }
  }
  const [x0, y0, x1, y1] = s.meta.bbox;
  map.fitBounds([[y0, x0], [y1, x1]]);
}

function buildPresets(s) {
  if (!s.presets.length) return;
  const box = document.getElementById("preset-buttons");
  for (const p of s.presets) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = p.name;
    b.dataset.edges = JSON.stringify(p.edges);
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => toggle(p.edges));
    box.append(b);
  }
  document.getElementById("presets").hidden = false;
}

async function main() {
  const s = await fetch("data/scenario.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  state.scenario = s;
  state.graph = buildGraph(s.network);
  state.lts0 = Int8Array.from(s.network.lts);
  state.base = computeFeatures(s, state.graph, state.lts0);
  state.current = state.base;
  state.uplift = s.lsoas.map(() => 0);

  document.title = `${s.meta.area} Cycle Scenarios`;
  document.getElementById("title").textContent = `${s.meta.area} cycle uptake`;
  document.getElementById("area-desc").textContent = s.meta.area_description;
  document.getElementById("disclaimer-text").textContent = s.meta.disclaimer.replace(/^Synthetic data\.\s*/, "");
  document.getElementById("model-fit").textContent = `R² ${s.model.r2.toFixed(2)}, cross-validated ${s.model.cv_r2.toFixed(2)}, n = ${s.model.n} LSOAs`;
  const repo = document.getElementById("repo-link");
  if (s.meta.repo_url) repo.href = s.meta.repo_url; else repo.remove();
  document.getElementById("legend-bar").style.background = `linear-gradient(90deg, ${ZONE_COLOURS.join(",")})`;
  document.getElementById("reset").addEventListener("click", () => { state.proposed.clear(); recompute(); });

  buildMap(s);
  buildPresets(s);
  render();
}

main().catch((err) => {
  console.error(err);
  document.getElementById("zone-info").textContent =
    "Could not load data/scenario.json. Run the pipeline, then serve this folder over HTTP (python -m http.server) rather than opening the file directly.";
});
