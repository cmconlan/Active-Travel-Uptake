// Checks the browser model (frontend/model.js) against Python. Run: node tests/frontend_parity.mjs
//  1. parity: JS features on the current network == Python's 2021 features, for every LSOA
//  2. no change -> no predicted uplift
//  3. upgrading every area street never makes a feature worse, and gives a non-negative uplift
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildGraph, computeFeatures, predictUplift, FEATURES } from "../frontend/model.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const scenario = JSON.parse(readFileSync(process.argv[2] ?? join(root, "frontend/data/scenario.json"), "utf8"));
const graph = buildGraph(scenario.network);
const lts = Int8Array.from(scenario.network.lts);
let failures = 0;
const fail = (msg) => { failures += 1; console.error("FAIL", msg); };

// 1. parity (Python rounds features to 4 dp in the CSV)
const base = computeFeatures(scenario, graph, lts);
scenario.lsoas.forEach((z, i) => {
  for (const f of FEATURES) {
    const js = base[i][f], py = z.expected[f];
    if (!(Math.abs(js - py) <= 1e-3)) fail(`${z.code} ${f}: js ${js} vs python ${py}`);
  }
});
console.log(`parity: ${scenario.lsoas.length} LSOAs x ${FEATURES.length} features checked`);

// 2. no change -> no uplift
base.forEach((b, i) => {
  const u = predictUplift(scenario.model, b, b);
  if (u !== 0) fail(`${scenario.lsoas[i].code}: uplift ${u} with no change`);
});

// 3. upgrade everything in the area
const up = Int8Array.from(lts);
for (const s of scenario.streets) up[s.e] = Math.min(up[s.e], scenario.params.upgrade_lts);
const t0 = performance.now();
const scen = computeFeatures(scenario, graph, up);
const ms = performance.now() - t0;
let totalExtra = 0;
scen.forEach((s, i) => {
  const b = base[i], z = scenario.lsoas[i];
  if (s.mean_route_lts > b.mean_route_lts + 1e-9) fail(`${z.code}: mean_route_lts got worse`);
  for (const f of ["share_routes_low_stress", "ls_reachable_pois", "ls_local_km"])
    if (s[f] < b[f] - 1e-9) fail(`${z.code}: ${f} got worse`);
  const u = predictUplift(scenario.model, b, s);
  if (!(u >= 0)) fail(`${z.code}: negative or NaN uplift ${u}`);
  totalExtra += (u / 100) * z.workers;
});
console.log(`upgrade-all scenario: recomputed in ${ms.toFixed(0)} ms, +${totalExtra.toFixed(0)} synthetic cycle commuters`);

if (failures) { console.error(`\n${failures} check(s) failed`); process.exit(1); }
console.log("all checks passed");
