// Feature and prediction logic for the scenario map. Mirrors pipeline/features.py and
// pipeline/model_spec.py; tests/frontend_parity.mjs checks it reproduces Python's 2021 features.
// Plain ES module with no dependencies, so it runs in the browser and in Node.

const FEATURES = ["mean_route_lts", "share_routes_low_stress", "ls_reachable_pois", "ls_local_km"];

// --- graph ---------------------------------------------------------------------------
export function buildGraph(network) {
  const nNodes = 1 + Math.max(...network.u, ...network.v);
  const adj = Array.from({ length: nNodes }, () => []);
  network.u.forEach((u, e) => {
    const v = network.v[e];
    if (u === v) return; // self-loops never carry a route
    adj[u].push(v, e);
    adj[v].push(u, e);
  });
  return { adj, len: Float64Array.from(network.len), nNodes };
}

// Minimal binary heap of [distance, node]
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(d, n) {
    const a = this.a; a.push([d, n]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]]; i = p;
    }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]]; i = m;
      }
    }
    return top;
  }
}

/** Distances from `source` over low-stress edges only, up to `cutoff` metres. */
function lowStressDijkstra(graph, isLow, source, cutoff) {
  const dist = new Map([[source, 0]]);
  const heap = new Heap();
  heap.push(0, source);
  const done = new Set();
  while (heap.size) {
    const [d, n] = heap.pop();
    if (done.has(n)) continue;
    done.add(n);
    const nb = graph.adj[n];
    for (let i = 0; i < nb.length; i += 2) {
      const e = nb[i + 1];
      if (!isLow[e]) continue;
      const nd = d + graph.len[e];
      if (nd > cutoff) continue;
      const m = nb[i];
      if (!dist.has(m) || nd < dist.get(m)) { dist.set(m, nd); heap.push(nd, m); }
    }
  }
  return dist;
}

// --- features ------------------------------------------------------------------------
/**
 * The model's four network features for every LSOA, given an LTS per edge.
 * Returns [{mean_route_lts, share_routes_low_stress, ls_reachable_pois, ls_local_km}, ...]
 */
export function computeFeatures(scenario, graph, lts) {
  const { params, dests, lsoas } = scenario;
  const lsMax = params.low_stress_max_lts;
  const isLow = Uint8Array.from(lts, (l) => (l <= lsMax ? 1 : 0));

  // A node is on the low-stress network if it touches a low-stress (non-loop) edge
  const onLow = new Uint8Array(graph.nNodes);
  scenario.network.u.forEach((u, e) => {
    const v = scenario.network.v[e];
    if (isLow[e] && u !== v) { onLow[u] = 1; onLow[v] = 1; }
  });
  const snap = (cands) => cands.find(([n]) => onLow[n]) ?? null; // nearest node on low-stress network
  const destSnap = dests.map((d) => snap(d.cand));

  return lsoas.map((z) => {
    // Fixed routes, re-scored
    let sumMean = 0, nLow = 0;
    for (const r of z.routes) {
      let w = 0, wl = 0, mx = 0;
      for (const e of r.e) {
        const l = graph.len[e];
        w += l; wl += l * lts[e];
        if (lts[e] > mx) mx = lts[e];
      }
      sumMean += wl / w;
      if (mx <= lsMax) nLow += 1;
    }
    const nr = z.routes.length;

    // Low-stress reachability within each category's cap
    let reach = 0;
    const o = snap(z.origin_cand);
    if (o) {
      const dist = lowStressDijkstra(graph, isLow, o[0], params.max_cap_m);
      dests.forEach((d, i) => {
        const s = destSnap[i];
        if (s && dist.has(s[0]) && o[1] + dist.get(s[0]) + s[1] <= params.caps[d.cat]) reach += 1;
      });
    }

    // Local low-stress km
    let localM = 0;
    for (const e of z.local) if (isLow[e]) localM += graph.len[e];

    return {
      mean_route_lts: nr ? sumMean / nr : NaN,
      share_routes_low_stress: nr ? nLow / nr : NaN,
      ls_reachable_pois: reach,
      ls_local_km: localM / 1000,
    };
  });
}

// --- prediction ----------------------------------------------------------------------
/**
 * Predicted uplift in cycle-to-work share (percentage points) from `base` -> `scen` features.
 * Only the change terms move in a scenario; level controls cancel out. The current low-stress
 * reachability plays the role of the baseline in the interaction. Improvements are never allowed
 * to reduce predicted cycling (the reachability effect and the total are floored at zero).
 */
export function predictUplift(model, base, scen) {
  const b = model.coefficients, sc = model.scaler, it = model.interaction;
  const zBase = (base[it.change.replace(/^d_/, "")] - sc[it.base].mean) / sc[it.base].sd;
  let pp = 0;
  for (const t of model.change_terms) {
    const f = t.replace(/^d_/, ""); // model term "d_x" is the change in feature "x"
    const d = (scen[f] - base[f]) / sc[t].sd;
    pp += t === it.change ? Math.max(0, b[t] + b[it.term] * zBase) * d : b[t] * d;
  }
  return Math.max(0, pp);
}

export { FEATURES };
