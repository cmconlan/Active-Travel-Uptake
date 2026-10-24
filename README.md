# Brighton cycle scenarios

**Add protected cycle lanes to central Brighton's streets and see where more people might cycle to work.**

An end-to-end urban analytics demo, built on real geography and real OpenStreetMap data:

1. Pull Brighton and Hove's LSOAs and its OSM cycle network as it stood in 2015 and in 2021.
2. Rate every street's Level of Traffic Stress (LTS 1–4) from its OSM tags.
3. Measure, for each LSOA, how traffic stress changed on routes to stations, shops, schools and the centre, and on the calm-street network around it.
4. Fit a first-difference model of change in cycle-to-work share on those changes.
5. Serve an interactive map where you upgrade streets and the browser recomputes everything live.

> ⚠️ **The commuting data is synthetic.** Steps 1–3 use real data. The census counts in step 4 are generated from planted effects, so the model has a known answer to recover. They do not describe real travel behaviour in Brighton and Hove, and the map says so prominently. A real study would swap in Census tables QS701EW (2011) and TS061 (2021) for `pipeline/synthetic_census.py`.

```
pipeline/
  fetch_lsoas.py       1  ONS LSOA 2021 boundaries + population-weighted centroids
  fetch_osm.py         1  OSM network + POIs at two dates (Overpass attic queries, with backoff)
  inspect_raw.py       1  sanity map of the downloads
  lts.py               2  Level of Traffic Stress per edge, 2015 LTS carried onto the 2021 network
  features.py          3  route-based LSOA features, 2015 vs 2021
  synthetic_census.py  4  SYNTHETIC 2011/2021 cycle-to-work counts from planted effects
  model_spec.py        4  the model's terms, shared by generator, model and front end
  train_model.py       4  OLS + robust SEs + CV, coefficient recovery report
  export_frontend.py   5  one compact scenario.json for the map
  run_pipeline.py         stages 2–5 in one go (add --fetch for stage 1)
frontend/              5  static Leaflet page; model.js mirrors the Python features
tests/                    LTS rules, model spec, JS/Python parity
```

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline/run_pipeline.py --fetch     # first time (the OSM pulls take a while)
python pipeline/run_pipeline.py             # afterwards
python -m pytest tests/ && node tests/frontend_parity.mjs
cd frontend && python -m http.server 8000   # open http://localhost:8000
```

Each stage can also be run on its own, as below. Settings live in `pipeline/config.py`.

### Stage 1: data acquisition (real geography, real OSM)

```bash
python pipeline/fetch_lsoas.py    # 2021 LSOAs for Brighton and Hove from the ONS Open Geography Portal
python pipeline/fetch_osm.py      # OSM cycle network + POIs as of 2015-01-01 and 2021-03-21 (Census day)
python pipeline/inspect_raw.py    # summary + data/raw/inspect_raw.png (2015 vs 2021 side by side)
```

| Output (`data/raw/`) | Contents |
|---|---|
| `lsoa_2021.gpkg` | LSOA 2021 boundaries (generalised, BGC), EPSG:27700 |
| `study_area.gpkg` | LA boundary plus a 1 km buffer, used to clip OSM pulls so edge LSOAs aren't truncated |
| `osm_network_<year>.graphml` / `.gpkg` | osmnx `bike` network with cycle-infrastructure tags kept |
| `pois_<year>.gpkg` | Bus stops, rail stations/halts, grocery (supermarket, convenience, greengrocer), schools, one point each |

Historical OSM comes from Overpass [attic queries](https://wiki.openstreetmap.org/wiki/Attic_data): the `[date:...]` setting returns OSM as it stood on that date. Each pull takes a few minutes and is cached in `./cache`. Rerunning skips existing outputs unless you pass `--force`.

**Caveat:** a change in OSM between 2015 and 2021 mixes real infrastructure change with mapping effort, since contributors added and retagged existing lanes too. That is fine for a demo, but it is worth saying out loud.

### Stage 2: Level of Traffic Stress per edge

```bash
python pipeline/lts.py            # -> data/processed/edges_lts.gpkg + lts_map.png
python -m pytest tests/           # the LTS rules, as test cases
```

Every edge gets a Level of Traffic Stress, from 1 (comfortable for most people, including children) to 4 (main roads in traffic, for confident cyclists only). The rules adapt Mekuria, Furth & Nixon (2012) to UK speeds and OSM tags:

| LTS | Rule (first match wins) |
|---|---|
| 1 | Cycleway; path, footway or pedestrian street with cycling designated/allowed; living street; separated track (`cycleway*=track`) |
| 2 | Painted lane on a non-main road under 40 mph; service road; local street at 20 mph (configurable: `LOCAL_20MPH_LTS`); path without cycling designation |
| 3 | Painted lane on a main road or at 40 mph; shared bus lane; tertiary road; local street at 30 mph |
| 4 | Main road (primary, secondary, trunk) with no provision; mixed traffic at 40+ mph or on 4+ lanes; painted lane at 50+ mph |

Choices that keep the 2015 to 2021 change honest:

- **Worst case on merged edges.** osmnx simplification can merge ways with different tags, so we take the busiest road class, the highest speed and the weakest provision.
- **Sides.** A two-way road takes the worse side, while a one-way road takes the side carrying traffic.
- **Missing speeds.** These are filled from the 2021 tag for the same OSM way where one exists, otherwise from a road-class default, using the same rule in both years. A 20 mph limit that was tagged in 2019 doesn't read as a 2015 to 2021 improvement.
- **2015 LTS on the 2021 network.** Each 2021 edge gets its 2015 LTS (`lts_2015`) by matching on the same node pair, or else on the nearest 2015 edge within 10 m. Edges with no match are flagged `not_in_earlier`. This lets features use one fixed set of routes scored in both years.

### Stage 3: route-based LSOA features

```bash
python pipeline/features.py       # -> data/processed/lsoa_features.csv/.gpkg, routes.csv, features_map.png
```

From each LSOA's population-weighted centroid, we find the shortest routes to:

- the nearest 3 rail stations (within 5 km)
- the nearest 3 grocery shops (within 3 km)
- the nearest 3 schools (within 3 km)
- the city centre (Clock Tower)

Bus stops aren't routed. They enter as a control, because good bus access competes with cycling.

**One network, two scorings.** Routes are found once on the 2021 network. They are then scored with the 2015 LTS and with the 2021 LTS, using the 2021 POI set for both years. So a change in a feature means a change in traffic stress, not a change in route choice or in which shops exist.

| Feature | Definition | Better |
|---|---|---|
| `mean_route_lts` | Mean over routes of length-weighted LTS | lower |
| `share_routes_low_stress` | Share of routes whose *worst* edge is LTS ≤ 2 (the weakest link) | higher |
| `low_stress_route_share` | Share of route length at LTS ≤ 2 | higher |
| `ls_detour` | Low-stress route length ÷ shortest route length, capped at 2 | lower |
| `ls_reachable_pois` | POIs reachable within their cap using only LTS ≤ 2 (plus `ls_reach_<category>`) | higher |
| `ls_island_km` | Size of the connected low-stress network around the origin | higher |
| `ls_local_km` | Low-stress km within 1 km network distance | higher |

Each feature is written as `<name>_2015`, `<name>_2021` and `d_<name>` (the change). Controls: `dist_centre_m`, `bus_stops_400m`.

`routes.csv` keeps each route's edge ids. That lets the front end re-score routes instantly when a user adds a lane.

**Simplifications:**

- Routing ignores one-way restrictions.
- 2021 edges with no 2015 counterpart score LTS 4 on 2015 routes and are excluded from the 2015 low-stress network.
- Origins and destinations can join the low-stress network from up to 250 m away.

### Stage 4: synthetic census and model

> ⚠️ **The census numbers are synthetic.** The LSOAs, network and features are real (OSM). The cycle-to-work counts are generated from those features using known, planted effects, so the model has a known answer to recover. They do not describe real travel behaviour in Brighton and Hove.

```bash
python pipeline/synthetic_census.py   # -> data/processed/census_synthetic.csv, synthetic_truth.json
python pipeline/train_model.py        # -> data/processed/model.json, model_report.csv, model_fit.png
```

**Model.** This is a first-difference regression of the change in cycle-to-work share from 2011 to 2021, in percentage points, per LSOA. The terms are defined once in `pipeline/model_spec.py`:

- **Network change:** `d_mean_route_lts`, `d_share_routes_low_stress`, `d_ls_reachable_pois`, `d_ls_local_km`. These are scaled by their SD but not centred, so a scenario with no change contributes nothing.
- **Interaction:** change in low-stress reachability × baseline reachability. Gains count for more where access was poor.
- **Level controls:** `dist_centre_m` and `bus_stops_400m`. In a differenced model these allow different trends, for example near the centre and where bus access is good.

These four were chosen from seven candidates using the correlation of their changes on Brighton's network. `low_stress_route_share` duplicates `mean_route_lts` (r = −0.76), `ls_detour` overlaps `ls_reachable_pois` (r = −0.68), and `ls_island_km` is heavy-tailed. All three remain in the features file.

The model is OLS with HC1 robust standard errors and 5-fold cross-validated R². The synthetic generator draws each LSOA's cycle commuter count as a binomial sample, so small-area sampling noise is realistic.

**Validation.** Over 200 synthetic draws, the estimates are unbiased and the 95% intervals cover the planted value 92–96% of the time. That checks the standard errors as well as the point estimates.

### Stage 5: scenario map

```bash
python pipeline/export_frontend.py   # -> frontend/data/scenario.json
node tests/frontend_parity.mjs       # browser model must reproduce Python's features
cd frontend && python -m http.server 8000
```

The map covers **central Brighton** (Seven Dials to the seafront to Kemptown; `FRONTEND_AREA` in config).

- **What you can do.** Click any street that isn't already LTS 1 to add a protected lane, which makes it LTS 1. You can also try a preset scheme.
- **Which zones get predictions.** LSOAs whose population-weighted centroid falls in the area. The other LSOAs are still part of the network that routes pass through.
- **What the browser recomputes on each click.** The four model features: it re-scores every stored route, re-runs a low-stress search out to 5 km, and re-sums local calm-street km. This takes tens of milliseconds.
- **What it shows.** Predicted extra cycle commuters per zone, which is `uplift × synthetic workers`.

**One implementation of each idea.**

- `frontend/model.js` recomputes the features for both "now" and "with lanes". The predicted change therefore comes only from your edits, never from a Python/JavaScript mismatch.
- `tests/frontend_parity.mjs` checks that "now" matches Python's 2021 features for every LSOA. It also checks that no change gives zero uplift and that upgrades never make a feature worse.

**Prediction rules.**

- Only the change terms move in a scenario; the level controls cancel out.
- The current low-stress reach acts as the baseline in the interaction.
- An improvement is never allowed to predict *less* cycling: the reach effect and the total are both floored at zero.
- Big schemes extrapolate beyond the model's training range, and the map says so.

## Deploy (GitHub Pages)

The Overpass pulls are too slow and too heavy on a shared server to run in CI. So the workflow publishes the **committed** `frontend/data/scenario.json`:

1. Run the pipeline locally.
2. Commit `frontend/data/scenario.json`. Everything in `data/` stays git-ignored.
3. Push to `main`. `.github/workflows/deploy.yml` runs the parity test and publishes `frontend/`.
4. In the repo, set **Settings → Pages → Source** to **GitHub Actions**.
5. Set `REPO_URL` in `pipeline/config.py`, so the map links back to this README.

The basemap is CARTO Positron, built on OpenStreetMap data, with attribution shown on the map.

## Making it real

| Synthetic or simplified here | Real-world version |
|---|---|
| Census cycle-to-work counts | 2011 QS701EW and 2021 TS061, with the ONS 2011→2021 LSOA lookup |
| Fixed routes to the nearest POIs | Census origin–destination flows (2011 WU03EW, 2021 ODWP01EW) to weight real commute destinations |
| One-way restrictions ignored | Directed routing that keeps contraflow cycling |
| No gradient | A DEM, such as OS Terrain 50; Brighton's hills matter |
| OSM change = infrastructure change | Council scheme records, to separate real change from mapping effort |

## Caveats (worth stating even for a demo)

- **The 2021 Census was taken during COVID-19 restrictions**, so its travel-to-work data is heavily distorted by home working. A real study would need to handle this, for example with the share of people still travelling.
- **The dates don't line up.** OSM snapshots are from 2015 and 2021, but the census years are 2011 and 2021. OSM coverage in 2011 was too patchy to use.
- **Lanes aren't placed at random.** Councils build where demand already exists, so naive estimates overstate the causal effect.
- **About 165 LSOAs is a small sample**, so the coefficients are noisy. The standard errors show it.
