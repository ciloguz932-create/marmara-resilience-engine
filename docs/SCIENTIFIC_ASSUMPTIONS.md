# Scientific Assumptions and Limitations — MRE-001

**Status: research/software-architecture prototype. Not a validated engineering
earthquake-risk model.**

This document exists so that no number produced by this software can be
mistaken for a scientific result. Every simplification below is deliberate. If
you are reading a chart from this engine, read this file first.

---

## 0. What MRE-001 is, and is not

**Is:** a working vertical slice proving that the pipeline

> scenario → hazard → probabilistic building damage → road disruption →
> changed network → hospital accessibility → intervention comparison →
> uncertainty-aware results

can be executed end to end, reproducibly, with correct software seams for
replacing each stage.

**Is not:**

- a prediction of what will happen in any earthquake
- a statement about any real building, road, or hospital
- a validated fragility, ground-motion, or accessibility model
- an optimization method with any optimality or correctness guarantee

**The city in MRE-001 is entirely synthetic.** It is not Istanbul, not
simplified Istanbul, and not anonymised Istanbul. It is invented from a random
seed. Results must never be presented on a map of a real place, or described
with a real place name, until a real-data layer exists and is documented here.

---

## 1. Synthetic world

| Assumption | Reality it departs from |
|---|---|
| ~1,000 buildings on a bounded 8×8 km plane | Real urban inventories are far larger, spatially clustered, and heavy-tailed in size |
| Building density ≈ 15.6 buildings/km², uniformly random | Orders of magnitude below a real urban area, and real buildings cluster along streets rather than scattering uniformly |
| Square footprints, side 8–24 m, sampled independently of height | Real footprints are irregular, aligned to parcels, and correlated with height and use |
| Regular grid road network with random edge dropout | Real networks are irregular, hierarchical, and topologically constrained by terrain and coastline |
| 3 hospitals with invented capacities | Real facility capacity, specialisation, and surge behaviour vary enormously |
| Population attached to point units | Real population is time-varying (day/night) and unevenly distributed within blocks |
| Flat plane, no terrain, no soil classes | Site amplification and topography strongly modulate real shaking |
| No liquefaction, landslide, fire-following, tsunami | These are major contributors to real earthquake loss |

Building attributes (construction year, floors, structural type, occupancy) are
sampled from configured distributions chosen for plausibility, **not** fitted to
any building stock survey.

### 1.1 Vulnerability index

Each building gets a heterogeneous, dimensionless vulnerability index:

```
v = type_factor · year_factor · (floors / 5)^0.18 · exp(N(0, 0.20))
v = clip(v, 0.35, 2.60)
```

Higher `v` means more fragile; fragility medians are **divided** by it.

- Type factors (masonry 1.40, RC frame 1.00, steel 0.70, …) are **invented**.
  Their *ordering* reflects broad engineering consensus; their *magnitudes* do
  not come from any study.
- Year breakpoints (1975, 1999) with factors (1.35, 1.10, 0.85) gesture at the
  idea that successive code generations improve performance. They are **not**
  tied to any specific national code revision, and must not be described as
  representing TDY-1975, TDY-1998, TBDY-2018, or any other real code.
- The lognormal term exists so that two otherwise identical buildings still
  differ — a synthetic *population*, not a synthetic archetype. It is not
  calibrated to observed within-class variability.
- Clipping at the bounds creates a small number of exact ties (6 buildings in
  the default city). This is a deliberate artefact of bounding, not a modelling
  claim.

The index collapses everything into one scalar. Real vulnerability depends on
irregularity, soft storeys, detailing, retrofit history, and workmanship, none
of which are represented.

### 1.2 Population units

Population lives on a **separate 15×15 grid**, deliberately decoupled from
buildings, with counts drawn from a lognormal spatial weighting summing to a
configured total.

This is a modelling choice, not a convenience: if demand were derived from
buildings, damage reduction and demand reduction would be confounded, and no
intervention comparison in a later phase could separate them.

Consequences: population does not sit inside buildings, does not vary between
day and night, and does not respond to the earthquake at all. Nobody evacuates,
relocates, or is trapped.

---

## 2. Hazard model

Intensity is computed as

```
d_eff = √(d_km² + h²)                       near-source saturation
I_med = I0 · exp(−decay_per_km · d_eff)     median field
I     = clip(I_med · exp(N(0, σ_ln)), 0, I_max)
```

where `d` is the distance to the source: to the **epicenter** for the default
`point_source_decay` model, or to the nearest point on the rupture segment for
`line_source_decay`. `h` (`near_source_saturation_km` = 3 km) prevents a
singularity at zero distance, playing the role a finite-fault depth term plays
in a real GMPE. `I_max` = 2.0 bounds the lognormal tail.

Default parameters: `I0` = 1.00, `decay_per_km` = 0.150, `σ_ln` = 0.25.

**On `decay_per_km` = 0.150.** This value was chosen so that the synthetic city
spans a meaningful damage gradient. At the initial 0.045 the field saturated the
fragility curves and 53% of buildings collapsed, which exercises nothing and
produces a degenerate slice. **This is a tuning choice made for a useful
prototype, not a calibration against observed attenuation.** No inference about
real attenuation rates may be drawn from it.

**Limitations:**

- `I` is a **dimensionless proxy**. It is deliberately not labelled PGA, PGV,
  SA(T), or MMI, because it is not calibrated to any of them, and converting it
  to such units would create false precision.
- Exponential distance decay is a stand-in for a ground-motion prediction
  equation (GMPE). Real GMPEs depend on magnitude, style of faulting, depth,
  site class (Vs30), and basin effects.
- Spatial correlation of the residual field is **not** modelled — each building
  draws an independent lognormal residual. Real ground-motion residuals are
  spatially correlated over hundreds of metres to kilometres, which means
  MRE-001 **understates** the probability of large clustered damage.
- Magnitude is descriptive metadata only; it does not enter the intensity
  calculation.
- A single scenario is used. No probabilistic seismic hazard analysis (PSHA),
  no rupture-set logic tree.

**Replacement path:** implement `mre.hazard.HazardField` — two methods,
`median_intensity()` and `sample()` — backed by **OpenQuake**, a real GMPE, or
an imported ShakeMap raster. Nothing downstream changes. This is the single
highest-value replacement in the whole engine.

---

## 3. Building damage (fragility)

Damage states: `NONE`, `SLIGHT`, `MODERATE`, `SEVERE`, `COLLAPSE`.

Exceedance probability for damage state *ds*:

```
P(D ≥ ds | I) = Φ( ln(I / (median_ds · v)) / β )
```

where `v` is the building's vulnerability index (higher = more fragile, hence
the median is **divided** by it) and `Φ` is the standard normal CDF. State
probabilities are consecutive differences of the exceedance curves:

```
P(NONE)     = 1 − P(D ≥ SLIGHT)
P(SLIGHT)   = P(D ≥ SLIGHT)   − P(D ≥ MODERATE)
P(MODERATE) = P(D ≥ MODERATE) − P(D ≥ SEVERE)
P(SEVERE)   = P(D ≥ SEVERE)   − P(D ≥ COLLAPSE)
P(COLLAPSE) = P(D ≥ COLLAPSE)
```

This requires the medians to increase strictly with severity, which the model
validates on construction. Default medians: SLIGHT 0.18, MODERATE 0.32,
SEVERE 0.52, COLLAPSE 0.75, with β = 0.55.

Damage is **sampled** from this distribution by inverse CDF, and the full
probability vector is retained alongside the sample — a sampled state alone
discards everything the model knows.

There is **no threshold rule** anywhere in the model: no "intensity > X means
collapse". At its median a building has exactly P = 0.5 of reaching that state,
never 1. `tests/test_fragility.py` guards this explicitly.

An **expected damage index** is also reported:

```
EDI = Σ_k P(k)·k / 4 ∈ [0, 1]
```

EDI is a continuous severity summary. It is **not** a loss ratio, and it does
not convert to repair cost, downtime, or casualties.

**Limitations — this is the most important section:**

- **The medians and β in `config/default.toml` are invented numbers.** They are
  not derived from Turkish building stock, not from the TBDY/TDY codes, not
  from HAZUS, not from any empirical damage database.
- The lognormal form is a conventional and defensible *shape*, but shape alone
  is not a model. Uncalibrated parameters make the output non-quantitative.
- The vulnerability index is a single scalar summarising construction year,
  floors, and structural type. Real vulnerability depends on irregularity,
  soft storeys, detailing quality, retrofit history, and construction
  workmanship, none of which are represented.
- Damage is sampled **independently per building** given intensity. Real damage
  is correlated through shared construction era, contractor, and site
  conditions. MRE-001 therefore **understates** the variance of aggregate loss.
- No casualty model. Occupancy is carried for future use and for accessibility
  demand weighting only. **This engine does not estimate deaths or injuries.**

**Do not state that any individual building will collapse.** The model produces
a probability under invented parameters for an invented building.

**Replacement path:** implement `mre.buildings.FragilityModel`. The engine
consumes damage-state probabilities and samples; a validated curve set drops in
without touching the Monte Carlo driver. `tests/test_fragility.py` tests the
model in isolation, so a replacement can be held to the same contract.

---

## 4. Road disruption

A link's probability of being affected is

```
n_l      = collapsed buildings whose footprint lies within R of link l
p_l      = clip( baseline_l + k · n_l · susceptibility_l , 0, 1 )
affected ~ Bernoulli(p_l)
state    = DEGRADED with probability q, else CLOSED   (given affected)
```

with R = 30 m, k = 0.20, q = 0.5, baseline = 0.01. `susceptibility_l` is a
per-class multiplier (arterial 0.60, collector 1.00, local 1.40) expressing that
narrow streets are blocked by less debris — **invented**, like everything else
here.

`CLOSED` links are removed from the graph; `DEGRADED` links remain with travel
time divided by `degraded_speed_factor` = 0.4. The baseline network is never
mutated — the disrupted graph is a copy — so before/after comparison stays
possible.

**A structural consequence of the synthetic city:** at 15.6 buildings/km² and
R = 30 m, only ~44% of links (322 of 740) have *any* building within reach.
Disruption is therefore concentrated on a minority of links, and even total
collapse of every building leaves most of the network open. In a real city,
with far higher density, debris blockage would be pervasive. **The prototype
substantially understates network disruption**, and the disruption it does
produce is a property of the synthetic layout as much as of the model.

**Limitations:**

- Debris blockage is modelled purely by proximity count. Real blockage depends
  on building height, setback, road width, and collapse mechanism.
- Bridges, tunnels, viaducts, and embankments are **not** modelled as distinct
  vulnerable assets, though in real events they dominate network disruption.
- The road network is a **regular grid** with ~12% of edges randomly removed
  (subject to staying connected). Real networks are irregular, hierarchical, and
  constrained by terrain and coastline. Grid topology has unusually high
  redundancy, which further understates disruption.
- Only `COLLAPSE` contributes debris. In reality partial collapse and falling
  cladding from `SEVERE` buildings also block roads.
- `criticality` is edge betweenness on the undamaged network. It describes how
  many shortest paths use a link; it is not a prediction of importance during a
  real emergency response.
- Link failures are sampled independently given the damage field.
- No time dimension: no debris clearance, no progressive reopening, no
  emergency response logistics. The result is a snapshot immediately after the
  event.
- No traffic assignment or congestion. Travel times are free-flow, which
  **understates** post-event travel times, since surviving links carry
  redirected demand.

---

## 5. Hospital accessibility

Each population unit is assigned to its shortest-travel-time hospital via
multi-source Dijkstra on the post-event graph.

Statistics are **population-weighted**. Mean and median cover the **reachable
population only**: units with no path, or with travel time above
`max_travel_time_min`, are excluded and counted separately.

This exclusion is deliberate and load-bearing. Averaging in an infinity gives an
infinite or NaN mean; averaging in the threshold value silently understates the
loss. Both would hide exactly the effect the engine exists to measure. Where
nobody is reachable at all, the mean is reported as `nan`, never `0.0`, which
would read as instant access.

A route counts as **disrupted** when its travel time rises by more than
`disrupted_route_tolerance_min` (0.01 min) or it becomes unreachable. Routes
already unreachable before the event are not counted again — they did not
change.

**Limitations:**

- Assignment is by travel time only; hospital capacity does **not** constrain
  assignment. Reported `hospital_utilisation` is therefore a *demand pressure
  indicator*, not a queueing result — values above 1.0 mean assigned demand
  exceeds emergency capacity, not that patients were turned away.
- **Utilisation scale artefact.** The synthetic population (≈120,000) and the
  synthetic hospital capacities (205 emergency beds total) were chosen
  independently, giving ≈585 people per emergency bed. Utilisation therefore
  lands in the hundreds rather than near 1. This is an artefact of two
  unrelated parameter sets, **not a finding about emergency capacity**.
  Utilisation is interpretable here only as a *relative* comparison between
  hospitals and between scenarios. Reconciling the two parameter sets is
  deferred; the numbers are reported as-is with the artefact declared alongside
  them in every summary.
- A population unit co-located with a hospital node has travel time 0 and
  remains reachable even with every road destroyed. Correct given the model,
  but it means a small amount of population is structurally immune to road
  disruption.
- Hospitals are assumed fully functional after the event. In reality hospitals
  are themselves vulnerable structures with power, water, and staffing
  dependencies. This is a significant optimistic bias.
- Demand is proportional to population, not to modelled injuries.
- Units beyond `max_travel_time_min` are counted as unreachable and excluded
  from mean/median, so improvements are not masked by infinities. This
  threshold is a reporting convention, not a clinical one — it is unrelated to
  any "golden hour" evidence base.
- Access is road-only: no helicopter, maritime, or pedestrian evacuation.

---

## 6. Monte Carlo and uncertainty

- Default 1,000 realisations; configurable.
- One master seed drives **named streams** (`city.roads`, `city.buildings`,
  `hazard:i`, `damage:i`, `disruption:i`), derived via
  `SeedSequence(entropy=seed, spawn_key=(crc32(name),))`. Names rather than
  sequential spawning means a stage consuming a different number of draws never
  shifts another stage, and adding a stage later cannot perturb existing
  results. CRC32 rather than `hash()` because `hash()` is salted per process and
  would silently destroy cross-run reproducibility.
- Realisation *i* is identical whether run alone or within a batch, so results
  are order-independent and survive future parallelisation.
### 6.1 Three distinct things, not to be conflated

| | What it is | Captured here? |
|---|---|---|
| **Monte Carlo sampling variability** | Spread from finite sampling of the model's own random steps: hazard residual, damage draw, link-state draw. Shrinks as *n* grows. | **Yes** — this is what the percentiles show |
| **Model uncertainty (epistemic)** | Uncertainty in the model *form* and in the invented parameter values. Does not shrink with *n*. | **No** — not represented at all |
| **Synthetic assumptions** | The city, network, population, and parameters are invented. No amount of simulation makes them describe a real place. | **No** — a fixed premise, not an uncertainty |

The reported percentiles are **empirical quantiles of the simulated
distribution**. They are:

- **not confidence intervals** — there is no estimator, no sampling
  distribution of a parameter, and no coverage claim;
- **not predictive intervals** for a real earthquake;
- a description of the model's internal variability only.

**Running 1,000 realisations does not make the model scientifically validated.**
More realisations narrow the bands, which makes the *simulation* more precise
about *its own assumptions* while saying nothing about the world. Precision is
not accuracy here, and reporting narrower bands from a larger *n* must never be
presented as increased confidence in the result.

**Consequently, the percentile bands from MRE-001 are far narrower than genuine
uncertainty about a real earthquake.** They describe the model, not the world.

### 6.2 Baseline pairing

The pre-earthquake network is a property of the city, not of a realisation, so
baseline accessibility is deterministic and computed once. Every realisation is
compared against **that same baseline**, giving a paired before/after
comparison on a single city.

Accessibility is **never** compared across two unrelated realisations: the
difference would mix the effect of the earthquake with the difference between
two random draws.

---

## 7. Interventions (Phase 5)

Three prototype types: `BUILDING_RETROFIT`, `ROAD_HARDENING`,
`HOSPITAL_SUPPORT`. Phase 5 adds a budget-constrained decision layer that
enumerates every feasible portfolio and ranks them by one objective. It changes
no scientific stage.

**How effects are modelled.** An intervention returns a modified *copy* of the
synthetic city, changing per-entity parameters only:

- `BUILDING_RETROFIT` — divides a targeted building's vulnerability index by
  `fragility_median_uplift` (equivalently, uplifts its fragility medians),
  floored at the synthetic vulnerability clip.
- `ROAD_HARDENING` — multiplies a targeted link's baseline closure probability
  and its debris susceptibility by `closure_probability_multiplier`.
- `HOSPITAL_SUPPORT` — multiplies a targeted hospital's emergency capacity by
  `emergency_capacity_multiplier`.

**Primary objective.** A single interpretable metric: **expected unreachable
population** (lower is better). Secondary metrics — travel time, closures,
collapses, and a prototype service-pressure index — are reported *separately*.
They are deliberately never combined into a weighted score, which would hide
value judgements inside an arbitrary number.

**Common random numbers.** Every portfolio is evaluated at the same seed and the
same realisation streams as the no-intervention baseline. Because each stochastic
draw is a fixed-size sample named `(seed, stage, realisation)` and never keyed by
a parameter value, realisation *i* of every portfolio sees the identical
intensity field and identical uniform draws. Benefit is therefore the **paired**
difference `baseline[i] − portfolio[i]`, reported as a distribution (mean,
P05/P50/P95, and probability of improvement), never a single "best" number. This
also yields a clean monotonicity property: under common random numbers, fewer
collapses or fewer closures can only remove disruption, so a structural
intervention never makes any paired realisation *worse*.

**Targeting is a documented heuristic, not an optimum.** Retrofit targets the
most vulnerable buildings; hardening targets the links most frequently *closed*
in a Monte Carlo pre-pass (where the network actually fails, which pure
centrality misses); support targets the most service-pressured hospitals. None
of these is claimed optimal.

**Out-of-sample target selection (validation split).** Data-driven hardening
targets are chosen on a **selection** subset of the realisations
(`selection_fraction`, default 30%), and every reported metric is computed on the
disjoint **evaluation** subset (the remaining 70%). Selecting targets and scoring
them on the *same* realisations would be an in-sample bias: the links that
happened to fail most in a given draw would be exactly the links hardened, and
the measured benefit would be partly a fit to that noise. The split removes that
bias. It is a deterministic, seed-derived permutation, so it is reproducible and
non-overlapping, and common random numbers are preserved *within* the evaluation
set (baseline and every portfolio share the same evaluation realisations). This
is a **prototype validation split** — it removes an in-sample selection artefact
inside the synthetic slice. It is **not** a statistical validation of the model
and says nothing about real-world validity. (In the default city the
out-of-sample hardening benefit is essentially unchanged from the in-sample
figure, which indicates the benefit was not an artefact of selection.)

**Two honest findings of the default synthetic city, not to be smoothed over:**

- **Hospital support cannot change the primary objective.** Under the Phase 4
  accessibility model, hospital capacity does **not** gate assignment (§5), so
  adding emergency capacity changes *who is over-subscribed*, never *who can be
  reached*. `HOSPITAL_SUPPORT` therefore moves only the prototype
  service-pressure index and yields exactly zero expected-unreachable-population
  benefit. This is surfaced in every output, never disguised as accessibility
  gain. It carries the same population/capacity scale artefact described in §5.
- **Benefit lives in the tail.** Unreachability is rare in the redundant
  synthetic grid (nonzero in ~9% of baseline realisations), so even an effective
  intervention shows benefit only in that minority of realisations and near-zero
  in the P05–P95 band; the effect appears in the *mean*. Reporting the mean
  benefit without its low probability of improvement would overstate a typical
  realisation.

**Limitations:**

- Unit costs are invented and are not engineering cost estimates.
- Effect multipliers are assumed, not measured.
- The search is a deterministic exhaustive enumeration of a tiny candidate set
  with **no optimality guarantee**; it is a strategy seam an MILP/optimizer can
  replace.
- Benefit is measured in accessibility terms only — no monetised loss, no
  casualties averted, no equity or distributional analysis of who benefits.
- Selection heuristics and effects depend on the synthetic layout as much as on
  the interventions themselves.
- Political, legal, and implementation feasibility are entirely absent.

Label all such output as **"Prototype intervention optimization"**.

---

## 8. Coordinate reference system

MRE-001 uses **EPSG:32635 (WGS 84 / UTM zone 35N)** for the synthetic study
area.

**Rationale:** a projected, metre-based CRS is required so that distances,
travel times, and adjacency radii are metrically meaningful. UTM 35N covers
24°E–30°E, which contains western Marmara and most of Istanbul, making it a
natural placeholder.

**This is provisional and explicitly NOT a Marmara-wide decision.** The Marmara
region straddles the UTM 35N/36N boundary at 30°E, so a single UTM zone
distorts one end of a region-wide study area. A final CRS decision (candidates
include EPSG:5254 TUREF/TM30, a custom LAEA, or per-zone tiling) is deferred
until real data extents are known.

Because the city is synthetic, its coordinates are arbitrary offsets within this
CRS chosen to fall in a plausible numeric range. **They do not correspond to
real locations and must not be plotted on a basemap of the Marmara region.**

---

## 9. What would be required to make this scientific

Not a roadmap promise — a statement of the gap:

1. Validated, region-specific fragility curves with documented provenance.
2. A real GMPE with site amplification and spatially correlated residuals.
3. A real building inventory with structural attributes and quantified
   attribute uncertainty.
4. Real network topology including bridges and tunnels as distinct assets.
5. Hospital functionality modelling (structural, non-structural, lifelines).
6. Time-dependent response: debris clearance, casualty arrival, surge capacity.
7. Independent validation against observed damage from a past event.
8. Epistemic uncertainty via a logic tree over models and parameters.
9. External peer review.

Until items 1–4 exist, output is a demonstration of software architecture.

---

## 10. Real-data pilot (Phase 6)

Phase 6 adds a **real-data pilot** on a real İstanbul study area, alongside the
synthetic slice. It does **not** make the engine a validated model. The governing
rule, held throughout: **a value is either taken from a cited source, or marked
`UNKNOWN` — never invented.** The pilot runs in a distinct `REAL_PILOT` mode
(`mre.real.CityMode`); the synthetic city remains the default.

**Study area.** Zeytinburnu district core, İstanbul (~2.6 km²). Chosen because it
is the canonical published İstanbul earthquake-risk pilot (JICA–İMM *Earthquake
Master Plan for İstanbul*, 2002; Boğaziçi/KOERI seismic risk assessment pilot),
sits on soft soil by the Marmara Sea, and faces the Main Marmara Fault Princes'
Islands segment.

**What is REAL** (OpenStreetMap, © OpenStreetMap contributors, **ODbL**; fetched
via Overpass, recorded in `provenance.json`):

- building footprints and centroids (real polygons, extruded as-is in Blender);
- storey counts where `building:levels` is tagged (~92% of the pilot);
- occupancy where the `building` tag is determinable;
- the road network and its **OSM node topology** (connectivity is real, not
  reconstructed); road classes from the `highway` tag;
- hospital locations and names.

**What is `UNKNOWN`** (recorded, never filled): construction year and structural
system for every building; storey count / occupancy where the tag is absent;
hospital capacity and bed counts. These appear literally as `UNKNOWN` in the
output GPKG attribute columns, with `*_source` columns marking provenance.

**What is PROTOTYPE** (assigned and labelled, *not* real):

- **Vulnerability.** Open data carries no structural attributes, so every real
  building is given a *uniform* prototype vulnerability index. The damage field
  therefore varies only with the scenario intensity gradient, never with an
  invented per-building fragility. Labelled `PROTOTYPE_UNIFORM`.
- **Fragility parameters.** The same invented medians/β as the synthetic engine
  (§3), applied to real footprints. The damage layer is **"prototype fragility on
  real geometry"**, not a real damage, loss, or casualty estimate.
- **Intensity.** A dimensionless *scenario-derived intensity proxy* (§2), **not**
  PGA/PGV/SA(T)/MMI. The fault **system and segment location are real and cited**
  (Princes' Islands segment, ~18.5 km south of the pilot); the proxy field's
  magnitude (`intensity_at_source`) and decay (`decay_per_km`) are **prototype
  tuning parameters**, exactly as §2 already documents them to be — chosen so the
  pilot spans a meaningful gradient, and carrying **no** ground-motion meaning at
  the real fault distance.
- **Hospital emergency capacity.** OSM has no capacity, so a prototype emergency
  capacity is assigned. Note the *primary* accessibility objective — who can
  reach a hospital — does **not** use capacity at all; only the secondary
  service-pressure metric does, and it is reported as prototype.
- **Demand.** No licensed real population distribution is used. Demand is a
  **uniform proxy grid** (`population_source = "UNIFORM_PROXY"`), deliberately
  decoupled from buildings (as in the synthetic slice). Accessibility results
  describe geographic/network access, **not** real population impact, and the
  "population" counts are proxy weights, not people.

**Documented approximations** (real geometry, simplified where the model
requires it): building footprints enter the disruption-adjacency model as their
**area-equivalent square** (the engine's `Building.footprint`); road links are
**junction-to-junction** spans of the real network (real topology, straightened
geometry in the GIS/Blender output); free-flow travel times use the engine's
prototype class speeds; edge criticality is not computed for the pilot
(`NOT_COMPUTED`).

**Sources not machine-ingested in this pilot.** AFAD instrumental catalogue, MTA
active-fault line geometry, and İBB building/geotechnical datasets are recorded
as **context/citations** in the scenario provenance rather than ingested as
layers — they require portal interaction or licensing this pilot does not
perform. They are marked accordingly, not silently substituted.

**Validation.** Every build runs `mre.real.validation` (valid geometry, single
consistent CRS, no NaN in required fields, no non-positive road lengths, road
connectivity, entities within the study area, non-empty demand proxy,
reproducibility, provenance completeness) and writes `validation.json`; results
are not presented unless it passes.

**Bottom line.** The real-data pilot demonstrates the *pipeline* running on real
geometry with honest provenance and integrity labelling. The **geometry, network,
and hospital locations are real; the physics is prototype**. It is a research
prototype, not an operational İstanbul earthquake prediction or a real
damage/loss estimate for any real building, road, or hospital.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-10 | Initial version for MRE-001 Phase 1 skeleton. |
| 2026-08-10 | Phase 3: documented the vulnerability index, decoupled population units, the point/line hazard formulation and the `decay_per_km` tuning choice, the fragility difference-of-exceedance formulation and EDI, and the road-disruption density consequence. |
| 2026-08-10 | Phase 4: documented population-weighted accessibility and the explicit exclusion of unreachable population from travel-time statistics; the hospital utilisation scale artefact; the separation of Monte Carlo sampling variability from model uncertainty and synthetic assumptions; and baseline pairing. |
| 2026-08-15 | Phase 5: documented the implemented intervention layer — per-entity effect modelling, the single expected-unreachable-population objective with separately-reported secondary metrics, common-random-numbers paired evaluation and its monotonicity property, data-driven hardening targeting, and the two honest findings (hospital support cannot move the primary objective; benefit lives in the tail). |
| 2026-08-15 | Phase 5.1: added the out-of-sample target-selection split — data-driven hardening targets chosen on a 30% selection subset, all metrics reported on the disjoint 70% evaluation subset, CRN preserved within the evaluation set. Removes in-sample selection bias; not a statistical validation. |
| 2026-08-16 | Phase 6: added the real-data pilot (§10). Real OSM geometry (Zeytinburnu, ODbL) for buildings/roads/hospitals with full provenance and UNKNOWN marking; prototype fragility/intensity applied to real geometry; uniform demand proxy; sourced Main Marmara Fault scenario with a proxy intensity field. Not a validated model or a real prediction. |
