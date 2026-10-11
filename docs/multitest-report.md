# Multi-address segmentation test — 7 Lahore landmarks

Date: 2026-10-11. Method: each public landmark geocoded via OSM Nominatim,
Esri World Imagery z19 chip (512px, ~150m) fetched, `predict_best()` run
(v2 generalist + v3 dense specialist, higher-confidence OK wins).
Satellite + overlay images saved in this folder (`*-sat.jpg`,
`*-overlay.jpg`); numeric results in `results.json`.

No real home addresses used — all public commercial/monumental places.

| # | Landmark (area) | Model | Conf | Frac | Verdict |
|---|---|---|---|---|---|
| 1 | Packages Mall (DHA/Walton) | v3 | 0.623 | 0.170 | POOR — paints the parking lot as roof, misses the actual mall roof |
| 2 | Emporium Mall (Johar Town) | v3 | 0.690 | 0.167 | MIXED — blobs sit on the building but track no real roof edges |
| 3 | Model Town Park (Model Town) | v2 | 0.000 | 0.000 | HONEST UNSURE — it is a park, no roofs; correctly refused |
| 4 | Fortress Square (Cantt) | v3 | 0.777 | 0.405 | POOR — paints the road/street, misses the mall buildings |
| 5 | Badshahi Mosque (Old City) | v3 | 0.754 | 0.358 | POOR — paints the courtyard, misses the domed monument |
| 6 | Liberty Market (Gulberg) | v3 | 0.551 | 0.041 | POOR — pin landed on a roundabout; model painted the roads |
| 7 | Shalimar Gardens (Shalimar) | v3 | 0.765 | 0.306 | POOR — paints garden lawns and paths as roofs |

## Where the model handles well vs poorly

**Handles (partially): small residential rooftops.** On the E378 Nishat
Colony street test, v3 produces discrete roof patches on real houses
with no road painting (user verdict: "near but not accurate" — close,
not clean per-house separation).

**Fails: everything that is not a small residential roof.**
Large flat commercial roofs (malls), domes and courtyards (Badshahi),
asphalt parking lots, roads, garden lawns — the model overfires on big
flat textured surfaces and underfires on the actual large structures.
It was trained only on residential rooftops; malls, monuments, and
parks are out-of-distribution.

**Honest bright spot:** on Model Town Park it returned UNSURE with zero
confidence instead of hallucinating — the honesty gate works.

## What this means for the product

- The residential use case (a homeowner's street, user-confirmed pin)
  is the only one the model is built for. Landmark pins are adversarial
  by nature (centroid of a mall/park, not a house roof).
- For the championship demo: always demo on residential addresses.
  Commercial/monumental pins will embarrass the model.
- Path to better: the failure mode is consistent (big flat non-roof
  surfaces), so targeted negative training data (parking lots, lawns,
  courtyards, roads) would directly attack it — blocked on labeling
  effort/GPU, not on ideas.
