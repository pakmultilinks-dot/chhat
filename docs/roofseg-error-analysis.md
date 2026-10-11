# rooftop_seg error analysis (honest, 2026-10-11)

Final model: TinyUNet v2 (4-level, 1.9M params, from scratch).
**Test IoU 0.3604 | Test Dice 0.4562** on leak-free spatial splits
(west=train / middle=val / east=test per region, buffer zones dropped).
Target was 0.55 — not reached. This document says why, proudly and precisely.

## What the model gets RIGHT

- Coarse built-up vs not-built-up: on test chips it reliably finds where the
  buildings are (confidence 0.66–0.73 on 6/6 test chips, all status OK).
- Large isolated buildings (DHA Phase 5 villas): mask shape roughly correct.
- Confidence gating works: blank/uncertain inputs return UNSURE, never a fake mask.
- On a real Nishat Colony house (E378 test): status OK, it found roof-like
  regions — but see below.

## What the model gets WRONG (user-verified on a real house)

1. **Merges neighboring houses into one blob.** The #1 complaint, confirmed on
   the verify grid: where the true mask shows 10–12 separate rectangles, the
   model predicts 1–3 big blobs. Row-houses become a single smear.
2. **Spills into streets.** Blob edges bleed past roof edges onto roads.
3. **Blobby boundaries.** Predicted edges are rounded; true roofs are rectangular.
4. **Low recall in dense areas.** On the E378 image (dense Nishat Colony) it
   detected only ~10% building pixels and 23 components while the image shows
   far more rooftops. Confidence 0.58 — barely above the 0.55 UNSURE line.
5. **Misses small houses** (test chip dha5_test_0004: found 2 blobs, missed most).

## WHY — the measured causes, in order

1. **The labels themselves merge houses.** OSM footprints for touching row-houses
   are adjacent polygons; rasterized at 0.6 m/px they become one blob. The model
   cannot learn separation that the labels do not contain. Boundary-weighted
   loss (edge band 5–7x) was tried twice from the best checkpoint and never beat
   plain BCE+Dice on val — the signal to separate simply is not in the labels.
2. **Residual label noise caps the ceiling.** Beyond the corrected systematic
   shift (dha −1,−2; dha5 +2,−2; johartown +6,−2 px, measured and split-half
   stable), individual OSM footprints are stale (demolished/built since mapping).
   Val IoU plateaued at ~0.28–0.30 across every configuration tried
   (lr 1e-3 → 3e-4, 0.5M → 1.9M params, boundary loss on/off). That plateau is a
   data ceiling, not a training bug.
3. **Small training set for the capacity.** 256 train chips for 1.9M params;
   augmentation (flips, rot90, brightness/contrast) helps but cannot invent data.
4. **Evaluation got stricter, honestly.** Wave 5 reported 0.41 — but on a random
   split where overlapping neighboring chips leaked across train/test. The
   spatial split used now is leak-free; 0.36 is the true number. We report the
   true number.
5. **Resolution.** z18 (0.6 m/px) downsampled to 256 px; thin gaps between houses
   are 1–3 px wide — near the noise floor.

## What would actually improve it (judge-ready answer)

1. **Instance-separated labels:** erode-then-watershed the rasterized OSM
   polygons so touching houses become separate instances with a forced gap;
   train with the boundary-weighted loss (already implemented, `--bw` flag).
2. **More Lahore regions** (5–10 quadrants, not 3) and z19 imagery where free.
3. **Full-resolution tiled inference** instead of resize-to-256 (keeps thin gaps).
4. **Watershed/split post-processing** on the probability map as an inference
   step — cheap, no retraining.
5. **GPU training** for longer schedules and proper hyperparameter search.

## Bottom line for the pitch

"We trained our own rooftop model on open Lahore data, measured and fixed the
label misalignment ourselves, evaluate it on truly held-out geography, and it
honestly scores IoU 0.36 — it finds built-up areas reliably but still merges
adjacent houses, which is exactly what our next label-separation step targets.
Every prediction carries a confidence score, and low-confidence outputs are
withheld rather than faked."
