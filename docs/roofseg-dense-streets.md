# Dense-street fine-tuning (honest report)

## Why

The v2 segmenter (trained on DHA / DHA Phase 5 / Johar Town, all planned
areas with spaced houses) failed visibly on dense wall-to-wall Lahore
streets: on the user's own street near Milad Chowk, Nishat Colony, it
painted 70% of the image green, including roads. The user judged it
"too wrong".

Root cause: the model had never seen dense-street morphology, and OSM
building footprints — the label source — are essentially ABSENT in dense
Lahore residential areas (probed 2026-10-11, per km2: Nishat Colony 7,
Walton 2, Samanabad 3, Ichhra 23, Model Town 4; only planned areas like
Defence Mor 542 and Johar Town 260 have coverage). There was nothing to
"snap" in Nishat Colony.

## What we built instead: clean, human-verified dense labels

`scripts/build_dense_clean.py`:

1. **Imagery**: Esri World Imagery at **z19** (~0.3 m/px) over three dense
   Lahore Cantt blocks (nishat_b, nishat_w, cantt_n; neighborhood centers
   only, no private addresses). Note: z20 was tried first but Esri
   returns flat gray placeholder tiles over Lahore — z19 is the max
   usable zoom (measured 2026-10-11).
2. **Candidate masks**: morphological roof proposals (gradient + Otsu +
   closing + eroded seeds + connected components, size/aspect filtered)
   with two cleaning steps:
   - **Vegetation filter**: drop components with high Excess Green Index
     (2G-R-B >= 25; calibrated: roofs ~12-17, trees ~37-45), so tree
     canopies and fields are never labeled as roofs.
   - **Edge-aware boundary refinement**: move every mask boundary pixel
     to the nearest Canny edge within 4px, keeping the safe interior.
     Labels trace visible rooflines instead of blobby approximations.
3. **Human verification**: all 202 candidates viewed on contact sheets;
   auto-rejected dominant-blob chips (>40% of mask in one component);
   a human kept **79 chips** where green sits on actual roofs, houses
   are separated, no road/vegetation spill. 8 random keeps spot-checked
   at full resolution — all passed.

`scripts/keep_dense_chips.py` writes the 79 approved chips to
`data/roofseg_dense/{train,val,test}` (24/27/28) with a geographic
x-zone split per region (no buffer dropped — documented: boundary chips
may share a 64px strip across splits; the primary generalization proof
is the E378 street test ~1 km away, not this split).

## Fine-tuning

`chhat/roofseg/finetune_dense.py`: start from `models/rooftop_seg.pt`
(v2), Adam lr 1e-4, BCE+Dice, augmentation, early stopping on dense-val
IoU. One careful run (plus one joint run, see below) — no endless loops.

## Results (honest)

| Model | Dense test IoU | Planned test IoU | E378 street (Milad Chowk) |
|---|---|---|---|
| v2 (planned-only) | ~0.13 (zero-shot) | 0.36-0.39 | paints 70% green incl. roads — "too wrong" |
| v3 (dense-only fine-tune) | **0.3846** (Dice 0.40) | 0.10 (forgotten) | discrete roof patches, 10% green — visibly better, still sparse |

The dense-only fine-tune (v3, `models/rooftop_seg_dense.pt`) is a real,
visible leap on dense streets but catastrophically forgot planned areas.

## Joint model (in progress)

To get ONE model good on both, a joint fine-tune on mixed data
(dense chips 4x oversampled + planned chips, from v2 weights, lr 1e-4)
is training: `models/rooftop_seg_joint.pt`. Result pending — reported
honestly when done, including the E378 overlay comparison. If it does
not beat v2/v3 where it matters, we say so and keep the specialists.
