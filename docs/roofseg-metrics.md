# rooftop_seg metrics (honest)

- Model: TinyUNet v2 from scratch (4-level U-Net), 1,927,009 params, 256x256, CPU. No pretrained backbone.
- Data: OSM footprints + Esri World Imagery (z18), 3 Lahore regions (DHA / DHA Phase 5 / Johar Town). train 256, val 165, test 153
- Labels: measured systematic OSM->Esri shift corrected per region (dha -1,-2; dha5 +2,-2; johartown +6,-2 px), estimated from mask-boundary vs image-gradient agreement, split-half stable.
- Split: SPATIAL (west/middle/east zones per region, one-stride buffer dropped) — no overlapping-chip leakage between train/val/test.
- Augmentation: flips, 90-degree rotations, brightness/contrast jitter.
- Loss: BCE + Dice (lr 3e-4, weight decay 1e-4). Boundary-weighted BCE fine-tuning
  (edge band 5-7x, targeting house separation) was tried twice from this checkpoint
  but never beat it on val — final weights are the BCE+Dice ones. Best val IoU: 0.2952
- **Test IoU: 0.3604 | Test Dice: 0.4562**
- Caveat: residual label noise (stale/demolished OSM footprints, per-building edits) caps the achievable score; predictions are confidence-gated (see chhat/roofseg/predict.py).

---

## Dense-street fine-tune (2026-10-11, accuracy wave)

- **Why**: v2 failed visibly on dense wall-to-wall Lahore streets (user verdict
  "too wrong" on his own street near Milad Chowk: painted 70% of the image
  green including roads). OSM footprints are near-absent in dense areas
  (Nishat Colony 7/km2), so no OSM labels to refine there.
- **Clean dense set**: 79 human-verified z19 chips (Nishat Colony / Lahore
  Cantt), labels = morphological proposals + vegetation filter (ExG) +
  edge-aware boundary snapping to Canny edges. Method + review protocol in
  `docs/roofseg-dense-streets.md`. Split 24/27/28 (train/val/test).
- **v3** (`models/rooftop_seg_dense.pt`): fine-tune from v2, lr 1e-4, 25 epochs.
  Dense-test **IoU 0.3846 / Dice 0.3998**. E378 street: green 70% -> 10%,
  discrete roof patches instead of whole-neighborhood painting — a real
  visible leap, but still sparse (misses roofs) and not yet crisp per-house
  masks. **Catastrophic forgetting**: planned-area IoU collapsed to 0.10.
- **Joint model** (`models/rooftop_seg_joint.pt`, training): fine-tune from v2
  on mixed data (dense 4x oversampled + planned), aiming for one model good
  on both. Result pending; reported honestly when done.
