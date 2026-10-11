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
