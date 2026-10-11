# rooftop_seg metrics (honest)

- Model: TinyUNet v2 from scratch (4-level U-Net), 1,927,009 params, 256x256, CPU. No pretrained backbone.
- Data: OSM footprints + Esri World Imagery (z18), 3 Lahore regions (DHA / DHA Phase 5 / Johar Town). train 256, val 165, test 153
- Labels: measured systematic OSM->Esri shift corrected per region (dha -1,-2; dha5 +2,-2; johartown +6,-2 px), estimated from mask-boundary vs image-gradient agreement, split-half stable.
- Split: SPATIAL (west/middle/east zones per region, one-stride buffer dropped) — no overlapping-chip leakage between train/val/test.
- Augmentation: flips, 90-degree rotations, brightness/contrast jitter.
- Loss: BCE + Dice. Best val IoU: 0.2792
- **Test IoU: 0.3161 | Test Dice: 0.4115**
- Caveat: residual label noise (stale/demolished OSM footprints, per-building edits) caps the achievable score; predictions are confidence-gated (see chhat/roofseg/predict.py).
