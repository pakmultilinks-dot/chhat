# rooftop_seg metrics (honest)

- Model: TinyUNet from scratch, 467,169 params, 256x256, CPU
- Data: OSM footprints + Esri imagery, Lahore. train 540, val 68, test 68
- Best val IoU: 0.4197
- **Test IoU: 0.4082 | Test Dice: 0.5325**
- Caveat: OSM/Esri label noise (misalignment, stale footprints) caps the achievable score; predictions are confidence-gated (see chhat/roofseg/predict.py).
