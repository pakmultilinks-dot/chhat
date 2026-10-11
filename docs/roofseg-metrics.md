# rooftop_seg metrics (honest)

- Model: TinyUNet from scratch, 467,169 params, 256x256, CPU
- Data: OSM footprints + Esri imagery, Lahore. train 128, val 16, test 17
- Best val IoU: 0.3914
- **Test IoU: 0.4602 | Test Dice: 0.5428**
- Caveat: OSM/Esri label noise (misalignment, stale footprints) caps the achievable score; predictions are confidence-gated (see chhat/roofseg/predict.py).
