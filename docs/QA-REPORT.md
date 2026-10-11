# Chhat — QA Report (Wave 6, 2026-10-11)

All numbers below are measured, not claimed. Demo-mode figures are labeled.

## 1. Unit tests
- `python -m unittest discover -s tests` → **5/5 pass** (parser, pipeline, numpy parity)

## 2. Address parser (trained by us)
- Model: BiLSTM token tagger, trained on 10,000 synthetic Pakistani addresses
- Held-out test (2,000 addresses): token accuracy 1.00, **entity F1 1.00**
- 24 hand-written tricky cases (messy commas, `H#`, Urdu-mixed, reversed order,
  sector form, lowercase): **29/29 pass** (5 clean + 24 tricky)
- Live check: `House 123, Street 4, Block C, Nishat Colony, Lahore` →
  house=123, street=4, block=C, colony=Nishat Colony, city=Lahore ✓
- Honest caveat: the 1.00 F1 is on synthetic data (same distribution as train);
  the 24 tricky hand-written cases are the real robustness proof.

## 3. Rooftop segmentation (trained by us)
- Model: TinyUNet from scratch, 467,169 params, 256×256, CPU
- Data: 676 chips, 3 Lahore regions (DHA Phase 5 / Model Town / Johar Town),
  OSM footprints + Esri imagery; train 540 / val 68 / test 68
- **Test IoU 0.4082 | Test Dice 0.5325**
- Live check: `scripts/roofseg_verify.py` on 6 DHA test chips →
  all OK, confidence 0.73–0.85, grid saved to `docs/roofseg-verify.png`
- Honest caveat: OSM/Esri label noise (misalignment, stale footprints) caps the
  score; predictions are **confidence-gated** — low confidence returns UNSURE
  instead of a fake mask (anti-hallucination by design).

## 4. End-to-end pipeline (local, DEMO mode — no API keys)
`scripts/nishat_e2e.py`, 3 Nishat Colony address variants:
- Parse: all 3 correct (clean, `H#45 St 7`, `Makan 89, Gali No 2`)
- Geocode: skipped honestly (no key) → labeled DEMO pin
- Roof: DEMO 178.4 m², usable 73.7 m²
- Layout: 18 × 550W panels → 9.9 kW
- Quote: **Rs 1,424,500 | saves Rs 585,051/yr | payback 2.4 yrs**
- **Chhat Score: 77/100**

## 5. Live deployment
- Homepage https://chhat-pakmultilinks-dot.vercel.app/ → HTTP 200, ~2.3s ✓
- `/api/parse`, `/api/analyze` → **timeout** (Vercel API rate limit from wave 3;
  expected reset ~2026-10-11 21:36 UTC). Not fought further per plan.
- Fallback: local demo runs the full pipeline (steps above).

## 6. Breakage found and fixed this wave
- `scripts/roofseg_verify.py` crashed with `ModuleNotFoundError: chhat` when run
  as `python scripts/roofseg_verify.py` from repo root → added `sys.path`
  bootstrap. Verified: runs clean, 5/5 unit tests still pass.
- No other breakage found.

## 7. Open blockers (need the user)
- `GOOGLE_MAPS_API_KEY` / `GOOGLE_SOLAR_API_KEY` for live geocoding + real roof
  data. Everything runs in labeled DEMO mode until then.
- Partner price list + 5–10 install records for quote validation story.

---

## Product-build QA (2026-10-11, post wave-9)

### 1. Unit tests: 15/15 green
- parser (incl. 57 multi-city tricky cases), numpy-parity, pipeline,
  golden demos (6/6), NEW roofseg honesty-gate tests (blank/uniform ->
  UNSURE, absurd-fraction cap).

### 2. Dual-model roof AI (`predict_best`) — verified live on Flask
- Dense Lahore pin (31.4951, 74.3953, near Milad Chowk): **v3** selected,
  conf 0.657, overlay rendered.
- DHA planned pin (31.4700, 74.3250): **v2** selected, conf 0.597.
- Karachi pin (24.8607, 67.0011): **v2** selected, conf 0.608.
- `/api/roofseg` returns `model_used` + `model_label`; UI shows
  "Model used" line in EN + UR.

### 3. Anti-hallucination hardening (bug found and fixed this wave)
- Found: v3 returned OK (conf 0.59, 82% of frame "roof") on blank/gray
  images — overconfident on out-of-distribution input.
- Fixed: `predict()` now returns UNSURE (no mask) when the frame is
  blank/uniform (std < 2) or building_fraction > 0.75 (absurd for a 150m
  chip). Verified: blank -> UNSURE, gray -> UNSURE, real E378 chip ->
  OK/v3/conf 0.655 with mask.

### 4. End-to-end `/api/analyze` on 3 cities (user-pinned, DEMO roof mode)
- Lahore (31.4951, 74.3953): score 84, 18 panels, 9.9 kW,
  Rs 1,424,500, payback 2.5 yrs.
- Karachi (24.8607, 67.0011): score 72, 18 panels, 9.9 kW,
  Rs 1,424,500, payback 2.6 yrs.
- Islamabad (33.6844, 73.0479): score 72, 18 panels, 9.9 kW,
  Rs 1,424,500, payback 2.5 yrs.
- All carry `"demo": true`; UI shows the red DEMO badge.

### 5. UI checks (live server)
- `/` and `/?lang=ur` return 200; Urdu strings render (tagline, golden
  card, model labels).
- Header step pills now show all 4 steps in the active language.
- All buttons wired: go, confirm, seeAi, backReport, goldenGo, ovToggle.
- `/golden-demos.json` serves 200 (offline fallback data present).

### 6. Honest limitations (unchanged, documented in docs/PRODUCT.md)
- Rooftop masks approximate: v2 IoU 0.3604, v3 dense IoU 0.3846; dense
  streets not cleanly per-house separated (user verdict: "near but not
  accurate").
- Exact house needs the user's map pin; text address alone is not enough.
- Demo-mode figures are estimates until GOOGLE_SOLAR_API_KEY exists.
- No deployment performed (user order).
