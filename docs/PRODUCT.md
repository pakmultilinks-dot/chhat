# Chhat (چھت) — Product

AI rooftop solar intelligence for Pakistan. A web application: type your
home address, our AI parses it, you confirm the pin on the map, our own
trained rooftop models find roofs on real satellite imagery, and you get
an animated panel layout, a PKR quote, and a Chhat Score. Urdu + English.

Repo: `github.com/pakmultilinks-dot/chhat`, branch `dev/satellite-eye`.
No deployment is configured (user order: accuracy first, no deploy).

## How to run it locally

```bash
cd ~/workspace/chhat-build
.venv/bin/python -m chhat.web.app
# open http://localhost:5000  (Urdu: http://localhost:5000/?lang=ur)
```

Requirements: the repo venv (`.venv`) with torch, flask, opencv, numpy.
Model weights ship in `models/` (`rooftop_seg.pt`, `rooftop_seg_dense.pt`,
`address_parser.pt` / `.npz`).

## The full user flow

1. **Address** — paste a Pakistani home address. Our own trained
   BiLSTM parser (F1 1.00 on 2,000 synthetic test addresses, 57 tricky
   hand-written cases across Lahore, Karachi, Islamabad, Rawalpindi,
   Faisalabad) breaks it into house / street / block / colony / city chips.
2. **Location** — a Leaflet map with a draggable pin. The user confirms
   their exact rooftop. (Without a `GOOGLE_MAPS_API_KEY`, geocoding falls
   back to the user pin; the map starts centered on Lahore.)
3. **Report** — Chhat Score ring (0-100), KPIs (system kW, 550W panels,
   yearly generation, payback), an animated SVG panel layout, and a
   net-billing-era PKR quote (equipment cost, yearly savings, 25-year
   savings). Roof figures come from the Solar API when a key exists,
   otherwise from the labeled DEMO roof model.
4. **Roof AI** — "See what our AI sees": real Esri World Imagery (open
   data, z19) fetched live for the pinned point, read by our two trained
   segmenters. `predict_best` runs both and keeps the higher-confidence OK
   result: **v2 generalist** for planned areas, **v3 dense-street
   specialist** for dense areas. The UI shows the satellite image, the
   green roof-mask overlay (toggleable), a confidence bar, which model
   answered, and the imagery source. Below confidence 0.55 — or on blank
   frames or absurd masks — the model returns UNSURE with no mask, and the
   UI shows "needs human review" in both languages.

**Offline fallback:** if the backend is unreachable, an "Offline demo
mode" card offers 10 precomputed golden demos (famous public landmarks —
never real homes) that render the full report with zero backend calls.
Data is embedded in the page and served at `/golden-demos.json`.

## Honest limitations (read before demoing)

- **Rooftop masks are approximate.** v2 test IoU 0.3604 (honest spatial
  split); v3 dense-test IoU 0.3846. On dense wall-to-wall streets the mask
  finds roof areas but does not cleanly separate every individual house.
  The user's own verdict on his street: "near but not accurate."
- **No exact-house guarantee from text alone.** The parser understands the
  address; the exact rooftop needs the user's map-pin confirmation.
- **Demo mode figures are estimates.** Without `GOOGLE_SOLAR_API_KEY`,
  roof area / shading come from a synthetic-but-realistic Lahore roof
  model, always shown under a red DEMO badge. The segmenter itself always
  reads real Esri imagery.
- **Parser F1 1.00 is on synthetic data** from the same generator family —
  real-world messy addresses may behave worse; 57 tricky cases guard this.
- **Imagery needs internet.** Esri tiles are fetched live; offline, only
  the golden demos work.
