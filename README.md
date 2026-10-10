# ☀️ Chhat (چھت) — AI Rooftop Solar Intelligence for Pakistan

**Satellite eye** of the Makaan AI platform. Paste a Pakistani home address,
get your roof's solar score, system size, PKR quote, and payback period.

## How it works

```
address text
  → Pakistani address parser (OUR trained BiLSTM token classifier)
  → Google Geocoding API → lat/lng  (geocoding utility, not intelligence)
  → user confirms/drags map pin
  → OUR rooftop segmentation model reads the roof  (in training;
     imagery: Google Solar API or open aerial datasets — swappable input)
  → OUR layout optimizer → panel placement avoiding tanks/mumty
  → OUR net-billing quote engine → PKR quote + payback
  → Chhat Score 0-100
```

**Design rule: our models are the brain.** External APIs are utilities
(geocoding) or swappable imagery inputs — never the intelligence.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install flask pyyaml requests numpy

# train the address parser (CPU, ~15 min)
.venv/bin/python -m chhat.parser.train --n-train 8000 --epochs 12

# run tests
.venv/bin/python -m unittest discover tests

# run the web app
.venv/bin/python -m chhat.web.app   # http://localhost:5000
```

## Environment keys (optional but recommended)

| Key | Purpose |
|---|---|
| `GOOGLE_MAPS_API_KEY` | address → lat/lng geocoding (utility, not intelligence) |
| `GOOGLE_SOLAR_API_KEY` | **optional** roof imagery source (imagery input only) |

Without keys the app runs in **DEMO mode**: the address parser still works
(trained model, local), geocoding is replaced by a draggable map pin, and the
roof is a realistic synthetic Lahore rooftop clearly labeled DEMO in code and UI.
Our own segmentation model (wave 4) will read live roofs from any imagery
source — no vendor lock-in by design.

## Project layout

- `chhat/parser/` — synthetic data generator, BiLSTM tagger, training, inference
- `chhat/geocode/` — Google Geocoding client
- `chhat/solar/` — Google Solar API client (+ labeled demo mode)
- `chhat/layout/` — panel layout optimizer
- `chhat/quote/` — net-billing quote engine, `prices.yaml` holds ALL numbers
- `chhat/web/` — Flask UI (English/اردو)

## Demo script (2 minutes)

1. Paste: `House 123, Street 4, Block C, Nishat Colony, Lahore`
2. See parsed fields (house/street/block/colony/city), drag pin to the roof
3. Confirm → Chhat Score, panel layout on the roof, PKR quote, payback

## Honest limitations

- The address parser is trained on **synthetic** addresses (templates from real
  Lahore patterns). Real-world messy addresses will be harder; the map-pin
  confirmation step is the safety net, and outputs are confidence-gated.
- Our rooftop segmentation model is in training (wave 4, open aerial data).
  Until it lands, roof geometry without an imagery key is synthetic demo data,
  clearly labeled. Imagery sources are swappable by design — no vendor is
  the brain.
- Quote prices in `prices.yaml` are Oct-2026 market estimates; verify before
  customer-facing use.
- This is the Satellite eye only (Makain AI roadmap: Designer, 3D, Guardian).
