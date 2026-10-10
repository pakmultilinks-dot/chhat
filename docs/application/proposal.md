# Chhat (چھت) — AI Rooftop Solar Intelligence for Pakistan

**Category:** AI in Urban Tech
**Team:** Abdur Rehman, M. Bilal, Daud, Awais — Lahore Garrison University
**Repository:** https://github.com/pakmultilinks-dot/chhat
**Live demo:** _(deployment link added on deploy; local demo: `python -m chhat.web.app`)_
**Contact:** Abdur Rehman

---

## 1. The problem

Pakistan's electricity crisis is not a forecast, it is this week. Lahore faces
rolling load-shedding, and residential tariffs have reached **Rs 47.69/unit**,
roughly **155% higher than 2021**. A record 17 GW of solar panels was imported
in 2024, yet most homeowners who want solar never get there. Why? Quote
friction: every installer demands a site visit, quotes take days, and since
**NEPRA replaced 1:1 net metering with net billing in October 2026**, every
old sizing calculator is wrong. Under net billing, exported units earn only
**Rs 8–11** while a self-consumed unit saves **Rs 35–65**. Oversizing a system
for export, the old rule of thumb, now loses money. No Pakistani tool models
this.

## 2. The solution

**Chhat** (چھت, "roof") turns a home address into a solar decision in under a
minute:

1. The homeowner pastes their address, e.g.
   `House 123, Street 4, Block C, Nishat Colony, Lahore`.
2. Our **trained Pakistani-address parser** structures it
   (house / street / block / colony / city).
3. The address is geocoded; the user confirms the exact rooftop with a
   draggable map pin.
4. **Our rooftop segmentation model** reads the roof from aerial imagery:
   usable area, shading, and obstructions. Imagery is a swappable input —
   Google Solar API or open aerial datasets — never the intelligence.
5. Our **layout optimizer** places 550W panels around real Pakistani rooftop
   obstructions (water tanks, mumty/stair bulkheads, parapets).
6. The **net-billing quote engine** sizes the system for *self-consumption*,
   not export, and returns: **Chhat Score (0–100), system kW, panel count,
   PKR quote, yearly savings, and payback period** — in English or اردو.

## 3. What is actually AI here

We claim no algorithmic breakthrough. The intelligence is applied and honest:

- **Trained address parser.** Pakistani addresses have no standard format. We
  generated 10,000 synthetic addresses from real Lahore patterns (Nishat
  Colony, DHA, Model Town, Johar Town, …) and trained a BiLSTM token
  classifier to extract structured fields. Test results: **token accuracy
  1.00, entity F1 1.00** (2,000 held-out synthetic addresses). The map-pin
  confirmation step is the safety net for messy real-world input.
- **Our own rooftop segmentation model (in training).** Aerial imagery goes
  in; usable roof area, shading, and obstructions come out — trained by us on
  open aerial datasets, not borrowed from anyone's API.
- **Layout optimization.** Panel placement under geometric constraints with
  obstruction avoidance — the piece generic calculators skip.
- **Net-billing sizing model.** The objective function changed in October
  2026: maximize self-consumed kWh (worth Rs 47.69) instead of total
  generation. Our quote engine encodes this; old tools do not.
- **Anti-hallucination by design.** Every output is confidence-gated: the
  parser shows its fields, geocoding shows its confidence, the map pin lets
  the user correct the location. The system never pretends certainty it
  does not have.
- Google Solar API, where used, is an **optional imagery source** — a tap we
  can open or close. It is never the intelligence.

## 4. Validation plan

- **Partner ground truth:** our solar-industry partner provides 5–10
  completed installations (address, system kW, actual generation). We report
  mean estimation error honestly — the number goes on the slide either way.
- **Address accuracy:** parser tested on messy real-world formats (missing
  fields, Urdu script, reversed order); geocode confidence is shown to the
  user, never hidden.
- **Price accuracy:** every rupee in the quote comes from `chhat/quote/prices.yaml`
  (Oct-2026 market rates), re-verified against partner pricing before the final.

## 5. Why now, why us

- The net-billing rule change is **days old**. Every competitor's math is stale.
- Load-shedding is **live this week**. Demand has never been higher.
- No Pakistani product does pre-sale rooftop analysis; global tools
  (Project Sunroof) do not serve Pakistan.
- We enter with a real solar partner: validation data in, qualified leads out.
  Distribution exists on day one.

## 6. Business model (one paragraph)

Homeowners get free scores. Qualified leads (high Chhat Score + confirmed
intent) flow to vetted installer partners, starting with ours, who pay per
lead or per closed deal. Completed-installation data feeds back into the
model, so estimates get sharper over time.

## 7. Timeline to the final (Oct 24)

| Dates | Milestone |
|---|---|
| Oct 11–12 | Application submitted; parser hardened on messy addresses |
| Oct 13–15 | Live Solar API integration (key permitting); layout optimizer v2 |
| Oct 16–18 | Partner validation: measured error on real installations |
| Oct 19–21 | Urdu UI polish, demo hardening, fixed-address rehearsal set |
| Oct 22–23 | Pitch rehearsal, Q&A drill, backup offline demo |
| Oct 24 | Final at FAST-NUCES Lahore |

## 8. Honest limitations

- The address parser is trained on **synthetic** data; real-world addresses
  are messier. The draggable map pin is the accuracy backstop, and every
  output is confidence-gated rather than hallucinated.
- Roof imagery is a swappable input (Google Solar API or open aerial
  datasets). Our own segmentation model — the intelligence that reads the
  roof — is in training; until it lands, roof geometry in the demo is
  clearly-labeled **synthetic demo data**.
- Quote figures are **estimates** from Oct-2026 market rates; a site visit
  still confirms the final quote. Chhat kills quote friction, not the engineer.
- This submission is the **Satellite eye**; the Makaan AI roadmap adds
  Designer, 3D, and Guardian eyes after the championship.
