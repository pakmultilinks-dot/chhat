# Chhat — 5-minute stage pitch outline

## Minute 0:00–0:45 — The pain (make them feel it)
- "Raise your hand if your electricity bill shocked you this year."
- Rs 47.69/unit, up ~155% since 2021. Load-shedding this week.
- "And if you wanted solar, what happened? A salesman, a site visit, a
  quote in 5 days — sized with math that died this month."

## Minute 0:45–1:30 — The rule change (the edge)
- NEPRA killed 1:1 net metering in October 2026. Export now earns Rs 8–11;
  self-use saves Rs 35–65.
- Every old calculator maximizes the wrong thing. Show the two numbers
  side by side on one slide.

## Minute 1:30–3:30 — LIVE DEMO (the gasp)
- Paste: `House 123, Street 4, Block C, Nishat Colony, Lahore`
- Show parsed fields appear (the trained parser working).
- Drag the pin to the roof, confirm.
- Panels animate onto the roof around the water tank. Chhat Score appears.
- Read the quote: system kW, PKR cost, payback years. In Urdu with one click.
- _Backup: pre-recorded video if the network dies._

## Minute 3:30–4:15 — What is actually AI
- Trained address parser (BiLSTM, entity F1 1.00 on 2,000 test addresses).
- Layout optimizer under real rooftop constraints.
- Net-billing sizing model — the math nobody else updated.
- Honest line: "Google gives us the satellite photo. The intelligence that
  turns it into a quote for a Pakistani roof is ours."

## Minute 4:15–5:00 — Business + close
- Free scores for homeowners; qualified leads to installer partners
  (ours is customer zero) who pay per lead/close.
- Partner installation data validates us — measured error, published.
- Close: "Type your address tonight. Know your roof by breakfast."

---

## Judge Q&A drill — 10 hard questions

1. **"Isn't this just the Google Solar API?"**
   Google sells raw roof data. No Pakistani product turns it into a quote,
   nobody models net billing, nobody handles water tanks and mumty
   structures, nobody parses Pakistani addresses, nobody speaks Urdu.
   The API is our supplier, not our product.

2. **"What's actually AI here, and what's just API calls?"**
   The address parser is a model we trained (metrics on the slide). The
   layout optimizer solves panel placement under constraints. The sizing
   model encodes the new net-billing economics. Satellite data is Google's;
   everything that turns it into a decision is ours.

3. **"Your parser is trained on synthetic data. Real addresses are chaos."**
   True — that's why the map pin exists. The parser gets you 90% there in
   one second; the draggable pin closes the last 10%. We show geocode
   confidence instead of hiding it.

4. **"How accurate are the generation estimates?"**
   Validated against [N] real installations from our partner; mean error
   [X]%. (Fill after validation runs Oct 16–18. We publish the number
   whatever it is.)

5. **"Net billing just changed — prove your math."**
   Objective: maximize self-consumed kWh × Rs 47.69 + exported kWh × Rs 9.5.
   Old tools maximized total kWh. Under the new tariff, a 10 kW system
   oversized for export can pay back *slower* than a right-sized 6 kW
   system. We show both on the quote screen.

6. **"What stops an installer from copying this in a month?"**
   Two moats: (a) the validation loop — partner install data keeps making
   our estimates sharper; (b) distribution — we are already the lead
   pipe for a real installer. A copy has neither on day one.

7. **"Why not just use Project Sunroof?"**
   It doesn't serve Pakistan. No Urdu, no Pakistani tariffs, no net
   billing, no local rooftops.

8. **"Your team is four people; the form says 1–3."**
   (Resolve before the final — confirm the competing three with organizers.)

9. **"What happens without internet / without the API key?"**
   The parser runs fully offline (local model). Geocoding falls back to the
   map pin. Roof data falls back to clearly-labeled demo mode. The product
   degrades gracefully instead of dying.

10. **"Is this a business or a science project?"**
    A business with a science core. Free consumer tool → qualified leads →
    installer revenue → validation data → better product. The loop is the
    company.
