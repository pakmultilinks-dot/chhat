# Chhat — Battle Card (memorize these 10)

1. **What Chhat is:** paste your home address → our AI finds your exact rooftop,
   reads the roof, places solar panels around your water tank, and gives you a
   system size, PKR quote, and payback — in under a minute, in Urdu or English.

2. **Number 1 — the pain:** electricity is **Rs 47.69/unit, up ~155% since 2021**,
   and Lahore is in rolling load-shedding. Everyone wants solar; nobody can get
   a straight answer.

3. **Number 2 — the edge:** NEPRA killed net metering in Oct 2026. Export now
   earns **Rs 8–11** while a self-consumed unit saves **Rs 35–65**. Every old
   calculator oversizes for export and loses money. Ours sizes for
   self-consumption. Nobody else models this.

4. **Number 3 — the proof:** our trained address parser scores **entity F1 1.00**
   and passes 29 tricky real-world cases; our trained rooftop model runs on
   Lahore imagery. We train our own — no core intelligence rented from anyone.

5. **"Isn't this just the Google API?"** Google is an *optional imagery supplier*,
   never the brain. The parser, the rooftop segmentation, the layout optimizer,
   and the net-billing quote engine are all trained/built by us. Imagery is
   swappable — open aerial data works too.

6. **"Your segmentation IoU is only 0.41?"** Honest answer: label noise caps it,
   and that's why every prediction is **confidence-gated** — low confidence says
   UNSURE instead of hallucinating a roof. Anti-hallucination is a feature, not
   a footnote.

7. **"F1 1.00 sounds fake."** Agree out loud: it's on synthetic data. Then show
   the 24 hand-written nasty cases — missing commas, `H#`, Urdu-mixed, reversed
   order — all passing. That's the real test.

8. **"Aurora Solar already does this."** Aurora serves the US and Europe. Nobody
   serves Pakistani rooftops, the net-billing regime, Urdu users, or PKR
   economics. We are Aurora for the 3 billion people Aurora doesn't serve.

9. **"Where's the business?"** Homeowners get free quotes; qualified leads flow
   to installer partners who pay per lead/close — our solar partner is customer
   zero. Their install data feeds back and sharpens our estimates.

10. **The closer (60-second demo):** type a judge's own home address → parsed
    fields pop up → roof mask draws itself → 18 panels placed → 77/100,
    9.9 kW, Rs 1,424,500, payback 2.4 years. Their house, their numbers.
