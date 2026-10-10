# Chhat — 60-Second Stage Demo Script

**Setup before the demo:** open the app on the projector. Have the Urdu toggle
ready. If `GOOGLE_SOLAR_API_KEY` is set you get live roof data; otherwise the
red **DEMO DATA** badge shows and the flow still works end to end.

## The 60 seconds

**0:00 — The hook (10s)**
> "Electricity is Rs 47.69 a unit and the light goes out every day. Everyone
> wants solar, but getting a quote takes a week. Watch this."

Type into the address box: `House 123, Street 4, Block C, Nishat Colony, Lahore`
Click **Analyze my roof**.

**0:10 — The parser (10s)**
> "First our trained AI reads the address — even messy ones like
> `H#45 St 7` or `Makan 89 Gali 2`."

Point at the chips: House / Street / Block / Colony / City appear one by one.

**0:20 — The pin (10s)**
> "Now I drop the pin on my exact roof."

Drag the map pin slightly, click **Confirm location**.

**0:30 — The gasp (20s)**
> "And there is my roof — with the panels placed around my water tank."

Panels pop onto the roof schematic one by one. The Chhat Score dial pulses:
**score / 100**. Read the cards aloud: system kW, panel count, yearly
generation, payback years, system cost in PKR, 25-year savings.

**0:50 — The closer (10s)**
> "Type any address in Pakistan. Sixty seconds. No salesman, no site visit.
> That is Chhat — छत."

## If the API key is missing

Say it proudly, do not hide it:
> "We are on demo roof data today — see the red badge. With the Solar API key
> this reads your real roof from satellite. The pipeline is identical."

## Backup lines (if something stalls)

- Parser hiccup: "Pakistani addresses are messy — that is exactly why we
  trained our own parser on 10,000 of them. Let me try the clean format."
- Slow network: "The map tiles are loading — the analysis itself runs locally."
- Judge asks about accuracy: "We validate every estimate against real
  installations from our solar partner and publish the measured error."
