# Chhat — Championship Submission Checklist (copy-paste ready)

Fill these into the championship form. Every answer below is final text.

---

## 1. Project title
Chhat (چھت) — AI Rooftop Solar Intelligence for Pakistan

## 2. Category
AI in Urban Tech

## 3. Team members (4)
1. Abdur Rehman — Lahore Garrison University
2. M. Bilal — Lahore Garrison University
3. Daud — Lahore Garrison University
4. Awais — Lahore Garrison University

## 4. Project proposal (.md)
File: `docs/application/proposal.md` in the repo (paste its full contents).

## 5. Five keywords
1. rooftop-solar
2. net-billing
3. geospatial-AI
4. urdu-nlp
5. energy-access

(One-line meanings in `docs/application/keywords.md`.)

## 6. GitHub link
https://github.com/pakmultilinks-dot/chhat

NOTE for the form: all championship work is on branch **`dev/satellite-eye`**
(direct link: https://github.com/pakmultilinks-dot/chhat/tree/dev/satellite-eye).
`main` is intentionally untouched until the team lead reviews the branch.

## 7. Live demo
https://chhat-pakmultilinks-dot.vercel.app/

STATUS (re-verified 2026-10-11 17:05 PKT): homepage loads fine (HTTP 200, ~2s).
`/api/parse` and `/api/analyze` return HTTP 501 — the live deployment is STALE
(built Oct 10 during the deploy wave; it serves an old weights-check handler,
not the Flask app). Fix: ONE redeploy of branch `dev/satellite-eye` after the
Vercel API rate limit resets ~2026-10-11 21:36 UTC (~02:36 PKT Oct 12).
Do not spam redeploys before then.
If the form is submitted before the reset, use the fallback below.

FALLBACK: local demo — clone the repo, check out `dev/satellite-eye`,
`pip install -r requirements.txt`, run the web app; full pipeline runs in
labeled DEMO mode without API keys. Exact steps in README.md.

## 8. What to demo on stage (60 seconds)
1. Paste: `House 123, Street 4, Block C, Nishat Colony, Lahore`
2. Show parsed fields (house/street/block/colony/city) as chips
3. Confirm the map pin on the rooftop
4. Roof mask overlay appears → 18 panels placed → score 77/100 →
   9.9 kW, Rs 1,424,500, payback 2.4 years
Full script: `docs/DEMO.md` (includes no-key fallback lines).

## 9. Pre-submit sanity checks
- [ ] Repo link opens and shows the project
- [ ] Branch `dev/satellite-eye` visible on GitHub
- [ ] Demo URL responds (or note the local-demo fallback)
- [ ] proposal.md pasted in full, keywords exactly 5
- [ ] Team names spelled as registered with the university
