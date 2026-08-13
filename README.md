# Real-Estate Agent Lead Enrichment — Proof of Concept

A lead fills in a Facebook lead form (name, email, phone). This tool researches
that person, decides whether they're genuinely a working real-estate agent,
gathers evidence, scores the lead 0–100, assigns a **P1–P6 priority**, and
writes a plain-text report.

This is the PoC stage: output is a `.txt` file per lead. Once approved, the same
structured result maps straight onto **ActiveCampaign** custom fields + a contact
note (see "Going live" below).

## What it finds for each lead

- Is this actually a real-estate agent? (with confidence)
- Their agency + agent profile page
- 2–3 most recent **active listings**
- A recent **sold** listing (~last 90 days)
- Whether they run a **team** (and rough size)
- Roughly how many **years** they've been an agent
- Active **social/professional profiles**
- Flags when a **personal email** was used (expected for agents who don't want
  it hitting their work inbox — treated as normal, not a red flag)

## Scoring rubric (deterministic — easy to tune)

Encoded in `score()` in `enrich.py` so the logic is transparent and adjustable:

| Signal | Points |
|---|---|
| Confirmed active agent | +20 |
| 3+ active listings (1–2 → +10) | +20 |
| 2+ sold in ~90 days (1 → +10) | +20 |
| Works in a team | +15 |
| 7+ yrs experience (5–6 → +12, 3–4 → +8) | +15 |
| 2+ social profiles (1 → +5) | +10 |

| Score | Priority |
|---|---|
| 80–100 | **P1** — call first |
| 65–79 | P2 |
| 50–64 | P3 |
| 35–49 | P4 |
| 20–34 | P5 |
| No evidence they're an agent | **P6** — verify manually |

A strong agent (3–4 active listings, 2–3 recent solds, a team, 5–7+ years) lands
~80–90 → **P1**. No evidence of being an agent → **P6**.

## Setup (first time on a new machine)

```bash
# 1. create a virtual environment and install dependencies
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. add your own Anthropic API key
cp .env.example .env
#    then open .env and paste your key in place of the placeholder
#    (get a key at https://console.anthropic.com -> API keys)
```

The `.env` file holds the API key and is git-ignored — it is NOT included in
this package. Each person supplies their own key.

## Run it

```bash
# one lead
./.venv/bin/python enrich.py --name "Jane Smith" --email jane@gmail.com \
    --phone "021 555 1234" --location "Auckland, NZ"

# batch a Facebook lead-form export
./.venv/bin/python enrich.py --csv leads.csv
```

Reports are written to `reports/<name>.txt`. See
`reports/SAMPLE_Jane_Smith.txt` for an example of the output format (synthetic
data — generated without an API key for illustration).

## How it works

1. **Research pass** — Claude (Opus 4.8) with the web-search tool builds a
   written dossier on the lead. This is the judgment-heavy step (deciding
   whether someone's genuinely an agent, weighing conflicting evidence), so
   it stays on Opus.
2. **Structure pass** — a second call (Haiku 4.5 — mechanical extraction
   from a dossier Opus already wrote, doesn't need Opus-level reasoning)
   extracts validated evidence fields (listings, solds, team, tenure,
   socials).
3. **Score** — Python applies the rubric above (deterministic, no model
   guessing on the final score) and assigns a priority.
4. **Report** — writes the `.txt`.

## Going live (ActiveCampaign)

The `Evidence` + `Score` objects already hold everything needed. To integrate:
swap the `render()`/file-write step for ActiveCampaign API calls that (a) set
custom fields (`lead_score`, `priority`, `is_agent`, `agency`, `active_listings`,
`recent_sold`, `has_team`, `years_experience`), and (b) attach the dossier as a
contact note. Trigger the script from the AC webhook / Facebook lead-form
automation instead of the CLI.
