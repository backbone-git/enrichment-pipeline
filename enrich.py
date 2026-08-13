#!/usr/bin/env python3
"""
Real-estate agent lead enrichment — proof of concept.

A lead comes in from a Facebook lead form (name + email + phone). This tool
runs web research on that person, decides whether they're genuinely a working
real-estate agent, pulls evidence (active listings, recent solds, team, social
profiles, tenure), scores the lead 0-100, assigns a P1-P6 priority, and writes
a plain-text report.

Output is a .txt file per lead. When this goes live, the same EnrichmentResult
object maps straight onto ActiveCampaign custom fields + a contact note.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python enrich.py --name "Jane Smith" --email jane@gmail.com --phone "021 555 1234" --location "Auckland, NZ"
    python enrich.py --csv leads.csv          # batch a Facebook lead-form export
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional

import anthropic
from pydantic import BaseModel, Field

try:  # load ANTHROPIC_API_KEY from a local .env if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "claude-opus-4-8"
REPORTS_DIR = Path("reports")


# --------------------------------------------------------------------------- #
# 1. Input
# --------------------------------------------------------------------------- #
@dataclass
class Lead:
    name: str
    email: str
    phone: str = ""
    location: str = ""  # optional hint, e.g. region the form was targeting

    @property
    def email_is_personal(self) -> bool:
        """A work agent filling a form with a personal email is a key signal."""
        domain = self.email.split("@")[-1].lower().strip()
        free = {
            "gmail.com", "googlemail.com", "outlook.com", "hotmail.com",
            "live.com", "yahoo.com", "yahoo.co.nz", "icloud.com", "me.com",
            "xtra.co.nz", "proton.me", "protonmail.com", "aol.com",
        }
        return domain in free


# --------------------------------------------------------------------------- #
# 2. Evidence schema (what the research pass must return)
# --------------------------------------------------------------------------- #
class Listing(BaseModel):
    address: str = Field(description="Listing address or headline")
    price: str = Field(default="", description="Asking/sold price if shown")
    url: str = Field(default="", description="Link to the listing")


class SocialProfile(BaseModel):
    platform: str
    url: str


class Evidence(BaseModel):
    is_real_estate_agent: bool = Field(
        description="True only if web evidence shows this person actively works as a real-estate agent."
    )
    confidence: str = Field(description='One of: "high", "medium", "low".')
    agency_name: str = Field(default="", description="Brokerage/agency they work for")
    agency_website: str = Field(default="")
    profile_url: str = Field(default="", description="Their agent profile page")
    years_experience: Optional[int] = Field(
        default=None, description="Years working as an agent, if determinable"
    )
    active_listings_count: int = Field(default=0)
    recent_listings: List[Listing] = Field(default_factory=list)
    sold_last_90_days_count: int = Field(default=0)
    recent_sold: List[Listing] = Field(default_factory=list)
    has_team: bool = Field(default=False)
    team_size: int = Field(default=0, description="People clearly working with them (0 if unknown/solo)")
    social_profiles: List[SocialProfile] = Field(default_factory=list)
    notes: str = Field(default="", description="Anything else relevant to lead quality")


# --------------------------------------------------------------------------- #
# 3. Research pass — Claude + web search builds a dossier
# --------------------------------------------------------------------------- #
RESEARCH_SYSTEM = """You are a research analyst qualifying inbound leads for a real-estate \
firm that recruits working real-estate agents. You are given a person's contact details \
from a lead form. Use web search to find out everything you can about them and decide \
whether they are genuinely a practising real-estate agent.

Investigate and gather concrete evidence for:
- Whether they are an active real-estate agent (and how confident you are).
- The agency/brokerage they work for and their agent profile page.
- Their active listings right now (find 2-3 of the most recent, with address + price + link).
- A recently sold listing (ideally within the last ~90 days).
- Whether they work in a team and roughly how many people are with them.
- Roughly how many years they have been an agent.
- Their active social/professional profiles (LinkedIn, Instagram, Facebook business page, etc.).

Search by name plus likely qualifiers (real estate, the agency, the region). The email may be \
a personal address, so don't rely on the email domain to find them. Be honest about uncertainty: \
if you cannot find evidence the person is an agent, say so clearly rather than guessing. Cite the \
sources you used. End with a clear written dossier covering every point above."""

STRUCTURE_SYSTEM = """Extract the research dossier into a single JSON object. Only record facts \
supported by the dossier; if something was not found, use empty string / 0 / null / empty list and \
set is_real_estate_agent accordingly. Do not invent listings, teams, or tenure.

Return ONLY the JSON object (no prose, no markdown fences), with exactly this shape:
{
  "is_real_estate_agent": true,
  "confidence": "high|medium|low",
  "agency_name": "",
  "agency_website": "",
  "profile_url": "",
  "years_experience": null,
  "active_listings_count": 0,
  "recent_listings": [{"address": "", "price": "", "url": ""}],
  "sold_last_90_days_count": 0,
  "recent_sold": [{"address": "", "price": "", "url": ""}],
  "has_team": false,
  "team_size": 0,
  "social_profiles": [{"platform": "", "url": ""}],
  "notes": ""
}"""


def research(client: anthropic.Anthropic, lead: Lead) -> str:
    """Run the web-research pass, returning a plain-text dossier."""
    user = (
        f"Lead from a Facebook lead form:\n"
        f"- Name: {lead.name}\n"
        f"- Email: {lead.email} ({'personal' if lead.email_is_personal else 'work/domain'} address)\n"
        f"- Phone: {lead.phone or 'n/a'}\n"
        f"- Location hint: {lead.location or 'n/a'}\n\n"
        f"Research this person and produce the dossier."
    )
    messages = [{"role": "user", "content": user}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}]

    for _ in range(6):  # bound the server-side tool loop (pause_turn)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=RESEARCH_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp.content},
            ]
            continue
        break

    return "\n".join(b.text for b in resp.content if b.type == "text").strip()


def structure(client: anthropic.Anthropic, dossier: str) -> Evidence:
    """Second pass: turn the dossier into validated Evidence (no tools).

    We ask for plain JSON and validate with Pydantic rather than using strict
    structured outputs — the constrained-decoding grammar for this nested schema
    can time out to compile, and a normal JSON pass is plenty reliable here.
    """
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=STRUCTURE_SYSTEM,
        messages=[{"role": "user", "content": dossier}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # Be forgiving: strip ``` fences and grab the outermost { ... }.
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return Evidence.model_validate(json.loads(text))


# --------------------------------------------------------------------------- #
# 4. Scoring rubric (deterministic — easy for the client to tune)
# --------------------------------------------------------------------------- #
@dataclass
class Score:
    points: int
    priority: str
    priority_label: str
    breakdown: List[str] = field(default_factory=list)


def score(ev: Evidence, lead: Lead) -> Score:
    pts = 0
    bd: List[str] = []

    def add(n: int, reason: str):
        nonlocal pts
        pts += n
        bd.append(f"  {'+' if n >= 0 else ''}{n:>3}  {reason}")

    # No evidence of being an agent -> not scorable, drops to P6.
    if not ev.is_real_estate_agent or ev.confidence == "low":
        bd.append("   --  No clear evidence this person is a working real-estate agent")
        return Score(min(pts, 15), "P6", "Not clearly an agent — verify manually", bd)

    add(20, f"Confirmed real-estate agent (confidence: {ev.confidence})")

    if ev.active_listings_count >= 3:
        add(20, f"{ev.active_listings_count} active listings")
    elif ev.active_listings_count >= 1:
        add(10, f"{ev.active_listings_count} active listing(s)")
    else:
        bd.append("    0  No active listings found")

    if ev.sold_last_90_days_count >= 2:
        add(20, f"{ev.sold_last_90_days_count} sold in ~last 90 days")
    elif ev.sold_last_90_days_count >= 1:
        add(10, f"{ev.sold_last_90_days_count} recent sold")
    else:
        bd.append("    0  No recent solds found")

    if ev.team_size >= 1 or ev.has_team:
        add(15, f"Works in a team (~{ev.team_size or 'size unknown'})")

    if ev.years_experience is not None:
        if ev.years_experience >= 7:
            add(15, f"{ev.years_experience} yrs experience")
        elif ev.years_experience >= 5:
            add(12, f"{ev.years_experience} yrs experience")
        elif ev.years_experience >= 3:
            add(8, f"{ev.years_experience} yrs experience")
        else:
            add(3, f"{ev.years_experience} yrs experience (early career)")

    n_social = len(ev.social_profiles)
    if n_social >= 2:
        add(10, f"{n_social} active social profiles")
    elif n_social == 1:
        add(5, "1 active social profile")

    pts = max(0, min(pts, 100))

    if pts >= 80:
        prio, label = "P1", "Call first — strong, active agent"
    elif pts >= 65:
        prio, label = "P2", "High priority"
    elif pts >= 50:
        prio, label = "P3", "Good — worth a call"
    elif pts >= 35:
        prio, label = "P4", "Moderate"
    else:
        prio, label = "P5", "Low — light evidence"
    return Score(pts, prio, label, bd)


# --------------------------------------------------------------------------- #
# 5. Report
# --------------------------------------------------------------------------- #
def render(lead: Lead, ev: Evidence, sc: Score, dossier: str) -> str:
    L = []
    w = L.append
    w("=" * 70)
    w("REAL-ESTATE AGENT LEAD — ENRICHMENT REPORT")
    w("=" * 70)
    w(f"Generated:  {date.today().isoformat()}")
    w("")
    w("LEAD")
    w(f"  Name:      {lead.name}")
    w(f"  Email:     {lead.email}  ({'personal' if lead.email_is_personal else 'work/domain'} address)")
    w(f"  Phone:     {lead.phone or 'n/a'}")
    w(f"  Location:  {lead.location or 'n/a'}")
    w("")
    w("-" * 70)
    w(f"  SCORE:     {sc.points}/100")
    w(f"  PRIORITY:  {sc.priority}  —  {sc.priority_label}")
    w("-" * 70)
    w("")
    w("VERDICT")
    w(f"  Is a real-estate agent:  {'YES' if ev.is_real_estate_agent else 'NO / UNCLEAR'}  (confidence: {ev.confidence})")
    if lead.email_is_personal:
        w("  Note: lead used a PERSONAL email — common when an agent doesn't want")
        w("        this landing in their work inbox. Treated as expected, not a red flag.")
    w("")
    w("EVIDENCE")
    w(f"  Agency:          {ev.agency_name or 'unknown'}")
    if ev.agency_website:
        w(f"  Agency site:     {ev.agency_website}")
    if ev.profile_url:
        w(f"  Agent profile:   {ev.profile_url}")
    yexp = ev.years_experience if ev.years_experience is not None else "unknown"
    w(f"  Experience:      {yexp} years")
    w(f"  Team:            {'Yes' if (ev.has_team or ev.team_size) else 'No / solo'}"
      + (f" (~{ev.team_size} people)" if ev.team_size else ""))
    w("")
    w(f"  Active listings: {ev.active_listings_count}")
    for x in ev.recent_listings:
        w(f"     - {x.address}{(' — ' + x.price) if x.price else ''}")
        if x.url:
            w(f"       {x.url}")
    w(f"  Sold (~90 days): {ev.sold_last_90_days_count}")
    for x in ev.recent_sold:
        w(f"     - {x.address}{(' — ' + x.price) if x.price else ''}")
        if x.url:
            w(f"       {x.url}")
    w("")
    w("  Social profiles:")
    if ev.social_profiles:
        for s in ev.social_profiles:
            w(f"     - {s.platform}: {s.url}")
    else:
        w("     - none found")
    if ev.notes:
        w("")
        w("  Notes:")
        for line in ev.notes.splitlines():
            w(f"     {line}")
    w("")
    w("SCORE BREAKDOWN")
    L.extend(sc.breakdown)
    w("")
    w("=" * 70)
    w("RESEARCH DOSSIER (raw)")
    w("=" * 70)
    w(dossier)
    w("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 6. Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class EnrichmentResult:
    lead: Lead
    evidence: Evidence
    score: Score
    dossier: str
    report: str


def enrich_one(client: anthropic.Anthropic, lead: Lead) -> EnrichmentResult:
    """Research + structure + score one lead. Pure compute, no I/O — callers
    decide whether/where to persist the result (local file, AC, etc.)."""
    print(f"  -> researching {lead.name} ...", file=sys.stderr)
    dossier = research(client, lead)
    print(f"  -> structuring evidence ...", file=sys.stderr)
    ev = structure(client, dossier)
    sc = score(ev, lead)
    report = render(lead, ev, sc, dossier)
    return EnrichmentResult(lead, ev, sc, dossier, report)


def write_report(result: EnrichmentResult) -> str:
    """CLI persistence: write the report to reports/<name>.txt."""
    REPORTS_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in result.lead.name).strip("_") or "lead"
    out = REPORTS_DIR / f"{safe}.txt"
    out.write_text(result.report, encoding="utf-8")
    print(f"  -> {result.score.priority} ({result.score.points}/100)  written to {out}", file=sys.stderr)
    return str(out)


def leads_from_csv(path: str) -> List[Lead]:
    leads = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = {k.lower().strip(): (v or "").strip() for k, v in row.items()}
            if not r.get("name") or not r.get("email"):
                continue
            leads.append(Lead(
                name=r["name"], email=r["email"],
                phone=r.get("phone", ""), location=r.get("location", ""),
            ))
    return leads


def main():
    ap = argparse.ArgumentParser(description="Real-estate agent lead enrichment (PoC)")
    ap.add_argument("--name")
    ap.add_argument("--email")
    ap.add_argument("--phone", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--csv", help="Batch: CSV with columns name,email,phone,location")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY=sk-ant-...")

    if args.csv:
        leads = leads_from_csv(args.csv)
    elif args.name and args.email:
        leads = [Lead(args.name, args.email, args.phone, args.location)]
    else:
        ap.error("Provide --name and --email, or --csv leads.csv")

    client = anthropic.Anthropic()
    print(f"Enriching {len(leads)} lead(s) with {MODEL}\n", file=sys.stderr)
    for lead in leads:
        try:
            write_report(enrich_one(client, lead))
        except Exception as e:  # keep a batch going if one lead fails
            print(f"  !! failed for {lead.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
