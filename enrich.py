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
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional

import anthropic
import httpx
from pydantic import BaseModel, Field, model_validator

try:  # load ANTHROPIC_API_KEY from a local .env if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "claude-opus-4-8"  # research() — the judgment-heavy call, kept on Opus
STRUCTURE_MODEL = "claude-haiku-4-5-20251001"  # structure() — mechanical extraction from an already-written dossier, doesn't need Opus
REPORTS_DIR = Path("reports")

# A hung request (rare, but seen in practice — one research call ran 20+
# minutes with no error) shouldn't be able to block a batch or a scheduled
# pipeline run indefinitely. This bounds every individual request made by
# the client — not the whole research() loop, which can itself make up to 6
# such requests on pause_turn, so the real worst-case ceiling is several
# times this number. On timeout, anthropic raises APITimeoutError, which the
# per-lead try/except in main()/pipeline.py catches so the batch keeps going.
#
# A single float here isn't a reliable hard ceiling — it maps to equal
# connect/read/write/pool timeouts, and a read timeout only fires on gaps
# between bytes, so it can be reset indefinitely by keep-alive-style trickle
# data from the server during a genuinely very long generation (observed:
# a request ran 14+ minutes past a nominal 300s timeout with no error).
# Explicit bounds close that gap. max_retries=0 matters too: the SDK retries
# retryable failures (including timeouts) a few times by default, each
# retry getting its own fresh timeout window — silently multiplying both
# wait time and cost. We already have our own retry mechanism (a failed
# lead just stays tagged for the next scheduled poll), so we don't want the
# SDK retrying inside a single run on top of that.
REQUEST_TIMEOUT_SECONDS = 300.0


def make_client() -> anthropic.Anthropic:
    timeout = httpx.Timeout(connect=10.0, read=REQUEST_TIMEOUT_SECONDS, write=30.0, pool=10.0)
    return anthropic.Anthropic(timeout=timeout, max_retries=0)


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
    years_experience: Optional[float] = Field(
        default=None, description="Years working as an agent, if determinable (fractional OK, e.g. 5.5)"
    )
    active_listings_count: Optional[int] = Field(
        default=None, description="null if not confirmed (e.g. search ran out before checking) — 0 only if confirmed to be zero"
    )
    recent_listings: List[Listing] = Field(default_factory=list)
    sold_last_90_days_count: Optional[int] = Field(
        default=None, description="null if not confirmed (e.g. search ran out before checking) — 0 only if confirmed to be zero"
    )
    recent_sold: List[Listing] = Field(default_factory=list)
    has_team: Optional[bool] = Field(
        default=None, description="null if not confirmed either way — true/false only if actually determined"
    )
    team_size: Optional[int] = Field(
        default=None, description="null if not confirmed — 0 only if confirmed to be solo, no team"
    )
    social_profiles: List[SocialProfile] = Field(default_factory=list)
    notes: str = Field(default="", description="Anything else relevant to lead quality")

    # Defensive hardening, added after three separate live runs each spent
    # real research cost only to crash on structure()'s output not matching
    # the schema exactly (e.g. the model returning 5.75 for years_experience,
    # or null for a field we'd typed as a bare int). Rather than keep
    # patching one field at a time as each new mismatch surfaces — expensive,
    # since the crash happens after the costly research call already ran —
    # round any stray float down to int for the fields that should always be
    # whole numbers, before Pydantic's normally-strict validation runs.
    @model_validator(mode="before")
    @classmethod
    def _coerce_whole_number_fields(cls, data):
        if not isinstance(data, dict):
            return data
        for key in ("active_listings_count", "sold_last_90_days_count", "team_size"):
            value = data.get(key)
            if isinstance(value, float):
                data[key] = round(value)
        return data


# --------------------------------------------------------------------------- #
# 3. Research pass — Claude + web search builds a dossier
# --------------------------------------------------------------------------- #
RESEARCH_SYSTEM = """You are a research analyst qualifying inbound leads for a real-estate \
firm that recruits working real-estate agents. You are given a person's contact details \
from a lead form. Use web search to find out everything you can about them and decide \
whether they are genuinely a practising real-estate agent.

Investigate and gather concrete evidence for:
- Whether they are an active real-estate agent (and how confident you are).
- The agency/brokerage they CURRENTLY work for and their current agent profile page — not a \
past employer. Agents change agencies, and search results can show stale, cached content from a \
page that no longer reflects reality. If you find more than one agency associated with the \
person, don't just report the first or most-prominent one — use the fetch tool to actually load \
the agency's own official profile/team page directly rather than trusting a search snippet about \
it. If a claimed "current" profile page errors out or looks removed when fetched, treat that as \
real evidence the person has left that agency, not as a dead end to ignore. Prefer whichever \
evidence carries the most recent dates (recent listings, recent award mentions, a live team page) \
over older material that may be stale or cached.
- Their active listings right now (find 2-3 of the most recent, with address + price + link).
- A recently sold listing (ideally within the last ~90 days).
- Whether they work in a team and roughly how many people are with them.
- Roughly how many years they have been an agent.
- Their active social/professional profiles (LinkedIn, Instagram, Facebook business page, etc.).

Search by name plus likely qualifiers (real estate, the agency, the region). The email may be \
a personal address, so don't rely on the email domain to find them. Be honest about uncertainty: \
if you cannot find evidence the person is an agent, say so clearly rather than guessing. Cite the \
sources you used. End with a clear written dossier covering every point above.

Once you have solid evidence for each point above, stop searching — you do not need to \
exhaustively check every listing site, social network, or directory a well-established agent \
might appear on. A handful of good sources is enough; prioritize finishing over completeness."""

STRUCTURE_SYSTEM = """Extract the research dossier into a single JSON object. Only record facts \
supported by the dossier; set is_real_estate_agent accordingly. Do not invent listings, teams, or tenure.

For active_listings_count, sold_last_90_days_count, has_team, and team_size specifically: the dossier will \
often distinguish between "confirmed none" and "not confirmed / ran out of search budget before \
checking" (e.g. it may say something was blocked by a search-tool limit, or that it simply wasn't \
investigated). This distinction matters a lot downstream, so preserve it exactly:
  - Use `null` when the dossier did not actually confirm the answer either way — including "the \
    search tool hit its usage cap before this could be checked." Do NOT default to 0/false just \
    because a number wasn't stated.
  - Use a real 0 / false ONLY when the dossier explicitly states the count is zero / no team was \
    found, as an actual confirmed finding, not merely the absence of a number in the text.

Return ONLY the JSON object (no prose, no markdown fences), with exactly this shape:
{
  "is_real_estate_agent": true,
  "confidence": "high|medium|low",
  "agency_name": "",
  "agency_website": "",
  "profile_url": "",
  "years_experience": null,
  "active_listings_count": null,
  "recent_listings": [{"address": "", "price": "", "url": ""}],
  "sold_last_90_days_count": null,
  "recent_sold": [{"address": "", "price": "", "url": ""}],
  "has_team": null,
  "team_size": null,
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
    tools = [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 15},
        # No per-use fee (token cost only, ~2,500 tokens for a typical page) — capped low since
        # this is meant for targeted verification of a claimed "current agency" page, not broad
        # fetching. max_content_tokens guards against an unexpectedly large page.
        #
        # use_cache=False matters a lot for what this tool is actually for here: by default
        # web_fetch may serve Anthropic's own cached copy of a page rather than the live one
        # ("may not always reflect the latest version available at the URL" per Anthropic's
        # docs) — observed directly: fetching an agent's ex-employer's profile page returned
        # 200 OK with full content, when the live page 404s. That's exactly backwards for
        # "verify this person still works here." use_cache requires web_fetch_20260309+.
        {
            "type": "web_fetch_20260309",
            "name": "web_fetch",
            "max_uses": 3,
            "max_content_tokens": 8000,
            "use_cache": False,
        },
    ]

    for round_num in range(1, 7):  # bound the server-side tool loop (pause_turn)
        round_start = time.monotonic()
        print(f"     [research round {round_num}] calling {MODEL} (streaming) ...", file=sys.stderr)
        # Streaming, not a single blocking create() call: a long non-streaming
        # request can hit infra-level timeouts (proxies between us and the
        # model) independent of any client-side timeout we configure, once
        # the server goes quiet for too long mid-generation. A continuous
        # stream of events keeps the connection visibly alive throughout, so
        # nothing in between kills it prematurely. See
        # https://docs.anthropic.com/en/api/errors#long-requests.
        #
        # Streaming alone would remove our own ceiling too, though — nothing
        # about it stops a genuinely slow-but-progressing generation from
        # running indefinitely. So we still enforce REQUEST_TIMEOUT_SECONDS
        # ourselves, checked against wall-clock time on every event, and
        # deliberately abort past that point rather than trusting either the
        # infrastructure or an open-ended stream to bound it for us.
        deadline = round_start + REQUEST_TIMEOUT_SECONDS
        with client.messages.stream(
            model=MODEL,
            max_tokens=8000,
            system=RESEARCH_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            tools=tools,
            messages=messages,
        ) as stream:
            for _event in stream:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"research round {round_num} exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s while "
                        f"streaming — aborting deliberately rather than waiting indefinitely"
                    )
            resp = stream.get_final_message()
        elapsed = time.monotonic() - round_start
        n_searches = sum(1 for b in resp.content if getattr(b, "type", "") == "server_tool_use")
        print(
            f"     [research round {round_num}] {elapsed:.1f}s  "
            f"stop_reason={resp.stop_reason}  searches_this_round={n_searches}  usage={resp.usage}",
            file=sys.stderr,
        )
        # Which URLs web_fetch actually pulled, and how large each one was —
        # added after a single round unexpectedly added ~635k input tokens
        # with only 3 fetches, well beyond the ~2,500 tokens/page estimate
        # for a typical page. This tells us exactly which URL was the culprit
        # next time, instead of guessing.
        for b in resp.content:
            if getattr(b, "type", "") != "web_fetch_tool_result":
                continue
            content = getattr(b, "content", None)
            url = getattr(content, "url", "?")
            if getattr(content, "type", "") == "web_fetch_tool_result_error":
                print(f"     [research round {round_num}] web_fetch ERROR  url={url}  error_code={getattr(content, 'error_code', '?')}", file=sys.stderr)
                continue
            doc = getattr(content, "content", None)
            source = getattr(doc, "source", None)
            data = getattr(source, "data", "") if source else ""
            size = len(data) if isinstance(data, str) else "n/a (binary)"
            print(f"     [research round {round_num}] web_fetch OK  url={url}  content_chars={size}", file=sys.stderr)
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

    Runs on STRUCTURE_MODEL (Haiku), not the research model — this is
    mechanical extraction from a dossier Opus already wrote, not a judgment
    call, so it doesn't need Opus-level reasoning. Meaningfully cheaper for
    no real quality cost.
    """
    resp = client.messages.create(
        model=STRUCTURE_MODEL,
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
    incomplete: bool = False  # True if scored with one or more categories unconfirmed — see score()


def score(ev: Evidence, lead: Lead) -> Score:
    pts = 0
    bd: List[str] = []
    unconfirmed: List[str] = []

    def add(n: int, reason: str):
        nonlocal pts
        pts += n
        bd.append(f"  {'+' if n >= 0 else ''}{n:>3}  {reason}")

    # No evidence of being an agent -> not scorable, drops to P6.
    if not ev.is_real_estate_agent or ev.confidence == "low":
        bd.append("   --  No clear evidence this person is a working real-estate agent")
        return Score(min(pts, 15), "P6", "Not clearly an agent — verify manually", bd)

    add(20, f"Confirmed real-estate agent (confidence: {ev.confidence})")

    # active_listings_count / sold_last_90_days_count / has_team are all
    # Optional: None means the research didn't actually confirm the answer
    # (e.g. ran out of search budget), a real 0/false means it did and found
    # none. Unconfirmed categories are skipped rather than scored as zero —
    # a lead shouldn't be penalized for a gap in the research — and flagged
    # via `incomplete` so the score is never silently treated as final.
    if ev.active_listings_count is None:
        bd.append("   ??  Active listings not confirmed (research incomplete)")
        unconfirmed.append("active listings")
    elif ev.active_listings_count >= 3:
        add(20, f"{ev.active_listings_count} active listings")
    elif ev.active_listings_count >= 1:
        add(10, f"{ev.active_listings_count} active listing(s)")
    else:
        bd.append("    0  No active listings found (confirmed)")

    if ev.sold_last_90_days_count is None:
        bd.append("   ??  Recent solds not confirmed (research incomplete)")
        unconfirmed.append("recent solds")
    elif ev.sold_last_90_days_count >= 2:
        add(20, f"{ev.sold_last_90_days_count} sold in ~last 90 days")
    elif ev.sold_last_90_days_count >= 1:
        add(10, f"{ev.sold_last_90_days_count} recent sold")
    else:
        bd.append("    0  No recent solds found (confirmed)")

    if ev.has_team is None and not ev.team_size:
        bd.append("   ??  Team status not confirmed (research incomplete)")
        unconfirmed.append("team status")
    elif (ev.team_size or 0) >= 1 or ev.has_team:
        add(15, f"Works in a team (~{ev.team_size or 'size unknown'})")

    if ev.years_experience is not None:
        yexp_display = round(ev.years_experience, 1)
        if ev.years_experience >= 7:
            add(15, f"{yexp_display} yrs experience")
        elif ev.years_experience >= 5:
            add(12, f"{yexp_display} yrs experience")
        elif ev.years_experience >= 3:
            add(8, f"{yexp_display} yrs experience")
        else:
            add(3, f"{yexp_display} yrs experience (early career)")

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

    incomplete = len(unconfirmed) > 0
    if incomplete:
        bd.append(f"   !!  Score may be understated — {', '.join(unconfirmed)} could not be confirmed")

    return Score(pts, prio, label, bd, incomplete=incomplete)


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
    if sc.incomplete:
        w("  ⚠  INCOMPLETE — one or more categories could not be confirmed (see")
        w("     SCORE BREAKDOWN below). This score may be understated; consider a")
        w("     manual follow-up rather than treating it as final.")
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
    yexp = round(ev.years_experience) if ev.years_experience is not None else "unknown"
    w(f"  Experience:      {yexp} years")
    if ev.has_team is None and not ev.team_size:
        w("  Team:            not confirmed")
    else:
        w(f"  Team:            {'Yes' if (ev.has_team or ev.team_size) else 'No / solo'}"
          + (f" (~{ev.team_size} people)" if ev.team_size else ""))
    w("")
    w(f"  Active listings: {ev.active_listings_count if ev.active_listings_count is not None else 'not confirmed'}")
    for x in ev.recent_listings:
        w(f"     - {x.address}{(' — ' + x.price) if x.price else ''}")
        if x.url:
            w(f"       {x.url}")
    w(f"  Sold (~90 days): {ev.sold_last_90_days_count if ev.sold_last_90_days_count is not None else 'not confirmed'}")
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

    client = make_client()
    print(f"Enriching {len(leads)} lead(s) with {MODEL}\n", file=sys.stderr)
    for lead in leads:
        try:
            write_report(enrich_one(client, lead))
        except Exception as e:  # keep a batch going if one lead fails
            print(f"  !! failed for {lead.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
