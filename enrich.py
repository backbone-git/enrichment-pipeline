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
import re
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

MODEL = "claude-opus-4-8"  # research_pass1()/research_pass2() — judgment-heavy, kept on Opus
STRUCTURE_MODEL = "claude-haiku-4-5-20251001"  # structure() — mechanical extraction from an already-written dossier, doesn't need Opus
REPORTS_DIR = Path("reports")

# A hung request (rare, but seen in practice — one research call ran 20+
# minutes with no error) shouldn't be able to block a batch or a scheduled
# pipeline run indefinitely. This bounds every individual request made by
# the client — not a whole research pass, which can itself make up to 6
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


# Bias (not a hard filter) toward NZ results — cuts down on wasted
# disambiguation effort against international namesakes (observed directly:
# a "Gary Noakes" search surfaced an unrelated financial adviser in
# Adelaide, Australia, that the model had to spend a search ruling out).
NZ_LOCATION = {"type": "approximate", "country": "NZ"}

# The known-good NZ real-estate aggregator sites — reliably carry rich,
# structured agent data (listings, sales history, reviews) in one place when
# a profile exists, so restricting the first research pass to these is both
# cheaper and more reliable than an open, undirected search. Deliberately
# does NOT include agency sites themselves or general search — those matter
# most for the *current employer* question, which needs pass 2's open
# search precisely because a very recent agency change may not be reflected
# on any of these aggregators yet.
KNOWN_AGENT_SITES = [
    "trademe.co.nz",
    "realestate.co.nz",
    "homes.co.nz",
    "ratemyagent.co.nz",
    "oneroof.co.nz",
]


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
# 3. Research — two passes instead of one open-ended call.
#
# Live testing showed cost and failure risk scaling with how much real
# information existed about a lead — exactly backwards, since a
# well-documented, active agent is the profile we most want to score well.
# A single open-ended call has to search, disambiguate, verify, and
# synthesize everything about a person in one unbounded turn, and the more
# there is to find, the more that turn costs and the more likely it is to
# time out. Splitting into a cheap, targeted first pass and an optional,
# narrowly-scoped second pass keeps cheap leads cheap and bounds the
# expensive work to exactly the one question generic search handles worst.
# --------------------------------------------------------------------------- #
RESEARCH_SYSTEM_PASS1 = """You are a research analyst qualifying inbound leads for a real-estate \
firm that recruits working real-estate agents. You are given a person's contact details from a \
lead form. Your web search is restricted to the major NZ real-estate aggregator sites (Trade Me \
Property, realestate.co.nz, homes.co.nz, RateMyAgent, OneRoof) — these reliably carry rich, \
structured agent data (listings, sales history, reviews) in one place when a profile exists, so \
this pass should be quick and cheap.

Investigate and gather concrete evidence for:
- Whether they are an active real-estate agent (and how confident you are).
- The agency/brokerage associated with them on these sites. This may be out of date — a separate \
pass will specifically verify their CURRENT employer — so just report what these sites show \
without exhaustively chasing verification here.
- Their active listings right now (find 2-3 of the most recent, with address + price + link).
- A recently sold listing (ideally within the last ~90 days).
- Whether they work in a team and roughly how many people are with them.
- Roughly how many years they have been an agent.
- Their active social/professional profiles, if shown on these sites.

Search by name plus likely qualifiers (real estate, the agency, the region) — but also run at \
least one search on the raw email address and, if given, the raw phone number, verbatim. These \
often surface an agent's own listing/profile pages directly in the snippet (including specifics \
like a stated active-listing count) that a name-only search misses. (Don't rely on the email \
*domain* to guess an employer, though — a personal address like gmail.com tells you nothing.) Be \
honest about uncertainty: \
if you cannot find evidence the person is an agent on these sites, say so clearly rather than \
guessing — they may still be a genuine agent who simply isn't well represented here, that's fine, \
don't stretch to compensate. Cite the sources you used. End with a clear written dossier covering \
every point above.

Once you have solid evidence for each point, stop searching — a handful of good sources is enough."""

RESEARCH_SYSTEM_PASS2 = """You are continuing a real-estate lead qualification, focused \
specifically on verifying the person's CURRENT employer. An earlier pass (restricted to major \
aggregator sites) already produced a first dossier, included in the user message below — treat \
its agency claim as a starting hypothesis, not a confirmed fact.

Agents change agencies, and search results — even a directly fetched page — can show stale, \
cached content that no longer reflects reality. Your job:
- Use web search plus the fetch tool to confirm who they currently work for. Actually fetch the \
agency's own official profile/team page directly rather than trusting a search snippet about it — \
a claimed "current" profile page that errors out or looks removed when fetched is itself real \
evidence the person has left that agency.
- If you find a different, more recent agency than the one in the first pass, that supersedes it \
— report the more recent one as current, and note the discrepancy explicitly.
- Once the agency question is settled, if you have fetch budget left, spend it confirming the \
active listing count directly, from BOTH of these when a URL for them exists (from the first \
pass's dossier, or find one): the person's Trade Me Property agent profile \
(trademe.co.nz/a/property/agent/<Name>), and their realestate.co.nz agent profile \
(realestate.co.nz/agent/<id>/<name>) — its page states the count outright (e.g. "Displaying 1-1, \
out of 1 active listings"). Fetch both directly rather than trusting a search snippet or either \
site's own summary/stats widget for this — observed directly: homes.co.nz's agent-stats data can \
disagree with a live, freshly-fetched realestate.co.nz page (reporting 0 active listings when \
realestate.co.nz's own page — fetched live, not cached — showed 1), so a single aggregator's \
number, or a search snippet, isn't enough to call it "confirmed zero." If the two fetched pages \
disagree, report the higher/more-specific figure as current and note the discrepancy explicitly \
rather than silently picking one. Note: RateMyAgent blocks direct fetching (returns a 403) — \
don't spend a fetch call on it, only use it via search if it comes up. If there's still budget \
after that, also try to fill in anything else the first pass flagged as not found or unconfirmed \
(recent sales, team, social profiles) — but the agency question is always the priority; don't \
spend budget on anything else at the expense of settling that one.
- For social profiles specifically, if you still have search calls left after everything above: \
try one or two searches like site:instagram.com and site:facebook.com against their name plus \
the confirmed agency (e.g. "Mel Farani Barfoot Thompson"). Only report a profile as theirs if the \
bio/name in the result clearly matches this specific person (real-estate branding, their agency \
name, their suburb) — common names return a lot of unrelated accounts, and an unverified guess is \
worse than reporting none found. Skip this entirely if you're low on search calls; it's the \
lowest-priority item in this pass.

End with a written summary that clearly states the confirmed current agency and cites the source \
that confirms it — or clearly states you could not confirm one, if that's the honest answer."""

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


def _lead_intro(lead: Lead) -> str:
    return (
        f"Lead from a Facebook lead form:\n"
        f"- Name: {lead.name}\n"
        f"- Email: {lead.email} ({'personal' if lead.email_is_personal else 'work/domain'} address)\n"
        f"- Phone: {lead.phone or 'n/a'}\n"
        f"- Location hint: {lead.location or 'n/a'}\n\n"
    )


def _run_research(client: anthropic.Anthropic, system_prompt: str, tools: list, user: str, label: str) -> str:
    """Shared agentic-turn runner behind both research passes: streaming,
    deadline-enforced, bounded pause_turn loop, real-time + post-round
    diagnostics. `label` just prefixes log lines ("pass1"/"pass2")."""
    messages = [{"role": "user", "content": user}]

    for round_num in range(1, 7):  # bound the server-side tool loop (pause_turn)
        round_start = time.monotonic()
        print(f"     [{label} round {round_num}] calling {MODEL} (streaming) ...", file=sys.stderr)
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
        tool_call_count = 0
        with client.messages.stream(
            model=MODEL,
            max_tokens=8000,
            system=system_prompt,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            tools=tools,
            messages=messages,
        ) as stream:
            for event in stream:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{label} round {round_num} exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s while "
                        f"streaming — aborting deliberately rather than waiting indefinitely"
                    )
                # Real-time progress, not just post-round: a tool call
                # starting is the clearest sign of forward progress, and the
                # thing we most need visibility into if a round times out —
                # added after two leads timed out with zero diagnostic
                # output, since the post-round print below only ever fires
                # on a round that actually completes.
                block = getattr(event, "content_block", None)
                if getattr(event, "type", "") == "content_block_start" and getattr(block, "type", "") == "server_tool_use":
                    tool_call_count += 1
                    print(
                        f"     [{label} round {round_num}] tool call #{tool_call_count}: "
                        f"{getattr(block, 'name', '?')} ({time.monotonic() - round_start:.0f}s in) ...",
                        file=sys.stderr,
                    )
            resp = stream.get_final_message()
        elapsed = time.monotonic() - round_start
        n_searches = sum(1 for b in resp.content if getattr(b, "type", "") == "server_tool_use")
        print(
            f"     [{label} round {round_num}] {elapsed:.1f}s  "
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
                print(f"     [{label} round {round_num}] web_fetch ERROR  url={url}  error_code={getattr(content, 'error_code', '?')}", file=sys.stderr)
                continue
            doc = getattr(content, "content", None)
            source = getattr(doc, "source", None)
            data = getattr(source, "data", "") if source else ""
            size = len(data) if isinstance(data, str) else "n/a (binary)"
            print(f"     [{label} round {round_num}] web_fetch OK  url={url}  content_chars={size}", file=sys.stderr)
        if resp.stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp.content},
            ]
            continue
        break

    return "\n".join(b.text for b in resp.content if b.type == "text").strip()


def research_pass1(client: anthropic.Anthropic, lead: Lead) -> str:
    """Cheap, fast pass: search restricted to known NZ real-estate
    aggregator sites only, no fetch. Enough to get a verdict and a rough
    picture for most leads on its own."""
    tools = [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 6,
            "allowed_domains": KNOWN_AGENT_SITES,
            "user_location": NZ_LOCATION,
            # _20260209+ defaults to routing calls through code execution
            # ("dynamic filtering") to trim results before they hit context —
            # and that code-execution loop isn't bounded by max_uses at all
            # (only web_search_requests/web_fetch_requests count against it).
            # Observed directly: a lead's pass 1 got stuck making repeated
            # code_execution/bash_code_execution calls with growing gaps
            # between them (39s -> 71s -> 138s -> 266s) until the 300s
            # deadline killed it — inside the "cheap" 6-search pass. Forcing
            # direct invocation removes that failure mode entirely rather
            # than trying to bound a mechanism we don't control the shape of.
            "allowed_callers": ["direct"],
        },
    ]
    user = _lead_intro(lead) + "Research this person and produce the dossier."
    return _run_research(client, RESEARCH_SYSTEM_PASS1, tools, user, label="pass1")


def research_pass2(client: anthropic.Anthropic, lead: Lead, pass1_dossier: str) -> str:
    """Targeted, more expensive pass — only run for leads pass 1 already
    found agent evidence for. Open search + fetch (cache bypassed), aimed
    specifically at confirming the CURRENT employer, since that's the one
    question the known aggregator sites answer worst (a very recent agency
    change may not be reflected on any of them yet)."""
    tools = [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 8,
            "user_location": NZ_LOCATION,
            "allowed_callers": ["direct"],  # see research_pass1 for why
        },
        # No per-use fee (token cost only, ~2,500 tokens for a typical page) — capped low since
        # this is meant for targeted verification (the claimed "current agency" page, plus up to
        # two listing-count sources: Trade Me and realestate.co.nz), not broad fetching.
        # max_content_tokens guards against an unexpectedly large page.
        #
        # use_cache=False matters a lot for what this tool is actually for here: by default
        # web_fetch may serve Anthropic's own cached copy of a page rather than the live one
        # ("may not always reflect the latest version available at the URL" per Anthropic's
        # docs) — observed directly: fetching an agent's ex-employer's profile page returned
        # 200 OK with full content, when the live page 404s. That's exactly backwards for
        # "verify this person still works here." use_cache requires web_fetch_20260309+.
        #
        # allowed_callers: web_fetch_20260209+ defaults to code-execution-mediated
        # calls too, same as web_search — see research_pass1 for the failure this caused.
        {
            "type": "web_fetch_20260309",
            "name": "web_fetch",
            "max_uses": 4,  # employer page + Trade Me + realestate.co.nz, plus one spare
            "max_content_tokens": 8000,
            "use_cache": False,
            "allowed_callers": ["direct"],
        },
    ]
    user = (
        _lead_intro(lead)
        + "Here is what an initial pass (restricted to major NZ real-estate aggregator sites) "
          "already found:\n\n---\n" + pass1_dossier + "\n---\n\n"
        + "Your job now: verify this person's CURRENT employer specifically — see your "
          "instructions above."
    )
    return _run_research(client, RESEARCH_SYSTEM_PASS2, tools, user, label="pass2")


# The only fields in this schema where null is intentional/expected — see
# Evidence's field descriptions. Everything else is a plain string (or a
# list), and the model will sometimes return null for one of those it has
# no value for (e.g. a listing's price when unstated, or a whole list when
# it found nothing) — natural given how strongly the rest of this schema
# says "use null when unconfirmed," but it crashes Pydantic validation for
# fields that were never meant to be nullable, no matter how deeply nested
# (recent_listings[i].price, social_profiles[i].url, etc.). Rather than
# patch one field at a time as each new one surfaces after an
# already-costly research call, sanitize generically before validation:
_NULLABLE_EVIDENCE_FIELDS = {"active_listings_count", "sold_last_90_days_count", "has_team", "team_size", "years_experience"}
_LIST_EVIDENCE_FIELDS = {"recent_listings", "recent_sold", "social_profiles"}
_NUMERIC_EVIDENCE_FIELDS = {"active_listings_count", "sold_last_90_days_count", "team_size", "years_experience"}


def _sanitize_nulls(obj):
    """Recursively: null on an intentionally-nullable field stays null; null
    on a list field becomes []; null on anything else becomes ""."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if v is None:
                if k in _NULLABLE_EVIDENCE_FIELDS:
                    result[k] = None
                elif k in _LIST_EVIDENCE_FIELDS:
                    result[k] = []
                else:
                    result[k] = ""
            else:
                result[k] = _sanitize_nulls(v)
        return result
    if isinstance(obj, list):
        return [_sanitize_nulls(x) for x in obj]
    return obj


def _coerce_numeric_strings(data: dict) -> dict:
    """A numeric field can come back as a non-numeric string (e.g. "several",
    "5+") — conversational rather than malformed, but Pydantic won't coerce
    it. Pull out a leading number if there is one; otherwise treat it the
    same as "not confirmed" (None) rather than crash on it."""
    for key in _NUMERIC_EVIDENCE_FIELDS:
        value = data.get(key)
        if isinstance(value, str):
            match = re.search(r"[\d.]+", value)
            data[key] = float(match.group()) if match else None
    return data


def _normalize_verdict_fields(data: dict) -> dict:
    """The two fields score()'s P6 safety check depends on — is_real_estate_agent
    and confidence — get normalized defensively, in different directions on
    purpose: the dossier's own verdict language is three-way (YES/NO/UNCLEAR),
    but the schema is strictly bool, so anything not a clean true/false
    collapses to False (treating "unclear" as "not confirmed" is the safe
    direction of error — the dangerous one would be defaulting an uncertain
    case to True). confidence has no enum constraint at the type level, so
    anything other than exactly "high"/"medium"/"low" collapses to "low".
    This isn't just crash-prevention: without it, an unrecognized confidence
    value like "very uncertain" silently fails score()'s `confidence == "low"`
    check and slips through to a normal (wrong) score instead of P6 — no
    crash, no error, nothing to notice. That's worse than a crash."""
    agent = data.get("is_real_estate_agent")
    if isinstance(agent, str):
        data["is_real_estate_agent"] = agent.strip().lower() in ("true", "yes", "y", "1")
    elif not isinstance(agent, bool):
        data["is_real_estate_agent"] = False

    confidence = data.get("confidence")
    normalized = str(confidence).strip().lower() if confidence is not None else ""
    data["confidence"] = normalized if normalized in ("high", "medium", "low") else "low"

    notes = data.get("notes")
    if isinstance(notes, list):
        data["notes"] = "; ".join(str(n) for n in notes)

    return data


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
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # A clean crash either way, but log what we actually got before it's
        # lost — this is the one failure mode that isn't a known field-level
        # shape mismatch, so there's nothing to sanitize; better to fail with
        # the raw text on hand than a bare "line 1 column 1" JSONDecodeError.
        print(f"     [structure] failed to parse JSON, raw text was: {text[:500]!r}", file=sys.stderr)
        raise
    data = _coerce_numeric_strings(_normalize_verdict_fields(data))
    return Evidence.model_validate(_sanitize_nulls(data))


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
    decide whether/where to persist the result (local file, AC, etc.).

    Two-pass research: a cheap pass restricted to known NZ real-estate
    aggregator sites first, then a structure() call to check whether it's
    even worth continuing. Only leads with actual agent evidence get a
    second, more expensive pass (open search + fetch) — narrowly targeted at
    the one question aggregator sites answer worst: who's their CURRENT
    employer. Non-agents (and low-confidence cases) stay on the cheap path
    entirely. See RESEARCH_SYSTEM_PASS1/PASS2 and the build plan for why.
    """
    print(f"  -> researching {lead.name} (pass 1: known NZ sites) ...", file=sys.stderr)
    pass1_dossier = research_pass1(client, lead)

    print(f"  -> structuring pass-1 evidence ...", file=sys.stderr)
    ev = structure(client, pass1_dossier)
    dossier = pass1_dossier

    if ev.is_real_estate_agent and ev.confidence != "low":
        print(f"  -> researching {lead.name} (pass 2: verify current agency) ...", file=sys.stderr)
        pass2_dossier = research_pass2(client, lead, pass1_dossier)
        dossier = pass1_dossier + "\n\n=== PASS 2 (current-agency verification) ===\n\n" + pass2_dossier
        print(f"  -> structuring combined evidence ...", file=sys.stderr)
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
