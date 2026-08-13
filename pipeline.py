#!/usr/bin/env python3
"""
Pipeline orchestrator.

Polls ActiveCampaign for contacts tagged "Pending Enrichment" (applied by
the existing AC automation as its last step, after it creates the deal),
runs the enrichment tool against each, creates a per-lead Google Doc
dossier, writes the results back onto the AC contact and every one of its
deals (regardless of status), then swaps the tag to "Enriched".

Never creates a contact or a deal — those stay the Zap's and the AC
automation's job respectively (see build plan for why). If a tagged contact
has no deal yet, the tag is left in place and it's retried on the next
scheduled run rather than erroring.

Run standalone for a manual/local pass:
    python pipeline.py
Scheduled via .github/workflows/enrich.yml in production.
"""

import sys
from typing import List

import anthropic

import ac_client
import config
import docs_client
from enrich import Lead, enrich_one, make_client

try:  # load ANTHROPIC_API_KEY / AC_API_URL / AC_API_KEY / Google config
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _contact_field_value(field_values: List[dict], field_id) -> str:
    if not field_id:
        return ""
    for fv in field_values:
        if str(fv.get("field")) == str(field_id):
            return fv.get("value", "")
    return ""


def _build_lead(contact: dict) -> Lead:
    first = contact.get("firstName", "")
    last = contact.get("lastName", "")
    name = f"{first} {last}".strip() or contact.get("email", "unknown")
    return Lead(name=name, email=contact.get("email", ""), phone=contact.get("phone", ""))


def _extra_info_lines(field_values: List[dict]) -> List[str]:
    lines = []
    for label, field_id in config.EXTRA_INFO_FIELD_IDS.items():
        value = _contact_field_value(field_values, field_id)
        if value:
            lines.append(f"{label}: {value}")
    return lines


def _note_body(result, extra_lines: List[str], doc_url: str) -> str:
    """Curated subset of Evidence, per the build plan — verdict + 6 evidence
    fields + Doc link. The raw dossier, itemized listings/socials, and
    ev.notes stay Doc-only."""
    ev = result.evidence
    lines = list(extra_lines)
    if lines:
        lines.append("")
    lines.append(
        f"Verdict: {'YES' if ev.is_real_estate_agent else 'NO / UNCLEAR'} "
        f"(confidence: {ev.confidence})"
    )
    lines.append(f"Agency: {ev.agency_name or 'unknown'}")
    if ev.agency_website:
        lines.append(f"Agency site: {ev.agency_website}")
    if ev.profile_url:
        lines.append(f"Agent profile: {ev.profile_url}")
    yexp = round(ev.years_experience) if ev.years_experience is not None else "unknown"
    lines.append(f"Experience: {yexp} years")
    if ev.has_team is None and not ev.team_size:
        lines.append("Team: not confirmed")
    else:
        lines.append(
            "Team: " + ("Yes" if (ev.has_team or ev.team_size) else "No / solo")
            + (f" (~{ev.team_size} people)" if ev.team_size else "")
        )
    active = ev.active_listings_count if ev.active_listings_count is not None else "not confirmed"
    sold = ev.sold_last_90_days_count if ev.sold_last_90_days_count is not None else "not confirmed"
    lines.append(f"Active listings: {active}")
    lines.append(f"Sold in ~90 days: {sold}")
    if result.score.incomplete:
        lines.append("")
        lines.append("⚠ INCOMPLETE — one or more categories above could not be confirmed by the")
        lines.append("research (e.g. search budget ran out). Score may be understated — worth a")
        lines.append("manual look rather than trusting this priority as final.")
    lines.append("")
    lines.append(f"Full dossier: {doc_url}")
    return "\n".join(lines)


def process_contact(client: anthropic.Anthropic, contact: dict) -> None:
    contact_id = contact["id"]
    lead = _build_lead(contact)
    print(f"-> {lead.name} <{lead.email}>  (AC contact {contact_id})", file=sys.stderr)

    result = enrich_one(client, lead)
    doc_url = docs_client.create_dossier_doc(lead.name, result.report)

    field_values = ac_client.get_contact_field_values(contact_id)
    extra_lines = _extra_info_lines(field_values)
    note_text = _note_body(result, extra_lines, doc_url)
    priority_label = config.PRIORITY_LABELS[result.score.priority]

    # Written regardless of eligibility — the pipeline never creates or
    # withholds a deal; that gating (where possible at all) happens
    # upstream in the AC automation, before research even runs.
    contact_values = {
        "score": str(result.score.points),  # Enrichment Score is TEXT on Contact
        "priority": priority_label,
        "doc_link": doc_url,
    }
    ac_client.update_contact_fields(contact_id, config.CONTACT_FIELD_IDS, contact_values)
    ac_client.add_note("Subscriber", contact_id, note_text)

    deals = ac_client.get_deals_for_contact(contact_id)
    if not deals:
        print(f"   no deal yet for {lead.name} — leaving tag, will retry next poll", file=sys.stderr)
        return  # leave "Pending Enrichment" tag in place; don't swap yet

    deal_values = {
        "score": result.score.points,  # Enrichment Score is NUMBER on Deal
        "priority": priority_label,
        "doc_link": doc_url,
    }
    for deal in deals:
        ac_client.update_deal_fields(deal["id"], config.DEAL_FIELD_IDS, deal_values)
        ac_client.add_note("Deal", deal["id"], note_text)

    ac_client.swap_tag(contact_id, remove=config.TAG_PENDING, add=config.TAG_DONE)
    print(f"   {result.score.priority} ({result.score.points}/100) — done", file=sys.stderr)


def main():
    client = make_client()
    contacts = ac_client.get_contacts_by_tag(config.TAG_PENDING)
    print(f"Found {len(contacts)} contact(s) tagged {config.TAG_PENDING!r}", file=sys.stderr)
    for contact in contacts:
        try:
            process_contact(client, contact)
        except Exception as e:  # keep the batch going if one lead fails
            print(f"  !! failed for contact {contact.get('id')}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
