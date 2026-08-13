#!/usr/bin/env python3
"""
Pipeline configuration — AC tag names and custom field IDs.

All IDs below are real, created against the live independentagent AC
account on 2026-08-12/13 (Phase 3 setup). Contact and deal custom fields
are separate AC objects with their own numeric ID sequences — same label,
different IDs, addressed separately since AC has no shared-field concept
across object types.

Both "Enrichment" field groups (contact group_id=6, deal group_id=7) exist
so these fields are visually grouped together in AC's UI rather than mixed
in with the ~20 pre-existing fields. Note: contact custom fields auto-join
a default group (group_id=1) on creation and have to be *moved* into a
different group (delete the existing groupMember, create a new one) — a
plain second POST to /groupMembers 409s. Deal custom fields did not exhibit
this auto-join behavior; a direct POST worked first try.
"""

TAG_PENDING = "Pending Enrichment"  # AC tag id 24
TAG_DONE = "Enriched"  # AC tag id 25

# AC custom field IDs — logical name -> numeric field ID.
CONTACT_FIELD_IDS = {
    "score": 23,  # "Enrichment Score" (text)
    "priority": 25,  # "Enrichment Priority" (dropdown, 6 options — see PRIORITY_LABELS)
    "doc_link": 24,  # "Enrichment Doc Link" (text)
}

DEAL_FIELD_IDS = {
    "score": 18,  # "Enrichment Score" (number)
    "priority": 20,  # "Enrichment Priority" (dropdown, 6 options — see PRIORITY_LABELS)
    "doc_link": 19,  # "Enrichment Doc Link" (text)
}

# Self-report "are you a real estate agent?" signal — NOT read by pipeline.py
# itself (see build plan: that gating only ever happens upstream, in the AC
# automation, before research runs). Recorded here purely as the answer to
# the plan's open question, for whoever configures that automation's branch
# condition: contact field id=3 ("Stage of Career", a combined qualifying +
# experience-tier dropdown), self-reported non-agent = value exactly
# "I’m not a real estate agent" (curly apostrophe, as AC stores it).
SELF_REPORT_FIELD_ID = 3
SELF_REPORT_NOT_AGENT_VALUE = "I’m not a real estate agent"

# Source-varying extra fields already mapped into AC by each source's Zap —
# label -> AC contact custom field ID. Confirmed from the live account:
# ScoreApp populates these three. Any field listed here that's blank for a
# given contact is simply omitted from that contact's Note (other sources,
# e.g. the plain Facebook lead form, don't populate them at all).
EXTRA_INFO_FIELD_IDS = {
    "ScoreApp Score": 20,
    "ScoreApp Results PDF": 22,
    # "ScoreCard Answers" (id 21, textarea — the full raw quiz answers) is
    # deliberately left out: verbose free text, better suited to being
    # viewed directly on the AC contact than repeated in a curated Note.
    # Add it here if you'd rather it flow through.
}

# The six Enrichment Priority dropdown option values, exactly as they must
# be written to AC — full descriptive label, not the bare P1..P6 code.
PRIORITY_LABELS = {
    "P1": "P1 — Call first — strong, active agent",
    "P2": "P2 — High priority",
    "P3": "P3 — Good — worth a call",
    "P4": "P4 — Moderate",
    "P5": "P5 — Low — light evidence",
    "P6": "P6 — Not clearly an agent — verify manually",
}
