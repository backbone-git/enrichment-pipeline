#!/usr/bin/env python3
"""
ActiveCampaign v3 REST API client.

Scope, per the build plan: the pipeline never creates a contact or a deal —
both are created upstream (Zap + AC automation). This client only reads
contacts tagged "Pending Enrichment", updates contact + open-deal custom
fields and notes, and swaps the tag to "Enriched".

Endpoint shapes were checked against ActiveCampaign's own API reference and
then live-tested against the real independentagent AC account during
Phase 3 setup. Two things that changed from the original guesses:
  - Deals are targeted regardless of open/won/lost status — deliberately
    not filtering by status (see config.py / build plan: rather than trust
    an unconfirmed status enum, every deal on the contact gets the same
    field/note update).
  - Listing a contact's tags is GET /contacts/{id}/contactTags (nested
    resource) — the filters[contact] query param on the flat /contactTags
    endpoint is silently ignored by this account, not an error, so it would
    have returned the wrong contacts' tags without ever raising.
"""

import os
from typing import Dict, List, Optional

import requests

try:  # load AC_API_URL / AC_API_KEY from a local .env if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = os.environ.get("AC_API_URL", "").rstrip("/")
API_TOKEN = os.environ.get("AC_API_KEY", "")


def _headers() -> dict:
    return {"Api-Token": API_TOKEN, "Content-Type": "application/json"}


def _url(path: str) -> str:
    return f"{BASE_URL}/api/3/{path.lstrip('/')}"


def _get(path: str, params: Optional[dict] = None) -> dict:
    r = requests.get(_url(path), headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    r = requests.post(_url(path), headers=_headers(), json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _put(path: str, body: dict) -> dict:
    r = requests.put(_url(path), headers=_headers(), json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> None:
    r = requests.delete(_url(path), headers=_headers(), timeout=30)
    r.raise_for_status()


# --------------------------------------------------------------------------- #
# Tags — the pipeline's queue mechanism
# --------------------------------------------------------------------------- #
def get_tag_id(tag_name: str) -> Optional[int]:
    data = _get("tags", params={"search": tag_name})
    for tag in data.get("tags", []):
        if tag["tag"] == tag_name:
            return int(tag["id"])
    return None


def get_contacts_by_tag(tag_name: str) -> List[dict]:
    """Contacts currently carrying `tag_name` — the poll query for
    "Pending Enrichment"."""
    tag_id = get_tag_id(tag_name)
    if tag_id is None:
        return []
    data = _get("contacts", params={"tagid": tag_id})
    return data.get("contacts", [])


def get_contact_field_values(contact_id: str) -> List[dict]:
    """A contact's custom field values (self-report answer, source-specific
    extra info, etc.) — fetched separately since the list-by-tag response
    doesn't embed them. VERIFY this nested-resource path against the live
    account during Phase 3."""
    data = _get(f"contacts/{contact_id}/fieldValues")
    return data.get("fieldValues", [])


def swap_tag(contact_id: str, remove: str, add: str) -> None:
    """Remove `remove` (if present) and add `add`. Both tags must already
    exist in AC (created during Phase 3 setup)."""
    add_id = get_tag_id(add)
    if add_id is None:
        raise RuntimeError(f"AC tag {add!r} does not exist — create it before running the pipeline")

    remove_id = get_tag_id(remove)
    if remove_id is not None:
        existing = _get(f"contacts/{contact_id}/contactTags").get("contactTags", [])
        for ct in existing:
            if str(ct.get("tag")) == str(remove_id):
                _delete(f"contactTags/{ct['id']}")

    _post("contactTags", {"contactTag": {"contact": contact_id, "tag": add_id}})


# --------------------------------------------------------------------------- #
# Contact — field values use the separate /fieldValues resource
# --------------------------------------------------------------------------- #
def update_contact_fields(contact_id: str, field_ids: Dict[str, int], values: Dict[str, str]) -> None:
    """`field_ids` maps our logical field names (e.g. "score", "priority",
    "doc_link") to AC's numeric custom field IDs — see config.py. `values`
    holds the value to write for each of those logical names."""
    for key, value in values.items():
        field_id = field_ids[key]
        _post("fieldValues", {"fieldValue": {"contact": contact_id, "field": field_id, "value": value}})


# --------------------------------------------------------------------------- #
# Deal — custom field values are embedded in the deal object itself
# --------------------------------------------------------------------------- #
def get_deals_for_contact(contact_id: str) -> List[dict]:
    """Every deal on this contact, regardless of status — see module
    docstring for why status isn't filtered on."""
    data = _get("deals", params={"filters[contact]": contact_id})
    return data.get("deals", [])


def update_deal_fields(deal_id: str, field_ids: Dict[str, int], values: Dict[str, str]) -> None:
    fields = [{"customFieldId": field_ids[key], "fieldValue": value} for key, value in values.items()]
    _put(f"deals/{deal_id}", {"deal": {"fields": fields}})


# --------------------------------------------------------------------------- #
# Notes — one shared endpoint for both contacts and deals
# --------------------------------------------------------------------------- #
def add_note(reltype: str, relid: str, text: str) -> None:
    """`reltype` is "Subscriber" for a contact or "Deal" for a deal."""
    _post("notes", {"note": {"note": text, "reltype": reltype, "relid": relid}})
