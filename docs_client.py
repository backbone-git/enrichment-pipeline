#!/usr/bin/env python3
"""
Google Docs client — creates one dossier Doc per lead, titled with the
lead's name, inside a shared Drive folder. Used by pipeline.py; the
enrich.py CLI path is unaffected and still writes local .txt reports.

Auth: a Google service account (unattended — no OAuth consent flow to
babysit on a schedule). GOOGLE_SERVICE_ACCOUNT_JSON may be either a path to
the key file, or the raw JSON key content itself (so it can be dropped
straight into a GitHub Actions secret without a file on disk).
"""

import json
import os
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build

try:  # load config from a local .env if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

_docs_service = None
_drive_service = None


def _load_credentials() -> service_account.Credentials:
    raw = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    if os.path.isfile(raw):
        return service_account.Credentials.from_service_account_file(raw, scopes=SCOPES)
    return service_account.Credentials.from_service_account_info(json.loads(raw), scopes=SCOPES)


def _get_services():
    global _docs_service, _drive_service
    if _docs_service is None:
        creds = _load_credentials()
        _docs_service = build("docs", "v1", credentials=creds)
        _drive_service = build("drive", "v3", credentials=creds)
    return _docs_service, _drive_service


def create_dossier_doc(title: str, report_text: str, folder_id: Optional[str] = None) -> str:
    """Create a Google Doc titled `title` containing `report_text`, inside
    `folder_id` (defaults to the GOOGLE_DRIVE_FOLDER_ID env var — must be a
    folder inside a Shared Drive the service account is a member of).
    Returns the Doc's URL.

    Created directly via the Drive API with the target folder as `parents`
    at creation time, rather than the more obvious "create via the Docs API,
    then move it" — a bare service account has zero Drive storage quota of
    its own, so `documents.create()` (which always lands in the caller's own
    My Drive root first) 403s regardless of what folder you intend to move
    it to afterward. Creating straight into a Shared Drive folder sidesteps
    this entirely, since the file's storage is attributed to the Shared
    Drive, not the service account. `supportsAllDrives=True` is required on
    every call that touches Shared Drive content.

    No de-duplication on title: Google Docs are addressed by ID, not title,
    so multiple docs sharing a lead's name (e.g. a re-enriched repeat lead)
    is expected and fine — matches the Note's "accumulate, don't overwrite"
    behavior on the AC side.
    """
    docs, drive = _get_services()
    folder_id = folder_id or os.environ["GOOGLE_DRIVE_FOLDER_ID"]

    file = drive.files().create(
        body={"name": title, "mimeType": "application/vnd.google-apps.document", "parents": [folder_id]},
        supportsAllDrives=True,
        fields="id",
    ).execute()
    doc_id = file["id"]

    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": report_text}}]},
    ).execute()

    return f"https://docs.google.com/document/d/{doc_id}/edit"
