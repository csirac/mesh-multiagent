from __future__ import annotations

import os
import base64
import json
import datetime
import tempfile
import subprocess
from email.message import EmailMessage
from typing import List, Dict, Optional, Any, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

def _resolve_path(path: str) -> str:
    """Expand ~ using real home from /etc/passwd, not $HOME."""
    from ..paths import resolve_path
    return resolve_path(path)


def _normalize_attachments(val) -> list[dict]:
    """Normalize attachments to a list of dicts with 'path' keys.

    Handles LLMs passing a bare string, list of strings, single dict,
    or already-correct list of dicts.
    """
    if not val:
        return []
    if isinstance(val, str):
        return [{"path": val}]
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return [
            item if isinstance(item, dict) else {"path": str(item)}
            for item in val
        ]
    return []


class GmailClient:
    """
    Minimal Gmail API client for CLI usage, with high-level methods suitable
    to expose as tools to an LLM.
    """

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send",

        # Calendar scopes:
        "https://www.googleapis.com/auth/calendar.readonly",  # list events
        "https://www.googleapis.com/auth/calendar.events",    # create/modify events
    ]

    def __init__(self, credentials_file="credentials.json", token_file="token.json", confirmation_mode: str = "cli"):
        """
        confirmation_mode:
          - "cli": use interactive CLI prompts for send/reply/modify
          - "web": never block; return pending_confirmation payloads for web UI
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.creds: Optional[Credentials] = None
        self.service = None
        self._authorize()
        self.confirmation_mode = confirmation_mode
        self.user_confirm = (confirmation_mode == "cli")

    # =======================
    # Auth / initialization
    # =======================

    def _authorize(self):
        """Authorize and build Gmail service. Fail gracefully if credentials missing."""
        if not os.path.exists(self.credentials_file):
            print(f"[GmailClient] credentials file not found: {self.credentials_file}")
            print(
                "Download an OAuth client JSON from Google Cloud Console and save it "
                "there, or pass credentials_file=... to GmailClient()."
            )
            self.service = None
            return

        if os.path.exists(self.token_file):
            self.creds = Credentials.from_authorized_user_file(
                self.token_file, self.SCOPES
            )

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    print(f"[GmailClient] Failed to refresh credentials: {e}")
                    self.service = None
                    return
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.SCOPES
                    )
                    self.creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"[GmailClient] OAuth flow failed: {e}")
                    self.service = None
                    return

            try:
                with open(self.token_file, "w") as token:
                    token.write(self.creds.to_json())
            except Exception as e:
                print(f"[GmailClient] Failed to write token file: {e}")

        try:
            self.service = build("gmail", "v1", credentials=self.creds)
        except Exception as e:
            print(f"[GmailClient] Failed to build Gmail service: {e}")
            self.service = None

    @property
    def ready(self) -> bool:
        """True if the client is usable (service created)."""
        return self.service is not None

    def _get_default_from(self) -> Optional[str]:
        """Return the default From header ("Name <email>") from Gmail settings, if any.

        We query users.settings.sendAs() once and cache the default display name + address.
        If anything fails, return None and let Gmail fill a bare address.
        """
        if not self.ready:
            return None

        cached = getattr(self, '_default_from', None)
        if cached is not None:
            return cached

        try:
            settings = (
                self.service.users()
                .settings()
                .sendAs()
                .list(userId="me")
                .execute()
            )
            send_as = settings.get('sendAs', []) or []
            default = None
            for s in send_as:
                if s.get('isDefault'):
                    default = s
                    break
            if default is None and send_as:
                default = send_as[0]

            if not default:
                self._default_from = None
                return None

            name = default.get('displayName') or ''
            email = default.get('sendAsEmail') or ''
            if name and email:
                self._default_from = f"{name} <{email}>"
            elif email:
                self._default_from = email
            else:
                self._default_from = None
        except Exception as e:
            print(f"[GmailClient] Failed to fetch default From settings: {e}")
            self._default_from = None

        return self._default_from

    # =======================
    # Internal low-level API
    # =======================

    def _get_recent_messages(
        self,
        n: int = 10,
        label_ids: Optional[List[str]] = None,
        use_priority_inbox: bool = False,
        mark_as_read: bool = False,
    ) -> List[Dict]:
        """
        Low-level: Return up to n most recent messages with full plain-text body.

        Args:
            n: Max number of messages to fetch.
            label_ids: Optional list of label IDs to filter (defaults to ["INBOX"]).
            use_priority_inbox: If True, restricts to Gmail's Primary/priority-style inbox
                                by using a category:primary query.
            mark_as_read: If True, mark the fetched messages as read (remove UNREAD label).
        """
        if not self.ready:
            print("[GmailClient] Not initialized (missing credentials or auth failed).")
            return []

        if label_ids is None:
            label_ids = ["INBOX"]

        # Approximate Gmail "priority" / Primary inbox.
        query = "category:primary" if use_priority_inbox else None

        try:
            result = (
                self.service.users()
                .messages()
                .list(
                    userId="me",
                    labelIds=label_ids,
                    maxResults=n,
                    q=query,
                )
                .execute()
            )

            messages = result.get("messages", [])
            if not messages:
                return []

            # Optionally mark these messages as read
            if mark_as_read:
                try:
                    self.service.users().messages().batchModify(
                        userId="me",
                        body={
                            "ids": [m["id"] for m in messages],
                            "removeLabelIds": ["UNREAD"],
                        },
                    ).execute()
                except HttpError as e:
                    print(f"[GmailClient] Failed to mark messages as read: {e}")

            detailed: List[Dict] = []
            for msg in messages:
                full_msg = (
                    self.service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg["id"],
                        format="full",
                    )
                    .execute()
                )

                headers = self._extract_headers(full_msg)
                body = self._get_plain_text_body(full_msg)
                labels = full_msg.get("labelIds", []) or []
                is_unread = "UNREAD" in labels

                detailed.append(
                    {
                        "id": full_msg["id"],
                        "threadId": full_msg.get("threadId"),
                        "snippet": full_msg.get("snippet"),
                        "from": headers.get("From"),
                        "to": headers.get("To"),
                        "cc": headers.get("Cc"),
                        "subject": headers.get("Subject"),
                        "date": headers.get("Date"),
                        "body": body,
                        "labels": labels,
                        "is_unread": is_unread,
                    }
                )

            return detailed

        except HttpError as error:
            print(f"[GmailClient] Error while fetching messages: {error}")
            return []

    def _get_body_part(self, message: Dict, wanted_mime: str) -> Optional[str]:
        """
        Find and decode the first part with the given MIME type.
        Supports nested multipart/* structures.
        """
        payload = message.get("payload", {})
        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})

        # Simple case: message itself is the wanted type
        if mime_type == wanted_mime:
            data = body.get("data")
            return self._decode_body(data)

        # Multipart: walk recursively
        parts = payload.get("parts", [])
        return self._walk_parts_for_mime(parts, wanted_mime)


    def _walk_parts_for_mime(self, parts: list, wanted_mime: str) -> Optional[str]:
        for part in parts or []:
            mime_type = part.get("mimeType", "")
            body = part.get("body", {})

            if mime_type == wanted_mime:
                data = body.get("data")
                if data:
                    return self._decode_body(data)

            # Nested multipart/*
            if mime_type.startswith("multipart/"):
                sub = self._walk_parts_for_mime(part.get("parts", []), wanted_mime)
                if sub:
                    return sub

        return None

    def _get_body_text(self, message: Dict) -> str:
        """
        Prefer text/plain, but fall back to text/html converted to text.
        """
        # 1) Try plain text
        text = self._get_body_part(message, "text/plain")
        if text:
            return text

        # 2) Fallback: HTML -> text
        html = self._get_body_part(message, "text/html")
        if not html:
            return ""  # nothing usable

        return self._html_to_text(html)  # you'll implement this

    def _html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Get visible text with some basic newlines
        text = soup.get_text("\n")
        # Optionally normalize whitespace
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    
    def _collect_attachments(self, message: dict, save_dir: str | None = None) -> list[dict]:
        """Walk the message payload and collect attachments. Optionally save to disk.

        Returns a list of dicts with keys: filename, mimeType, size, path (may be None).
        """
        if not self.ready:
            return []

        payload = message.get("payload", {}) or {}
        attachments: list[dict] = []

        def walk_parts(parts: list[dict] | None) -> None:
            for part in parts or []:
                mime_type = part.get("mimeType", "")
                filename = part.get("filename") or ""
                body = part.get("body", {}) or {}

                # Gmail uses non-empty filename to mark attachments
                if filename:
                    attachment_id = body.get("attachmentId")
                    data = body.get("data")

                    # Large attachments are fetched via attachments().get
                    if attachment_id and not data:
                        try:
                            att = (
                                self.service.users()
                                .messages()
                                .attachments()
                                .get(
                                    userId="me",
                                    messageId=message.get("id"),
                                    id=attachment_id,
                                )
                                .execute()
                            )
                            data = att.get("data")
                        except HttpError as e:
                            print(f"[GmailClient] Failed to fetch attachment {filename}: {e}")
                            data = None

                    content_bytes = b""
                    if data:
                        try:
                            content_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
                        except Exception as e:
                            print(f"[GmailClient] Failed to decode attachment {filename}: {e}")

                    local_path: str | None = None
                    if save_dir and content_bytes:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            local_path = os.path.join(save_dir, filename)
                            with open(local_path, "wb") as f:
                                f.write(content_bytes)
                        except OSError as e:
                            print(f"[GmailClient] Failed to write attachment {filename}: {e}")
                            local_path = None

                    attachments.append(
                        {
                            "filename": filename,
                            "mimeType": mime_type,
                            "size": body.get("size"),
                            "path": local_path,
                        }
                    )

                # Recurse into nested multiparts
                if mime_type.startswith("multipart/"):
                    walk_parts(part.get("parts", []) or [])

        if payload.get("parts"):
            walk_parts(payload.get("parts", []) or [])

        return attachments

    def _get_message(self, message_id: str) -> Optional[Dict]:
        """
        Low-level: Get a single message (full) with plain-text body,
        mark it as read (remove UNREAD label) if present, and collect attachments.
        """
        if not self.ready:
            print("[GmailClient] Not initialized (missing credentials or auth failed).")
            return None

        try:
            full_msg = (
                self.service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="full",
                )
                .execute()
            )

            headers = self._extract_headers(full_msg)
            body = self._get_body_text(full_msg)
            labels = full_msg.get("labelIds", []) or []
            is_unread = "UNREAD" in labels

            # Collect attachments and optionally save them under a stable directory
            attachments = self._collect_attachments(
                full_msg,
                save_dir=_resolve_path("~/.gmail_attachments"),
            )

            # Best-effort: mark as read
            if is_unread:
                try:
                    self.service.users().messages().modify(
                        userId="me",
                        id=message_id,
                        body={"removeLabelIds": ["UNREAD"]},
                    ).execute()
                except HttpError as e:
                    print(f"[GmailClient] Failed to mark message {message_id} as read: {e}")

            return {
                "id": full_msg["id"],
                "threadId": full_msg.get("threadId"),
                "snippet": full_msg.get("snippet"),
                "from": headers.get("From"),
                "to": headers.get("To"),
                "cc": headers.get("Cc"),
                "subject": headers.get("Subject"),
                "date": headers.get("Date"),
                "body": body,
                "labels": labels,
                "is_unread": is_unread,
                "attachments": attachments,
            }

        except HttpError as error:
            print(f"[GmailClient] Error while fetching message {message_id}: {error}")
            return None

    def _build_raw_create_message(
        self,
        to: str,
        subject: str,
        body_text: str,
        sender: str = "me",
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        cc: Optional[List[str] | str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the base64url-encoded Gmail API message body.

        Shared by messages().send() and drafts().create() — both accept the
        same {"raw": ..., "threadId": ...} payload (drafts nest it under
        {"message": ...}).

        attachments: list of dicts with at least a "path" key, and optionally
        "filename" and "mimeType". Example:
            {"path": "/tmp/file.pdf", "filename": "IR-memo.pdf", "mimeType": "application/pdf"}
        """
        if not self.ready:
            print("[GmailClient] Not initialized (missing credentials or auth failed).")
            return None

        message = EmailMessage()
        message["To"] = to
        # Prefer explicit sender if provided and not just "me"; otherwise
        # use the default send-as display name from Gmail settings.
        if sender and sender != "me":
            message["From"] = sender
        else:
            default_from = self._get_default_from()
            if default_from:
                message["From"] = default_from
        message["Subject"] = subject

        if cc:
            if isinstance(cc, str):
                message["Cc"] = cc
            else:
                message["Cc"] = ", ".join(cc)

        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
        if references:
            message["References"] = references

        message.set_content(body_text)

        # Attach any files provided
        for att in attachments or []:
            path = att.get("path")
            if not path:
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError as e:
                print(f"[GmailClient] Failed to read attachment {path}: {e}")
                continue

            filename = att.get("filename") or os.path.basename(path)
            mime_type = att.get("mimeType") or "application/octet-stream"
            maintype, _, subtype = mime_type.partition("/")
            if not maintype or not subtype:
                maintype, subtype = "application", "octet-stream"

            message.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        create_message: Dict[str, Any] = {"raw": encoded_message}
        if thread_id:
            create_message["threadId"] = thread_id

        return create_message

    def _send_email_raw(
        self,
        to: str,
        subject: str,
        body_text: str,
        sender: str = "me",
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        cc: Optional[List[str] | str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> Optional[Dict]:
        """Low-level: Send an email (optionally with attachments) without confirmation.

        If thread_id and in_reply_to are provided, Gmail will treat this as a reply.
        """
        create_message = self._build_raw_create_message(
            to=to,
            subject=subject,
            body_text=body_text,
            sender=sender,
            thread_id=thread_id,
            in_reply_to=in_reply_to,
            references=references,
            cc=cc,
            attachments=attachments,
        )
        if create_message is None:
            return None

        try:
            sent = (
                self.service.users()
                .messages()
                .send(userId="me", body=create_message)
                .execute()
            )
            return sent
        except HttpError as error:
            print(f"[GmailClient] Error while sending email: {error}")
            return None

    def _create_draft_raw(
        self,
        to: str,
        subject: str,
        body_text: str,
        sender: str = "me",
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
        cc: Optional[List[str] | str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> Optional[Dict]:
        """Low-level: Create a Gmail draft. Never sends anything.

        Same MIME construction as _send_email_raw; the payload goes to
        drafts().create() instead of messages().send(). threadId (inside the
        message body) makes a reply draft thread under the original.
        """
        create_message = self._build_raw_create_message(
            to=to,
            subject=subject,
            body_text=body_text,
            sender=sender,
            thread_id=thread_id,
            in_reply_to=in_reply_to,
            references=references,
            cc=cc,
            attachments=attachments,
        )
        if create_message is None:
            return None

        try:
            draft = (
                self.service.users()
                .drafts()
                .create(userId="me", body={"message": create_message})
                .execute()
            )
            return draft
        except HttpError as error:
            print(f"[GmailClient] Error while creating draft: {error}")
            return None

    def _expand_relative_date_query(self, query: str) -> str:
        """
        Translate simple relative-date operators (currently: newer_than:N[smhdwmy])
        into absolute Unix-timestamp-based Gmail 'after:' constraints.

        If no supported operator is present, returns the query unchanged.
        """
        import re as _re

        if not query:
            return query

        pattern = _re.compile(r"newer_than:(\d+)([smhdwmy])?", _re.IGNORECASE)
        m = pattern.search(query)
        if not m:
            return query

        amount = int(m.group(1))
        unit = (m.group(2) or "d").lower()

        # Map unit to a timedelta (seconds/minutes/hours/days/weeks/years).
        if unit == "s":
            delta = datetime.timedelta(seconds=amount)
        elif unit == "m":
            delta = datetime.timedelta(minutes=amount)
        elif unit == "h":
            delta = datetime.timedelta(hours=amount)
        elif unit == "d":
            delta = datetime.timedelta(days=amount)
        elif unit == "w":
            delta = datetime.timedelta(weeks=amount)
        elif unit == "y":
            # Rough year approximation is fine for a relative search window.
            delta = datetime.timedelta(days=365 * amount)
        else:
            delta = datetime.timedelta(days=amount)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        after_ts = int((now_utc - delta).timestamp())

        # Remove all newer_than: tokens from the original query.
        cleaned = pattern.sub("", query)
        cleaned = " ".join(cleaned.split())  # normalize whitespace

        extra = f"after:{after_ts}"
        if cleaned:
            return f"{cleaned} {extra}"
        else:
            return extra

    def search_emails(
        self,
        query: str,
        limit: int = 20,
        mark_as_read: bool = False,
    ) -> List[Dict]:
        """
        TOOL-LIKE: Search emails using a Gmail query string, returning preview only.

        The `query` uses standard Gmail search syntax, e.g.:
            "from:alice subject:meeting has:attachment after:2025/01/01"

        Args:
            query: Gmail search query string.
            limit: Max number of matching messages to return.
            mark_as_read: If True, mark the fetched messages as read.

        Returns:
            List of dicts with:
                id, threadId, snippet, from, to, cc, subject, date,
                labels, is_unread
        """
        if not self.ready:
            print("[GmailClient] Not initialized (missing credentials or auth failed).")
            return []

        # Expand UI-style relative date operators (e.g., newer_than:1d).
        effective_query = self._expand_relative_date_query(query)

        try:
            result = (
                self.service.users()
                .messages()
                .list(
                    userId="me",
                    q=effective_query,
                    maxResults=limit,
                )
                .execute()
            )
            messages = result.get("messages", [])
            if not messages:
                return []

            # Optionally mark these messages as read
            if mark_as_read:
                try:
                    self.service.users().messages().batchModify(
                        userId="me",
                        body={
                            "ids": [m["id"] for m in messages],
                            "removeLabelIds": ["UNREAD"],
                        },
                    ).execute()
                except HttpError as e:
                    print(f"[GmailClient] Failed to mark messages as read: {e}")

            results: List[Dict] = []
            for msg in messages:
                full_msg = (
                    self.service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg["id"],
                        format="full",
                    )
                    .execute()
                )

                headers = self._extract_headers(full_msg)
                body = self._get_plain_text_body(full_msg)
                labels = full_msg.get("labelIds", []) or []
                is_unread = "UNREAD" in labels

                preview_lines = (body or "").splitlines()[:3]
                body_preview = "\n".join(preview_lines)

                results.append(
                    {
                        "id": full_msg["id"],
                        "threadId": full_msg.get("threadId"),
                        "snippet": full_msg.get("snippet"),
                        "from": headers.get("From"),
                        "to": headers.get("To"),
                        "cc": headers.get("Cc"),
                        "subject": headers.get("Subject"),
                        "date": headers.get("Date"),
                        "labels": labels,
                        "is_unread": is_unread,
                        # body_preview is available but intentionally not returned
                    }
                )

            return results
        except HttpError as error:
            print(f"[GmailClient] Error while searching messages: {error}")
            return []

    # =======================
    # High-level “tool” API
    # =======================
    def list_recent_emails(
        self,
        limit: int = 10,
        priority_inbox: bool = True,
        mark_as_read: bool = False,
    ) -> List[Dict]:
        """
        TOOL-LIKE: List the most recent emails with **preview only**.

        Args:
            limit: Max number of recent messages to return.
            priority_inbox: If True, restrict to Gmail's Primary/priority-style inbox.
            mark_as_read: If True, mark the fetched messages as read.

        Returns:
            List of dicts with:
                id, threadId, snippet, from, to, cc, subject, date,
                is_unread, labels
        """
        msgs = self._get_recent_messages(
            n=limit,
            use_priority_inbox=priority_inbox,
            mark_as_read=mark_as_read,
        )
        result: List[Dict] = []

        for m in msgs:
            body = m.get("body") or ""
            preview_lines = body.splitlines()[:3]
            body_preview = "\n".join(preview_lines)

            result.append(
                {
                    "id": m.get("id"),
                    "threadId": m.get("threadId"),
                    "snippet": m.get("snippet"),
                    "from": m.get("from"),
                    "to": m.get("to"),
                    "cc": m.get("cc"),
                    "subject": m.get("subject"),
                    "date": m.get("date"),
                    # "body_preview": body_preview,
                    "is_unread": m.get("is_unread", False),
                    "labels": m.get("labels", []),
                }
            )

        return result

    def list_emails_from_date(
        self,
        date: datetime.date | str,
        label_ids: Optional[List[str]] = None,
        timezone: Optional[str] = None,
    ) -> List[Dict]:
        """
        TOOL-LIKE: List all emails from a given calendar date.

        The date is interpreted in a specific timezone, which by default is
        America/Chicago. This function intentionally does NOT include the
        message body in the returned data.

        Args:
            date: Either a datetime.date or a 'YYYY-MM-DD' string.
            label_ids: Optional label filter (defaults to ['INBOX']).
            timezone: IANA timezone name (e.g. 'America/Chicago').
                      If None, defaults to 'America/Chicago'.

        Returns:
            List of dicts with:
                id, threadId, snippet, from, to, cc, subject, date,
                labels, is_unread
        """
        if not self.ready:
            print("[GmailClient] Not initialized (missing credentials or auth failed).")
            return []

        # Normalize date
        if isinstance(date, str):
            year, month, day = map(int, date.split("-"))
            date_obj = datetime.date(year, month, day)
        else:
            date_obj = date

        if label_ids is None:
            label_ids = ["INBOX"]

        # Timezone: default to America/Chicago if not provided
        if timezone is None:
            local_tz = ZoneInfo("America/Chicago")
        else:
            local_tz = ZoneInfo(timezone)

        # Local midnight start/end for that calendar date in the chosen timezone
        start_local = datetime.datetime(
            date_obj.year, date_obj.month, date_obj.day, 0, 0, 0, tzinfo=local_tz
        )
        end_local = start_local + datetime.timedelta(days=1)

        # Convert to UTC and to Unix timestamps (seconds)
        start_utc = start_local.astimezone(datetime.timezone.utc)
        end_utc = end_local.astimezone(datetime.timezone.utc)

        after_ts = int(start_utc.timestamp())
        before_ts = int(end_utc.timestamp())

        # Gmail query using Unix timestamps (UTC-based, unambiguous)
        q = f"after:{after_ts} before:{before_ts}"

        try:
            all_messages: List[Dict] = []
            page_token = None

            while True:
                result = (
                    self.service.users()
                    .messages()
                    .list(
                        userId="me",
                        labelIds=label_ids,
                        q=q,
                        pageToken=page_token,
                    )
                    .execute()
                )

                messages = result.get("messages", [])
                if not messages:
                    break

                for msg in messages:
                    full_msg = (
                        self.service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=msg["id"],
                            format="full",  # could be 'metadata', but 'full' is fine if you want flexibility
                        )
                        .execute()
                    )
                    headers = self._extract_headers(full_msg)
                    labels = full_msg.get("labelIds", []) or []
                    is_unread = "UNREAD" in labels

                    all_messages.append(
                        {
                            "id": full_msg["id"],
                            "threadId": full_msg.get("threadId"),
                            "snippet": full_msg.get("snippet"),
                            "from": headers.get("From"),
                            "to": headers.get("To"),
                            "cc": headers.get("Cc"),
                            "subject": headers.get("Subject"),
                            "date": headers.get("Date"),
                            "labels": labels,
                            "is_unread": is_unread,
                            # body intentionally omitted
                        }
                    )

                page_token = result.get("nextPageToken")
                if not page_token:
                    break

            return all_messages

        except HttpError as error:
            print(f"[GmailClient] Error while fetching messages from date {date_obj}: {error}")
            return []

    def get_email(self, message_id: str) -> Optional[Dict]:
        """
        TOOL-LIKE: Get a single email by ID with **full body**.
        """
        return self._get_message(message_id)

    def _build_pending_tool_response(self, kind: str, preview: dict) -> dict:
        """Return a JSON-serializable payload indicating pending confirmation.

        This is used for the web UI, where CLI-based input() prompts are not possible.
        The outer chat/tool wrapper is added by dispatch_tool_call; here we only
        return the payload so that ChatEngine can detect `status=pending_confirmation`
        directly from tool_msg["content"].
        """
        payload = {
            "status": "pending_confirmation",
            "kind": kind,
            "preview": preview,
        }
        return payload

    def send_email_with_confirmation(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str] | str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> Optional[Dict]:
        """
        TOOL-LIKE: Send an email, but ALWAYS ask the human for confirmation.

        - In CLI mode, uses interactive input() prompts.
        - In web mode, returns a "pending_confirmation" tool payload for the UI.
        """
        # Common preview context
        preview = {
            "to": to,
            "subject": subject,
            "body": body,
            "cc": cc,
            "attachments": attachments or [],
        }

        if self.confirmation_mode == "web":
            # Do not block; caller should surface this preview to the user.
            return self._build_pending_tool_response("gmail_send", preview)

        print("\n=== Email send request ===")
        print(f"To: {to}")
        if cc:
            if isinstance(cc, str):
                print(f"Cc: {cc}")
            else:
                print(f"Cc: {', '.join(cc)}")
        print(f"Subject: {subject}")
        if attachments:
            print("Attachments:")
            for att in attachments:
                path = att.get("path") or "(no path)"
                fname = att.get("filename") or "(no filename)"
                mime = att.get("mimeType") or "(no mimeType)"
                print(f"  - {fname} ({mime}) :: {path}")
        print("Body:")
        print("-" * 40)
        print(body)
        print("-" * 40)

        while True:
            if self.user_confirm:
                answer = input("Send this email? [y/N/e]: ").strip().lower()
            else:
                answer = "n"

            if answer in ("y", "yes"):
                result = self._send_email_raw(
                    to=to,
                    subject=subject,
                    body_text=body,
                    cc=cc,
                    attachments=attachments,
                )
                if result:
                    print(f"Email sent. Gmail message ID: {result['id']}")
                else:
                    print("Failed to send email.")
                return result
            elif answer in ("e", "edit"):
                print("Opening editor to edit email body before sending...")
                body = self._edit_body_in_editor(body)
                print("Sending edited email...")
                result = self._send_email_raw(
                    to=to,
                    subject=subject,
                    body_text=body,
                    cc=cc,
                    attachments=attachments,
                )
                if result:
                    print(f"Email sent (after edit). Gmail message ID: {result['id']}")
                else:
                    print("Failed to send email after edit.")
                return result
            else:
                print("Email NOT sent.")
                return f"Email NOT sent, cancelled by user. {answer}"

    def reply_to_email_with_confirmation(
        self,
        message_id: str,
        body: str,
        cc: Optional[List[str] | str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> Optional[Dict]:
        """
        TOOL-LIKE: Reply to an existing email, with confirmation.

        - In CLI mode, uses interactive input() prompts.
        - In web mode, returns a "pending_confirmation" tool payload for the UI.
        """
        if not self.ready:
            print("[GmailClient] Not initialized (missing credentials or auth failed).")
            return None

        try:
            original = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as e:
            print(f"[GmailClient] Failed to fetch original message {message_id}: {e}")
            return None

        headers = self._extract_headers(original)
        thread_id = original.get("threadId")

        to = headers.get("Reply-To") or headers.get("From")
        if not to:
            print("[GmailClient] Original message has no From/Reply-To; cannot determine recipient.")
            return None

        subject = headers.get("Subject") or ""
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        orig_message_id = headers.get("Message-ID")
        references = headers.get("References")
        if orig_message_id:
            references = (references + " " + orig_message_id).strip() if references else orig_message_id

        preview = {
            "to": to,
            "subject": subject,
            "body": body,
            "cc": cc,
            "attachments": attachments or [],
            "thread_id": thread_id,
            "in_reply_to": orig_message_id,
            "references": references,
            "original_message_id": message_id,
        }

        if self.confirmation_mode == "web":
            return self._build_pending_tool_response("gmail_reply", preview)

        print("\n=== Email reply request ===")
        print(f"Replying to message ID: {message_id}")
        print(f"To: {to}")
        if cc:
            if isinstance(cc, str):
                print(f"Cc: {cc}")
            else:
                print(f"Cc: {', '.join(cc)}")
        print(f"Subject: {subject}")
        if attachments:
            print("Attachments:")
            for att in attachments:
                path = att.get("path") or "(no path)"
                fname = att.get("filename") or "(no filename)"
                mime = att.get("mimeType") or "(no mimeType)"
                print(f"  - {fname} ({mime}) :: {path}")
        print("Body:")
        print("-" * 40)
        print(body)
        print("-" * 40)

        while True:
            if self.user_confirm:
                answer = input("Send this reply? [y/N/e]: ").strip().lower()
            else:
                answer = "n"

            if answer in ("y", "yes"):
                result = self._send_email_raw(
                    to=to,
                    subject=subject,
                    body_text=body,
                    thread_id=thread_id,
                    in_reply_to=orig_message_id,
                    references=references,
                    cc=cc,
                    attachments=attachments,
                )
                if result:
                    print(f"Reply sent. Gmail message ID: {result['id']}")
                else:
                    print("Failed to send reply.")
                return result
            elif answer in ("e", "edit"):
                print("Opening editor to edit reply body before sending...")
                body = self._edit_body_in_editor(body)
                print("Sending edited reply...")
                result = self._send_email_raw(
                    to=to,
                    subject=subject,
                    body_text=body,
                    thread_id=thread_id,
                    in_reply_to=orig_message_id,
                    references=references,
                    cc=cc,
                    attachments=attachments,
                )
                if result:
                    print(f"Reply sent (after edit). Gmail message ID: {result['id']}")
                else:
                    print("Failed to send reply after edit.")
                return result
            else:
                print("Reply NOT sent.")
                return f"Reply NOT sent, cancelled by user. {answer}"

    # =======================
    # Internal helpers
    # =======================

    @staticmethod
    def _extract_headers(message: Dict) -> Dict[str, str]:
        payload = message.get("payload", {})
        headers_list = payload.get("headers", [])
        return {h["name"]: h["value"] for h in headers_list}

    def _get_plain_text_body(self, message: Dict) -> str:
        """
        Try to extract the plain-text body from the Gmail message structure.
        """
        payload = message.get("payload", {})

        # Simple text/plain
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data")
            return self._decode_body(data)

        # Multipart: walk parts to find text/plain
        parts = payload.get("parts", [])
        return self._walk_parts_for_plain_text(parts)

    def _walk_parts_for_plain_text(self, parts) -> str:
        for part in parts or []:
            mime_type = part.get("mimeType", "")
            if mime_type == "text/plain":
                data = part.get("body", {}).get("data")
                text = self._decode_body(data)
                if text:
                    return text

            # Nested multipart
            if part.get("parts"):
                text = self._walk_parts_for_plain_text(part.get("parts"))
                if text:
                    return text

        return ""  # nothing found

    @staticmethod
    def _decode_body(data: Optional[str]) -> str:

        if not data:
            return ""
        try:
            decoded_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _edit_body_in_editor(self, body: str) -> str:
        """Open the given body text in $EDITOR and return the edited text."""
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False, encoding="utf-8") as tf:
            path = tf.name
            tf.write(body or "")
            tf.flush()
        try:
            try:
                subprocess.call([editor, path])
            except FileNotFoundError:
                # Fallback: try a bare shell if editor is something more complex
                subprocess.call(editor, shell=True)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # =======================
    # Tool dispatch
    # =======================

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "1", "yes", "y"):
                return True
            if v in ("false", "0", "no", "n"):
                return False
        return default

    def dispatch_tool_call(self, tool_call: Any) -> Tuple[Dict[str, str], bool]:
        """
        Handle Gmail-related tool calls from LLM.

        Tools expected:

        - gmail_list_recent:
            args: {
                "limit": int (optional, default 10),
                "priority_inbox": bool (optional, default True),
                "mark_as_read": bool (optional, default False)
            }

        - gmail_send_message:
            args: {
                "to": str,
                "subject": str,
                "body": str,
                "cc": str | [str] (optional)
            }

        - gmail_get_email:
            args: { "message_id": str }

        - gmail_reply_to:
            args: {
                "message_id": str,
                "body": str,
                "cc": str | [str] (optional)
            }

        - gmail_list_from_date (optional):
            args: {
                "date": "YYYY-MM-DD",
                "label_ids": [str] (optional)
            }

        - gmail_search_emails:
            args: {
                "query": str,
                "limit": int (optional, default 20),
                "mark_as_read": bool (optional, default False)
            }

        Returns:
            (tool_response_message, is_error)
        """
        # --- Normalize access to tool_call fields (same style as your Exa dispatch) ---
        try:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            call_id = tool_call.id
        except AttributeError:
            try:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                call_id = tool_call.get("id", "")
            except Exception:
                return ({
                    "role": "tool",
                    "tool_call_id": "",
                    "content": "Malformed tool call: missing fields."
                }, True)

        # --- Parse JSON args ---
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception as e:
            return ({
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Invalid JSON in tool arguments: {e} args: {raw_args}"
            }, True)

        is_error = False
        output: Any

        # --- Dispatch by tool name ---
        if name == "gmail_list_recent":
            limit = args.get("limit", 10)
            try:
                limit = int(limit)
            except Exception:
                limit = 10  # fall back, don't hard error

            priority_inbox = self._coerce_bool(args.get("priority_inbox", True), True)
            mark_as_read = self._coerce_bool(args.get("mark_as_read", False), False)

            output = self.list_recent_emails(
                limit=limit,
                priority_inbox=priority_inbox,
                mark_as_read=mark_as_read,
            )

        elif name == "gmail_send_message":
            to = (args.get("to") or "").strip()
            subject = (args.get("subject") or "").strip()
            body = args.get("body") or ""
            cc = args.get("cc")
            attachments = _normalize_attachments(args.get("attachments"))

            missing = []
            if not to:
                missing.append("to")
            if not subject:
                missing.append("subject")
            if not body:
                missing.append("body")

            if missing:
                output = f"Missing required parameter(s): {', '.join(missing)}"
                is_error = True
            else:
                # This will prompt on the CLI for confirmation before sending
                result = self.send_email_with_confirmation(
                    to=to,
                    subject=subject,
                    body=body,
                    cc=cc,
                    attachments=attachments,
                )
                output = result or "Email was not sent (cancelled or failed)."

        elif name == "gmail_get_email":
            message_id = (args.get("message_id") or "").strip()
            if not message_id:
                output = "Missing required parameter: message_id"
                is_error = True
            else:
                msg = self.get_email(message_id)
                if msg is None:
                    output = f"Failed to fetch email with id: {message_id}"
                    is_error = True
                else:
                    output = msg

        elif name == "gmail_reply_to":
            message_id = (args.get("message_id") or "").strip()
            body = args.get("body") or ""
            cc = args.get("cc")
            attachments = _normalize_attachments(args.get("attachments"))

            missing = []
            if not message_id:
                missing.append("message_id")
            if not body:
                missing.append("body")

            if missing:
                output = f"Missing required parameter(s): {', '.join(missing)}"
                is_error = True
            else:
                result = self.reply_to_email_with_confirmation(
                    message_id=message_id,
                    body=body,
                    cc=cc,
                    attachments=attachments,
                )
                output = result or "Reply was not sent (cancelled or failed)."

        elif name == "gmail_list_from_date":
            date_str = (args.get("date") or "").strip()
            if not date_str:
                output = "Missing required parameter: date (expected 'YYYY-MM-DD')"
                is_error = True
            else:
                label_ids = args.get("label_ids")
                if not isinstance(label_ids, list):
                    label_ids = None
                output = self.list_emails_from_date(date=date_str, label_ids=label_ids)
                
        elif name == "gmail_search_emails":
            print( "dtc: " )
            print( args )
            query = (args.get("query") or "").strip()
            if not query:
                output = "Missing required parameter: query"
                is_error = True
            else:
                limit = args.get("limit", 20)
                try:
                    limit = int(limit)
                except Exception:
                    limit = 20

                mark_as_read = self._coerce_bool(args.get("mark_as_read", False), False)

                output = self.search_emails(
                    query=query,
                    limit=limit,
                    mark_as_read=mark_as_read,
                )
        else:
            output = f"Unknown Gmail tool: {name}"
            is_error = True

        # --- Normalize output to a string for the chat API ---
        if not isinstance(output, str):
            try:
                content = json.dumps(output, ensure_ascii=False)
            except TypeError:
                content = str(output)
        else:
            content = output

        return ({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }, is_error)

