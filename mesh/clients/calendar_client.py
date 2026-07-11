# calendar_client.py

from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from zoneinfo import ZoneInfo


class CalendarClient:
    """
    Minimal Google Calendar client for CLI usage, with high-level methods
    suitable to expose as tools to an LLM.

    This version assumes you already have a valid `Credentials` object
    (e.g., from your GmailClient) that includes Calendar scopes:
      - https://www.googleapis.com/auth/calendar.readonly
      - https://www.googleapis.com/auth/calendar.events
    """

    def __init__(self, creds: Credentials, confirmation_mode: str = "cli"):
        self.creds = creds
        self.service = None
        self._build_service()
        self.confirmation_mode = confirmation_mode
        self.user_confirm = (confirmation_mode == "cli")

    def _build_service(self):
        try:
            self.service = build("calendar", "v3", credentials=self.creds)
        except Exception as e:
            print(f"[CalendarClient] Failed to build Calendar service: {e}")
            self.service = None

    @property
    def ready(self) -> bool:
        return self.service is not None

    # =======================
    # High-level methods
    # =======================

    def list_upcoming_events(
        self,
        max_results: int = 10,
        calendar_id: str = "primary",
    ) -> List[Dict]:
        """
        TOOL-LIKE: List upcoming calendar events from the primary calendar.

        Returns:
            List of dicts with:
                id, summary, description, start, end, location, attendees
        """
        if not self.ready:
            print("[CalendarClient] Not initialized.")
            return []

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        try:
            events_result = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=now,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except HttpError as error:
            print(f"[CalendarClient] Error while listing upcoming events: {error}")
            return []

        events = events_result.get("items", [])
        result: List[Dict] = []
        for ev in events:
            result.append(
                {
                    "id": ev.get("id"),
                    "summary": ev.get("summary"),
                    "description": ev.get("description"),
                    "start": ev.get("start"),
                    "end": ev.get("end"),
                    "location": ev.get("location"),
                    "attendees": ev.get("attendees", []),
                    "recurringEventId": ev.get("recurringEventId"),  # master id if this is an instance
                    "recurrence": ev.get("recurrence"),              # only on master
                }
            )
        return result

    def list_events_on_date(
        self,
        date: datetime.date | str,
        calendar_id: str = "primary",
        timezone: str = "America/Chicago",
    ) -> List[Dict]:
        """
        TOOL-LIKE: List all events on a given calendar date in the given timezone.

        Args:
            date: Either a datetime.date or a 'YYYY-MM-DD' string.
            calendar_id: Google Calendar ID (default 'primary').
            timezone: IANA timezone name (default 'America/Chicago').

        Returns:
            List of dicts with:
                id, summary, description, start, end, location, attendees
        """
        if not self.ready:
            print("[CalendarClient] Not initialized.")
            return []

        # Normalize date
        if isinstance(date, str):
            year, month, day = map(int, date.split("-"))
            date_obj = datetime.date(year, month, day)
        else:
            date_obj = date

        local_tz = ZoneInfo(timezone)

        start_local = datetime.datetime(
            date_obj.year, date_obj.month, date_obj.day, 0, 0, 0, tzinfo=local_tz
        )
        end_local = start_local + datetime.timedelta(days=1)

        start_utc = start_local.astimezone(datetime.timezone.utc).isoformat()
        end_utc = end_local.astimezone(datetime.timezone.utc).isoformat()

        try:
            events_result = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=start_utc,
                    timeMax=end_utc,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except HttpError as error:
            print(f"[CalendarClient] Error while listing events on date {date_obj}: {error}")
            return []

        events = events_result.get("items", [])
        result: List[Dict] = []
        for ev in events:
            result.append(
                {
                    "id": ev.get("id"),
                    "summary": ev.get("summary"),
                    "description": ev.get("description"),
                    "start": ev.get("start"),
                    "end": ev.get("end"),
                    "location": ev.get("location"),
                    "attendees": ev.get("attendees", []),
                    "recurringEventId": ev.get("recurringEventId"),
                    "recurrence": ev.get("recurrence"),
                }
            )
        return result

    def delete_event_with_confirmation(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> dict | None:
        """
        TOOL-LIKE: Delete a calendar event, with support for:
          - deleting a single occurrence of a recurring event, OR
          - deleting the entire recurring series.

        In CLI mode, prompts interactively.
        In web confirmation mode, deletes directly without extra prompts,
        defaulting to a single occurrence.
        """
        if not self.ready:
            print("[CalendarClient] Not initialized.")
            return None

        # Fetch event details first so we can show context and detect recurrence
        try:
            ev = (
                self.service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
        except HttpError as error:
            print(f"[CalendarClient] Error fetching event {event_id}: {error}")
            return None

        # Minimal human-readable summary
        summary = ev.get("summary")
        start = ev.get("start")
        end = ev.get("end")
        location = ev.get("location")
        description = ev.get("description")
        recurring_event_id = ev.get("recurringEventId")

        is_instance = recurring_event_id is not None  # True if this is an occurrence in a series
        master_id = recurring_event_id if is_instance else ev.get("id")

        print("\n=== Calendar event delete request ===")
        print(f"ID: {event_id}")
        print(f"Summary: {summary}")
        print(f"Start: {json.dumps(start)}")
        print(f"End:   {json.dumps(end)}")
        if location:
            print(f"Location: {location}")
        if description:
            print(f"Description: {description}")

        if is_instance:
            print("\nThis event is part of a recurring series.")
            print(f"Series master ID: {master_id}")
        print("-" * 40)

        # In web confirmation mode, delete directly (single occurrence only)
        if self.confirmation_mode == "web":
            delete_id = event_id
            scope = "single occurrence"
            try:
                self.service.events().delete(
                    calendarId=calendar_id,
                    eventId=delete_id,
                ).execute()
                print(f"Event deleted ({scope}) [web mode]. ID: {delete_id}")
                return {
                    "status": "deleted",
                    "event_id": delete_id,
                    "scope": scope,
                }
            except HttpError as error:
                print(f"[CalendarClient] Error deleting event {delete_id}: {error}")
                return None

        delete_id = event_id
        scope = "single occurrence"

        # If this is an instance in a series, ask whether to delete occurrence or entire series
        if is_instance:
            while True:
                if self.user_confirm:
                    choice = input(
                        "Delete (o)nly this occurrence, (s)eries (all occurrences), or (c)ancel? [o/s/C]: "
                    ).strip().lower()
                else:
                    choice = "c"
                if choice in ("o", "occurrence", "this"):
                    delete_id = event_id
                    scope = "single occurrence"
                    break
                elif choice in ("s", "series", "all"):
                    delete_id = master_id
                    scope = "entire series"
                    break
                elif choice in ("c", "n", "", "cancel"):
                    print("Event NOT deleted.")
                    return None
                else:
                    print("Please answer 'o' (occurrence), 's' (series), or 'c' (cancel).")

        # Final confirmation
        while True:
            if self.user_confirm:
                answer = input(f"Confirm delete {scope}? [y/N]: ").strip().lower()
            else:
                answer = "N, access disabled"
            if answer in ("y", "yes"):
                try:
                    self.service.events().delete(
                        calendarId=calendar_id,
                        eventId=delete_id,
                    ).execute()
                    print(f"Event deleted ({scope}). ID: {delete_id}")
                    return {
                        "status": "deleted",
                        "event_id": delete_id,
                        "scope": scope,
                    }
                except HttpError as error:
                    print(f"[CalendarClient] Error deleting event {delete_id}: {error}")
                    return None
            else:
                print("Event NOT deleted. {answer}")
                return None


    def _build_pending_tool_response(self, kind: str, preview: dict) -> dict:
        """Return a JSON-serializable payload indicating pending confirmation.

        Used by the web UI; CLI mode continues to prompt interactively. The
        outer tool wrapper is added later by dispatch_tool_call so that
        ChatEngine can see `status=pending_confirmation` in the top-level JSON.
        """
        payload = {
            "status": "pending_confirmation",
            "kind": kind,
            "preview": preview,
        }
        return payload

    def create_event_with_confirmation(
        self,
        summary: str,
        start: str,
        end: str,
        timezone: str = "America/Chicago",
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        recurrence: Optional[List[str]] = None,
        reminders: Optional[Dict[str, Any]] = None,
        calendar_id: str = "primary",
    ) -> Optional[Dict]:
        """
        TOOL-LIKE: Create a calendar event, but ALWAYS ask for confirmation.

        - In CLI mode, uses interactive input() prompts.
        - In web mode, returns a "pending_confirmation" tool payload for the UI.

        Args:
            summary: Event title.
            start: Start datetime in 'YYYY-MM-DDTHH:MM:SS' (interpreted in `timezone`).
            end:   End   datetime in 'YYYY-MM-DDTHH:MM:SS' (interpreted in `timezone`).
            timezone: IANA timezone name (default 'America/Chicago').
            description: Optional event description.
            location: Optional location string.
            attendees: Optional list of attendee email addresses.
            recurrence: Optional list of RRULE strings.
            reminders: Optional dict, e.g.:
            {
               "useDefault": false,
                "overrides": [
                {"method": "popup", "minutes": 10},
                {"method": "email", "minutes": 30}
                 ]
            }
            calendar_id: Calendar ID (default 'primary').

        Returns:
            The created event dict, or None if cancelled/failed.
        """
        if not self.ready:
            print("[CalendarClient] Not initialized.")
            return None

        attendees_list = [{"email": a} for a in (attendees or [])]

        event_body = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start, "timeZone": timezone},
            "end": {"dateTime": end, "timeZone": timezone},
            "attendees": attendees_list,
        }

        if recurrence:
            event_body["recurrence"] = recurrence

        if reminders is not None:
            event_body["reminders"] = reminders
            
        preview = {
            "summary": summary,
            "start": start,
            "end": end,
            "timezone": timezone,
            "description": description,
            "location": location,
            "attendees": attendees or [],
            "recurrence": recurrence or [],
            "reminders": reminders,
            "calendar_id": calendar_id,
        }

        if self.confirmation_mode == "web":
            return self._build_pending_tool_response("calendar_create", preview)

        print("\n=== Calendar event create request ===")
        print(json.dumps(event_body, indent=2))
        print("-" * 40)

        while True:
            if self.user_confirm:
                answer = input("Create this event? [y/N]: ").strip().lower()
            else:
                answer = "N"
                
            if answer in ("y", "yes"):
                try:
                    created = (
                        self.service.events()
                        .insert(calendarId=calendar_id, body=event_body)
                        .execute()
                    )
                    print(f"Event created. ID: {created.get('id')}")
                    return created
                except HttpError as error:
                    print(f"[CalendarClient] Error while creating event: {error}")
                    return None
            elif answer in ("n", "no", ""):
                print("Event NOT created.")
                return None
            else:
                print("Please answer 'y' or 'n'.")

    # =======================
    # Tool dispatch
    # =======================

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def dispatch_tool_call(self, tool_call: Any) -> Tuple[Dict[str, str], bool]:
        """
        Handle Calendar-related tool calls from LLM.

        Tools expected:

        - calendar_list_upcoming:
            args: {
                "max_results": int (optional, default 10)
            }

        - calendar_list_on_date:
            args: {
                "date": "YYYY-MM-DD",
                "timezone": "IANA timezone string" (optional, default "America/Chicago")
            }

        - calendar_create_event:
            args: {
                "summary": str,
                "start": "YYYY-MM-DDTHH:MM:SS",
                "end": "YYYY-MM-DDTHH:MM:SS",
                "description": str (optional),
                "location": str (optional),
                "attendees": [str] (optional),
                "timezone": "IANA timezone string" (optional, default "America/Chicago")
            }   
        - calendar_delete_event:
            args: {
                "event_id": str,
                "calendar_id": str (optional, default "primary")
            }

        Returns:
            (tool_response_message, is_error)
        """
        # Normalize access to tool_call fields (same style as Gmail dispatch)
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

        # Parse JSON args
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

        if name == "calendar_list_upcoming":
            max_results = self._coerce_int(args.get("max_results", 10), 10)
            output = self.list_upcoming_events(max_results=max_results)

        elif name == "calendar_list_on_date":
            date_str = (args.get("date") or "").strip()
            if not date_str:
                output = "Missing required parameter: date (expected 'YYYY-MM-DD')"
                is_error = True
            else:
                timezone = (args.get("timezone") or "America/Chicago").strip()
                output = self.list_events_on_date(date=date_str, timezone=timezone)

        elif name == "calendar_create_event":
            summary = (args.get("summary") or "").strip()
            start = (args.get("start") or "").strip()
            end = (args.get("end") or "").strip()
            description = args.get("description")
            location = args.get("location")
            attendees = args.get("attendees") or []
            timezone = (args.get("timezone") or "America/Chicago").strip()
            recurrence = args.get("recurrence") or None
            reminders = args.get("reminders") or None
            
            missing = []
            if not summary:
                missing.append("summary")
            if not start:
                missing.append("start")
            if not end:
                missing.append("end")

            if missing:
                output = f"Missing required parameter(s): {', '.join(missing)}"
                is_error = True
            else:
                result = self.create_event_with_confirmation(
                    summary=summary,
                    start=start,
                    end=end,
                    description=description,
                    location=location,
                    attendees=attendees,
                    timezone=timezone,
                    recurrence=recurrence,
                    reminders=reminders
                )
                output = result or "Event was not created (cancelled or failed)."

        elif name == "calendar_delete_event":
            event_id = (args.get("event_id") or "").strip()
            calendar_id = (args.get("calendar_id") or "primary").strip() or "primary"

            if not event_id:
                output = "Missing required parameter: event_id"
                is_error = True
            else:
                result = self.delete_event_with_confirmation(
                    event_id=event_id,
                    calendar_id=calendar_id,
                )
                output = result or "Event was not deleted (cancelled or failed)."

        else:
            output = f"Unknown Calendar tool: {name}"
            is_error = True

        # Normalize output to a string for the chat API
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

