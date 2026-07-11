"""
Canvas LMS REST API client for mesh-tool.

Provides course management capabilities via the Canvas REST API.
Token stored at ~/.mesh/canvas_token.json with bearer auth.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

TOKEN_FILE = os.path.join(
    __import__("pwd").getpwuid(os.getuid()).pw_dir, ".mesh", "canvas_token.json"
)
DEFAULT_BASE_URL = "https://canvas.tamu.edu"


class CanvasClient:
    """Canvas LMS REST API client with pagination, rate limiting, and error handling."""

    def __init__(self, token_path: str = TOKEN_FILE):
        self._token_path = token_path
        self._config: dict[str, Any] = {}
        self._session = requests.Session()
        self._rate_limit_remaining: int | None = None
        self._load_config()

    def _load_config(self) -> None:
        if os.path.exists(self._token_path):
            with open(self._token_path) as f:
                self._config = json.load(f)
            token = self._config.get("access_token", "")
            if token:
                self._session.headers["Authorization"] = f"Bearer {token}"
        else:
            self._config = {}

    def _save_config(self) -> None:
        os.makedirs(os.path.dirname(self._token_path), exist_ok=True)
        with open(self._token_path, "w") as f:
            json.dump(self._config, f, indent=2)

    @property
    def base_url(self) -> str:
        return self._config.get("base_url", DEFAULT_BASE_URL).rstrip("/")

    def is_available(self) -> bool:
        return bool(self._config.get("access_token"))

    def get_active_course(self) -> int | None:
        cid = self._config.get("active_course_id")
        if cid is not None:
            return int(cid)
        return None

    def set_active_course(self, course_id: int) -> None:
        self._config["active_course_id"] = course_id
        self._save_config()

    def set_token(self, access_token: str, base_url: str | None = None) -> None:
        """Set token in memory only. Call save_config() after validation."""
        self._config["access_token"] = access_token
        if base_url:
            self._config["base_url"] = base_url
        elif "base_url" not in self._config:
            self._config["base_url"] = DEFAULT_BASE_URL
        self._config.setdefault("user_id", None)
        self._config.setdefault("active_course_id", None)
        self._session.headers["Authorization"] = f"Bearer {access_token}"

    def save_config(self) -> None:
        """Persist current config to disk. Public wrapper for post-validation save."""
        self._save_config()

    def resolve_course(self, course_id: int | str | None) -> int:
        if course_id is not None:
            return int(course_id)
        active = self.get_active_course()
        if active is not None:
            return active
        raise ValueError(
            "No course_id provided and no active course set. "
            "Use canvas_set_course to set an active course, or pass --course_id."
        )

    # -- HTTP layer ----------------------------------------------------------

    def _check_rate_limit(self) -> None:
        if self._rate_limit_remaining is not None:
            if self._rate_limit_remaining <= 20:
                logger.warning("Canvas rate limit critical (%d remaining), pausing 30s",
                               self._rate_limit_remaining)
                time.sleep(30)
            elif self._rate_limit_remaining <= 100:
                delay = max(0.5, (100 - self._rate_limit_remaining) * 0.1)
                logger.info("Canvas rate limit low (%d remaining), backoff %.1fs",
                            self._rate_limit_remaining, delay)
                time.sleep(delay)

    def _update_rate_limit(self, response: requests.Response) -> None:
        remaining = response.headers.get("X-Rate-Limit-Remaining")
        if remaining is not None:
            try:
                self._rate_limit_remaining = int(float(remaining))
            except (ValueError, TypeError):
                pass

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> requests.Response:
        self._check_rate_limit()
        url = f"{self.base_url}/api/v1{path}"
        resp = self._session.request(method, url, params=params, json=json_body, timeout=30)
        self._update_rate_limit(resp)
        self._raise_for_status(resp)
        return resp

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        code = resp.status_code
        try:
            body = resp.json()
            msg = body.get("errors", body.get("message", resp.text))
        except Exception:
            msg = resp.text[:500]
        error_map = {
            401: f"Authentication failed (401). Check your Canvas access token. {msg}",
            403: f"Insufficient permissions (403). {msg}",
            404: f"Resource not found (404). {msg}",
            422: f"Validation error (422). {msg}",
        }
        raise CanvasAPIError(error_map.get(code, f"Canvas API error ({code}): {msg}"), code)

    def _get_json(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", path, params=params).json()

    def _post_json(self, path: str, json_body: dict | None = None) -> Any:
        return self._request("POST", path, json_body=json_body).json()

    def _put_json(self, path: str, json_body: dict | None = None) -> Any:
        return self._request("PUT", path, json_body=json_body).json()

    # -- Pagination ----------------------------------------------------------

    def _get_paginated(
        self,
        path: str,
        params: dict | None = None,
        limit: int = 50,
    ) -> list[dict]:
        params = dict(params or {})
        per_page = min(limit, 100)
        params["per_page"] = per_page
        collected: list[dict] = []
        url = f"{self.base_url}/api/v1{path}"

        while url and len(collected) < limit:
            self._check_rate_limit()
            resp = self._session.get(url, params=params, timeout=30)
            self._update_rate_limit(resp)
            self._raise_for_status(resp)
            data = resp.json()
            if isinstance(data, list):
                collected.extend(data)
            else:
                collected.append(data)

            # Follow Link: <url>; rel="next"
            url = None
            params = None  # params are baked into the next URL
            link_header = resp.headers.get("Link", "")
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break

        return collected[:limit]

    # -- API methods ---------------------------------------------------------

    def get_self(self) -> dict:
        return self._get_json("/users/self")

    def list_courses(
        self,
        enrollment_state: str = "active",
        limit: int = 50,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "enrollment_state": enrollment_state,
            "include[]": "total_students",
        }
        return self._get_paginated("/courses", params=params, limit=limit)

    def get_course(self, course_id: int | str | None = None) -> dict:
        cid = self.resolve_course(course_id)
        return self._get_json(
            f"/courses/{cid}",
            params={"include[]": "total_students"},
        )


    # -- Assignments -----------------------------------------------------------

    def list_assignments(
        self, course_id: int | str | None = None, bucket: str | None = None, limit: int = 50,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {}
        if bucket:
            params["bucket"] = bucket
        return self._get_paginated(f"/courses/{cid}/assignments", params=params, limit=limit)

    def get_assignment(self, assignment_id: int | str, course_id: int | str | None = None) -> dict:
        cid = self.resolve_course(course_id)
        return self._get_json(f"/courses/{cid}/assignments/{assignment_id}")

    # -- Students / Enrollments ------------------------------------------------

    def list_students(
        self, course_id: int | str | None = None, enrollment_type: str = "student", limit: int = 200,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {"enrollment_type[]": enrollment_type}
        return self._get_paginated(f"/courses/{cid}/users", params=params, limit=limit)

    def get_student(
        self, user_id: int | str, course_id: int | str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        params = {"include[]": ["enrollments", "total_activity_time", "last_login"]}
        return self._get_json(f"/courses/{cid}/users/{user_id}", params=params)

    def get_grades(
        self, course_id: int | str | None = None, student_id: int | str | None = None, limit: int = 200,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {"type[]": "StudentEnrollment"}
        if student_id:
            params["user_id"] = student_id
        return self._get_paginated(f"/courses/{cid}/enrollments", params=params, limit=limit)

    # -- Submissions -----------------------------------------------------------

    def list_submissions(
        self,
        assignment_id: int | str,
        course_id: int | str | None = None,
        include: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {}
        if include:
            params["include[]"] = include.split(",") if "," in include else include
        return self._get_paginated(
            f"/courses/{cid}/assignments/{assignment_id}/submissions", params=params, limit=limit,
        )

    def get_submission(
        self,
        assignment_id: int | str,
        student_id: int | str,
        course_id: int | str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        return self._get_json(
            f"/courses/{cid}/assignments/{assignment_id}/submissions/{student_id}",
            params={"include[]": ["submission_comments", "rubric_assessment"]},
        )

    # -- Modules ---------------------------------------------------------------

    def list_modules(self, course_id: int | str | None = None, limit: int = 50) -> list[dict]:
        cid = self.resolve_course(course_id)
        return self._get_paginated(
            f"/courses/{cid}/modules", params={"include[]": "items_count"}, limit=limit,
        )

    def list_module_items(
        self, module_id: int | str, course_id: int | str | None = None, limit: int = 50,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        return self._get_paginated(f"/courses/{cid}/modules/{module_id}/items", limit=limit)

    # -- Pages -----------------------------------------------------------------

    def list_pages(
        self,
        course_id: int | str | None = None,
        sort: str | None = None,
        published: bool | None = None,
        limit: int = 50,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {}
        if sort:
            params["sort"] = sort
        if published is not None:
            params["published"] = str(published).lower()
        return self._get_paginated(f"/courses/{cid}/pages", params=params, limit=limit)

    def get_page(self, page_url: str, course_id: int | str | None = None) -> dict:
        cid = self.resolve_course(course_id)
        return self._get_json(f"/courses/{cid}/pages/{page_url}")

    # -- Announcements ---------------------------------------------------------

    def list_announcements(self, course_id: int | str | None = None, limit: int = 20) -> list[dict]:
        cid = self.resolve_course(course_id)
        return self._get_paginated(
            f"/courses/{cid}/discussion_topics",
            params={"only_announcements": "true"},
            limit=limit,
        )

    # -- Files -----------------------------------------------------------------

    def list_files(
        self,
        course_id: int | str | None = None,
        search_term: str | None = None,
        content_types: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {}
        if search_term:
            params["search_term"] = search_term
        if content_types:
            params["content_types[]"] = content_types
        return self._get_paginated(f"/courses/{cid}/files", params=params, limit=limit)

    def download_file(self, file_id: int | str, save_path: str | None = None) -> dict:
        file_info = self._get_json(f"/files/{file_id}")
        url = file_info.get("url")
        if not url:
            raise CanvasAPIError("No download URL in file response", 0)
        dl_resp = self._session.get(url, timeout=120, stream=True)
        if not dl_resp.ok:
            raise CanvasAPIError(f"Download failed ({dl_resp.status_code})", dl_resp.status_code)
        if not save_path:
            home = __import__("pwd").getpwuid(os.getuid()).pw_dir
            save_dir = os.path.join(home, ".mesh", "canvas_downloads")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_info.get("display_name", f"file_{file_id}"))
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in dl_resp.iter_content(8192):
                f.write(chunk)
        return {
            "path": save_path,
            "filename": file_info.get("display_name"),
            "size": file_info.get("size"),
        }

    # -- Analytics -------------------------------------------------------------

    def get_analytics(self, course_id: int | str | None = None, analytics_type: str = "activity") -> Any:
        cid = self.resolve_course(course_id)
        type_map = {
            "activity": f"/courses/{cid}/analytics/activity",
            "assignments": f"/courses/{cid}/analytics/assignments",
            "student_summaries": f"/courses/{cid}/analytics/student_summaries",
        }
        path = type_map.get(analytics_type)
        if not path:
            raise ValueError(
                f"Unknown analytics type: {analytics_type}. Use: activity, assignments, student_summaries"
            )
        return self._get_json(path)

    # -- Rubrics ---------------------------------------------------------------

    def list_rubrics(self, course_id: int | str | None = None, limit: int = 50) -> list[dict]:
        cid = self.resolve_course(course_id)
        return self._get_paginated(f"/courses/{cid}/rubrics", limit=limit)

    def get_rubric(
        self, rubric_id: int | str, course_id: int | str | None = None, include: str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {}
        if include:
            params["include[]"] = include
        return self._get_json(f"/courses/{cid}/rubrics/{rubric_id}", params=params)

    # -- Quizzes ---------------------------------------------------------------

    def list_quizzes(
        self, course_id: int | str | None = None, search_term: str | None = None, limit: int = 50,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        params: dict[str, Any] = {}
        if search_term:
            params["search_term"] = search_term
        return self._get_paginated(f"/courses/{cid}/quizzes", params=params, limit=limit)

    def get_quiz(self, quiz_id: int | str, course_id: int | str | None = None) -> dict:
        cid = self.resolve_course(course_id)
        return self._get_json(f"/courses/{cid}/quizzes/{quiz_id}")

    def list_quiz_questions(
        self, quiz_id: int | str, course_id: int | str | None = None, limit: int = 50,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        return self._get_paginated(f"/courses/{cid}/quizzes/{quiz_id}/questions", limit=limit)

    def list_quiz_submissions(
        self, quiz_id: int | str, course_id: int | str | None = None, limit: int = 200,
    ) -> list[dict]:
        cid = self.resolve_course(course_id)
        url = f"{self._base_url}/api/v1/courses/{cid}/quizzes/{quiz_id}/submissions"
        params = {"per_page": min(limit, 100)}
        all_subs: list[dict] = []
        while url and len(all_subs) < limit:
            resp = self._session.get(url, params=params, timeout=30)
            params = None
            self._update_rate_limit(resp)
            self._raise_for_status(resp)
            data = resp.json()
            if isinstance(data, dict) and "quiz_submissions" in data:
                all_subs.extend(data["quiz_submissions"])
            elif isinstance(data, list):
                all_subs.extend(data)
            url = None
            link = resp.headers.get("Link", "")
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
        return all_subs[:limit]


    # -- Write: Assignments ----------------------------------------------------

    def create_assignment(
        self,
        name: str,
        course_id: int | str | None = None,
        due_at: str | None = None,
        points_possible: float | None = None,
        description: str | None = None,
        submission_types: list[str] | None = None,
        published: bool = False,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload: dict[str, Any] = {"assignment": {"name": name, "published": published}}
        if due_at:
            payload["assignment"]["due_at"] = due_at
        if points_possible is not None:
            payload["assignment"]["points_possible"] = points_possible
        if description:
            payload["assignment"]["description"] = description
        if submission_types:
            payload["assignment"]["submission_types"] = submission_types
        return self._post_json(f"/courses/{cid}/assignments", json_body=payload)

    def update_assignment(
        self,
        assignment_id: int | str,
        course_id: int | str | None = None,
        **kwargs: Any,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload = {"assignment": {k: v for k, v in kwargs.items() if v is not None}}
        return self._put_json(f"/courses/{cid}/assignments/{assignment_id}", json_body=payload)

    # -- Write: Grading --------------------------------------------------------

    def grade_submission(
        self,
        assignment_id: int | str,
        student_id: int | str,
        grade: str,
        course_id: int | str | None = None,
        comment: str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload: dict[str, Any] = {"submission": {"posted_grade": grade}}
        if comment:
            payload["comment"] = {"text_comment": comment}
        return self._put_json(
            f"/courses/{cid}/assignments/{assignment_id}/submissions/{student_id}",
            json_body=payload,
        )

    def bulk_grade(
        self,
        assignment_id: int | str,
        grades: dict[str, dict],
        course_id: int | str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        grade_data = {}
        for student_id, info in grades.items():
            entry: dict[str, Any] = {"posted_grade": info["grade"]}
            if "comment" in info:
                entry["text_comment"] = info["comment"]
            grade_data[str(student_id)] = entry
        return self._post_json(
            f"/courses/{cid}/assignments/{assignment_id}/submissions/update_grades",
            json_body={"grade_data": grade_data},
        )

    # -- Write: Announcements --------------------------------------------------

    def post_announcement(
        self,
        title: str,
        message: str,
        course_id: int | str | None = None,
        delayed_post_at: str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload: dict[str, Any] = {
            "title": title,
            "message": message,
            "is_announcement": True,
        }
        if delayed_post_at:
            payload["delayed_post_at"] = delayed_post_at
        return self._post_json(f"/courses/{cid}/discussion_topics", json_body=payload)

    # -- Write: Pages ----------------------------------------------------------

    def create_page(
        self,
        title: str,
        body: str,
        course_id: int | str | None = None,
        published: bool = False,
        editing_roles: str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload: dict[str, Any] = {
            "wiki_page": {"title": title, "body": body, "published": published}
        }
        if editing_roles:
            payload["wiki_page"]["editing_roles"] = editing_roles
        return self._post_json(f"/courses/{cid}/pages", json_body=payload)

    def update_page(
        self,
        page_url: str,
        course_id: int | str | None = None,
        **kwargs: Any,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload = {"wiki_page": {k: v for k, v in kwargs.items() if v is not None}}
        return self._put_json(f"/courses/{cid}/pages/{page_url}", json_body=payload)

    # -- Write: Modules --------------------------------------------------------

    def create_module(
        self,
        name: str,
        course_id: int | str | None = None,
        position: int | None = None,
        prerequisite_module_ids: list[int] | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload: dict[str, Any] = {"module": {"name": name}}
        if position is not None:
            payload["module"]["position"] = position
        if prerequisite_module_ids:
            payload["module"]["prerequisite_module_ids"] = prerequisite_module_ids
        return self._post_json(f"/courses/{cid}/modules", json_body=payload)

    def add_module_item(
        self,
        module_id: int | str,
        item_type: str,
        course_id: int | str | None = None,
        content_id: int | str | None = None,
        title: str | None = None,
        position: int | None = None,
        page_url: str | None = None,
        external_url: str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        item: dict[str, Any] = {"type": item_type}
        if content_id is not None:
            item["content_id"] = int(content_id)
        if title:
            item["title"] = title
        if position is not None:
            item["position"] = position
        if page_url:
            item["page_url"] = page_url
        if external_url:
            item["external_url"] = external_url
        return self._post_json(
            f"/courses/{cid}/modules/{module_id}/items", json_body={"module_item": item}
        )

    def publish_module(
        self,
        module_id: int | str,
        published: bool,
        course_id: int | str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        return self._put_json(
            f"/courses/{cid}/modules/{module_id}",
            json_body={"module": {"published": published}},
        )

    # -- Write: File Upload ----------------------------------------------------

    def upload_file(
        self,
        local_path: str,
        course_id: int | str | None = None,
        folder_path: str = "/",
        name: str | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        if not os.path.isfile(local_path):
            raise CanvasAPIError(f"File not found: {local_path}", 0)
        file_size = os.path.getsize(local_path)
        file_name = name or os.path.basename(local_path)
        import mimetypes
        content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"

        # Step 1: Request upload URL
        step1 = self._post_json(
            f"/courses/{cid}/files",
            json_body={
                "name": file_name,
                "size": file_size,
                "content_type": content_type,
                "parent_folder_path": folder_path,
            },
        )
        upload_url = step1.get("upload_url")
        upload_params = step1.get("upload_params", {})
        if not upload_url:
            raise CanvasAPIError("Canvas did not return an upload URL", 0)

        # Step 2: Upload file data
        with open(local_path, "rb") as f:
            files = {"file": (file_name, f, content_type)}
            resp = requests.post(
                upload_url, data=upload_params, files=files, timeout=300, allow_redirects=False,
            )

        # Step 3: Confirm (follow redirect or POST to confirm URL)
        if resp.status_code in (301, 302, 303):
            confirm_url = resp.headers.get("Location")
            if confirm_url:
                confirm_resp = self._session.get(confirm_url, timeout=30)
                self._raise_for_status(confirm_resp)
                return confirm_resp.json()
        elif resp.ok:
            return resp.json()

        self._raise_for_status(resp)
        return resp.json()

    # -- Write: Messaging ------------------------------------------------------

    def inbox_message(
        self,
        recipients: list[str],
        subject: str,
        body: str,
        course_id: int | str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "recipients": recipients,
            "subject": subject,
            "body": body,
        }
        if course_id is not None:
            cid = self.resolve_course(course_id)
            payload["context_code"] = f"course_{cid}"
        return self._post_json("/conversations", json_body=payload)

    # -- Write: Quizzes --------------------------------------------------------

    def create_quiz(
        self,
        title: str,
        course_id: int | str | None = None,
        description: str | None = None,
        quiz_type: str = "assignment",
        time_limit: int | None = None,
        due_at: str | None = None,
        published: bool = False,
    ) -> dict:
        cid = self.resolve_course(course_id)
        quiz: dict[str, Any] = {"title": title, "quiz_type": quiz_type, "published": published}
        if description:
            quiz["description"] = description
        if time_limit is not None:
            quiz["time_limit"] = time_limit
        if due_at:
            quiz["due_at"] = due_at
        return self._post_json(f"/courses/{cid}/quizzes", json_body={"quiz": quiz})

    def create_quiz_question(
        self,
        quiz_id: int | str,
        question_text: str,
        question_type: str,
        course_id: int | str | None = None,
        question_name: str | None = None,
        points_possible: float | None = None,
        answers: list[dict] | None = None,
    ) -> dict:
        cid = self.resolve_course(course_id)
        question: dict[str, Any] = {
            "question_text": question_text,
            "question_type": question_type,
        }
        if question_name:
            question["question_name"] = question_name
        if points_possible is not None:
            question["points_possible"] = points_possible
        if answers:
            question["answers"] = answers
        return self._post_json(
            f"/courses/{cid}/quizzes/{quiz_id}/questions",
            json_body={"question": question},
        )

    def update_quiz(
        self,
        quiz_id: int | str,
        course_id: int | str | None = None,
        **kwargs: Any,
    ) -> dict:
        cid = self.resolve_course(course_id)
        payload = {"quiz": {k: v for k, v in kwargs.items() if v is not None}}
        return self._put_json(f"/courses/{cid}/quizzes/{quiz_id}", json_body=payload)


class CanvasAPIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code
