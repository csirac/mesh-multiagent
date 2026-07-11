from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import (
    async_playwright,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from playwright_stealth import Stealth

class BrowserClient:
    def __init__(self):
        self.p = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None

        self._ui_snapshot_id = 0
        self._ui_by_id: dict[int, Dict[str, Any]] = {}

        # Jitter tuning knobs
        self.jitter_action_ms = (400, 850)
        self.jitter_nav_ms = (1800, 2500)
        self._prev_key_to_id: dict[str, int] = {}
        self._prev_key_to_abbrev: dict[str, dict] = {}
        self._prev_max_id: int = 0

        # Approximate last mouse position for smoother moves (None = unknown)
        self._last_mouse_pos: tuple[float, float] | None = None

    @classmethod
    async def create(
            cls,
            *,
            user_data_dir: str,                 # REQUIRED for persistent profile
            profile_directory: str = "Default", # "Default", "Profile 1", ...
            headless: bool = False,             # False=headed, True=headless (uses new headless mode in Playwright 1.40+)
    ) -> "BrowserClient":
        instance = cls()
        instance.p = await async_playwright().start()

        # Persistent context == uses the on-disk Chrome profile at user_data_dir
        instance.context = await instance.p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",     # <-- system Chrome
            headless=headless,
            viewport={"width": 1080, "height": 1080},
            args=[
                "--window-size=1080,1080",
                f"--profile-directory={profile_directory}",
            ],
        )

        instance.browser = instance.context.browser

        stealth = Stealth()
        await stealth.apply_stealth_async(instance.context)

        # Persistent contexts may already have a page open
        pages = instance.context.pages
        instance.page = pages[0] if pages else await instance.context.new_page()

        instance._ui_snapshot_id = 0
        instance._ui_by_id = {}
        return instance

    async def close(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.p:
            await self.p.stop()
        # Reset state so is_open() returns False
        self.p = None
        self.browser = None
        self.context = None
        self.page = None
        self._ui_snapshot_id = 0
        self._ui_by_id = {}
        self._prev_key_to_id = {}
        self._prev_key_to_abbrev = {}
        self._prev_max_id = 0
        self._last_mouse_pos = None

    async def is_open(self) -> bool:
        """Check if the browser session is currently open and usable."""
        if self.page is None:
            return False
        try:
            # Try a simple operation to verify the page is still responsive
            _ = self.page.url
            return True
        except Exception:
            return False

    async def open(
        self,
        *,
        user_data_dir: str,
        profile_directory: str = "Default",
        headless: bool = False,             # False=headed, True=headless (uses new headless mode in Playwright 1.40+)
    ) -> None:
        """Open a browser session on this instance if not already open."""
        if await self.is_open():
            raise RuntimeError("Browser session is already open")

        self.p = await async_playwright().start()

        self.context = await self.p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",
            headless=headless,
            viewport={"width": 1080, "height": 1080},
            args=[
                "--window-size=1080,1080",
                f"--profile-directory={profile_directory}",
            ],
        )

        self.browser = self.context.browser

        stealth = Stealth()
        await stealth.apply_stealth_async(self.context)

        pages = self.context.pages
        self.page = pages[0] if pages else await self.context.new_page()

        self._ui_snapshot_id = 0
        self._ui_by_id = {}

    # -------------------------
    # Jitter helpers
    # -------------------------
    async def _jitter(self, min_ms: int, max_ms: int) -> None:
        if max_ms <= 0:
            return
        if min_ms < 0:
            min_ms = 0
        if max_ms < min_ms:
            max_ms = min_ms
        ms = random.randint(min_ms, max_ms)
        await self.page.wait_for_timeout(ms)

    async def _post_action(self) -> None:
        a, b = self.jitter_action_ms
        await self._jitter(a, b)

    async def _post_nav(self) -> None:
        a, b = self.jitter_nav_ms
        await self._jitter(a, b)

    # -------------------------
    # Locators by snapshot id
    # -------------------------
    def _sel_for_id(self, element_id: int) -> str:
        return f'[data-agent-id="{int(element_id)}"]'

    async def _loc_for_id(self, element_id: int):
        loc = self.page.locator(self._sel_for_id(element_id))
        c = await loc.count()
        if c == 0:
            raise ValueError(f"Element id={element_id} not found in DOM (stale snapshot?).")
        if c > 1:
            raise ValueError(f"Element id={element_id} matched {c} nodes (id system broken).")
        return loc

    async def _move_mouse_to_element(self, loc) -> None:
        """Move the mouse to a random point inside the element's bounding box, with visible, human-ish motion."""
        try:
            box = await loc.bounding_box()
        except Exception:
            box = None

        if not box:
            return  # nothing we can do

        # Target point: random location inside the element
        target_x = box.get("x", 0) + box.get("width", 0) * random.uniform(0.2, 0.8)
        target_y = box.get("y", 0) + box.get("height", 0) * random.uniform(0.2, 0.8)

        # If we have no idea where the mouse is, just move there in several steps from an off-screen-ish guess
        if getattr(self, "_last_mouse_pos", None) is None:
            steps = random.randint(8, 18)
            for i in range(1, steps + 1):
                frac = i / steps
                x = target_x * frac + (target_x - 200) * (1 - frac)
                y = target_y * frac + (target_y - 200) * (1 - frac)
                await self.page.mouse.move(x, y)
                await self.page.wait_for_timeout(random.randint(10, 35))
        else:
            start_x, start_y = self._last_mouse_pos
            steps = random.randint(10, 24)
            for i in range(1, steps + 1):
                frac = i / steps
                # small random wobble around the line
                x = start_x + (target_x - start_x) * frac + random.uniform(-1.5, 1.5)
                y = start_y + (target_y - start_y) * frac + random.uniform(-1.5, 1.5)
                await self.page.mouse.move(x, y)
                await self.page.wait_for_timeout(random.randint(10, 35))

        self._last_mouse_pos = (target_x, target_y)

    # -------------------------
    # Navigation / lifecycle
    # -------------------------
    async def goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded")
        await self._post_nav()

    async def back(self) -> None:
        await self.page.go_back(wait_until="domcontentloaded")
        await self._post_nav()

    async def get_url(self) -> str:
        return self.page.url

    async def wait(self, ms: int) -> None:
        await self.page.wait_for_timeout(ms)

    async def snapshot_text(
        self,
        *,
        limit: int = 200,
        min_len: int = 5,
        max_len: int = 400,
        filter: str = None,
    ) -> list[dict]:
        if filter == "":
            filter = None

        return await self.page.evaluate(
            """({limit, minLen, maxLen, filterText}) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
              const filterNorm = (filterText || '').toLowerCase();

              const isVisible = (el) => {
                const s = getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden' || s.pointerEvents === 'none') return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
              };

              const scoreModal = (el) => {
                // Prefer "top-most" modal-like things.
                const s = getComputedStyle(el);
                const z = parseInt(s.zIndex || '0', 10) || 0;
                const ariaModal = (el.getAttribute('aria-modal') || '') === 'true' ? 100000 : 0;
                const roleDialog = (el.getAttribute('role') || '') === 'dialog' ? 50000 : 0;
                const isDialogTag = el.tagName.toLowerCase() === 'dialog' ? 60000 : 0;
                return z + ariaModal + roleDialog + isDialogTag;
              };

              const pushText = (out, seen, text, meta) => {
                const t = norm(text);
                if (!t || t.length < minLen) return false;
                if (filterNorm && !t.toLowerCase().includes(filterNorm)) return false;

                const clipped = t.slice(0, Math.max(0, maxLen || 0));
                if (!clipped) return false;

                // de-dupe by clipped text
                if (seen.has(clipped)) return false;
                seen.add(clipped);

                out.push({
                  kind: "text",
                  text: clipped,
                  ...meta
                });
                return true;
              };

              const out = [];
              const seen = new Set();

              // -------------------------
              // 1) Modal / overlay first
              // -------------------------
              const modalCandidates = Array.from(document.querySelectorAll(
                'dialog,[role="dialog"],[aria-modal="true"],[class*="modal"],[class*="Modal"],[data-component*="modal"]'
              )).filter(isVisible);

              modalCandidates.sort((a,b) => scoreModal(b) - scoreModal(a));

              for (const modal of modalCandidates) {
                if (out.length >= limit) break;

                // First, try the modal's own innerText (usually the best summary)
                pushText(out, seen, modal.innerText || modal.textContent || "", { source: "modal" });
                if (out.length >= limit) break;

                // Then try key text-ish nodes inside the modal for more detail
                const innerSel = 'h1,h2,h3,h4,h5,h6,p,li,dt,dd,small,strong,em,td,th,[aria-label]';
                const inner = Array.from(modal.querySelectorAll(innerSel)).filter(isVisible);
                for (const el of inner) {
                  if (out.length >= limit) break;
                  const aria = norm(el.getAttribute('aria-label') || '');
                  const t0 = norm(el.innerText || el.textContent || '');
                  const t = t0 || aria;
                  pushText(out, seen, t, { source: "modal" });
                }
              }

              // -------------------------
              // 2) General page text
              // -------------------------
              if (out.length < limit) {
                const sel = 'h1,h2,h3,h4,h5,h6,p,li,dt,dd,small,strong,em,td,th,[aria-label]';
                const els = Array.from(document.querySelectorAll(sel));

                for (const el of els) {
                  if (out.length >= limit) break;
                  if (!isVisible(el)) continue;

                  // Skip text that is inside a modal we already processed (optional but helps)
                  if (modalCandidates.length) {
                    let node = el;
                    let insideModal = false;
                    while (node && node !== document.body) {
                      if (modalCandidates.includes(node)) { insideModal = true; break; }
                      node = node.parentElement;
                    }
                    if (insideModal) continue;
                  }

                  const aria = norm(el.getAttribute('aria-label') || '');
                  const t0 = norm(el.innerText || el.textContent || '');
                  const t = t0 || aria;

                  pushText(out, seen, t, { source: "page" });
                }
              }

              return out;
            }""",
            {
                "limit": limit,
                "minLen": min_len,
                "maxLen": max_len,
                "filterText": filter,
            },
        )
    
    async def snapshot_fast(
        self,
        *,
        limit: int = 200,
        filter: str = None,
        delta: bool = False,
        include_key: bool = False,
    ) -> Any:
        """Snapshot actionable UI controls with stable ids.

        Ordering priorities inside a single snapshot:
          1) Controls inside visible modals
          2) Controls in the main content region ([role=main], #content, etc.)
          3) Nav/sidebars/other chrome

        IMPORTANT: In full mode (delta=False), we:
          - Build the full list of visible controls in that priority order,
          - Apply any label filter over the full list,
          - Then apply `limit`.

        That means a filter like "Add Content to LLM Test Module (Sandbox)" will
        match even if the corresponding button lives deep in the DOM, beyond the
        usual limit used for unfiltered snapshots.
        """

        self._ui_snapshot_id += 1
        snapshot_id = self._ui_snapshot_id

        prev_key_to_id = getattr(self, "_prev_key_to_id", {}) or {}
        prev_max_id = int(getattr(self, "_prev_max_id", 0) or 0)

        # JS side: build full, unbounded list of visible controls in priority order.
        full_items = await self.page.evaluate(
            """({snapshotId, prevKeyToId, prevMaxId}) => {
              for (const el of document.querySelectorAll('[data-agent-id]')) {
                el.removeAttribute('data-agent-id');
              }

              const isVisibleNow = (el) => {
                const s = window.getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden' || s.pointerEvents === 'none') return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
              };

              const sel = 'a[href],button,input,select,textarea,[role],[onclick],[tabindex]';
              const allEls = Array.from(document.querySelectorAll(sel));

              // Modal-first control discovery: collect visible modals, then
              // prioritize controls that are descendants of those modals.
              const modalCandidates = Array.from(document.querySelectorAll(
                'dialog,[role="dialog"],[aria-modal="true"],[class*="modal"],[class*="Modal"],[data-component*="modal"]'
              )).filter(isVisibleNow);

              // Reordered list of actionable elements: modal controls first,
              // then region-important controls, then global nav / sidebars last.
              const els = [];
              const seen = new Set();

              const isRegionImportant = (el) => {
                // Prefer controls inside ARIA main landmarks or Canvas content wrappers.
                const main = el.closest('[role="main"], #content, #content-wrapper, #not_right_side');
                if (main) return true;
                return false;
              };

              const isNavOrSidebar = (el) => {
                // Heuristic: global nav, course nav, sidebars, footers.
                const navLike = el.closest('nav, #global_nav, #left-side, #right-side, .ic-app-nav-toggle-and-crumbs');
                if (navLike) return true;
                const cls = (el.className || '').toString();
                if (/ic-app-nav|ic-app-course-menu|ic-Layout-watermark/.test(cls)) return true;
                return false;
              };

              // Buckets for non-modal controls.
              const regionEls = [];
              const otherEls = [];

              // 1) Controls inside visible modals (highest priority)
              for (const modal of modalCandidates) {
                for (const el of allEls) {
                  if (!seen.has(el) && modal.contains(el)) {
                    seen.add(el);
                    els.push(el);
                  }
                }
              }

              // 2) Partition remaining controls into region-important and others
              for (const el of allEls) {
                if (seen.has(el)) continue;
                if (isNavOrSidebar(el)) {
                  otherEls.push(el);
                } else if (isRegionImportant(el)) {
                  regionEls.push(el);
                } else {
                  otherEls.push(el);
                }
              }

              // 3) Push region-important controls next
              for (const el of regionEls) {
                if (!seen.has(el)) {
                  seen.add(el);
                  els.push(el);
                }
              }

              // 4) Finally put nav / sidebar / misc controls at the end
              for (const el of otherEls) {
                if (!seen.has(el)) {
                  seen.add(el);
                  els.push(el);
                }
              }

              const actionsFor = (tag, role, href) => {
                tag = (tag || '').toLowerCase();
                role = (role || '').toLowerCase();
                href = href || '';
                if (tag === 'input' || tag === 'textarea' || role === 'combobox' || role === 'textbox' || role === 'searchbox')
                  return ['fill','press','click'];
                if (tag === 'select' || role === 'listbox')
                  return ['select','click'];
                if (tag === 'a' && href)
                  return ['click'];
                return ['click'];
              };

              const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

              // Strip embedded URLs from labels; we already expose href separately.
              const stripUrls = (s) => {
                if (!s) return '';
                return norm(s.replace(/https?:\/\/\S+/g, ' '));
              };

              // Trim trailing pure-numeric tokens: "X X 10 0" -> "X X".
              const trimNumericTail = (s) => {
                const parts = (s || '').split(' ');
                while (parts.length && /^[0-9]+$/.test(parts[parts.length - 1])) {
                  parts.pop();
                }
                return parts.join(' ');
              };

              // Collapse duplicated prefix: "A B C A B C 10 0" -> "A B C".
              const collapseDuplicatePrefix = (s) => {
                const parts = (s || '').split(' ');
                if (parts.length < 6) return s; // need at least 3+3
                // Try prefix lengths from up to 8 words down to 3.
                for (let len = Math.min(8, Math.floor(parts.length / 2)); len >= 3; len--) {
                  let ok = true;
                  for (let i = 0; i < len; i++) {
                    if (parts[i] !== parts[len + i]) {
                      ok = false;
                      break;
                    }
                  }
                  if (ok) {
                    const head = parts.slice(0, len).join(' ');
                    const tail = parts.slice(2 * len).join(' ');
                    return (head + (tail ? (' ' + tail) : '')).trim();
                  }
                }
                return s;
              };

              const bestLabel = (el) => {
                const aria = (el.getAttribute('aria-label') || '').trim();
                const text = (el.textContent || '').trim().replace(/\s+/g,' ');
                const ph = (el.getAttribute('placeholder') || '').trim();
                const nm = (el.getAttribute('name') || '').trim();
                const href = (el.getAttribute('href') || '').trim();

                const role = (el.getAttribute('role') || '').trim();
                const normLower = (s) => (s || '').toLowerCase().replace(/\s+/g,' ').trim();

                // Prefer explicit dialog titles when present (e.g. Canvas "Add Item to ..." modals).
                if (role === 'dialog') {
                  const titleEl = el.querySelector('.ui-dialog-title');
                  if (titleEl) {
                    const dialogTitle = normLower(titleEl.textContent || '');
                    if (dialogTitle) {
                      return dialogTitle.slice(0, 200);
                    }
                  }
                }

                const raw = (aria || text || ph || nm || href).slice(0, 300);
                return collapseDuplicatePrefix(trimNumericTail(stripUrls(raw)));
              };

              const isEnabled = (el) => {
                const ariaDisabled = (el.getAttribute('aria-disabled') || '').toLowerCase() === 'true';
                const disabledProp = !!el.disabled;
                return !(ariaDisabled || disabledProp);
              };

              const normLower = (s) => (s || '').toLowerCase().replace(/\s+/g,' ').trim();

              const signatureFor = (tag, role, href, ariaLabel, placeholder, nameAttr, label) => {
                return [
                  normLower(tag),
                  normLower(role),
                  normLower(href),
                  normLower(nameAttr),
                  normLower(placeholder),
                  normLower(ariaLabel),
                  normLower(label),
                ].join('|');
              };

              let nextId = Math.max(0, prevMaxId);
              const usedPrevIds = new Set();
              const sigCounts = Object.create(null);

              const allVisible = [];
              for (const el of els) {
                if (!isVisibleNow(el)) continue;

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const href = el.getAttribute('href') || '';
                const ariaLabel = el.getAttribute('aria-label') || '';
                const placeholder = el.getAttribute('placeholder') || '';
                const nameAttr = el.getAttribute('name') || '';
                const label = bestLabel(el);

                const sig = signatureFor(tag, role, href, ariaLabel, placeholder, nameAttr, label);
                const occ = (sigCounts[sig] || 0);
                sigCounts[sig] = occ + 1;

                const key = `${sig}#${occ}`;

                let id = prevKeyToId[key];
                if (typeof id === 'number' && id > 0 && !usedPrevIds.has(id)) {
                  usedPrevIds.add(id);
                } else {
                  nextId += 1;
                  id = nextId;
                }

                el.setAttribute('data-agent-id', String(id));

                const text = (el.textContent || '').trim().replace(/\s+/g,' ').slice(0, 120);

                allVisible.push({
                  snapshotId,
                  id,
                  key,
                  kind: 'dom',
                  tag,
                  role,
                  href,
                  ariaLabel,
                  placeholder,
                  nameAttr,
                  text,
                  label,
                  actions: actionsFor(tag, role, href),
                  enabled: isEnabled(el),
                  selector: `[data-agent-id="${id}"]`,
                });
              }

              return { items: allVisible, maxId: nextId };
            }""",
            {"snapshotId": snapshot_id, "prevKeyToId": prev_key_to_id, "prevMaxId": prev_max_id},
        )

        items_full: list[dict] = full_items["items"]
        self._prev_max_id = int(full_items.get("maxId") or 0)

        # keep full items for execution/debug
        self._ui_by_id = {int(e["id"]): e for e in items_full}

        # id reuse map for next time
        self._prev_key_to_id = {e["key"]: int(e["id"]) for e in items_full if e.get("key")}

        # -----------------------------
        # Build *UNFILTERED* abbrev list (baseline for delta tracking)
        # -----------------------------
        ui_abbrev_all: list[dict] = []
        for e in items_full:
            label = e.get("label") or ""
            e_short = {"id": int(e["id"]), "label": label, "type": e.get("tag", "")}

            e_short["key"] = e.get("key", "")
            if not include_key:
                # we'll remove it right before returning (see below)
                pass

            if e.get("tag") == "a":
                e_short["type"] = "link"
            elif e.get("tag") != "button":
                e_short["actions"] = e.get("actions") or []

            if not e.get("enabled", True):
                e_short["enabled"] = False

            ui_abbrev_all.append(e_short)

        # normalize filter
        if filter == "":
            filter = None
        f = filter.lower() if isinstance(filter, str) and filter else None

        # helper: apply filter to a list (by label, full-list then limit)
        def _filter_list(lst: list[dict], *, limit: int | None) -> list[dict]:
            if not f:
                return lst[: limit or None]

            # Basic substring match
            base = [x for x in lst if f in (x.get("label", "").lower())]

            # Optional: light token-based soft match (half tokens)
            tokens = [t for t in f.split() if t]
            if tokens:
                for x in lst:
                    lab = (x.get("label", "").lower())
                    if f in lab:
                        continue
                    hits = sum(1 for t in tokens if t in lab)
                    if hits >= (len(tokens) + 1) // 2:
                        base.append(x)

            # Deduplicate while preserving order
            seen_ids: set[int] = set()
            out: list[dict] = []
            for x in base:
                xid = int(x.get("id", 0))
                if xid in seen_ids:
                    continue
                seen_ids.add(xid)
                out.append(x)

            return out[: limit or None]

        # -----------------------------
        # Delta mode (computed vs UNFILTERED baseline)
        # Baseline storage is always unfiltered; returned payload can be filtered.
        # -----------------------------
        if delta:
            prev_all = getattr(self, "_prev_key_to_abbrev", {}) or {}
            cur_all = {e.get("key", ""): e for e in ui_abbrev_all if e.get("key")}

            added_all = [cur_all[k] for k in cur_all.keys() if k not in prev_all]
            updated_all = [cur_all[k] for k in cur_all.keys() if k in prev_all and cur_all[k] != prev_all[k]]

            removed_keys_all = [k for k in prev_all.keys() if k not in cur_all]

            # return removed element info from the *previous* snapshot
            removed_all = []
            for k in removed_keys_all:
                old = dict(prev_all[k])
                old["key"] = k
                old["removed"] = True
                removed_all.append(old)

            # Store UNFILTERED current baseline
            self._prev_key_to_abbrev = cur_all

            added_out = _filter_list(added_all, limit=limit)
            updated_out = _filter_list(updated_all, limit=limit)
            removed_out = _filter_list(removed_all, limit=limit)

            for e in added_out + updated_out + removed_out:
                if not include_key:
                    e.pop("key", None)

            return {
                "added": added_out,
                "updated": updated_out,
                "removed": removed_out,
            }

        # -----------------------------
        # Full mode: filter over full list, then apply limit
        # -----------------------------
        self._prev_key_to_abbrev = {e["key"]: e for e in ui_abbrev_all if e.get("key")}
        out = _filter_list(ui_abbrev_all, limit=limit)
        if not include_key:
            for e in out:
                e.pop("key", None)
        return out


    async def list_frames(self) -> list[dict]:
        """Debug helper: list all frames for the current page."""
        frames = self.page.frames
        out: list[dict] = []
        for idx, f in enumerate(frames):
            try:
                url = f.url
            except Exception:
                url = ""
            try:
                name = f.name
            except Exception:
                name = ""
            out.append({
                "index": idx,
                "is_main": (f == self.page.main_frame),
                "url": url,
                "name": name,
            })
        return out

    async def debug_frame_controls(self, frame_index: int, *, limit: int = 500) -> list[dict]:
        """
        Debug helper: dump actionable-ish elements from a single frame.

        Includes both visible and hidden / off-viewport elements, and marks
        visibility-related flags so we can see what the normal agent snapshot
        would filter out.
        """
        frames = self.page.frames
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"frame_index out of range: {frame_index} (have {len(frames)} frames)")

        frame = frames[frame_index]

        items = await frame.evaluate(
            """({limit}) => {
              const sel = 'a[href],button,input,select,textarea,[role],[onclick],[tabindex]';
              const allEls = Array.from(document.querySelectorAll(sel));

              // Modal-first ordering for debug output: we want to see which
              // controls live under modals, but we still include both visible
              // and hidden elements, as the docstring promises.
              const modalCandidates = Array.from(document.querySelectorAll(
                'dialog,[role="dialog"],[aria-modal="true"],[class*="modal"],[class*="Modal"],[data-component*="modal"]'
              ));

              const els = [];
              const seen = new Set();

              // 1) Controls that are descendants of any modal candidate
              for (const modal of modalCandidates) {
                for (const el of allEls) {
                  if (!seen.has(el) && modal.contains(el)) {
                    seen.add(el);
                    els.push(el);
                  }
                }
              }

              // 2) All remaining controls outside modals
              for (const el of allEls) {
                if (!seen.has(el)) {
                  seen.add(el);
                  els.push(el);
                }
              }

              const isVisibleNow = (el) => {
                const s = window.getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden' || s.pointerEvents === 'none') return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
              };

              const isInViewport = (el, minPx = 4) => {
                const r = el.getBoundingClientRect();
                const vw = window.innerWidth || document.documentElement.clientWidth;
                const vh = window.innerHeight || document.documentElement.clientHeight;
                const ix = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0));
                const iy = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
                const area = ix * iy;
                return { inViewport: area >= minPx, visibleArea: area };
              };

              const bestLabel = (el) => {
                const aria = (el.getAttribute('aria-label') || '').trim();
                const text = (el.textContent || '').trim().replace(/\\s+/g,' ');
                const ph = (el.getAttribute('placeholder') || '').trim();
                const nm = (el.getAttribute('name') || '').trim();
                const href = (el.getAttribute('href') || '').trim();
                return (aria || text || ph || nm || href).slice(0, 300);
              };

              const actionsFor = (tag, role, href) => {
                tag = (tag || '').toLowerCase();
                role = (role || '').toLowerCase();
                href = href || '';
                if (tag === 'input' || tag === 'textarea' || role === 'combobox' || role === 'textbox' || role === 'searchbox')
                  return ['fill','press','click'];
                if (tag === 'select' || role === 'listbox')
                  return ['select','click'];
                if (tag === 'a' && href)
                  return ['click'];
                return ['click'];
              };

              const boxOf = (el) => {
                const r = el.getBoundingClientRect();
                return { x: r.x, y: r.y, width: r.width, height: r.height };
              };

              const out = [];
              for (const el of els) {
                if (out.length >= limit) break;

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const href = el.getAttribute('href') || '';
                const ariaLabel = el.getAttribute('aria-label') || '';
                const placeholder = el.getAttribute('placeholder') || '';
                const nameAttr = el.getAttribute('name') || '';
                const text = (el.textContent || '').trim().replace(/\\s+/g,' ').slice(0, 200);
                const label = bestLabel(el);

                const visibleNow = isVisibleNow(el);
                const vp = isInViewport(el, 4);

                out.push({
                  tag,
                  role,
                  href,
                  ariaLabel,
                  placeholder,
                  nameAttr,
                  text,
                  label,
                  actions: actionsFor(tag, role, href),
                  visibleNow,
                  inViewport: vp.inViewport,
                  visibleArea: vp.visibleArea,
                  box: boxOf(el),
                });
              }

              return out;
            }""",
            {"limit": limit},
        )

        for e in items:
            e["frame_index"] = frame_index
        return items
    async def debug_frame_tree(self, frame_index: int, *, limit: int = 500) -> dict:
        """
        Debug helper: hierarchical snapshot of actionable controls in a single frame.

        Walks the DOM tree and returns a nested container/control structure, with
        a simple actionCount for how many controls were included. This is meant
        for offline inspection and comparison against debug_frame_controls.
        """
        frames = self.page.frames
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"frame_index out of range: {frame_index} (have {len(frames)} frames)")

        frame = frames[frame_index]

        result = await frame.evaluate(
            """({limit}) => {
              const selActionable = 'a[href],button,input,select,textarea,[role],[onclick],[tabindex]';

              const isVisibleNow = (el) => {
                const s = window.getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden' || s.pointerEvents === 'none') return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
              };

              const isActionable = (el) => {
                const tag = el.tagName.toLowerCase();
                const role = (el.getAttribute('role') || '').toLowerCase();
                if (tag === 'a' && el.getAttribute('href')) return true;
                if (tag === 'button') return true;
                if (tag === 'input' || tag === 'select' || tag === 'textarea') return true;
                if (['button','link','menuitem','tab','option'].includes(role)) return true;
                if (el.hasAttribute('onclick') || el.hasAttribute('tabindex')) return true;
                return false;
              };

              const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

              // Strip embedded URLs from labels; we already expose href separately.
              const stripUrls = (s) => {
                if (!s) return '';
                return norm(s.replace(/https?:\/\/\S+/g, ' '));
              };

              // Trim trailing pure-numeric tokens: "X X 10 0" -> "X X".
              const trimNumericTail = (s) => {
                const parts = (s || '').split(' ');
                while (parts.length && /^[0-9]+$/.test(parts[parts.length - 1])) {
                  parts.pop();
                }
                return parts.join(' ');
              };

              // Collapse duplicated prefix: "A B C A B C 10 0" -> "A B C".
              const collapseDuplicatePrefix = (s) => {
                const parts = (s || '').split(' ');
                if (parts.length < 6) return s; // need at least 3+3
                // Try prefix lengths from up to 8 words down to 3.
                for (let len = Math.min(8, Math.floor(parts.length / 2)); len >= 3; len--) {
                  let ok = true;
                  for (let i = 0; i < len; i++) {
                    if (parts[i] !== parts[len + i]) {
                      ok = false;
                      break;
                    }
                  }
                  if (ok) {
                    const head = parts.slice(0, len).join(' ');
                    const tail = parts.slice(2 * len).join(' ');
                    return (head + (tail ? (' ' + tail) : '')).trim();
                  }
                }
                return s;
              };

              const bestLabel = (el) => {
                const tag = el.tagName.toLowerCase();
                const role = (el.getAttribute('role') || '').toLowerCase();

                const aria = norm(el.getAttribute('aria-label') || '');
                const text = norm(el.textContent || '');
                const ph = norm(el.getAttribute('placeholder') || '');
                const nm = norm(el.getAttribute('name') || '');
                const href = norm(el.getAttribute('href') || '');

                // Prefer explicit dialog titles when present (e.g. Canvas "Add Item to ..." modals).
                if (role === 'dialog') {
                  const titleEl = el.querySelector('.ui-dialog-title');
                  if (titleEl) {
                    const dialogTitle = norm(titleEl.textContent || '');
                    if (dialogTitle) {
                      return dialogTitle.slice(0, 200);
                    }
                  }
                }

                const containerishTags = ['main','section','article','nav','header','footer','aside','ul','ol','li','div','form'];
                const isContainer = containerishTags.includes(tag);

                const shortLabelFromDescendant = () => {
                  const cand = el.querySelector('a[href],button,[role="button"],[role="link"]');
                  if (!cand) return '';
                  const aria2 = norm(cand.getAttribute('aria-label') || '');
                  const text2 = norm(cand.textContent || '');
                  const ph2 = norm(cand.getAttribute('placeholder') || '');
                  const nm2 = norm(cand.getAttribute('name') || '');
                  const href2 = norm(cand.getAttribute('href') || '');
                  const lab = aria2 || text2 || ph2 || nm2 || href2;
                  return lab.slice(0, 160);
                };

                // Prefer aria-label for nav and landmark-ish containers when present
                if (aria && (tag === 'nav' || ['banner','main','contentinfo','complementary','region'].includes(role))) {
                  return aria.slice(0, 160);
                }

                let base = aria || text || ph || nm || href;

                // For generic containers with long text and no aria, try using a child link/button label.
                if (!aria && isContainer && base && base.length > 80) {
                  const childLab = shortLabelFromDescendant();
                  if (childLab) {
                    base = childLab;
                  }
                }

                if (!base) return '';

                // Generic cleanups: strip URLs, numeric tails, duplicated prefixes.
                base = stripUrls(base);
                base = trimNumericTail(base);
                base = collapseDuplicatePrefix(base);

                // Final truncation to keep labels manageable
                return base.slice(0, 200);
              };

              const isContainerishTag = (tag) => {
                return ['main','section','article','nav','header','footer','aside','ul','ol','li','div','form'].includes(tag);
              };

              let actionCount = 0;

              const walk = (el) => {
                if (!isVisibleNow(el)) return null;

                const children = [];
                for (const child of el.children) {
                  const c = walk(child);
                  if (c) children.push(c);
                }

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const href = el.getAttribute('href') || '';
                const label = bestLabel(el);
                const actionable = isActionable(el);

                if (actionable) {
                  if (limit > 0 && actionCount >= limit) {
                    return null;
                  }
                  actionCount += 1;
                  return {
                    kind: 'control',
                    tag,
                    role,
                    href,
                    label,
                    children: [],
                  };
                }

                // Container node: keep only if it has actionable descendants.
                if (!children.length) {
                  return null;
                }

                // Collapse boring single-child wrappers like <div> / <span> with no special role.
                if ((tag === 'div' || tag === 'span') && !role && children.length === 1) {
                  return children[0];
                }

                // Only keep some container tags to avoid wrapping everything in generic divs.
                if (!isContainerishTag(tag)) {
                  // If this non-containerish node has exactly one child, just return the child.
                  if (children.length === 1) {
                    return children[0];
                  }
                  return {
                    kind: 'container',
                    tag,
                    role,
                    label,
                    children,
                  };
                }

                return {
                  kind: 'container',
                  tag,
                  role,
                  label,
                  children,
                };
              };

              const root = walk(document.body) || {
                kind: 'container',
                tag: 'body',
                role: '',
                label: '',
                children: [],
              };

              return { tree: root, actionCount };
            }""",
            {"limit": limit},
        )

        result["frame_index"] = frame_index
        return result


    async def debug_frame_tree_v2(self, frame_index: int, *, limit: int = 500) -> dict:
        """Debug helper: canonical hierarchical snapshot schema with ids on controls.

        This version shares the same id space as snapshot_fast by reusing the
        data-agent-id attributes that snapshot_fast assigns. It first runs a
        non-delta snapshot to ensure ids are present, then walks the DOM to
        build a hierarchical tree using those ids.
        """
        frames = self.page.frames
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"frame_index out of range: {frame_index} (have {len(frames)} frames)")

        # Ensure snapshot_fast has tagged actionable elements with data-agent-id
        # and established a stable id mapping for this snapshot.
        try:
            await self.snapshot_fast(limit=limit, filter=None, delta=False)
        except Exception:
            # If this fails we still attempt the tree; worst case, controls
            # may be missing ids and will be skipped.
            pass

        frame = frames[frame_index]

        result = await frame.evaluate(
            """({limit}) => {
              const isVisibleNow = (el) => {
                const s = window.getComputedStyle(el);
                if (s.display === 'none' || s.visibility === 'hidden' || s.pointerEvents === 'none') return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
              };

              const isActionable = (el) => {
                const tag = el.tagName.toLowerCase();
                const role = (el.getAttribute('role') || '').toLowerCase();
                if (tag === 'a' && el.getAttribute('href')) return true;
                if (tag === 'button') return true;
                if (tag === 'input' || tag === 'select' || tag === 'textarea') return true;
                if (['button','link','menuitem','tab','option'].includes(role)) return true;
                if (el.hasAttribute('onclick') || el.hasAttribute('tabindex')) return true;
                return false;
              };

              const actionsFor = (tag, role, href) => {
                tag = (tag || '').toLowerCase();
                role = (role || '').toLowerCase();
                href = href || '';
                if (tag === 'input' || tag === 'textarea' || role === 'combobox' || role === 'textbox' || role === 'searchbox')
                  return ['fill','press','click'];
                if (tag === 'select' || role === 'listbox')
                  return ['select','click'];
                if (tag === 'a' && href)
                  return ['click'];
                return ['click'];
              };

              const bestLabel = (el) => {
                const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

                // Strip embedded URLs from labels; we already expose href separately.
                const stripUrls = (s) => {
                  if (!s) return '';
                  return norm(s.replace(/https?:\/\/\S+/g, ' '));
                };

                // Trim trailing pure-numeric tokens: "X X 10 0" -> "X X".
                const trimNumericTail = (s) => {
                  const parts = (s || '').split(' ');
                  while (parts.length && /^[0-9]+$/.test(parts[parts.length - 1])) {
                    parts.pop();
                  }
                  return parts.join(' ');
                };

                // Collapse duplicated prefix: "A B C A B C 10 0" -> "A B C".
                const collapseDuplicatePrefix = (s) => {
                  const parts = (s || '').split(' ');
                  if (parts.length < 6) return s; // need at least 3+3
                  // Try prefix lengths from up to 8 words down to 3.
                  for (let len = Math.min(8, Math.floor(parts.length / 2)); len >= 3; len--) {
                    let ok = true;
                    for (let i = 0; i < len; i++) {
                      if (parts[i] !== parts[len + i]) {
                        ok = false;
                        break;
                      }
                    }
                    if (ok) {
                      const head = parts.slice(0, len).join(' ');
                      const tail = parts.slice(2 * len).join(' ');
                      return (head + (tail ? (' ' + tail) : '')).trim();
                    }
                  }
                  return s;
                };

                const tag = el.tagName.toLowerCase();

                // Prefer explicit dialog titles when present (e.g. Canvas "Add Item to ..." modals).
                const rawRole = el.getAttribute('role') || '';
                if (rawRole === 'dialog' || rawRole.toLowerCase() === 'dialog') {
                  const titleEl = el.querySelector('.ui-dialog-title');
                  if (titleEl) {
                    const dialogTitle = norm(titleEl.textContent || '');
                    if (dialogTitle) {
                      // Prefer dialog title over generic label; return early
                      return dialogTitle.slice(0, 200);
                    }
                  }
                }
                const role = (el.getAttribute('role') || '').toLowerCase();

                const aria = norm(el.getAttribute('aria-label') || '');
                const text = norm(el.textContent || '');
                const ph   = norm(el.getAttribute('placeholder') || '');
                const nm   = norm(el.getAttribute('name') || '');
                const href = norm(el.getAttribute('href') || '');

                const containerishTags = ['main','section','article','nav','header','footer','aside','ul','ol','li','div','form'];
                const isContainer = containerishTags.includes(tag);

                const shortLabelFromDescendant = () => {
                  const cand = el.querySelector('a[href],button,[role=\"button\"],[role=\"link\"]');
                  if (!cand) return '';
                  const aria2 = norm(cand.getAttribute('aria-label') || '');
                  const text2 = norm(cand.textContent || '');
                  const ph2   = norm(cand.getAttribute('placeholder') || '');
                  const nm2   = norm(cand.getAttribute('name') || '');
                  const href2 = norm(cand.getAttribute('href') || '');
                  const lab = aria2 || text2 || ph2 || nm2 || href2;
                  return lab.slice(0, 160);
                };

                // Prefer aria-label for nav and landmark-ish containers when present
                if (aria && (tag === 'nav' || ['banner','main','contentinfo','complementary','region'].includes(role))) {
                  return aria.slice(0, 160);
                }

                let base = aria || text || ph || nm || href;

                // For generic containers with long text and no aria, try using a child link/button label.
                if (!aria && isContainer && base && base.length > 80) {
                  const childLab = shortLabelFromDescendant();
                  if (childLab) {
                    base = childLab;
                  }
                }

                if (!base) return '';

                // Generic cleanups: strip URLs, numeric tails, duplicated prefixes.
                base = stripUrls(base);
                base = trimNumericTail(base);
                base = collapseDuplicatePrefix(base);

                // Final truncation to keep labels manageable
                return base.slice(0, 200);
              };

              const isContainerishTag = (tag) => {
                return ['main','section','article','nav','header','footer','aside','ul','ol','li','div','form'].includes(tag);
              };

              let actionCount = 0;

              const walk = (el) => {
                if (!isVisibleNow(el)) return null;

                const children = [];
                for (const child of el.children) {
                  const c = walk(child);
                  if (c) children.push(c);
                }

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const href = el.getAttribute('href') || '';
                const label = bestLabel(el);
                const actionable = isActionable(el);

                // Reuse ids from snapshot_fast via data-agent-id
                const idAttr = el.getAttribute('data-agent-id');
                const id = idAttr ? parseInt(idAttr, 10) : NaN;

                if (actionable && id && !Number.isNaN(id)) {
                  if (limit > 0 && actionCount >= limit) {
                    return null;
                  }
                  actionCount += 1;
                  return {
                    kind: 'control',
                    id,
                    tag,
                    role,
                    href,
                    label,
                    actions: actionsFor(tag, role, href),
                    children: [],
                  };
                }

                // Container node: keep only if it has actionable descendants.
                if (!children.length) {
                  return null;
                }

                // Collapse boring single-child wrappers like <div> / <span> with no special role.
                if ((tag === 'div' || tag === 'span') && !role && children.length === 1) {
                  return children[0];
                }

                // Only keep some container tags to avoid wrapping everything in generic divs.
                if (!isContainerishTag(tag)) {
                  // If this non-containerish node has exactly one child, just return the child.
                  if (children.length === 1) {
                    return children[0];
                  }
                  return {
                    kind: 'container',
                    tag,
                    role,
                    label,
                    children,
                  };
                }

                return {
                  kind: 'container',
                  tag,
                  role,
                  label,
                  children,
                };
              };

              const root = walk(document.body) || {
                kind: 'container',
                tag: 'body',
                role: '',
                label: '',
                children: [],
              };

              return { tree: root, action_count: actionCount };
            }""",
            {"limit": limit},
        )

        result["frame_index"] = frame_index
        return result



    async def debug_dump_page(self, output_path: str = "/tmp/browser_debug_dump.txt") -> str:
        """
        Dump all current page content to a debug file for debugging purposes.
        This includes:
        - Page URL
        - Page title
        - Full page HTML (truncated if too large)
        - Visible text blocks
        - Control snapshot
        - Frame tree (if available)
        
        Returns a summary message with the path to the dump file.
        """
        from datetime import datetime
        
        dump_lines = []
        timestamp = datetime.now().isoformat()
        
        dump_lines.append("=" * 80)
        dump_lines.append(f"Browser Debug Dump - {timestamp}")
        dump_lines.append("=" * 80)
        dump_lines.append("")
        
        # 1. Basic page info
        try:
            url = self.page.url
            title = await self.page.title()
            dump_lines.append("PAGE INFO:")
            dump_lines.append(f"  URL: {url}")
            dump_lines.append(f"  Title: {title}")
            dump_lines.append("")
        except Exception as e:
            dump_lines.append(f"PAGE INFO: Error getting page info: {e}")
            dump_lines.append("")
        
        # 2. Page HTML (truncated to avoid huge files)
        try:
            html = await self.page.content()
            html_preview = html[:50000]  # First 50KB
            dump_lines.append("PAGE HTML (first 50KB):")
            dump_lines.append("-" * 80)
            dump_lines.append(html_preview)
            if len(html) > 50000:
                dump_lines.append("")
                dump_lines.append(f"... (truncated, total size: {len(html)} bytes)")
            dump_lines.append("")
            dump_lines.append("")
        except Exception as e:
            dump_lines.append(f"PAGE HTML: Error getting HTML: {e}")
            dump_lines.append("")
        
        # 3. Visible text blocks
        try:
            text_blocks = await self.snapshot_text(limit=500, min_len=1, max_len=500)
            dump_lines.append("VISIBLE TEXT BLOCKS:")
            dump_lines.append("-" * 80)
            for i, block in enumerate(text_blocks, 1):
                source = block.get("source", "unknown")
                text = block.get("text", "")
                dump_lines.append(f"[{i}] ({source}): {text[:200]}")
                if len(text) > 200:
                    dump_lines.append(f"    ... (truncated, total {len(text)} chars)")
            dump_lines.append("")
        except Exception as e:
            dump_lines.append(f"VISIBLE TEXT: Error getting text: {e}")
            dump_lines.append("")
        
        # 4. Control snapshot
        try:
            controls = await self.snapshot_fast(limit=500, filter=None, delta=False)
            dump_lines.append("CONTROL SNAPSHOT:")
            dump_lines.append("-" * 80)
            dump_lines.append(f"Total controls: {len(controls)}")
            for i, control in enumerate(controls[:100], 1):  # First 100 controls
                control_id = control.get("id")
                label = control.get("label", "")[:100]
                control_type = control.get("type", "")
                actions = control.get("actions", [])
                dump_lines.append(f"[{i}] id={control_id} type={control_type} label={label}")
                if actions:
                    dump_lines.append(f"    actions: {', '.join(actions)}")
            if len(controls) > 100:
                dump_lines.append(f"... (showing first 100 of {len(controls)} controls)")
            dump_lines.append("")
        except Exception as e:
            dump_lines.append(f"CONTROL SNAPSHOT: Error getting controls: {e}")
            dump_lines.append("")
        
        # 5. Frame tree
        try:
            tree = await self.debug_frame_tree_v2(frame_index=0, limit=500)
            dump_lines.append("FRAME TREE (main frame):")
            dump_lines.append("-" * 80)
            import json
            dump_lines.append(json.dumps(tree, indent=2, default=str)[:30000])
            dump_lines.append("")
        except Exception as e:
            dump_lines.append(f"FRAME TREE: Error getting tree: {e}")
            dump_lines.append("")
        
        # Write to file
        dump_content = "\n".join(dump_lines)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(dump_content)
            return f"Debug dump written to {output_path} ({len(dump_content)} bytes)"
        except Exception as e:
            # Fallback to /tmp if write fails
            fallback_path = "/tmp/browser_debug_dump.txt"
            try:
                with open(fallback_path, "w", encoding="utf-8") as f:
                    f.write(dump_content)
                return f"Debug dump written to {fallback_path} (original path failed: {e})"
            except Exception as e2:
                return f"Failed to write debug dump: {e2}. Dump size: {len(dump_content)} bytes"


    # -------------------------
    # Extract quiz questions
    # -------------------------
    async def extract_quiz_questions(self) -> dict:
        """
        Extract all questions from a Canvas Classic Quiz edit page.
        
        Returns a JSON-serializable dict with:
        - quiz: metadata (title, id, question_count, etc.)
        - questions: list of question objects with position, text, type, points, etc.
        
        Raises RuntimeError if not on a Canvas quiz edit page.
        """
        if not self.page:
            raise RuntimeError("No active browser page")
        
        # First try: extract from JavaScript ENV variable
        try:
            env_data = await self.page.evaluate("""
                (() => {
                    // ENV is a global variable in Canvas
                    if (typeof ENV !== 'undefined') {
                        const quiz = ENV.QUIZ || {};
                        const questions = ENV.QUIZ_QUESTIONS || [];
                        
                        // Try to extract questions from DOM as fallback if needed
                        let dom_questions = [];
                        if (!questions || questions.length === 0) {
                            // Find question containers
                            const questionDivs = document.querySelectorAll(
                                'div[role="region"][aria-label*="Question"], ' +
                                'div.question_holder, ' +
                                'div.display_question'
                            );
                            
                            dom_questions = Array.from(questionDivs).map((div, idx) => {
                                const questionText = div.querySelector('.question_text')?.innerText ||
                                                    div.querySelector('.question')?.innerText ||
                                                    div.innerText.substring(0, 200);
                                const points = div.querySelector('.points_possible')?.innerText ||
                                              div.querySelector('[data-points]')?.getAttribute('data-points') ||
                                              '1';
                                
                                return {
                                    position: idx + 1,
                                    text: questionText.trim(),
                                    points: parseInt(points) || 1,
                                    type: 'unknown',
                                    id: div.getAttribute('data-question-id') || ''
                                };
                            });
                        }
                        
                        return {
                            quiz: {
                                id: quiz.id,
                                title: quiz.title,
                                question_count: quiz.question_count,
                                points_possible: quiz.points_possible,
                                url: window.location.href
                            },
                            questions: questions.length > 0 ? questions : dom_questions,
                            source: 'canvas_env'
                        };
                    }
                    return { error: "Not a Canvas quiz page (ENV not found)" };
                })()
            """)
            
            if 'error' in env_data:
                raise RuntimeError(env_data['error'])
                
            return env_data
            
        except Exception as e:
            # Fallback: DOM scraping approach
            try:
                questions_data = await self.page.evaluate("""
                    (() => {
                        const questions = [];
                        
                        // Method 1: Classic quiz question containers
                        const questionContainers = document.querySelectorAll(
                            'div[role="region"][aria-label*="Question"], ' +
                            'div.question, ' +
                            'div.display_question, ' +
                            'div.question_holder'
                        );
                        
                        for (let i = 0; i < questionContainers.length; i++) {
                            const container = questionContainers[i];
                            
                            // Extract question text
                            let questionText = '';
                            const textEl = container.querySelector('.question_text') ||
                                         container.querySelector('.question_content') ||
                                         container.querySelector('.question_body');
                            
                            if (textEl) {
                                questionText = textEl.innerText.trim();
                            } else {
                                // Fallback: get all text nodes, filter out UI elements
                                const allText = container.innerText;
                                const lines = allText.split('\\n').filter(line => 
                                    line.length > 10 && 
                                    !line.includes('Edit') && 
                                    !line.includes('Delete') &&
                                    !line.includes('Move To') &&
                                    !line.includes('pts')
                                );
                                questionText = lines.slice(0, 3).join(' ').trim();
                            }
                            
                            // Extract points
                            let points = 1;
                            const pointsEl = container.querySelector('.points_possible') ||
                                           container.querySelector('.points') ||
                                           container.querySelector('[data-points]');
                            if (pointsEl) {
                                const pointsText = pointsEl.innerText || pointsEl.getAttribute('data-points') || '';
                                const match = pointsText.match(/(\\d+)/);
                                points = match ? parseInt(match[1]) : 1;
                            }
                            
                            // Extract question type from CSS classes or data attributes
                            let questionType = 'unknown';
                            if (container.classList.contains('multiple_choice_question')) {
                                questionType = 'multiple_choice';
                            } else if (container.classList.contains('true_false_question')) {
                                questionType = 'true_false';
                            } else if (container.classList.contains('short_answer_question')) {
                                questionType = 'short_answer';
                            } else if (container.classList.contains('essay_question')) {
                                questionType = 'essay';
                            } else if (container.classList.contains('fill_in_multiple_blanks_question')) {
                                questionType = 'fill_in_multiple_blanks';
                            } else if (container.classList.contains('multiple_answers_question')) {
                                questionType = 'multiple_answers';
                            }
                            
                            questions.push({
                                position: i + 1,
                                text: questionText,
                                points: points,
                                type: questionType,
                                id: container.getAttribute('data-question-id') || container.id || `q${i+1}`,
                                container_html: container.outerHTML.substring(0, 500)  // truncated for debugging
                            });
                        }
                        
                        // Get quiz metadata from page
                        const quizTitle = document.querySelector('input[name="quiz[title]"]')?.value ||
                                         document.querySelector('h1, h2, h3')?.innerText ||
                                         document.title.replace(' - Canvas', '');
                        
                        return {
                            quiz: {
                                title: quizTitle,
                                question_count: questions.length,
                                url: window.location.href
                            },
                            questions: questions,
                            source: 'dom_scraping',
                            warning: questions.length === 0 ? 'No questions found in DOM' : null
                        };
                    })()
                """)
                
                if questions_data['questions']:
                    return questions_data
                else:
                    raise RuntimeError(f"No questions found: {questions_data.get('warning', 'Unknown error')}")
                    
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to extract quiz questions: {fallback_error}")


    # -------------------------
    # Hardened action methods (+ jitter)
    # -------------------------
    async def click(self, element_id: int) -> None:
        loc = await self._loc_for_id(element_id)
        await loc.scroll_into_view_if_needed()

        # Move mouse into the element area to look more human-like
        await self._move_mouse_to_element(loc)

        # Try clicking directly first
        try:
            await loc.click(timeout=1_500)
            await self._post_action()
            return
        except PlaywrightTimeoutError as e:
            error_msg = str(e)
            # Check if this is the "subtree intercepts pointer events" or viewport error
            if "intercepts pointer events" not in error_msg and "outside of the viewport" not in error_msg:
                # Not a facade or viewport issue, do the normal retry
                await self._post_action()
                await loc.click(timeout=1_500)
                await self._post_action()
                return

        # If we get here, we have a facade/overlay blocking the click.
        # Try to find a clickable parent (e.g., <label> containing the input).
        try:
            # Get the tag of the element we're trying to click
            tag = await loc.evaluate("el => el.tagName.toLowerCase()")
            
            if tag in ("input", "button"):
                # Check if there's a parent label
                parent_label = self.page.locator(f'[data-agent-id="{element_id}"]').locator("xpath=..")
                parent_label_count = await parent_label.count()
                if parent_label_count > 0:
                    parent_tag = await parent_label.evaluate("el => el.tagName.toLowerCase()")
                    if parent_tag == "label":
                        # Click the parent label instead
                        await parent_label.click(timeout=1_500)
                        await self._post_action()
                        return
        except Exception as e:
            # If anything fails in the label-based workaround, continue to JS fallback
            pass

        # As a last resort, click via JavaScript (bypasses pointer events)
        try:
            await loc.evaluate("el => el.click()")
            await self._post_action()
            return
        except Exception:
            # JS click failed too - raise the original error
            pass

        # All attempts failed - raise the original error
        await self._post_action()
        await loc.click(timeout=1_500)

    async def fill(self, element_id: int, text: str) -> None:
        loc = await self._loc_for_id(element_id)
        await loc.scroll_into_view_if_needed()

        await self._move_mouse_to_element(loc)

        await loc.focus()
        await self._jitter(40, 120)
        await loc.fill(text)
        await self._post_action()

    async def type(self, element_id: int, text: str, delay_ms: Optional[int] = None) -> None:
        loc = await self._loc_for_id(element_id)
        await loc.scroll_into_view_if_needed()

        # Move mouse into the element area to look more human-like
        await self._move_mouse_to_element(loc)

        await loc.focus()
        await self._jitter(40, 120)

        # If caller explicitly requested a fixed delay, honor it directly.
        if isinstance(delay_ms, int) and delay_ms > 0:
            await loc.type(text, delay=delay_ms)
            await self._post_action()
            return

        # Otherwise, use a slightly more human-ish pattern with
        # per-character jitter and occasional small pauses at spaces.
        for ch in text:
            if ch == " " and random.random() < 0.25:
                # brief "thinking" pause before some spaces
                await self.page.wait_for_timeout(random.randint(120, 400))

            per_char_delay = random.randint(40, 160)  # ~40–160ms between keystrokes
            await loc.type(ch, delay=per_char_delay)

        await self._post_action()

    async def press(self, element_id: int, key: str) -> None:
        loc = await self._loc_for_id(element_id)
        await loc.focus()
        await self._jitter(40, 120)
        await loc.press(key)

        # Enter is often “nav-like”
        if key.lower() in ("enter", "numpadenter"):
            await self._post_nav()
        else:
            await self._post_action()

    async def select(self, element_id: int, *, value: Optional[str] = None, label: Optional[str] = None) -> None:
        if (value is None) == (label is None):
            raise ValueError("select(): pass exactly one of value= or label=")

        loc = await self._loc_for_id(element_id)
        await loc.scroll_into_view_if_needed()

        await self._move_mouse_to_element(loc)

        if value is not None:
            await loc.select_option(value=value)
        else:
            await loc.select_option(label=label)

        await self._post_action()

    async def scroll_by(self, delta_y: float) -> None:
        await self.page.mouse.wheel(0, delta_y)
        await self._post_action()

    async def scroll_into_view(self, element_id: int) -> None:
        loc = await self._loc_for_id(element_id)
        await loc.scroll_into_view_if_needed()
        await self._post_action()

    # -------------------------
    # Read tools
    # -------------------------
    async def get_text(self, element_id: int) -> str:
        loc = await self._loc_for_id(element_id)
        return await loc.inner_text()

    async def get_value(self, element_id: int) -> str:
        loc = await self._loc_for_id(element_id)
        return await loc.input_value()

    async def is_visible(self, element_id: int) -> bool:
        loc = await self._loc_for_id(element_id)
        return await loc.is_visible()

    async def is_enabled(self, element_id: int) -> bool:
        loc = await self._loc_for_id(element_id)
        return await loc.is_enabled()

    # -------------------------
    # Tool dispatcher (now no jitter here; it’s in methods)
    # -------------------------
    async def dispatch_tool_call(self, tool_call: Any) -> Tuple[Dict[str, str], bool]:
        def tool_msg(call_id: str, content: Any, is_error: bool = False) -> Tuple[Dict[str, str], bool]:
            s = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return ({"role": "tool", "tool_call_id": call_id or "", "content": s}, is_error)

        def coerce_int(v: Any, field: str) -> int:
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
            raise TypeError(f"Parameter '{field}' must be int")

        def coerce_float(v: Any, field: str) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                return float(v.strip())
            raise TypeError(f"Parameter '{field}' must be a number")

        async def run_action_with_delta(action_coro, *, limit: int = 200):
            """
            Run an action (click/fill/select/scroll/etc.), then compute a controls
            delta via snapshot_fast(delta=True).

            Returns a tool_msg payload of the form:
              { "ok": true/false, "delta": {...} } or { "ok": false, "error": "..." }.
            """
            # 1) Execute the action itself
            try:
                await action_coro
            except Exception as e:
                # Do NOT advance delta baseline on failed actions.
                return tool_msg(
                    call_id,
                    {"ok": False, "error": f"{name} failed: {e}"},
                    True,
                )

            # 2) Compute delta relative to previous snapshot_fast baseline
            try:
                delta = await self.snapshot_fast(limit=limit, filter=None, delta=True)
            except Exception as e:
                # Action succeeded but delta computation failed; surface both states.
                return tool_msg(
                    call_id,
                    {
                        "ok": True,
                        "delta_error": f"snapshot_fast(delta=True) failed: {e}",
                    },
                    False,
                )

            return tool_msg(call_id, {"ok": True, "delta": delta}, False)

        # normalize tool_call shape
        call_id = ""
        name = None
        raw_args = None
        try:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            call_id = getattr(tool_call, "id", "") or ""
        except Exception:
            pass

        if not name:
            try:
                fn = tool_call["function"]
                name = fn["name"]
                raw_args = fn.get("arguments", "")
                call_id = tool_call.get("id", "") or ""
            except Exception:
                return tool_msg("", "Malformed tool call: missing fields.", True)

        if not isinstance(name, str) or not name.strip():
            return tool_msg(call_id, "Malformed tool call: missing function name.", True)
        name = name.strip()

        # parse args
        if raw_args is None or raw_args == "":
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    return tool_msg(call_id, "Tool arguments must be a JSON object/dict.", True)
            except Exception as e:
                return tool_msg(call_id, f"Invalid JSON in tool arguments: {e}. args: {raw_args}", True)
        else:
            return tool_msg(call_id, f"Unsupported tool arguments type: {type(raw_args)}", True)

        try:
            if name == "browser_snapshot_controls":
                limit = coerce_int(args.get("limit", 200), "limit")
                filter = args.get("filter")
                return tool_msg(call_id, await self.snapshot_fast(limit=limit, filter=filter, delta=False), False)
            if name == "browser_snapshot_controls_delta":
                limit = coerce_int(args.get("limit", 200), "limit")
                filter = args.get("filter")
                return tool_msg(call_id, await self.snapshot_fast(limit=limit, filter=filter, delta=True), False)

            if name == "browser_snapshot_controls_tree":
                # Hierarchical snapshot of a single frame, using canonical v2 schema
                # with ids + actions on control nodes. Intended for structural
                # reasoning and as the canonical control tree.
                frame_index = coerce_int(args.get("frame_index", 0), "frame_index")
                limit = coerce_int(args.get("limit", 500), "limit")
                filter = args.get("filter")
                tree = await self.debug_frame_tree_v2(frame_index, limit=limit)
                # Optional filter: prune tree to nodes whose labels contain substring.
                if isinstance(filter, str) and filter.strip():
                    f = filter.strip().lower()

                    def prune(node):
                        label = (node.get('label') or '').lower()
                        kids = [prune(c) for c in node.get('children', [])]
                        kids = [c for c in kids if c is not None]
                        node_out = dict(node)
                        node_out['children'] = kids
                        if f in label:
                            return node_out
                        if kids:
                            return node_out
                        return None

                    pruned = prune(tree.get('tree') or {})
                    tree = dict(tree)
                    tree['tree'] = pruned or {"kind": "container", "tag": "body", "role": "", "label": "", "children": []}

                return tool_msg(call_id, tree, False)

            if name == "browser_read_text":
                limit = coerce_int(args.get("limit", 200), "limit")
                min_len = coerce_int(args.get("min_len", 5), "min_len")
                max_len = coerce_int(args.get("max_len", 50), "max_len")
                filter = args.get("filter")
                if filter is not None and not isinstance(filter, str):
                    raise TypeError("Parameter 'filter' must be str")
                return tool_msg(
                    call_id,
                    await self.snapshot_text(limit=limit, min_len=min_len, max_len=max_len, filter=filter),
                    False,
                )

            if name == "browser_list_frames":
                return tool_msg(call_id, await self.list_frames(), False)

            if name == "browser_debug_frame_controls":
                frame_index = coerce_int(args.get("frame_index"), "frame_index")
                limit = coerce_int(args.get("limit", 500), "limit")
                return tool_msg(call_id, await self.debug_frame_controls(frame_index, limit=limit), False)

            if name == "browser_debug_dump":
                output_path = args.get("output_path", "/tmp/browser_debug_dump.txt")
                if not isinstance(output_path, str):
                    raise TypeError("Parameter 'output_path' must be str")
                return tool_msg(call_id, await self.debug_dump_page(output_path), False)

            if name == "browser_extract_quiz_questions":
                try:
                    result = await self.extract_quiz_questions()
                    return tool_msg(call_id, result, False)
                except Exception as e:
                    return tool_msg(call_id, {
                        "error": str(e),
                        "message": f"Failed to extract quiz questions: {e}"
                    }, True)

            if name == "browser_click":
                element_id = coerce_int(args.get("id"), "id")
                return await run_action_with_delta(
                    self.click(element_id)
                )

            if name == "browser_fill":
                text = args.get("text")
                if not isinstance(text, str):
                    raise TypeError("Parameter 'text' must be str")
                element_id = coerce_int(args.get("id"), "id")
                return await run_action_with_delta(
                    self.fill(element_id, text)
                )

            if name == "browser_type":
                text = args.get("text")
                if not isinstance(text, str):
                    raise TypeError("Parameter 'text' must be str")

                delay_arg = args.get("delay_ms", None)
                if delay_arg is None:
                    # Let BrowserClient.type choose a human-ish pattern by default.
                    delay_ms = None
                else:
                    delay_ms = coerce_int(delay_arg, "delay_ms")

                element_id = coerce_int(args.get("id"), "id")
                return await run_action_with_delta(
                    self.type(element_id, text, delay_ms=delay_ms)
                )

            if name == "browser_press":
                key = args.get("key")
                if not isinstance(key, str):
                    raise TypeError("Parameter 'key' must be str")
                element_id = coerce_int(args.get("id"), "id")
                return await run_action_with_delta(
                    self.press(element_id, key)
                )

            if name == "browser_select":
                element_id = coerce_int(args.get("id"), "id")
                raw_value = args.get("value", None)
                raw_label = args.get("label", None)

                # Treat empty strings as not provided, so the tool can send
                # exactly one of value/label in practice.
                value = raw_value if (isinstance(raw_value, str) and raw_value != "") else None
                label = raw_label if (isinstance(raw_label, str) and raw_label != "") else None

                return await run_action_with_delta(
                    self.select(element_id, value=value, label=label)
                )

            if name == "browser_scroll_by":
                delta_y = coerce_float(args.get("delta_y"), "delta_y")
                return await run_action_with_delta(
                    self.scroll_by(delta_y)
                )

            if name == "browser_scroll_into_view":
                element_id = coerce_int(args.get("id"), "id")
                return await run_action_with_delta(
                    self.scroll_into_view(element_id)
                )

            if name == "browser_wait":
                await self.wait(coerce_int(args.get("ms"), "ms"))
                return tool_msg(call_id, {"ok": True}, False)

            if name == "browser_session_status":
                is_open = await self.is_open()
                url = None
                if is_open:
                    try:
                        url = self.page.url
                    except Exception:
                        pass
                return tool_msg(call_id, {"open": is_open, "url": url}, False)

            if name == "browser_session_open":
                user_data_dir = args.get("user_data_dir")
                if not isinstance(user_data_dir, str) or not user_data_dir.strip():
                    raise TypeError("Parameter 'user_data_dir' must be non-empty str")
                profile_directory = args.get("profile_directory", "Default")
                if not isinstance(profile_directory, str):
                    raise TypeError("Parameter 'profile_directory' must be str")
                # headless: false=headed (default), true=headless mode
                headless_arg = args.get("headless", False)
                headless = headless_arg in (True, "true", "True", "new", 1)
                await self.open(
                    user_data_dir=user_data_dir,
                    profile_directory=profile_directory,
                    headless=headless,
                )
                return tool_msg(call_id, {"ok": True, "headless": headless}, False)

            if name == "browser_session_close":
                await self.close()
                return tool_msg(call_id, {"ok": True}, False)

            if name == "browser_get_url":
                return tool_msg(call_id, {"url": await self.get_url()}, False)

            if name == "browser_back":
                await self.back()
                return tool_msg(call_id, {"ok": True}, False)

            if name == "browser_goto":
                url = args.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise TypeError("Parameter 'url' must be non-empty str")
                await self.goto(url)
                return tool_msg(call_id, {"ok": True}, False)

            if name == "browser_get_text":
                return tool_msg(call_id, {"text": await self.get_text(coerce_int(args.get("id"), "id"))}, False)

            if name == "browser_get_value":
                return tool_msg(call_id, {"value": await self.get_value(coerce_int(args.get("id"), "id"))}, False)

            if name == "browser_is_visible":
                return tool_msg(call_id, {"visible": await self.is_visible(coerce_int(args.get("id"), "id"))}, False)

            if name == "browser_is_enabled":
                return tool_msg(call_id, {"enabled": await self.is_enabled(coerce_int(args.get("id"), "id"))}, False)

            return tool_msg(call_id, f"Unknown tool: {name}", True)

        except Exception as e:
            return tool_msg(call_id, f"{name} failed: {e}", True)