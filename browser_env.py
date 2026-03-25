"""
playwright browser environment in which an agent can operate and train

overall approach:
1. start playwright (synchronous api for now)
2. launch chromium browser (isolate context per run)
3. open a page
4. navigate to a url
5. observe and perform actions on the page (after ensuring dom loaded)
6. shut entire environment down
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright, Locator

from actions import ActionParseError, ParsedAction, parse_action

# locator query for interactive elements in DOM order; must stay in sync with click/type by index in v0
_INTERACTIVE_SELECTOR = (
    "a[href], button, input, textarea, select, "
    '[role="button"], [role="link"], [role="textbox"], [role="searchbox"], '
    '[role="checkbox"], [role="radio"], [role="tab"], [role="menuitem"]'
)
# keep in mind this won't get elements that are only interactive bc of a js event listener
# but don't have a role, e.g. button

_TRUNCATION_MARK = "\n...[truncated]"


@dataclass
class Observation:
    """Text observation for the LLM. Rn indices match _last_interactive_locators on BrowserEnv."""

    text: str
    n_elements: int
    truncated: bool


@dataclass
class BrowserEnv:
    """A single chromium session. Opens a browser, creates a context, and accesses a single page."""

    headless: bool = True # run browser in headless mode, i.e. no visible browser window/ui
    viewport_width: int = 1280
    viewport_height: int = 720

    _playwright: Optional[Playwright] = field(default=None, repr=False) # playwright object for starting/stopping playwright and interacting with browser
    _browser: Optional[Browser] = field(default=None, repr=False) # browser object for launching browser and creating contexts
    _context: Optional[BrowserContext] = field(default=None, repr=False) # context object for isolating browser state (cookies, etc.)
    _page: Optional[Page] = field(default=None, repr=False) # page object for interacting with a Page (single tab/window)
    
    # refreshes on each get_observation(); for v0, will matches with [i] lines in observation textfile (when i impl logs)
    _last_interactive_locators: List[Locator] = field(default_factory=list, repr=False)

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserEnv not started; call start() or use 'with BrowserEnv()' first")
        return self._page

    def start(self) -> None:
        """Launches chromium and opens a new page (not if already started)."""
        if self._playwright is not None:
            return
        self._playwright = sync_playwright().start() # starts playwright driver
        self._browser = self._playwright.chromium.launch(headless=self.headless) # opens up one chromium instance
        self._context = self._browser.new_context( # creates new isolated context
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        self._page = self._context.new_page() # single tab to enable goto and clicks later

    def stop(self) -> None:
        """Closes context, browser, and playwright driver in that order."""
        if self._context is not None:
            self._context.close()
        self._context = None
        self._page = None
        if self._browser is not None:
            self._browser.close()
        self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
        self._playwright = None
        self._last_interactive_locators = []

    def _list_interactive_locators(self) -> List[Locator]:
        """
        Returns interactive elements in dom order 
        (**in v0**, matches with observation indices. yes im aware this is hacky for now).
        """
        root = self.page.locator(_INTERACTIVE_SELECTOR)
        n = root.count()
        return [root.nth(i) for i in range(n)]

    def _describe_all_locators(self, selector: str) -> list:
        """
        Returns a list of description strings for all elements matching selector,
        in dom order. Uses a single page.evaluate() call instead of one per element,
        should be much faster and prevents stalling on pages with lots of interactive elements (e.g. wikipedia)
        """
        try:
            return self.page.evaluate(
                """(selector) => {
                    const els = Array.from(document.querySelectorAll(selector));
                    return els.map(el => {
                        const tag = el.tagName.toLowerCase();
                        const type = el.getAttribute('type') || '';
                        const role = el.getAttribute('role') || '';
                        const placeholder = el.getAttribute('placeholder') || '';
                        const al = el.getAttribute('aria-label') || '';
                        const name = el.getAttribute('name') || '';
                        let text = (el.innerText || '').trim().replace(/\\s+/g, ' ');
                        if (text.length > 120) text = text.slice(0, 117) + '...';
                        const bits = [tag + (type ? '[' + type + ']' : '')];
                        if (role) bits.push('role=' + role);
                        const label = [al, placeholder, name, text].find(s => s && s.length);
                        return bits.join(' ') + (label ? ' | ' + label : '');
                    });
                }""",
                selector,
            )
        except Exception:
            return []

    def get_observation(self, max_chars: int = 16000) -> Observation:
        """
        Build a text snapshot: url, title, then numbered interactive elements.
        (**in v0**, indices i match _last_interactive_locators[i] for a future action)
        
        Waits for any in-progress navigation to settle before reading the DOM,
        preventing stale locator queries after actions that trigger a new page load.
        title() is additionally wrapped defensively to handle the narrow race window where it could
        throw while mid-navigation. instead of crashing, the observation will say "loading..."
        and the loop will re-observe next step, which should be a fully loaded page.
        """
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass  # already settled or timed out, proceed anyway

        try:
            title = self.page.title()
        except Exception:
            title = "(loading...)"  # context destroyed mid-navigation, loop will re-observe next step

        self._last_interactive_locators = self._list_interactive_locators()
        lines: List[str] = [
            f"URL: {self.page.url}", # just title would be cleaner but know from testing on x.com
                                     # that title() is unreliable on spa's (where stuff like title is set by react)
            f"Title: {title}",
            f"Interactive elements: {len(self._last_interactive_locators)}",
            "",
        ]
        descs = self._describe_all_locators(_INTERACTIVE_SELECTOR)
        for i, desc in enumerate(descs):
            lines.append(f"[{i}] {desc}")

        body = "\n".join(lines)
        truncated = len(body) > max_chars
        if truncated:
            text = body[: max_chars - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK
        else:
            text = body
        return Observation(text=text, n_elements=len(self._last_interactive_locators), truncated=truncated)

    def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        Navigate the current page to specified url. 
        Considered successful if dom loads, otherwise raises if Playwright fails.
        """
        self.page.goto(url, wait_until=wait_until)

    def _resolve_locator(self, index: Optional[int], action_name: str):
        """
        Intended for use by click / type actions (or anything that needs index and locator)
        Validates index and returns (locator, None) on success, or (None, error_dict) on failure.
        Also scrolls the element into view so it's ready to interact with
        """
        if index is None:
            return None, {"ok": False, "error": f"{action_name}: missing index"}
        if not self._last_interactive_locators:
            return None, {"ok": False, "error": f"{action_name}: no observation available (call get_observation first)"}
        if index < 0 or index >= len(self._last_interactive_locators):
            return None, {"ok": False, "error": f"{action_name}: index {index} out of range (0–{len(self._last_interactive_locators) - 1})"}
        loc = self._last_interactive_locators[index]
        loc.scroll_into_view_if_needed(timeout=5000)
        return loc, None

    def execute_action(self, action: ParsedAction) -> dict:
        """
        Execute one ParsedAction against the current page. Returns dict with ok/error keys

        For click / type, action.index refers to the indices produced by the most
        recent get_observation() output
        """
        try:
            action_type = action.action_type

            if action_type == "stop":
                return {"ok": True, "done": True}

            if action_type == "back":
                self.page.go_back()
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
                return {"ok": True}

            if action_type == "scroll_up":
                self.page.mouse.wheel(0, -800)
                return {"ok": True}

            if action_type == "scroll_down":
                self.page.mouse.wheel(0, 800)
                return {"ok": True}

            if action_type == "goto":
                if not action.url:
                    return {"ok": False, "error": "goto: missing url"}
                self.goto(action.url)
                return {"ok": True}

            if action_type in ("click", "type"):
                loc, err = self._resolve_locator(action.index, action_type)
                if err:
                    return err
                if action_type == "click":
                    loc.click(timeout=5000)
                    try:
                        self.page.wait_for_load_state("domcontentloaded", timeout=8000)
                    except Exception:
                        pass
                else:
                    loc.click(timeout=5000)
                    loc.fill(action.text or "", timeout=5000)
                    if action.submit:
                        url_before = self.page.url
                        self.page.keyboard.press("Enter")
                        # wait for navigation to start (url change) before waiting for it to settle;
                        # without this, wait_for_load_state sees the departing page as already loaded
                        # and returns immediately, leaving the next observation on an empty mid-nav DOM
                        try:
                            self.page.wait_for_url(
                                lambda url: url != url_before, timeout=3000
                            )
                        except Exception:
                            pass  # url didn't change (e.g. in-page form): fall through
                        try:
                            self.page.wait_for_load_state("domcontentloaded", timeout=8000)
                        except Exception:
                            pass
                return {"ok": True}

            return {"ok": False, "error": f"unknown action type: {action_type}"}

        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def execute_action_from_line(self, line: str) -> dict:
        """Parse an action line and execute it. Returns dict with ok/error keys"""
        try:
            action = parse_action(line)
        except ActionParseError as e:
            return {"ok": False, "error": f"parse error: {e}"}
        return self.execute_action(action)

    def __enter__(self) -> "BrowserEnv":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# local test (needs playwright install chromium)
if __name__ == "__main__":
    with BrowserEnv(headless=True) as env:
        env.goto("https://google.com/search?q=playwright")
        obs = env.get_observation(max_chars=8000)
        print("n_elements:", obs.n_elements, "truncated:", obs.truncated)
        print(obs.text[:4000])

        print("\n--- executing: click 0 ---")
        res = env.execute_action_from_line("click 0")
        print("exec_res:", res)
        obs2 = env.get_observation(max_chars=2000)
        print("\nnew URL:", env.page.url)
        print(obs2.text[:1200])
