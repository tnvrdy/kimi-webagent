"""
TODO: create a browser environment in which a webagent can operate and train :3

overall approach:
1. start playwright (synchronous api for now)
2. launch chromium browser (isolate context per run)
3. open a page
4. navigate to a url
5. observe and perform actions on the page (after ensuring dom loaded)
6. shut entire environment down

basic lifecycle done: start chromium, goto a page, navigate, shut down env
now implementing text observations, then action parsing from a tiny action vocabulary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright, Locator

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

    @staticmethod
    def _describe_locator(index: int, loc: Locator) -> str:
        """Returns a short line for the model to parse: tag, type, role, best-effort label."""
        try:
            desc = loc.evaluate( # js to run inside playwright context to get description of each element
                """el => {
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
                }"""
            )
        except Exception as exc: # detached / cross-origin / timeout errors possible
            desc = f"<unavailable {type(exc).__name__}>"
        return f"[{index}] {desc}"

    def get_observation(self, max_chars: int = 16000) -> Observation:
        """
        Build a text snapshot: url, title, then numbered interactive elements.
        (**in v0**, indices i match _last_interactive_locators[i] for a future action)
        """
        self._last_interactive_locators = self._list_interactive_locators()
        lines: List[str] = [
            f"URL: {self.page.url}",
            f"Title: {self.page.title()}",
            f"Interactive elements: {len(self._last_interactive_locators)}",
            "",
        ]
        for i, loc in enumerate(self._last_interactive_locators):
            lines.append(self._describe_locator(i, loc))

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

    def __enter__(self) -> "BrowserEnv":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# local test (needs playwright install chromium)
if __name__ == "__main__":
    with BrowserEnv(headless=True) as env:
        env.goto("https://google.com")
        obs = env.get_observation(max_chars=8000)
        print("n_elements:", obs.n_elements, "truncated:", obs.truncated)
        print(obs.text[:4000])
