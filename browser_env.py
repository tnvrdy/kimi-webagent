"""
TODO: create a browser environment in which a webagent can operate and train :3

overall approach:
1. start playwright (synchronous api for now)
2. launch chromium browser (isolate context per run)
3. open a page
4. navigate to a url
5. observe and perform actions on the page (after ensuring dom loaded)
6. shut entire environment down

basic lifecycle: start chromium, goto a page, navigate, shut down env
i want browser side working before i impl text observations, action parsing from a tiny action vocabulary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


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
        print("title:", env.page.title())
        print("url:", env.page.url)
