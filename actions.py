"""
Allowed action vocabulary for browser agent; one line per action.

Parsed by parse_action(), intended to be executed later by BrowserEnv.execute_action()
Indices refer to the same ordering as get_observation() / BrowserEnv._last_interactive_locators for now

overall goal: make observation in browser env, llm reasons over observation+goal+history
to choose next discrete action to take, llm chooses an allowed action string, parser 
parses and validates it, browser executes it, make new observation, rinse and repeat

the vocab serves to keep the llm's output constrained and executable
"""

from __future__ import annotations

import re
import shlex # for v0 to tokenize action lines. will be replaced with json later
from dataclasses import dataclass
from typing import Literal, Optional

Action = Literal[
    "stop",
    "back",
    "scroll_up",
    "scroll_down",
    "goto",
    "click",
    "type",
]


# for LLM system prompt, keep in sync with parse_action()
ACTION_VOCABULARY = """
Allowed actions (one line each, lowercase):

  stop
      End the episode; no browser change.

  back
      Browser back navigation.

  scroll up | scroll down
      Page scroll (also accepted: scroll_up, scroll_down as a single token).

  goto <url>
      Navigate to an absolute http(s) URL (everything after the first space).

  click <index>
      Click interactive element [index] from the last observation list (0-based).

  type <index> <text> [submit]
      Fill element [index]. Use double quotes for multi-word text, e.g. type 2 "hello world".
      Optional trailing keyword submit (or true or enter) presses Enter after typing.
""".strip()


class ActionParseError(ValueError):
    """Raised when a line does not match the vocabulary."""


@dataclass(frozen=True)
class ParsedAction:
    action_type: Action
    index: Optional[int] = None
    url: Optional[str] = None
    text: Optional[str] = None
    submit: bool = False


def _is_submit_token(t: str) -> bool:
    return t.casefold() in ("submit", "true", "enter", "1")


def parse_action(line: str) -> ParsedAction:
    """
    Parse a single action line into a ParsedAction object.
    Raises ActionParseError on empty or invalid input line.
    """
    s = line.strip()
    if not s:
        raise ActionParseError("empty action line")

    lower = s.lower()
    if lower == "stop":
        return ParsedAction(action_type="stop")
    if lower == "back":
        return ParsedAction(action_type="back")
    if lower in ("scroll_up", "scroll-up"):
        return ParsedAction(action_type="scroll_up")
    if lower in ("scroll_down", "scroll-down"):
        return ParsedAction(action_type="scroll_down")

    parts = shlex.split(s, posix=True)
    if not parts:
        raise ActionParseError("empty action line")
    verb = parts[0].lower()

    if verb == "scroll":
        if len(parts) != 2:
            raise ActionParseError('scroll: use "scroll up" or "scroll down"')
        d = parts[1].lower()
        if d == "up":
            return ParsedAction(action_type="scroll_up")
        if d == "down":
            return ParsedAction(action_type="scroll_down")
        raise ActionParseError('scroll: second word must be "up" or "down"')

    if verb == "goto": # gets url as remainder of original line after "goto "
        m = re.match(r"^\s*goto\s+", s, re.IGNORECASE)
        if not m:
            raise ActionParseError("goto: missing URL")
        url = s[m.end() :].strip()
        if not url:
            raise ActionParseError("goto: missing URL")
        if not url.startswith(("http://", "https://")):
            raise ActionParseError("goto: URL must start with http:// or https://")
        return ParsedAction(action_type="goto", url=url)

    if verb == "click":
        if len(parts) != 2:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            raise ActionParseError('click: use click <index>, e.g. click 3')
        try:
            idx = int(parts[1])
        except ValueError as e:
            raise ActionParseError("click: index must be an integer") from e
        if idx < 0:
            raise ActionParseError("click: index must be >= 0")
        return ParsedAction(action_type="click", index=idx)

    if verb == "type":
        if len(parts) < 3:
            raise ActionParseError(
                'type: use type <index> <text> [submit], e.g. type 1 "hello" submit'
            )
        try:
            idx = int(parts[1])
        except ValueError as e:
            raise ActionParseError("type: index must be an integer") from e
        if idx < 0:
            raise ActionParseError("type: index must be >= 0")
        tail = parts[2:]
        submit = False
        if len(tail) >= 2 and _is_submit_token(tail[-1]):
            submit = True
            tail = tail[:-1]
        if not tail:
            raise ActionParseError("type: missing text")
        text = " ".join(tail)
        return ParsedAction(action_type="type", index=idx, text=text, submit=submit)

    raise ActionParseError(f"unknown action: {verb!r}")


if __name__ == "__main__":
    samples = [
        "stop",
        "back",
        "scroll up",
        "scroll_down",
        "goto https://example.com/path?q=1",
        "click 0",
        'type 2 hello',
        'type 2 "hello world" submit',
    ]
    for sample in samples:
        print(sample, "->", parse_action(sample))
