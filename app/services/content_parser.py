"""
content_parser.py
~~~~~~~~~~~~~~~~~
Fetches a URL (or accepts raw content) and returns a 4-tuple:
  - raw_html        : original HTML/text string
  - soup            : BeautifulSoup over the full HTML (for h-tag analysis)
  - clean_text      : body text with boilerplate (nav/header/footer/aside) stripped
  - first_paragraph : first meaningful paragraph text
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import httpx
from bs4 import BeautifulSoup, Tag

# Tags whose content we treat as boilerplate
_BOILERPLATE_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript"}

_FETCH_TIMEOUT_S = 10


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_and_parse(
    input_type: str, input_value: str
) -> Tuple[str, BeautifulSoup, str, str]:
    """Return (raw_html, soup, clean_text, first_paragraph).

    input_type : 'url' | 'text'
    input_value: a URL string OR raw HTML / plain-text content
    """
    if input_type == "url":
        raw_html = _fetch_url(input_value)
    else:
        raw_html = input_value

    is_html = _looks_like_html(raw_html)

    if is_html:
        soup = BeautifulSoup(raw_html, "html.parser")
        clean_text = _extract_clean_text(soup)
        first_paragraph = _extract_first_paragraph_html(soup)
    else:
        soup = BeautifulSoup("", "html.parser")  # empty — no h-tags to find
        clean_text = raw_html.strip()
        first_paragraph = _extract_first_paragraph_plain(raw_html)

    return raw_html, soup, clean_text, first_paragraph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_url(url: str) -> str:
    """Fetch a URL and return its text. Raises ValueError on any failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AEGIS-AEO-Bot/1.0; +https://aegis.ai)"
            )
        }
        response = httpx.get(
            url, follow_redirects=True, timeout=_FETCH_TIMEOUT_S, headers=headers
        )
        response.raise_for_status()
        return response.text
    except httpx.TimeoutException as exc:
        raise ValueError(f"Connection timeout after {_FETCH_TIMEOUT_S}s") from exc
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"HTTP {exc.response.status_code} from {url}") from exc
    except httpx.RequestError as exc:
        raise ValueError(str(exc)) from exc


def _looks_like_html(text: str) -> bool:
    """Heuristic: contains at least one HTML tag in the first 2 KB."""
    return bool(re.search(r"<[a-zA-Z][\s\S]*?>", text[:2000]))


def _extract_clean_text(soup: BeautifulSoup) -> str:
    """Strip boilerplate containers, then return joined text from body."""
    working = BeautifulSoup(str(soup), "html.parser")  # work on a copy
    for tag in working.find_all(_BOILERPLATE_TAGS):
        tag.decompose()

    # Prefer <main> or <article> if present; fall back to <body> or root
    container: Optional[Tag] = (
        working.find("main")
        or working.find("article")
        or working.find("body")
        or working
    )

    lines = []
    for element in container.descendants:  # type: ignore[union-attr]
        if isinstance(element, str):
            stripped = element.strip()
            if stripped:
                lines.append(stripped)

    return " ".join(lines)


def _extract_first_paragraph_html(soup: BeautifulSoup) -> str:
    """Return the text of the first <p> tag with non-trivial content."""
    for p_tag in soup.find_all("p"):
        # Skip paragraphs inside boilerplate containers
        if any(
            a.name in _BOILERPLATE_TAGS for a in p_tag.parents if hasattr(a, "name")
        ):
            continue
        text = p_tag.get_text(separator=" ", strip=True)
        # Must have at least 15 chars and 3 words to be considered real
        if len(text) >= 15 and len(text.split()) >= 3:
            return text
    return ""


def _extract_first_paragraph_plain(text: str) -> str:
    """For plain text: return the first double-newline-separated block."""
    blocks = re.split(r"\n{2,}", text.strip())
    for block in blocks:
        stripped = block.strip()
        if stripped and len(stripped.split()) >= 3:
            return stripped
    return text.strip()
