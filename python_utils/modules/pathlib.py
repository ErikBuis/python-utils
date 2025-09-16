from __future__ import annotations

import re

from ..custom.print import to_ascii


def slugify(text: str) -> str:
    """Make a string suitable for use in URLs and filenames.

    Convert to ASCII. Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Strip leading and trailing whitespace, dashes, and
    underscores.

    Taken from:
    https://github.com/django/django/blob/master/django/utils/text.py

    Args:
        text: The input string.

    Returns:
        The slugified string.
    """
    text = to_ascii(text)
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-_")
