import re
import unicodedata


def slugify(value: str, convert_ascii: bool = True) -> str:
    """Make a string suitable for use in URLs and filenames.

    Convert to ASCII if convert_ascii True. Convert spaces or repeated dashes
    to single dashes. Remove characters that aren't alphanumerics, underscores,
    or hyphens. Convert to lowercase. Also strip leading and trailing
    whitespace, dashes, and underscores.

    Taken from:
    https://github.com/django/django/blob/master/django/utils/text.py
    """
    value = str(value)
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        if convert_ascii
        else unicodedata.normalize("NFKC", value)
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")
