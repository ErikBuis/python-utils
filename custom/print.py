import shutil
from typing import Any


def stringify(
    obj: Any,
    indent_level: int = 0,
    max_line_len: int = 0,
    ignore: tuple[type, ...] | None = None,
) -> str:
    """Stringify an object with indentation and line breaks.

    Args:
        obj: The object to stringify.
        indent_level: The current indentation level. The indentation itself
            is displayed as 4 spaces per level. In most cases, you can leave
            this at 0. It is used internally for recursive calls.
        max_line_len: The maximum line length before breaking. If set to 0,
            the terminal width will be used.
        ignore: A list of types to ignore. If the object is of one of these
            types, it will be stringified as "type(obj)".

    Returns:
        The stringified object.
    """
    if max_line_len == 0:
        max_line_len = shutil.get_terminal_size().columns

    indent = " " * 4 * indent_level
    indent_next = " " * 4 * (indent_level + 1)

    if ignore is not None and isinstance(obj, ignore):
        return type(obj).__name__

    repr_obj = repr(obj)

    if len(repr_obj) <= max_line_len:
        return repr_obj

    if isinstance(obj, list):
        return (
            "[\n"
            + ",\n".join(
                indent_next
                + stringify(
                    item,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                )
                for item in obj
            )
            + "\n"
            + indent
            + "]"
        )

    if isinstance(obj, tuple):
        return (
            "(\n"
            + ",\n".join(
                indent_next
                + stringify(
                    item,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                )
                for item in obj
            )
            + "\n"
            + indent
            + ")"
        )

    if isinstance(obj, dict):
        return (
            "{\n"
            + ",\n".join(
                indent_next
                + repr(key)
                + ": "
                + stringify(
                    value,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                )
                for key, value in obj.items()
            )
            + "\n"
            + indent
            + "}"
        )

    return repr_obj
