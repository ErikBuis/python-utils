from __future__ import annotations

import shutil
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, overload


def to_ascii(text: str) -> str:
    """Convert Unicode string to plain ASCII.

    - Removes accents (e.g. "é" → "e").
    - Simplifies compatibility chars (e.g. "ﬃ" → "ffi").
    - Drops any non-ASCII characters (e.g. "ß" → "").

    Args:
        text: The input Unicode string.

    Returns:
        The converted ASCII string.
    """
    # Normalize with compatibility decomposition.
    normalized = unicodedata.normalize("NFKD", text)

    # Encode to ASCII, ignoring non-ASCII leftovers.
    return normalized.encode("ascii", "ignore").decode("ascii")


def stringify(
    obj: Any,
    indent_level: int = 0,
    max_line_len: int = 0,
    ignore: tuple[type, ...] | type | None = None,
) -> str:
    """Stringify an object with indentation and line breaks.

    Args:
        obj: The object to stringify.
        indent_level: The current indentation level. The indentation itself
            is displayed as 4 spaces per level. In most cases, you can leave
            this at 0. It is used internally for recursive calls.
        max_line_len: The maximum line length before breaking. If set to 0,
            the terminal width will be used.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".

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


def __stringify_fast_sequence_consecutive(
    obj: Sequence[Any],
    indent_level: int,
    max_line_len: int,
    ignore: tuple[type, ...] | type | None,
    line_len_left: int,
    cached_strs: list[tuple[list[str], int]],
) -> tuple[bool, list[str | int], int]:
    """Stringify the internal elements of a sequence.

    This function is used internally by __stringify_fast() to stringify the
    internal elements of a sequence. It is used when the elements are
    concatenated on the same line. It will try to concatenate all elements
    on the same line, but if that's not possible, it will return early.

    An example of what this function constructs is:
    'item 1', 'item 2', 'item 3', 'item 4', 'item 5'

    Args:
        obj: The sequence to stringify.
        indent_level: The current indentation level.
        max_line_len: The maximum line length before breaking.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".
        line_len_left: The number of characters that can still be added to the
            current line.
        cached_strs: A list of tuples containing cached strings. Each tuple
            represents one item in the sequence, and contains two things:
            1. The stringified item as a list of strings.
            2. The length of all strings in that list combined.
            Together, the stringified items must be no longer than the given
            line_len_left. This argument will be updated in-place.

    Returns:
        Tuple containing:
        - A boolean indicating whether the function managed to stringify all
            elements on the same line. If False, the function will return
            early. In this case, the second and third return values will be set
            to [] and the input line_len_left respectively.
        - The object as a list of strings and integers. Strings are the
            stringified objects, integers are indices into the list of cached
            strings (that may have been updated in-place). To reconstruct the
            final string, the strings must be joined in the order they appear
            in this list.
        - The number of characters that can still be added to the current line
            after the function has finished.

    Examples:
    >>> cached_strs = []
    >>> plan_succeeded, strs, line_len_left = (
    ...     __stringify_fast_sequence_consecutive(
    ...         ["item 1", "item 2", "item 3", "item 4", "item 5"],
    ...         indent_level=0,
    ...         max_line_len=80,
    ...         ignore=None,
    ...         line_len_left=80,
    ...         cached_strs=cached_strs,
    ...     )
    ... )
    >>> "".join(
    ...     s
    ...     for s_or_i in strs
    ...     for s in (
    ...         cached_strs[s_or_i][0] if isinstance(s_or_i, int) else [s_or_i]
    ...     )
    ... )
    "'item 1', 'item 2', 'item 3', 'item 4', 'item 5'"
    """
    input_line_len_left = line_len_left

    strs = []
    for i, item in enumerate(obj):
        if i < len(cached_strs):
            curr_strs_len = cached_strs[i][1]
        else:
            curr_strs_len = 0
            curr_strs = []
            for s in __stringify_fast(
                item,
                indent_level=indent_level,
                max_line_len=max_line_len,
                ignore=ignore,
                line_len_left=line_len_left,
                breaking_allowed=False,
            ):
                if s is None:
                    return False, [], input_line_len_left
                curr_strs.append(s)
                curr_strs_len += len(s)
                if line_len_left - curr_strs_len < 0:
                    return False, [], input_line_len_left
            cached_strs.append((curr_strs, curr_strs_len))

        strs.append(i)
        line_len_left -= curr_strs_len
        if i != len(obj) - 1:  # no comma after last item
            strs.append(", ")
            line_len_left -= 2
            if line_len_left < 0:
                return False, [], input_line_len_left

    return True, strs, line_len_left


def __stringify_fast_sequence_separated(
    obj: Sequence[Any],
    indent_level: int,
    max_line_len: int,
    ignore: tuple[type, ...] | type | None,
    line_len_left: int,
    cached_strs: list[tuple[list[str], int]],
) -> list[str]:
    """Stringify the internal elements of a sequence.

    This function is used internally by __stringify_fast() to stringify the
    internal elements of a sequence. It is used when the elements are put on
    separate lines. It will always succeed, but if the elements are too long
    to fit on a single line, the maximum line length WILL be exceeded.

    An example of what this function constructs is:
    "'item 1',\\n'item 2',\\n'item 3',\\n'item 4',\\n'item 5',"

    Args:
        obj: The sequence to stringify.
        indent_level: The current indentation level.
        max_line_len: The maximum line length before breaking.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".
        line_len_left: The number of characters that can still be added to the
            current line.
        cached_strs: A list of tuples containing cached strings. Each tuple
            represents one item in the sequence, and contains two things:
            1. The stringified item as a list of strings.
            2. The length of all strings in that list combined.
            Together, the stringified items must be no longer than the given
            line_len_left.

    Returns:
        The object as a list of strings. To reconstruct the final string, the
        strings must be joined in the order they appear in this list.

    Examples:
    >>> strs = __stringify_fast_sequence_separated(
    ...     ["item 1", "item 2", "item 3", "item 4", "item 5"],
    ...     indent_level=1,
    ...     max_line_len=80,
    ...     ignore=None,
    ...     line_len_left=76,
    ...     cached_strs=[],
    ... )
    >>> "".join(strs)
    "'item 1',\\n    'item 2',\\n    'item 3',\\n    'item 4',\\n    'item 5',"
    """
    strs = []
    for i, item in enumerate(obj):
        if i < len(cached_strs):
            curr_strs = cached_strs[i][0]
        else:
            curr_strs = __stringify_fast(
                item,
                indent_level=indent_level,
                max_line_len=max_line_len,
                ignore=ignore,
                line_len_left=line_len_left - 1,  # account for ","
                breaking_allowed=True,
            )

        strs.extend(curr_strs)
        strs.append(",")
        if i != len(obj) - 1:  # no line break after last item
            strs.append("\n")
            strs.append(" " * (4 * indent_level))

    return strs


def __stringify_fast_mapping_consecutive(
    obj: Mapping[Any, Any],
    indent_level: int,
    max_line_len: int,
    ignore: tuple[type, ...] | type | None,
    line_len_left: int,
    cached_strs: list[tuple[list[str], int]],
) -> tuple[bool, list[str | int], int]:
    """Stringify the internal elements of a mapping.

    This function is used internally by __stringify_fast() to stringify the
    internal elements of a mapping. It is used when the elements are
    concatenated on the same line. It will try to concatenate all elements
    on the same line, but if that's not possible, it will return early.

    An example of what this function constructs is:
    'key 1': 'value 1', 'key 2': 'value 2', 'key 3': 'value 3'

    Args:
        obj: The mapping to stringify.
        indent_level: The current indentation level.
        max_line_len: The maximum line length before breaking.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".
        line_len_left: The number of characters that can still be added to the
            current line.
        cached_strs: A list of tuples containing cached strings. Each tuple
            represents one key/value in the sequence, and contains two things:
            1. The stringified key/value as a list of strings.
            2. The length of all strings in that list combined.
            Together, the stringified keys/values must be no longer than the
            given line_len_left. This argument will be updated in-place.

    Returns:
        Tuple containing:
        - A boolean indicating whether the function managed to stringify all
            elements on the same line. If False, the function will return
            early. In this case, the second and third return values will be set
            to [] and the input line_len_left respectively.
        - The object as a list of strings and integers. Strings are the
            stringified objects, integers are indices into the list of cached
            strings (that may have been updated in-place). To reconstruct the
            final string, the strings must be joined in the order they appear
            in this list.
        - The number of characters that can still be added to the current line
            after the function has finished.

    Examples:
    >>> cached_strs = []
    >>> plan_succeeded, strs, line_len_left = (
    ...     __stringify_fast_mapping_consecutive(
    ...         {"key 1": "value 1", "key 2": "value 2", "key 3": "value 3"},
    ...         indent_level=0,
    ...         max_line_len=80,
    ...         ignore=None,
    ...         line_len_left=80,
    ...         cached_strs=cached_strs,
    ...     )
    ... )
    >>> "".join(
    ...     s
    ...     for s_or_i in strs
    ...     for s in (
    ...         cached_strs[s_or_i][0] if isinstance(s_or_i, int) else [s_or_i]
    ...     )
    ... )
    "'key 1': 'value 1', 'key 2': 'value 2', 'key 3': 'value 3'"
    """
    input_line_len_left = line_len_left

    strs = []
    for i, (key, value) in enumerate(obj.items()):
        key_i = i * 2
        value_i = i * 2 + 1

        if key_i < len(cached_strs):
            curr_strs_len = cached_strs[key_i][1]
        else:
            curr_strs_len = 0
            curr_strs = []
            for s in __stringify_fast(
                key,
                indent_level=indent_level,
                max_line_len=max_line_len,
                ignore=ignore,
                line_len_left=line_len_left,
                breaking_allowed=False,
            ):
                if s is None:
                    return False, [], input_line_len_left
                curr_strs.append(s)
                curr_strs_len += len(s)
                if line_len_left - curr_strs_len < 0:
                    return False, [], input_line_len_left
            cached_strs.append((curr_strs, curr_strs_len))

        strs.append(key_i)
        line_len_left -= curr_strs_len
        strs.append(": ")
        line_len_left -= 2
        if line_len_left < 0:
            return False, [], input_line_len_left

        if value_i < len(cached_strs):
            curr_strs_len = cached_strs[value_i][1]
        else:
            curr_strs_len = 0
            curr_strs = []
            for s in __stringify_fast(
                value,
                indent_level=indent_level,
                max_line_len=max_line_len,
                ignore=ignore,
                line_len_left=line_len_left,
                breaking_allowed=False,
            ):
                if s is None:
                    return False, [], input_line_len_left
                curr_strs.append(s)
                curr_strs_len += len(s)
                if line_len_left - curr_strs_len < 0:
                    return False, [], input_line_len_left
            cached_strs.append((curr_strs, curr_strs_len))

        strs.append(value_i)
        line_len_left -= curr_strs_len
        if i != len(obj) - 1:  # no comma after last item
            strs.append(", ")
            line_len_left -= 2
            if line_len_left < 0:
                return False, [], input_line_len_left

    return True, strs, line_len_left


def __stringify_fast_mapping_separated(
    obj: Mapping[Any, Any],
    indent_level: int,
    max_line_len: int,
    ignore: tuple[type, ...] | type | None,
    line_len_left: int,
    cached_strs: list[tuple[list[str], int]],
) -> list[str]:
    """Stringify the internal elements of a mapping.

    This function is used internally by __stringify_fast() to stringify the
    internal elements of a mapping. It is used when the elements are put on
    separate lines. It will always succeed, but if the elements are too long
    to fit on a single line, the maximum line length WILL be exceeded.

    An example of what this function constructs is:
    "'key 1': 'value 1',\\n'key 2': 'value 2',\\n'key 3': 'value 3',"

    Args:
        obj: The mapping to stringify.
        indent_level: The current indentation level.
        max_line_len: The maximum line length before breaking.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".
        line_len_left: The number of characters that can still be added to the
            current line.
        cached_strs: A list of tuples containing cached strings. Each tuple
            represents one key/value in the sequence, and contains two things:
            1. The stringified key/value as a list of strings.
            2. The length of all strings in that list combined.
            Together, the stringified keys/values must be no longer than the
            given line_len_left.

    Returns:
        The object as a list of strings. To reconstruct the final string, the
        strings must be joined in the order they appear in this list.

    Examples:
    >>> strs = __stringify_fast_mapping_separated(
    ...     {"key 1": "value 1", "key 2": "value 2", "key 3": "value 3"},
    ...     indent_level=1,
    ...     max_line_len=80,
    ...     ignore=None,
    ...     line_len_left=76,
    ...     cached_strs=[],
    ... )
    >>> "".join(strs)
    "'key 1': 'value 1',\\n    'key 2': 'value 2',\\n    'key 3': 'value 3',"
    """
    strs = []
    for i, (key, value) in enumerate(obj.items()):
        key_i = i * 2
        value_i = i * 2 + 1

        if key_i < len(cached_strs):
            curr_strs = cached_strs[key_i][0]
        else:
            curr_strs = __stringify_fast(
                key,
                indent_level=indent_level,
                max_line_len=max_line_len,
                ignore=ignore,
                line_len_left=line_len_left - 3,  # account for ": " + any char
                breaking_allowed=True,
            )

        strs.extend(curr_strs)
        strs.append(": ")

        if value_i < len(cached_strs):
            curr_strs = cached_strs[value_i][0]
        else:
            # Account for multi-line keys.
            for s in reversed(strs):
                line_len_left -= (
                    len(s) if "\n" not in s else len(s.rpartition("\n")[2])
                )

            curr_strs = __stringify_fast(
                value,
                indent_level=indent_level,
                max_line_len=max_line_len,
                ignore=ignore,
                line_len_left=line_len_left - 1,  # account for ","
                breaking_allowed=True,
            )

        strs.extend(curr_strs)
        strs.append(",")
        if i != len(obj) - 1:  # no line break after last item
            strs.append("\n")
            strs.append(" " * (4 * indent_level))

    return strs


@overload
def __stringify_fast(
    obj: list[Any],
    indent_level: int = 0,
    max_line_len: int = 0,
    ignore: tuple[type, ...] | type | None = None,
    line_len_left: int = -1,
    breaking_allowed: Literal[True] = ...,
) -> Iterable[str]:
    pass


@overload
def __stringify_fast(
    obj: list[Any],
    indent_level: int = 0,
    max_line_len: int = 0,
    ignore: tuple[type, ...] | type | None = None,
    line_len_left: int = -1,
    breaking_allowed: Literal[False] = ...,
) -> Iterable[str | None]:
    pass


def __stringify_fast(
    obj: Any,
    indent_level: int = 0,
    max_line_len: int = 0,
    ignore: tuple[type, ...] | type | None = None,
    line_len_left: int = -1,
    breaking_allowed: bool = True,
) -> Iterable[str | None]:
    """Stringify an object with indentation and line breaks.

    Args:
        obj: The object to stringify.
        indent_level: The current indentation level. The indentation itself
            is displayed as 4 spaces per level. Used internally for recursive
            calls.
        max_line_len: The maximum line length before breaking. If set to 0,
            the terminal width will be used.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".
        line_len_left: The number of characters that can still be added to the
            current line. Used internally for recursive calls.
        breaking_allowed: Whether breaking is allowed. If not allowed and a
            break would be necessary, None is yielded and the function will
            return immediately. Used internally for recursive calls.

    Yields:
        Stringified objects that can't be further broken down. If breaking
        is not allowed and a break would be necessary, None is yielded and
        the function will return immediately.
    """
    if max_line_len == 0:
        max_line_len = shutil.get_terminal_size().columns

    if line_len_left == -1:
        line_len_left = max_line_len - 4 * indent_level

    if ignore is not None and isinstance(obj, ignore):
        yield type(obj).__name__
        return

    if isinstance(
        obj, (str, bytes, bytearray, int, float, complex, bool, type(None))
    ):
        yield repr(obj)
        return

    indent = " " * (4 * indent_level)
    indent_next = " " * (4 * (indent_level + 1))

    if isinstance(obj, Sequence):
        # Put the opening bracket on the current line.
        if type(obj) is list:
            start_char = "["
            end_char = "]"
        elif type(obj) is tuple:
            start_char = "("
            end_char = ")"
        else:
            start_char = type(obj).__name__ + "("
            end_char = ")"

        yield start_char
        line_len_left -= len(start_char)

        # Initialize an object to cache stringified items in the sequence.
        cached_strs = []

        # PLAN A
        plan_succeeded = True

        if line_len_left >= 0:
            # First, we try plan A: try to stringify all elements by putting
            # them on the same line. Example:
            # ...bla... ["item 1", "item 2", "item 3", "item 4", "item 5"]
            plan_succeeded, strs, line_len_left = (
                __stringify_fast_sequence_consecutive(
                    obj,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                    line_len_left=line_len_left - len(end_char),
                    cached_strs=cached_strs,
                )
            )
            if plan_succeeded is True:
                # If we're here, plan A succeeded. We can simply yield the
                # string and we're done.
                for s_or_i in strs:
                    if isinstance(s_or_i, int):
                        yield from cached_strs[s_or_i][0]
                    else:
                        yield s_or_i
                yield end_char
                return
        else:
            # Here, we don't have enough space to even put the opening bracket
            # on the current line. Thus, we will continue to plan B.
            plan_succeeded = False

        # plan_succeeded is ALWAYS False here.
        if breaking_allowed is False:
            # If plan A failed, then breaking is necessary but not allowed, so
            # we should yield None and return immediately.
            yield None
            return

        # Add a line break and indent for the next line.
        yield f"\n{indent_next}"
        prev_line_len_left = line_len_left
        line_len_left = max_line_len - len(indent_next)

        # PLAN B
        plan_succeeded = True

        if line_len_left > prev_line_len_left:
            # Second, we try plan B: try to stringify all elements by putting
            # them on a single new line. Example:
            # ...bla... [
            #     "item 1", "item 2", "item 3", "item 4", "item 5"
            # ]
            # Since this is just plan A but with a longer line length, we can
            # reuse the cached strings from plan A.
            plan_succeeded, strs, line_len_left = (
                __stringify_fast_sequence_consecutive(
                    obj,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                    line_len_left=line_len_left,
                    cached_strs=cached_strs,
                )
            )
            if plan_succeeded is True:
                # If we're here, plan B succeeded. We can simply yield the
                # string and we're done.
                for s_or_i in strs:
                    if isinstance(s_or_i, int):
                        yield from cached_strs[s_or_i][0]
                    else:
                        yield s_or_i
                yield f"\n{indent}{end_char}"
                return
        else:
            # If the new line doesn't even have more space than the current
            # line, we know that plan B won't work because plan A also didn't
            # work. Thus, we will continue to plan C.
            plan_succeeded = False

        # plan_succeeded is ALWAYS False here.

        # PLAN C
        # Third, we try plan C: try to stringify all elements by putting them
        # on separate lines. Example:
        # ...bla... [
        #     "item 1",
        #     "item 2",
        #     "item 3",
        #     "item 4",
        #     "item 5",
        # ]
        yield from __stringify_fast_sequence_separated(
            obj,
            indent_level=indent_level + 1,
            max_line_len=max_line_len,
            ignore=ignore,
            line_len_left=line_len_left,
            cached_strs=cached_strs,
        )
        yield f"\n{indent}{end_char}"
        return

    if isinstance(obj, Mapping):
        # Put the opening bracket on the current line.
        if type(obj) is dict:
            start_char = "{"
            end_char = "}"
        else:
            start_char = type(obj).__name__ + "{"
            end_char = "}"

        yield start_char
        line_len_left -= len(start_char)

        # Initialize an object to cache stringified items in the mapping.
        cached_strs = []

        # PLAN A
        plan_succeeded = True

        if line_len_left >= 0:
            # First, we try plan A: try to stringify all elements by putting
            # them on the same line. Example:
            # ...bla... {"key 1": "value 1", "key 2": "value 2"}
            plan_succeeded, strs, line_len_left = (
                __stringify_fast_mapping_consecutive(
                    obj,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                    line_len_left=line_len_left - len(end_char),
                    cached_strs=cached_strs,
                )
            )
            if plan_succeeded is True:
                # If we're here, plan A succeeded. We can simply yield the
                # string and we're done.
                for s_or_i in strs:
                    if isinstance(s_or_i, int):
                        yield from cached_strs[s_or_i][0]
                    else:
                        yield s_or_i
                yield end_char
                return
        else:
            # Here, we don't have enough space to even put the opening bracket
            # on the current line. Thus, we will continue to plan B.
            plan_succeeded = False

        # plan_succeeded is ALWAYS False here.
        if breaking_allowed is False:
            # If plan A failed, then breaking is necessary but not allowed, so
            # we should yield None and return immediately.
            yield None
            return

        # Add a line break and indent for the next line.
        yield f"\n{indent_next}"
        prev_line_len_left = line_len_left
        line_len_left = max_line_len - len(indent_next)

        # PLAN B
        plan_succeeded = True

        if line_len_left > prev_line_len_left:
            # Second, we try plan B: try to stringify all elements by putting
            # them on a single new line. Example:
            # ...bla... {
            #     "key 1": "value 1", "key 2": "value 2"
            # }
            # Since this is just plan A but with a longer line length, we can
            # reuse the cached strings from plan A.
            plan_succeeded, strs, line_len_left = (
                __stringify_fast_mapping_consecutive(
                    obj,
                    indent_level=indent_level + 1,
                    max_line_len=max_line_len,
                    ignore=ignore,
                    line_len_left=line_len_left,
                    cached_strs=cached_strs,
                )
            )
            if plan_succeeded is True:
                # If we're here, plan B succeeded. We can simply yield the
                # string and we're done.
                for s_or_i in strs:
                    if isinstance(s_or_i, int):
                        yield from cached_strs[s_or_i][0]
                    else:
                        yield s_or_i
                yield f"\n{indent}{end_char}"
                return
        else:
            # If the new line doesn't even have more space than the current
            # line, we know that plan B won't work because plan A also didn't
            # work. Thus, we will continue to plan C.
            plan_succeeded = False

        # plan_succeeded is ALWAYS False here.

        # PLAN C
        # Third, we try plan C: try to stringify all elements by putting them
        # on separate lines. Example:
        # ...bla... {
        #     "key 1": "value 1",
        #     "key 2": "value 2",
        # }
        yield from __stringify_fast_mapping_separated(
            obj,
            indent_level=indent_level + 1,
            max_line_len=max_line_len,
            ignore=ignore,
            line_len_left=line_len_left,
            cached_strs=cached_strs,
        )
        yield f"\n{indent}{end_char}"
        return

    yield repr(obj)


def stringify_fast(
    obj: Any,
    indent_level: int = 0,
    max_line_len: int = 0,
    ignore: tuple[type, ...] | type | None = None,
) -> str:
    """Stringify an object with indentation and line breaks.

    Args:
        obj: The object to stringify.
        indent_level: The current indentation level. The indentation itself
            is displayed as 4 spaces per level. In most cases, you can leave
            this at 0. It is used internally for recursive calls.
        max_line_len: The maximum line length before breaking. If set to 0,
            the terminal width will be used.
        ignore: A type or tuple of types to ignore. If the object is of one of
            these types, it will be stringified as "type(obj)".

    Returns:
        The stringified object.
    """
    return "".join(__stringify_fast(obj, indent_level, max_line_len, ignore))
