import logging
import time
from typing import Any


logger = logging.getLogger(__name__)

__timers: dict[int, dict[str, Any]] = {}


def human_readable_time(
    time: int, significant_digits: int = 3, abbreviate: bool = True
) -> str:
    """Convert the given time to a human-readable string.

    If the time is shorter than one minute, the largest unit that can
    represent the time in the given amount of significant digits is used. If
    the time is longer than one minute, all units smaller than the time until
    seconds will be shown, no matter how many significant digits were
    requested.

    Args:
        time: Time in nanoseconds as returned by the difference between two
            time.perf_counter_ns() calls (or any of the other _ns variants).
        significant_digits: The number of significant digits to show.
        abbreviate: Whether to abbreviate the unit names (e.g. "ms" instead of
            "milliseconds").

    Returns:
        A human-readable string representing the given time.

    Examples:
        Under one minute:
        >>> human_readable_time(1234567890)
        '1.23 s'
        >>> human_readable_time(1234567890, significant_digits=2)
        '1.2 s'
        >>> human_readable_time(1234567890, significant_digits=1)
        '1 s'
        >>> human_readable_time(234567890)
        '235 ms'
        >>> human_readable_time(234567890, significant_digits=2)
        '0.23 s'  # note that 230 ms would have 3 significant digits
        >>> human_readable_time(12345678900)
        '12.3 s'
        >>> human_readable_time(12345678900, significant_digits=2)
        '12 s'
        >>> human_readable_time(12345678900, significant_digits=1)
        '12 s'  # only case where significant digits are not respected
        >>> human_readable_time(1995123456)
        '2.00 s'
        >>> human_readable_time(9995123456)
        '10.0 s'
        >>> human_readable_time(1234567890, abbreviate=False)
        '1.23 seconds'
        >>> human_readable_time(234567890, abbreviate=False)
        '235 milliseconds'

        Over or equal to one minute:
        >>> human_readable_time((3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000)
        '3h 14m 15s'  # no spaces are added before the units
        >>> human_readable_time(
        >>>     (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000,
        >>>     abbreviate=False,
        >>> )
        '3 hours 14 minutes 15 seconds'  # spaces are added before the units
        >>> human_readable_time(
        >>>     (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000 + 23456789,
        >>>     significant_digits=5,
        >>> )
        '3h 14m 15s'
        >>> human_readable_time(
        >>>     (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000 + 23456789,
        >>>     significant_digits=6,
        '3h 14m 15.2s'
        >>> human_readable_time(
        >>>     (3 * 60 * 60 + 7 * 60 + 15) * 1_000_000_000 + 23456789,
        >>>     significant_digits=6,
        >>> )
        '3h 7m 15.2s'  # note that the 0 before the 7 counts as a sig. digit
    """
    if significant_digits < 1:
        raise ValueError("significant_digits must be at least 1.")

    if time < 60 * 1_000_000_000:  # time is shorter than one minute
        time_str = str(time)

        # Round the time to the nearest number that can be represented with the
        # given number of significant digits.
        if (
            significant_digits < len(time_str)
            and time_str[significant_digits] >= "5"
        ):
            for i in range(significant_digits - 1, -1, -1):
                if time_str[i] != "9":
                    time_str = (
                        time_str[:i]
                        + str(int(time_str[i]) + 1)
                        + time_str[i + 1 :]
                    )
                    break
                time_str = time_str[:i] + "0" + time_str[i + 1 :]
            else:
                time_str = "1" + time_str

        # Determine the unit the time should be represented in.
        if (
            len(time_str) <= 3
            and significant_digits >= (len(time_str) - 1) % 3 + 1
        ):
            unit = "ns" if abbreviate else "nanoseconds"
            factorof10 = 0
        elif (
            len(time_str) <= 6
            and significant_digits >= (len(time_str) - 4) % 3 + 1
        ):
            unit = "µs" if abbreviate else "microseconds"
            factorof10 = 3
        elif (
            len(time_str) <= 9
            and significant_digits >= (len(time_str) - 7) % 3 + 1
        ):
            unit = "ms" if abbreviate else "milliseconds"
            factorof10 = 6
        else:
            unit = "s" if abbreviate else "seconds"
            factorof10 = 9

        # Assemble the time string.
        dot_idx = len(time_str) - factorof10
        if dot_idx > 0:
            if dot_idx >= significant_digits:
                time_str = time_str[:dot_idx]
            else:
                time_str = (
                    time_str[:dot_idx]
                    + "."
                    + time_str[dot_idx:significant_digits]
                )
        else:
            time_str = "0." + "0" * -dot_idx + time_str[:significant_digits]
        return f"{time_str} {unit}"

    # Time is longer than or equal to one minute.
    time_str = ""
    sigdits_used = 0
    if time >= 365_242198790 * 24 * 60 * 60:
        # Matt Parker said we could do better than leap years.
        years = str(time // (365_242198790 * 24 * 60 * 60))
        time_str += years
        time_str += "y " if abbreviate else " years "
        time %= 365_242198790 * 24 * 60 * 60
        sigdits_used += len(years)
    if time >= 24 * 60 * 60 * 1_000_000_000:
        days = str(time // (24 * 60 * 60 * 1_000_000_000))
        time_str += days
        time_str += "d " if abbreviate else " days "
        time %= 24 * 60 * 60 * 1_000_000_000
        sigdits_used += len(days) if sigdits_used == 0 else 3
    elif sigdits_used > 0:
        sigdits_used += 3
    if time >= 60 * 60 * 1_000_000_000:
        hours = str(time // (60 * 60 * 1_000_000_000))
        time_str += hours
        time_str += "h " if abbreviate else " hours "
        time %= 60 * 60 * 1_000_000_000
        sigdits_used += len(hours) if sigdits_used == 0 else 2
    elif sigdits_used > 0:
        sigdits_used += 2
    if time >= 60 * 1_000_000_000:
        minutes = str(time // (60 * 1_000_000_000))
        time_str += minutes
        time_str += "m " if abbreviate else " minutes "
        time %= 60 * 1_000_000_000
        sigdits_used += len(minutes) if sigdits_used == 0 else 2
    elif sigdits_used > 0:
        sigdits_used += 2
    seconds = str(time // 1_000_000_000)
    time_str += seconds
    time %= 1_000_000_000
    sigdits_used += len(seconds) if sigdits_used == 0 else 2
    if time > 0 and significant_digits > sigdits_used:
        time_str += "."
        time_str += str(time)[: significant_digits - sigdits_used]
    time_str += "s" if abbreviate else " seconds"

    return time_str


def start_timer(
    msg: str = "Timing code", logging_level: int = logging.DEBUG
) -> int:
    """Start a timer for timing code.

    Use stop_timer to stop the timer and log the time passed since start_timer
    was called.

    You may call this function multiple times to start multiple timers. To
    stop a timer, you must call stop_timer with the id returned by the
    corresponding call to start_timer.

    Args:
        msg: The message to log.
        logging_level: The logging level to use for the message.

    Returns:
        The id of the timer. Pass this id to stop_timer to stop the timer.
    """
    global __timers

    # Log the message with an ellipsis to indicate that the timer has started.
    msg = 4 * len(__timers) * " " + msg
    logger.log(logging_level, f"{msg}...")

    # Find a unique timer id.
    timer_id = 0
    while timer_id in __timers:
        timer_id += 1
    timer = {"logging_level": logging_level, "msg": msg, "start_time": None}
    __timers[timer_id] = timer

    # Start as late as possible to minimize the time between the start and stop
    # calls.
    timer["start_time"] = time.perf_counter_ns()
    return timer_id


def stop_timer(timer_id: int) -> int:
    """Stop a timer and log the time passed since start_timer was called.

    Args:
        timer_id: The id of the timer to stop. This id is returned by
            start_timer. If the timer has already been stopped, we will log an
            error event to the logger and return -1.

    Returns:
        The time passed since the corresponding call to start_timer in
        nanoseconds. If the timer id does not exist, -1 will be returned.
    """
    # Stop as soon as possible to minimize the time between the start and stop
    # calls.
    stop_time = time.perf_counter_ns()

    # Log the time passed since the corresponding call to start_timer.
    global __timers
    try:
        timer_config = __timers.pop(timer_id)
    except KeyError:
        logger.error(
            f"Timer with id {timer_id} does not exist or has already been"
            " stopped!"
        )
        return -1
    diff = stop_time - timer_config["start_time"]

    logger.log(
        timer_config["logging_level"],
        f"{timer_config['msg']} took {human_readable_time(diff)}",
    )

    return diff
