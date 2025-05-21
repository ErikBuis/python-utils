from math import floor

import torch


def xiaolin_wu_anti_aliasing(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Xiaolin Wu's line algorithm for drawing an anti-aliased line.

    This function is equivalent to Xiaolin Wu's line algorithm, but it is
    vectorized to process all steps on the major axis in parallel. Therefore,
    it is much faster than the naive implementation.

    Args:
        x0: X-coordinate of the first endpoint of the line segment.
        y1: Y-coordinate of the first endpoint of the line segment.
        x1: X-coordinate of the second endpoint of the line segment.
        y1: Y-coordinate of the second endpoint of the line segment.

    Returns:
        Tuple containing:
        - Pixel x-coordinates.
            Shape: [S]
        - Pixel y-coordinates.
            Shape: [S]
        - Pixel values between 0 and 1.
            Shape: [S]
    """
    # Determine if the line is steep.
    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep:
        # Swap the x and y coordinates to ensure the line is not steep.
        x0, y0, x1, y1 = y0, x0, y1, x1

    if x0 > x1:
        # Swap the start and end to ensure the line goes from left to right.
        x0, y0, x1, y1 = x1, y1, x0, y0

    # Calculate the gradient of the line segment.
    dx, dy = x1 - x0, y1 - y0
    gradient = dy / dx if dx else 1

    # Pre-process the beginning of the line segment.
    xpxl_begin = round(x0)
    xgap_begin = 1 - (x0 + 0.5 - xpxl_begin)

    # Pre-process the end of the line segment.
    xpxl_end = round(x1)
    xgap_end = x1 + 0.5 - xpxl_end

    # Initialize the return values.
    S = 2 * (xpxl_end - xpxl_begin + 1)
    pixels_x = torch.empty(S, dtype=torch.int64)
    pixels_y = torch.empty(S, dtype=torch.int64)
    vals = torch.empty(S, dtype=torch.float64)

    # Calculate values used in the main loop.
    x = torch.arange(xpxl_begin, xpxl_end + 1, dtype=torch.int64)  # [S // 2]
    intery = y0 + gradient * (x.double() - x0)  # [S // 2]
    ipart_intery = intery.floor().long()  # [S // 2]
    fpart_intery = intery - ipart_intery  # [S // 2]
    rfpart_intery = 1 - fpart_intery  # [S // 2]

    # Fill the return values.
    if steep:
        pixels_x[::2] = ipart_intery
        pixels_y[::2] = x
        pixels_x[1::2] = ipart_intery + 1
        pixels_y[1::2] = x
    else:
        pixels_x[::2] = x
        pixels_y[::2] = ipart_intery
        pixels_x[1::2] = x
        pixels_y[1::2] = ipart_intery + 1
    vals[::2] = rfpart_intery
    vals[1::2] = fpart_intery

    # Handle the beginning and end of the line segment.
    vals[:2] *= xgap_begin
    vals[-2:] *= xgap_end

    return pixels_x, pixels_y, vals


def xiaolin_wu_anti_aliasing_naive(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[list[int], list[int], list[float]]:
    """Xiaolin Wu's line algorithm for drawing an anti-aliased line.

    This function is made to be as close as possible to the pseudocode from
    the Wikipedia page about Xiaolin Wu's line algorithm. Source:
    https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm

    Args:
        x0: X-coordinate of the first endpoint of the line segment.
        y1: Y-coordinate of the first endpoint of the line segment.
        x1: X-coordinate of the second endpoint of the line segment.
        y1: Y-coordinate of the second endpoint of the line segment.

    Returns:
        Tuple containing:
        - List of pixel x-coordinates.
            Shape: [S]
        - List of pixel y-coordinates.
            Shape: [S]
        - List of pixel values between 0 and 1.
            Shape: [S]
    """

    def ipart(x: float) -> int:
        return floor(x)

    def round(x: float) -> int:
        return floor(x + 0.5)

    def fpart(x: float) -> float:
        return x - floor(x)

    def rfpart(x: float) -> float:
        return 1 - fpart(x)

    pixels_x, pixels_y, brightness = [], [], []

    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx else 1

    # Handle the first endpoint.
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)

    if steep:
        pixels_x.extend([ypxl1, ypxl1 + 1])
        pixels_y.extend([xpxl1, xpxl1])
        brightness.extend([rfpart(yend) * xgap, fpart(yend) * xgap])
    else:
        pixels_x.extend([xpxl1, xpxl1])
        pixels_y.extend([ypxl1, ypxl1 + 1])
        brightness.extend([rfpart(yend) * xgap, fpart(yend) * xgap])

    intery = yend + gradient

    # Handle the second endpoint.
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)

    if steep:
        pixels_x.extend([ypxl2, ypxl2 + 1])
        pixels_y.extend([xpxl2, xpxl2])
        brightness.extend([rfpart(yend) * xgap, fpart(yend) * xgap])
    else:
        pixels_x.extend([xpxl2, xpxl2])
        pixels_y.extend([ypxl2, ypxl2 + 1])
        brightness.extend([rfpart(yend) * xgap, fpart(yend) * xgap])

    # Main loop.
    if steep:
        for x in range(xpxl1 + 1, xpxl2):
            pixels_x.extend([ipart(intery), ipart(intery) + 1])
            pixels_y.extend([x, x])
            brightness.extend([rfpart(intery), fpart(intery)])
            intery += gradient
    else:
        for x in range(xpxl1 + 1, xpxl2):
            pixels_x.extend([x, x])
            pixels_y.extend([ipart(intery), ipart(intery) + 1])
            brightness.extend([rfpart(intery), fpart(intery)])
            intery += gradient

    return pixels_x, pixels_y, brightness
