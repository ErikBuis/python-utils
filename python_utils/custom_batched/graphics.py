import torch

from ..modules_batched.torch import arange_batched, replace_padding_batched


def xiaolin_wu_anti_aliasing_batched(
    x0: torch.Tensor, y0: torch.Tensor, x1: torch.Tensor, y1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Xiaolin Wu's line algorithm for drawing anti-aliased lines.

    Args:
        x0: X-coordinate of the first endpoint of the line segment.
            Shape: [B]
        y1: Y-coordinate of the first endpoint of the line segment.
            Shape: [B]
        x1: X-coordinate of the second endpoint of the line segment.
            Shape: [B]
        y1: Y-coordinate of the second endpoint of the line segment.
            Shape: [B]

    Returns:
        Tuple containing:
        - Pixel x-coordinates. Padded with zeros.
            Shape: [B, max(S_bs)]
        - Pixel y-coordinates. Padded with zeros.
            Shape: [B, max(S_bs)]
        - Pixel values between 0 and 1. Padded with zeros.
            Shape: [B, max(S_bs)]
        - The number of pixels in each line segment.
            Shape: [B]
    """
    device = x0.device

    # Determine if the line is steep.
    steep = torch.abs(y1 - y0) > torch.abs(x1 - x0)  # [B]

    # Swap the x and y coordinates to ensure the line is not steep.
    x0, y0, x1, y1 = torch.where(
        steep, torch.stack([y0, x0, y1, x1]), torch.stack([x0, y0, x1, y1])
    )

    # Swap the start and end to ensure the line goes from left to right.
    x0, y0, x1, y1 = torch.where(
        x0 > x1, torch.stack([x1, y1, x0, y0]), torch.stack([x0, y0, x1, y1])
    )

    # Calculate the gradient of the line segments.
    dx, dy = x1 - x0, y1 - y0  # [B], [B]
    gradient = torch.where(dx != 0, dy / dx, 1)  # [B]

    # Pre-process the beginning of the line segments.
    xpxl_begin = x0.round().long()  # [B]
    xgap_begin = 1 - (x0 + 0.5 - xpxl_begin)  # [B]

    # Pre-process the end of the line segments.
    xpxl_end = x1.round().long()  # [B]
    xgap_end = x1 + 0.5 - xpxl_end  # [B]

    # Initialize the return values.
    S_bs = 2 * (xpxl_end - xpxl_begin + 1)  # [B]
    max_S_bs = int(S_bs.max())
    B = len(S_bs)
    pixels_x = torch.empty((B, max_S_bs), device=device, dtype=torch.int64)
    pixels_y = torch.empty((B, max_S_bs), device=device, dtype=torch.int64)
    vals = torch.empty((B, max_S_bs), device=device, dtype=torch.float64)

    # Calculate values used in the main loop.
    x, _ = arange_batched(
        xpxl_begin, xpxl_end + 1, dtype=torch.int64
    )  # [B, max(S_bs) // 2], _
    intery = y0.unsqueeze(1) + gradient.unsqueeze(1) * (
        x.double() - x0.unsqueeze(1)
    )  # [B, max(S_bs) // 2]
    ipart_intery = intery.floor().long()  # [B, max(S_bs) // 2]
    fpart_intery = intery - ipart_intery  # [B, max(S_bs) // 2]
    rfpart_intery = 1 - fpart_intery  # [B, max(S_bs) // 2]

    # Fill the return values.
    pixels_x[:, ::2] = torch.where(steep.unsqueeze(1), ipart_intery, x)
    pixels_y[:, ::2] = torch.where(steep.unsqueeze(1), x, ipart_intery)
    pixels_x[:, 1::2] = torch.where(steep.unsqueeze(1), ipart_intery + 1, x)
    pixels_y[:, 1::2] = torch.where(steep.unsqueeze(1), x, ipart_intery + 1)
    vals[:, ::2] = rfpart_intery
    vals[:, 1::2] = fpart_intery

    # Handle the beginning and end of the line segments.
    arange_B = torch.arange(B, device=device)
    vals[:, :2] *= xgap_begin.unsqueeze(1)
    vals[arange_B, S_bs - 2] *= xgap_end
    vals[arange_B, S_bs - 1] *= xgap_end

    # Pad the return values.
    replace_padding_batched(pixels_x, S_bs, in_place=True)
    replace_padding_batched(pixels_y, S_bs, in_place=True)
    replace_padding_batched(vals, S_bs, in_place=True)

    return pixels_x, pixels_y, vals, S_bs
