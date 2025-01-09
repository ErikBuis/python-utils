import torch
from scipy.interpolate import griddata
from scipy.spatial import QhullError


def griddata_batched(
    points: torch.Tensor,
    values: torch.Tensor,
    P_bs: torch.Tensor,
    xi: torch.Tensor,
    X_bs: torch.Tensor,
    method: str = "linear",
    fill_value: float = torch.nan,
    rescale: bool = False,
) -> torch.Tensor:
    """
    A batched version of scipy.interpolate.griddata.

    Note that this method essentially loops over the batch and calls
    scipy.interpolate.griddata for each batch element. However, it does handle
    padding and rescaling of the input data and catches the exception that is
    raised when the convex hull of the input points is degenerate. These
    features are not provided by scipy.interpolate.griddata which is why we
    implement this method here.

    Args:
        points: Point coordinates to interpolate between.
            Padding could be arbitrary.
            Shape: [B, max(P_bs), D]
        values: Values per point coordinate. Padding could be arbitrary.
            Shape: [B, max(P_bs)]
        P_bs: The number of points to interpolate between in each batch.
            Shape: [B]
        xi: Points at which to interpolate data. Padding could be arbitrary.
            Shape: [B, max(X_bs), D]
        X_bs: The number of points at which to interpolate data in each batch.
            Shape: [B]
        method: Method of interpolation. Must be one of "linear", "nearest",
            or "cubic".
            nearest: Return the value at the data point closest to the point
                of interpolation. See NearestNDInterpolator for more details.
            linear: Tessellate the input point set to N-D simplices, and
                interpolate linearly on each simplex. See LinearNDInterpolator
                for more details.
            cubic (1-D): Return the value determined from a cubic spline.
            cubic (2-D): Return the value determined from a piecewise cubic,
                continuously differentiable (C1), and approximately
                curvature-minimizing polynomial surface. See
                CloughTocher2DInterpolator for more details.
        fill_value: Value used to fill in for requested points outside of the
            convex hull of the input points or if the convex hull of the input
            points is degenerate. If not provided, then the default is nan.
            This option has no effect for the "nearest" method.
        rescale: Whether to rescale points to the unit cube before performing
            interpolation. This is useful if some of the input dimensions have
            incommensurable units and differ by many orders of magnitude.

    Returns:
        Array of interpolated values. Padded with fill_value.
            Shape: [B, max(X_bs)]
    """
    B, max_X_bs, _ = xi.shape
    device = points.device

    # Initialize the output array.
    values_interpolated = torch.full((B, max_X_bs), fill_value, device=device)

    # Loop over the batch and interpolate the values.
    for b in range(B):
        if P_bs[b] == 0:
            continue  # no points to interpolate between
        try:
            values_interpolated[b, : X_bs[b]] = torch.from_numpy(
                griddata(
                    points[b, : P_bs[b]],
                    values[b, : P_bs[b]],
                    xi[b, : X_bs[b]],
                    method=method,
                    fill_value=fill_value,
                    rescale=rescale,
                )
            )
        except QhullError:  # the convex hull is degenerate
            pass  # leave the fill_value in place

    return values_interpolated
