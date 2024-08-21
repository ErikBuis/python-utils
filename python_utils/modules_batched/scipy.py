import torch
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError


def griddata_batched(
    points: torch.Tensor,
    values: torch.Tensor,
    num_points_per_batch: torch.Tensor,
    xi: torch.Tensor,
    num_xi_per_batch: torch.Tensor,
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
        points: Point coordinates to interpolate between. Padded with zeros if
            the batch is heterogeneous.
            Shape: [B, max(P_b), D]
        values: Values per point coordinate. Padded with zeros if the batch is
            heterogeneous.
            Shape: [B, max(P_b)]
        num_points_per_batch: The number of points to interpolate between in
            each batch.
            Shape: [B]
        xi: Points at which to interpolate data.
            Shape: [B, max(X_b), D]
        num_xi_per_batch: The number of points at which to interpolate data
            in each batch.
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
        Array of interpolated values.
            Shape: [B, max(X_b)]
    """
    B, max_X_b, _ = xi.shape
    device = points.device

    # Initialize the output array.
    values_interpolated = torch.full(
        (B, max_X_b), fill_value, device=device
    )  # [B, max(X_b)]

    # Loop over the batch and interpolate the values.
    for b in range(B):
        if num_points_per_batch[b] == 0:
            continue  # no points to interpolate between
        try:
            values_interpolated[b, : num_xi_per_batch[b]] = torch.from_numpy(
                griddata(
                    points[b, : num_points_per_batch[b]],
                    values[b, : num_points_per_batch[b]],
                    xi[b, : num_xi_per_batch[b]],
                    method=method,
                    fill_value=fill_value,
                    rescale=rescale,
                )
            )
        except QhullError:  # the convex hull is degenerate
            pass  # leave the fill_value in place

    return values_interpolated
