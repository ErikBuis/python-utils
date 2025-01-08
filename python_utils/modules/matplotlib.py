from typing import Any, Sequence

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import numpy as np
import numpy.typing as npt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


def get_scalar_mappable(
    values: Sequence[float] | npt.ArrayLike,
    colors: Sequence[str] | npt.ArrayLike,
    color_values: Sequence[float] | npt.ArrayLike | None = None,
    use_log_scale: bool = False,
    linear_width: float | None = None,
) -> matplotlib.cm.ScalarMappable:
    """Get a ScalarMappable with color map: colors[0] -> ... -> colors[-1].

    Args:
        values: The values to map to colors.
            Length: N
        colors: Colors to map the values to.
            The smallest value is mapped to colors[0].
            The largest value is mapped to colors[-1].
            For possible choices of colors, see:
            https://matplotlib.org/stable/gallery/color/named_colors.html
            Length: C
        color_values: With which values the middle colors correspond. Since
            the smallest value is mapped to colors[0] and the largest value is
            mapped to colors[-1], only the middle colors can be affected by
            this parameter. The values must be in [min(values), max(values)].
            If None, the colors will be evenly distributed. Useful if you want
            to unevenly distribute the colors.
            Length: C - 2
        use_log_scale: Whether to use a log scale for assigning colors.
        linear_width: The width of the linear part around zero if log scaling
            is used. If None, the linear width will be set to a small value
            automatically. Ignored if use_log_scale is False.

    Returns:
        ScalarMappable color map for the coefficients.
        - Use scalar_mappable.cmap to get the color map.
        - Use scalar_mappable.norm to get the norm.
            The norm maps the range [vmin, vmax] to [0, 1].
            To get vmin and vmax, use scalar_mappable.norm.vmin and
                scalar_mappable.norm.vmax respectively.
            To map a value from the range [vmin, vmax] to [0, 1], use
                scalar_mappable.norm(value).

    Examples:
        >>> # The following example will create a mapping to the colors red,
        >>> # green, and blue, which will be distributed as follows:
        >>> #                       0.7                            0.3
        >>> #  |------------------------------------------|------------------|
        >>> # red                                       green              blue
        >>> # The values will be mapped to the colors as follows:
        >>> #  |------------------------------------------|------------------|
        >>> #  1              2              3               4               5
        >>> # fully         orange        yellow           mostly         fully
        >>> #  red          -ish          /green            green          blue
        >>> get_scalar_mappable(
        >>>     [1, 2, 3, 4, 5],
        >>>     ["red", "green", "blue"],
        >>>     [0.7 * (5 - 1)],
        >>> )
    """
    values = np.array(values)
    colors = np.array(colors)
    if color_values is not None:
        color_values = np.array(color_values)
        if len(color_values) != len(colors) - 2:
            raise ValueError(
                "The number of color values must be len(colors) - 2."
            )
        if (
            color_values.min() < values.min()
            or color_values.max() > values.max()
        ):
            raise ValueError(
                "The color values must be between min(values) and max(values)."
            )

    vmin = values.min()
    vmax = values.max()

    # Use a log scale if specified, otherwise use a linear scale.
    # norm is a function that maps the range [vmin, vmax] to [0, 1].
    if use_log_scale:
        if linear_width is None:
            # Make the linear width as small as possible.
            linear_width = np.abs(values).min()
        norm = matplotlib.colors.AsinhNorm(
            vmin=vmin, vmax=vmax, linear_width=linear_width  # type: ignore
        )
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create a mapping from values to colors.
    name = "_".join(colors)
    if color_values is not None:
        all_norms = np.concatenate([[0], norm(color_values).data, [1]])
    else:
        all_norms = np.linspace(0, 1, len(colors))
    color_list = list(zip(all_norms, colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name=name, colors=color_list
    )

    # Create the ScalarMappable color map.
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def get_scalar_mappable_middle_white(
    values: Sequence[float] | npt.ArrayLike,
    from_color: str = "red",
    to_color: str = "green",
    use_log_scale: bool = False,
    zero_is_white: bool = False,
) -> matplotlib.cm.ScalarMappable:
    """Get a ScalarMappable with color map: from_color -> white -> to_color.

    Args:
        values: The values to map to colors.
            Length: N
        from_color: The color to map the smallest value(s) to.
        to_color: The color to map the largest value(s) to.
        use_log_scale: Whether to use a log scale.
        zero_is_white: If True, zero values will be mapped to white. Otherwise,
            the average between vmin and vmax will be mapped to white.

    Returns:
        ScalarMappable color map for the coefficients.
        - Use scalar_mappable.get_cmap() to get the color map.
        - Use scalar_mappable.norm to get the norm.
            The norm maps the range [vmin, vmax] to [0, 1].
            To get vmin and vmax, use scalar_mappable.norm.vmin and
                scalar_mappable.norm.vmax respectively.
            To map a value to the range [0, 1], use
                scalar_mappable.norm(value).
    """
    values = np.array(values)

    if zero_is_white:
        # If zero values should be mapped to white, then we need to make sure
        # that there is at least one negative and one positive value.
        vmin = values.min()
        vmax = values.max()
        if vmin > 0:
            values = np.concatenate([values, [-vmin]])
        elif vmax < 0:
            values = np.concatenate([values, [-vmax]])

    return get_scalar_mappable(
        values,
        (from_color, "white", to_color),
        color_values=(0,) if zero_is_white else None,
        use_log_scale=use_log_scale,
        linear_width=(
            np.sort(np.abs(values))[int(len(values) * 0.1)]
            if use_log_scale and zero_is_white
            else None
        ),
    )


class Arrow3D(FancyArrowPatch):
    """A 3D arrow that can be plotted on an Axes3D instance.

    Taken from:
    https://stackoverflow.com/questions/22867620/
    putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

    This class draws an arrow using an ArrowStyle. To draw an arrow, you should
    call arrow3D() instead of creating an instance of this class directly.

    The head and tail positions are fixed at the specified start and end points
    of the arrow, but the size and shape (in display coordinates) of the arrow
    does not change when the axis is moved or zoomed.
    """

    def __init__(
        self, x, y, z, dx, dy, dz, *args, **kwargs  # type: ignore
    ) -> None:
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def do_3d_projection(self, renderer=None) -> float:  # type: ignore
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform(
            (x1, x2), (y1, y2), (z1, z2), self.axes.M  # type: ignore
        )
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def arrow3D(
    ax: Axes3D,
    x: float | npt.ArrayLike,
    y: float | npt.ArrayLike,
    z: float | npt.ArrayLike,
    dx: float | npt.ArrayLike,
    dy: float | npt.ArrayLike,
    dz: float | npt.ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Add an 3D arrow to an Axes3D instance.

    Note: In contrast to other patches, the default capstyle and joinstyle for
    Arrow3D are set to "round".

    Args:
        ax: The Axes3D instance to add the arrow to.
        x: The x-coordinate of the tail of the arrow.
        y: The y-coordinate of the tail of the arrow.
        z: The z-coordinate of the tail of the arrow.
        dx: The difference in x-coordinates between the head and tail of the
            arrow.
        dy: The difference in y-coordinates between the head and tail of the
            arrow.
        dz: The difference in z-coordinates between the head and tail of the
            arrow.
        *args: Additional arguments to pass to the
            matplotlib.patches.FancyArrowPatch constructor.
        **kwargs: Additional keyword arguments to pass to the
            matplotlib.patches.FancyArrowPatch constructor. Here is a list of
            the available FancyArrowPatch properties:

        arrowstyle (str or matplotlib.patches.ArrowStyle, default: "simple"):
            The ArrowStyle with which the fancy arrow is drawn. If a string,
            it should be one of the available arrowstyle names, with optional
            comma-separated attributes. The optional attributes are meant to
            be scaled with the mutation_scale. The available names can be found
            at matplotlib.patches.ArrowStyle.
        connectionstyle (str or matplotlib.patches.ConnectionStyle or None,
            default: "arc3"): The ConnectionStyle with which the tail and head
            are connected. If a string, it should be one of the available
            connectionstyle names, with optional comma-separated attributes.
            The available names can be found at
            matplotlib.patches.ConnectionStyle.
        patchA (matplotlib.patches.Patch, default: None): The patch for the
            head of the arrow.
        patchB (matplotlib.patches.Patch, default: None): The patch for the
            tail of the arrow.
        shrinkA (float, default: 2): Shrinking factor of the tail of the arrow.
        shrinkB (float, default: 2): Shrinking factor of the head of the arrow.
        mutation_scale (float, default: 1): Value with which attributes of
            arrowstyle (e.g. head_length) will be scaled.
        mutation_aspect (None or float, default: None): The height of the
            rectangle will be squeezed by this value before the mutation and
            the mutated box will be stretched by the inverse of it.
        **kwargs: Additional keyword arguments to pass to the
            matplotlib.patches.Patch constructor. A list of the available Patch
            properties can be found at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch
    """
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
