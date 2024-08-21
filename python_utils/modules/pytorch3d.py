from collections import defaultdict

import torch
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Rotate, Transform3d, Translate


def transform_2D_points(
    points: torch.Tensor, transform: Transform3d
) -> torch.Tensor:
    """Transform a batch of 2D points with a transformation.

    Args:
        points: The points to transform. Should represent x and y coordinates.
            The z coordinate is assumed to be zero.
            Shape: [B, max(P_b), 2] or [P, 2]
        transform: The transformation to apply to the points.

    Returns:
        The transformed points. Represents x and y coordinates.
            Shape: [B, max(P_b), 3] or [P, 3]
    """
    padding_shape = points.shape[:-1] + (1,)
    device = points.device

    # Transform the points.
    points_3d = torch.concatenate(
        [points, torch.zeros(padding_shape, device=device)], dim=-1
    )  # [B, max(P_b), 3] or [P, 3]
    return transform.transform_points(
        points_3d, eps=1e-6
    )  # [B, max(P_b), 3] or [P, 3]


def get_matrix_rotate_vec_a_to_vec_b(
    from_n: torch.Tensor,
    to_n: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Find the matrices that rotate one set of vectors onto another.

    Useful for rotating a plane onto another plane in 3D.

    Warning: The returned matrix is transposed compared to the mathematical
        conventions! This is because the operation is batched: it is easier to
        multiply the batch of matrices with a batch column vectors when the
        matrices are already transposed.

    For the mathematical derivation, see:
    https://math.stackexchange.com/questions/180418/
    calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    Args:
        from_n: The vectors to rotate from. Must be normalized.
            Shape: [B, 3]
        to_n: The vectors to rotate to. Must be normalized.
            Shape: [B, 3]
        dtype: The data type of the rotation matrices.
        device: The device to store the rotation matrices on.

    Returns:
        The rotation matrices to rotate from from_n to to_n.
            Shape: [B, 3, 3]
    """
    device_ = torch.device(device) if device is not None else from_n.device
    B = len(from_n)
    zeros = torch.zeros(B, dtype=dtype, device=device_)  # [B]
    u = torch.cross(from_n, to_n, dim=1)  # [B, 3]
    c = torch.sum(from_n * to_n, dim=1, keepdim=True).unsqueeze(2)  # [B, 1, 1]
    v_x = torch.stack([
        torch.stack([zeros, -u[:, 2], u[:, 1]]),
        torch.stack([u[:, 2], zeros, -u[:, 0]]),
        torch.stack([-u[:, 1], u[:, 0], zeros]),
    ])  # [3, 3, B]
    v_x = v_x.permute(2, 1, 0)  # [B, 3, 3]
    I = torch.eye(3, dtype=dtype, device=device_)  # [3, 3]
    return torch.where(
        # Test whether the vectors are opposites.
        (from_n + to_n).abs() > 1e-7,
        # If the vectors are not opposites, perform a rotation around u.
        I + v_x + v_x.bmm(v_x) / (1 + c),
        # If the vectors are opposites, perform a 180 degree rotation around
        # any axis perpendicular to from_n.
        I - 2 * from_n.unsqueeze(2).bmm(from_n.unsqueeze(1)),
    )  # [B, 3, 3]


class ProjectToSurface(Transform3d):
    """A transformation that projects 3D points to a plane in 3D.

    This class is similar to the PyTorch3D classes in the transforms module in
    the sense that it will only store a homogeneous matrix internally when
    initialized. By doing this, multiple transformations can be applied to the
    in a single matrix multiplication when transform_points() is called.
    """

    def __init__(
        self,
        surface: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        """Create a transformation that projects 3D points to a plane in 3D.

        Args:
            surface: The surface(s) to project the points to. Can be one of:
                - A single surface represented by a tensor.
                    Shape: [4]
                - Multiple surfaces represented by a tensor.
                    Shape: [B, 4]
                Each surface is represented by a tuple (n_0, n_1, n_2, d),
                where the surface is represented by the equation:
                    n_0 * x + n_1 * y + n_2 * z = d
                - n = (n_0, n_1, n_2) is the normal vector of the surface.
                - d is the distance of the surface to the origin divided by
                    the length of the normal vector.
                - (x, y, z) are the coordinates of a point on the surface
            dtype: The data type of the transformation matrix.
            device: The device to store the transformation matrix on.
        """
        # Initialize the transformation.
        device_ = (
            torch.device(device) if device is not None else surface.device
        )
        super().__init__(dtype=dtype, device=device_)

        # Perform error handling.
        if surface.ndim == 1:
            self.surface = surface.unsqueeze(0)  # [1, 4]
        if surface.shape[-1] != 4 or surface.ndim > 2:
            raise ValueError(
                "surface must have shape [4] or [B, 4], but got"
                f" {surface.shape}"
            )
        surface = surface.to(device_, dtype)

        # Create the projection matrix.
        B = surface.shape[0]
        n0, n1, n2, d = surface.unbind(-1)  # [B], ...
        zeros = torch.zeros(B, dtype=dtype, device=device_)  # [B]
        ones = torch.ones(B, dtype=dtype, device=device_)  # [B]

        mat = torch.stack([
            torch.stack([1 - n0.square(), -n0 * n1, -n0 * n2, -n0 * d]),
            torch.stack([-n1 * n0, 1 - n1.square(), -n1 * n2, -n1 * d]),
            torch.stack([-n2 * n0, -n2 * n1, 1 - n2.square(), -n2 * d]),
            torch.stack([zeros, zeros, zeros, ones]),
        ])  # [4, 4, B]
        # We swap the first and second dimensions because pytorch3d expects
        # the matrix to look like:
        # M = [
        #     [Rxx, Ryx, Rzx, 0],
        #     [Rxy, Ryy, Rzy, 0],
        #     [Rxz, Ryz, Rzz, 0],
        #     [Tx,  Ty,  Tz,  1],
        # ]
        mat = mat.permute(2, 1, 0)  # [B, 4, 4]
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        raise RuntimeError(
            "The inverse of a projection matrix is not defined."
        )


class SurfaceToSurface(Transform3d):
    """A transformation that rotates a surface onto another surface.

    This class is similar to the PyTorch3D classes in the transforms module in
    the sense that it will only store a homogeneous matrix internally when
    initialized. By doing this, multiple transformations can be applied to the
    in a single matrix multiplication when transform_points() is called.
    """

    def __init__(
        self,
        from_surface: torch.Tensor,
        to_surface: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        """Create a transformation that rotates a surface onto another surface.

        The surfaces are rotated using the least amount of rotation possible.
        Intuitively, this means that when animating the rotation, the
        "movement" of the normal vector of the surface is minimal.

        Args:
            from_surface: The surface(s) to rotate from. Can be one of:
                - A single surface represented by a tensor.
                    Shape: [4]
                - Multiple surfaces represented by a tensor.
                    Shape: [B, 4]
                Each surface is represented by a tuple (n_0, n_1, n_2, d),
                where the surface is represented by the equation:
                    n_0 * x + n_1 * y + n_2 * z = d
                - n = (n_0, n_1, n_2) is the normal vector of the surface.
                - d is the distance of the surface to the origin divided by
                    the length of the normal vector.
                - (x, y, z) are the coordinates of a point on the surface
            to_surface: The surface(s) to rotate to. Can be one of:
                - A single surface represented by a tensor.
                    Shape: [4]
                - Multiple surfaces represented by a tensor.
                    Shape: [B, 4]
                Each surface is represented by a tuple (n_0, n_1, n_2, d),
                where the surface is represented by the equation:
                    n_0 * x + n_1 * y + n_2 * z = d
                - n = (n_0, n_1, n_2) is the normal vector of the surface.
                - d is the distance of the surface to the origin divided by
                    the length of the normal vector.
                - (x, y, z) are the coordinates of a point on the surface.
            dtype: The data type of the transformation matrix.
            device: The device to store the transformation matrix on.
        """
        # Initialize the transformation.
        device_ = (
            torch.device(device) if device is not None else from_surface.device
        )
        super().__init__(dtype=dtype, device=device_)

        # Perform error handling.
        if from_surface.ndim == 1:
            from_surface = from_surface.unsqueeze(0)  # [1, 4]
        if from_surface.shape[-1] != 4 or from_surface.ndim > 2:
            raise ValueError(
                "from_surface must have shape [4] or [B, 4], but got"
                f" {from_surface.shape}"
            )
        from_surface = from_surface.to(device_, dtype)
        if to_surface.ndim == 1:
            to_surface = to_surface.unsqueeze(0)  # [1, 4]
        if to_surface.shape[-1] != 4 or to_surface.ndim > 2:
            raise ValueError(
                "to_surface must have shape [4] or [B, 4], but got"
                f" {to_surface.shape}"
            )
        to_surface = to_surface.to(device_, dtype)
        if from_surface.shape[0] == 1 and to_surface.shape[0] != 1:
            from_surface = from_surface.expand(to_surface.shape[0], -1)
        if to_surface.shape[0] == 1 and from_surface.shape[0] != 1:
            to_surface = to_surface.expand(from_surface.shape[0], -1)
        if from_surface.shape[0] != to_surface.shape[0]:
            raise ValueError(
                "from_surface and to_surface must have the same batch size,"
                f" but got {from_surface.shape[0]} and {to_surface.shape[0]}"
            )

        # Calculate some intermediate values.
        from_n0, from_n1, from_n2, from_d = from_surface.unbind(-1)  # [B], ...
        to_n0, to_n1, to_n2, to_d = to_surface.unbind(-1)  # [B], ...
        from_n = torch.stack([from_n0, from_n1, from_n2], dim=1)  # [B, 3]
        to_n = torch.stack([to_n0, to_n1, to_n2], dim=1)  # [B, 3]
        from_n = from_n / torch.norm(from_n, dim=1, keepdim=True)  # [B, 3]
        to_n = to_n / torch.norm(to_n, dim=1, keepdim=True)  # [B, 3]

        # First translate the points from from_surface to the origin.
        # from_v is the point on from_surface closest to the origin.
        from_v = -from_n * from_d.unsqueeze(-1)  # [B, 3]
        matrix1 = Translate(
            -from_v, dtype=dtype, device=device_
        ).get_matrix()  # [B, 4, 4]

        # Then rotate the points from from_n to to_n.
        rot = get_matrix_rotate_vec_a_to_vec_b(
            from_n, to_n, dtype=dtype, device=device_
        )  # [B, 3, 3]
        matrix2 = Rotate(
            rot, dtype=dtype, device=device_
        ).get_matrix()  # [B, 4, 4]

        # Finally, translate the points from the origin to to_surface.
        # to_v is the point on to_surface closest to the origin.
        to_v = -to_n * to_d.unsqueeze(-1)  # [B, 3]
        matrix3 = Translate(
            to_v, dtype=dtype, device=device_
        ).get_matrix()  # [B, 4, 4]

        self._matrix = matrix1.bmm(matrix2).bmm(matrix3)  # [B, 4, 4]


def estimate_normals(
    pointclouds: torch.Tensor | Pointclouds,
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    use_symeig_workaround: bool = True,
) -> torch.Tensor:
    """Estimate the normals of a batch of point clouds.

    This function is equivalent to `estimate_pointcloud_normals` from
    PyTorch3D, but it handles the case where the number of points in a point
    cloud is smaller than the neighborhood size correctly. It does this by
    decreasing the neighborhood size for the point clouds that are too small.

    Warning: This function does not handle the case where the number of points
        in a point cloud is one or zero.
    """
    if (
        isinstance(pointclouds, torch.Tensor)
        and pointclouds.shape[1] <= 1
        or torch.any(pointclouds.num_points_per_cloud() <= 1)  # type: ignore
    ):
        raise ValueError(
            "The number of points in a point cloud must be at least two."
        )

    if isinstance(pointclouds, torch.Tensor):
        return estimate_pointcloud_normals(
            pointclouds,
            neighborhood_size=pointclouds.shape[1] - 1,
            disambiguate_directions=disambiguate_directions,
            use_symeig_workaround=use_symeig_workaround,
        )

    if torch.all(pointclouds.num_points_per_cloud() > neighborhood_size):
        return estimate_pointcloud_normals(
            pointclouds,
            neighborhood_size=neighborhood_size,
            disambiguate_directions=disambiguate_directions,
            use_symeig_workaround=use_symeig_workaround,
        )

    # The neighborhood_size argument must be smaller than the size of each
    # of the point clouds. estimate_pointcloud_normals throws an error if this
    # is not the case, which is really stupid because we do want to use the
    # requested neighborhood size for the point clouds that are large enough.
    # To work around this, we should split the point clouds into 1 + N groups:
    # 1. The point clouds that are large enough.
    # 2. For each point cloud that is too small, group it with other point
    #    clouds of the same size.
    counts = defaultdict(list)
    for i, num_points in enumerate(pointclouds.num_points_per_cloud()):
        if num_points > neighborhood_size:
            counts[neighborhood_size].append(i)
        else:
            counts[num_points - 1].append(i)

    normals = torch.zeros_like(pointclouds.points_padded())
    for neighborhood_size, idcs in counts.items():
        max_P_b = pointclouds[idcs].points_padded().shape[1]
        normals[idcs, :max_P_b] = estimate_pointcloud_normals(
            pointclouds[idcs],
            neighborhood_size=neighborhood_size,
            disambiguate_directions=disambiguate_directions,
            use_symeig_workaround=use_symeig_workaround,
        )

    return normals
