"""This file implements various 3D transformations.

The structure of the classes is inspired by the Transform3d class in PyTorch3D.
See https://github.com/facebookresearch/pytorch3d for the original code.
"""

from __future__ import annotations

import torch

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class Transform3D:
    """A batch of 3D transformation matrices."""

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create a batch of 3D transformation matrices."""
        self._matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(
            0
        )  # [1, 4, 4]

    def __getitem__(self, b: int) -> "Transform3D":
        """Get a single transformation matrix from the batch.

        Args:
            b: The index of the transformation matrix to get.

        Returns:
            The transformation matrix.
        """
        device = self._matrix.device
        dtype = self._matrix.dtype
        new = Transform3D(device=device, dtype=dtype)
        new._matrix = self._matrix[b].unsqueeze(0)
        return new

    def __len__(self) -> int:
        """Get the number of transformation matrices in the batch.

        Returns:
            The number of transformation matrices.
        """
        return len(self._matrix)

    @property
    def matrix(self) -> torch.Tensor:
        """The transformation matrices as a tensor.

        If the transform was composed from others, the matrix for the composite
        transform will be returned.

        Returns:
            The transformation matrices.
                Shape: [B, 4, 4]
        """
        return self._matrix

    def _multiply(self, other: "Transform3D") -> None:
        """Multiply this batch of matrices with another batch in-place.

        Args:
            other: The transformation matrices to multiply with.
                Shape: [B, 4, 4]
        """
        self._matrix = self._matrix.matmul(other.matrix)

    def multiply(self, other: "Transform3D") -> "Transform3D":
        """Multiply this batch of matrices with another batch out-of-place.

        Args:
            other: The transformation matrices to multiply with.
                Shape: [B, 4, 4]

        Returns:
            The composite transformation matrices.
        """
        device = self._matrix.device
        dtype = self._matrix.dtype
        new = Transform3D(device=device, dtype=dtype)
        new._matrix = self._matrix.matmul(other.matrix)
        return new

    @staticmethod
    def concat(transforms: list["Transform3D"]) -> "Transform3D":
        """Concatenate a list of transformation matrices into a single batch.

        Args:
            transforms: The list of transformation matrices to concatenate.

        Returns:
            The concatenated transformation matrices.
        """
        device = transforms[0]._matrix.device
        dtype = transforms[0]._matrix.dtype
        new = Transform3D(device=device, dtype=dtype)
        new._matrix = torch.concat([t.matrix for t in transforms])
        return new

    def transform_points(
        self, points: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Transform a batch of points with the transformation matrices.

        Args:
            points: The points to transform. Padding could be arbitrary.
                Shape: [B, max(P_bs), 3] or [P, 3]
            eps: A small value to prevent division by zero.

        Returns:
            The transformed points. Padding could be arbitrary and is not
            preserved.
                Shape: [B, max(P_bs), 3] or [P, 3]
        """
        # Perform error handling.
        points_batched = True
        if points.ndim == 2:
            if len(self._matrix) != 1:
                raise ValueError(
                    "If points is not batched, the transformation matrices"
                    " must have a batch size of 1."
                )
            points = points.unsqueeze(0)
            points_batched = False
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "points must have shape [B, max(P_bs), 3] or [P, 3], but got"
                f" {points.shape}"
            )
        B, max_P_bs, _ = points.shape
        device = points.device
        dtype = points.dtype

        # Transform the points.
        points = torch.concat(
            [points, torch.ones((B, max_P_bs, 1), device=device, dtype=dtype)],
            dim=2,
        )  # [B, max(P_bs), 4]
        points = points.matmul(self._matrix)  # [B, max(P_bs), 4]
        denom = points[:, :, 3:]  # [B, max(P_bs), 1]
        denom = torch.where(
            denom.abs() < eps, torch.full_like(denom, eps), denom
        )
        points /= denom  # [B, max(P_bs), 4]
        points = points[:, :, :3]  # [B, max(P_bs), 3]
        if not points_batched:
            return points.squeeze(0)
        return points

    def to(
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move the transformation matrices to a different device or data type.

        Warning: This method is in-place, meaning that the returned object is
        the same as the object that called this method.

        Args:
            device: The device to move the transformation matrices to.
            dtype: The data type to move the transformation matrices to.

        Returns:
            The transformation matrices on the new device or data type.
        """
        if device is None and dtype is None:
            raise ValueError("Either device or dtype must be specified.")

        self._matrix = self._matrix.to(device=device, dtype=dtype)
        return self

    def inverse(self) -> "Transform3D":
        """Get the inverse of the transformation matrices.

        Raises a RuntimeError if (one of) the matrices is/are not invertible.

        Returns:
            The inverse transformation matrices.
        """
        dtype = self._matrix.dtype
        device = self._matrix.device
        new = Transform3D(device=device, dtype=dtype)
        new._matrix = self._matrix.inverse()
        return new


class Translate(Transform3D):
    """A transformation that translates 3D points."""

    def __init__(
        self,
        x: torch.Tensor | float = 0,
        y: torch.Tensor | float = 0,
        z: torch.Tensor | float = 0,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create a transformation that translates 3D points.

        Args:
            x: The translation along the x-axis. Can be a scalar or tensor.
                Shape: [] or [B]
            y: The translation along the y-axis. Can be a scalar or tensor.
                Shape: [] or [B]
            z: The translation along the z-axis. Can be a scalar or tensor.
                Shape: [] or [B]
            device: The device to store the transformation matrix on.
            dtype: The data type of the transformation matrix.
        """
        super().__init__(device=device, dtype=dtype)

        # Perform error handling.
        x = torch.as_tensor(x, device=device, dtype=dtype)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        y = torch.as_tensor(y, device=device, dtype=dtype)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        z = torch.as_tensor(z, device=device, dtype=dtype)
        if z.ndim == 0:
            z = z.unsqueeze(0)
        if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
            raise ValueError(
                "x, y, and z must be scalars or 1D tensors, but x has shape"
                f" {x.shape}, y has shape {y.shape}, and z has shape {z.shape}"
            )
        B = max(len(x), len(y), len(z))
        if (
            len(x) != B
            and len(x) != 1
            or len(y) != B
            and len(y) != 1
            or len(z) != B
            and len(z) != 1
        ):
            raise ValueError(
                "x, y, and z must be broadcastable to the same shape, but x"
                f" has shape {x.shape}, y has shape {y.shape}, and z has shape"
                f" {z.shape}"
            )

        # Create the translation matrix.
        if len(x) != B:
            x = x.expand(B)
        if len(y) != B:
            y = y.expand(B)
        if len(z) != B:
            z = z.expand(B)
        mat = (
            torch.eye(4, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )  # [B, 4, 4]
        mat[:, 3, :3] = torch.stack([x, y, z], dim=1)  # [B, 3]
        self._matrix = mat  # [B, 4, 4]


class Scale(Transform3D):
    """A transformation that scales 3D points isotropically."""

    def __init__(
        self,
        x: torch.Tensor | float = 1,
        y: torch.Tensor | float = 1,
        z: torch.Tensor | float = 1,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create a transformation that scales 3D points isotropically.

        Args:
            x: The scaling factor along the x-axis. Can be a scalar or tensor.
                Shape: [] or [B]
            y: The scaling factor along the y-axis. Can be a scalar or tensor.
                Shape: [] or [B]
            z: The scaling factor along the z-axis. Can be a scalar or tensor.
                Shape: [] or [B]
            device: The device to store the transformation matrix on.
            dtype: The data type of the transformation matrix.
        """
        super().__init__(device=device, dtype=dtype)

        # Perform error handling.
        x = torch.as_tensor(x, device=device, dtype=dtype)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        y = torch.as_tensor(y, device=device, dtype=dtype)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        z = torch.as_tensor(z, device=device, dtype=dtype)
        if z.ndim == 0:
            z = z.unsqueeze(0)
        if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
            raise ValueError(
                "x, y, and z must be scalars or 1D tensors, but x has shape"
                f" {x.shape}, y has shape {y.shape}, and z has shape {z.shape}"
            )
        B = max(len(x), len(y), len(z))
        if (
            len(x) != B
            and len(x) != 1
            or len(y) != B
            and len(y) != 1
            or len(z) != B
            and len(z) != 1
        ):
            raise ValueError(
                "x, y, and z must be broadcastable to the same shape, but x"
                f" has shape {x.shape}, y has shape {y.shape}, and z has shape"
                f" {z.shape}"
            )

        # Create the scaling matrix.
        if len(x) != B:
            x = x.expand(B)
        if len(y) != B:
            y = y.expand(B)
        if len(z) != B:
            z = z.expand(B)
        mat = (
            torch.eye(4, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )  # [B, 4, 4]
        mat[:, 0, 0] = x
        mat[:, 1, 1] = y
        mat[:, 2, 2] = z
        self._matrix = mat  # [B, 4, 4]


class Rotate(Transform3D):
    """A transformation that rotates 3D points around an axis."""

    def __init__(
        self,
        rot: torch.Tensor,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create a transformation that rotates 3D points around an axis.

        Args:
            rot: The rotation matrices to apply to the points. Can be one of:
                - A single rotation matrix.
                    Shape: [3, 3]
                - Multiple rotation matrices.
                    Shape: [B, 3, 3]
            device: The device to store the transformation matrix on.
            dtype: The data type of the transformation matrix.
        """
        super().__init__(device=device, dtype=dtype)

        # Perform error handling.
        if rot.ndim == 2:
            rot = rot.unsqueeze(0)
        if rot.ndim != 3 or rot.shape[1:] != (3, 3):
            raise ValueError(
                f"rot must have shape [B, 3, 3] or [3, 3], but got {rot.shape}"
            )
        rot = rot.to(device=device, dtype=dtype)
        if not self.__is_valid_rotation_matrix(rot):
            raise ValueError("The matrix is not a valid rotation matrix.")

        # Create the rotation matrix.
        self._matrix[:, :3, :3] = rot  # [B, 4, 4]

    @staticmethod
    def __is_valid_rotation_matrix(
        rot: torch.Tensor, atol: float = 1e-7
    ) -> bool:
        """Check whether a matrix is a valid rotation matrix.

        A rotation matrix is valid if it is orthogonal. Mathematically:
        RR^T = I and det(R) = 1

        Args:
            rot: The batch of rotation matrices to check.
                Shape: [B, 3, 3]
            atol: The tolerance when checking whether the matrix is orthogonal.

        Returns:
            Whether the matrix is a valid rotation matrix.
        """
        device = rot.device
        dtype = rot.dtype
        B = len(rot)

        # Check whether the matrix is orthogonal.
        I = (  # noqa: E741
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )  # [B, 3, 3]
        if not torch.allclose(rot.bmm(rot.transpose(2, 1)), I, atol=atol):
            return False

        # Check whether the matrix has a determinant of 1.
        ones = torch.ones(B, device=device, dtype=dtype)
        if not torch.allclose(torch.det(rot), ones, atol=atol):
            return False

        return True


class RotateAxisAngle(Rotate):
    """A transformation that rotates 3D points around an axis by an angle."""

    def __init__(
        self,
        angle: torch.Tensor | float,
        axis: str = "X",
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create a transformation that rotates 3D points around an axis.

        Args:
            angle: Angle to rotate by in radians. Can be a scalar or tensor.
                Shape: [] or [B]
            axis: The axis to rotate around. Must be one of "X", "Y", or "Z".
            device: The device to store the transformation matrix on.
            dtype: The data type of the transformation matrix.
        """
        # Perform error handling.
        angle = torch.as_tensor(angle, device=device, dtype=dtype)
        if angle.ndim == 0:
            angle = angle.unsqueeze(0)
        if angle.ndim != 1:
            raise ValueError(
                "angle must be a scalar or a 1D tensor, but got shape"
                f" {angle.shape}"
            )
        B = len(angle)

        # Create the rotation matrix.
        c = angle.cos()  # [B]
        s = angle.sin()  # [B]
        zeros = torch.zeros(B, device=device, dtype=dtype)
        ones = torch.ones(B, device=device, dtype=dtype)
        if axis == "X":
            mat = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, c, -s]),
                torch.stack([zeros, s, c]),
            ])  # [3, 3, B]
        elif axis == "Y":
            mat = torch.stack([
                torch.stack([c, zeros, s]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([-s, zeros, c]),
            ])  # [3, 3, B]
        elif axis == "Z":
            mat = torch.stack([
                torch.stack([c, -s, zeros]),
                torch.stack([s, c, zeros]),
                torch.stack([zeros, zeros, ones]),
            ])  # [3, 3, B]
        else:
            raise ValueError(
                f"axis must be one of 'X', 'Y', or 'Z', but got '{axis}'"
            )
        # We swap dimension 1 and 2 because matmul() expects the matrix to
        # look like:
        # M = [
        #     [Rxx, Ryx, Rzx, 0],
        #     [Rxy, Ryy, Rzy, 0],
        #     [Rxz, Ryz, Rzz, 0],
        #     [Tx,  Ty,  Tz,  1],
        # ]
        mat = mat.permute(2, 1, 0)  # [B, 3, 3]
        super().__init__(mat, device=device, dtype=dtype)


class ProjectToSurface(Transform3D):
    """A transformation that projects 3D points to a plane in 3D."""

    def __init__(
        self,
        surface: torch.Tensor,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
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
                - (x, y, z) are the coordinates of a point on the surface.
            device: The device to store the transformation matrix on.
            dtype: The data type of the transformation matrix.
        """
        super().__init__(device=device, dtype=dtype)

        # Perform error handling.
        if surface.ndim == 1:
            self.surface = surface.unsqueeze(0)  # [1, 4]
        if surface.shape[-1] != 4 or surface.ndim > 2:
            raise ValueError(
                "surface must have shape [4] or [B, 4], but got"
                f" {surface.shape}"
            )
        surface = surface.to(device=device, dtype=dtype)

        # Create the projection matrix.
        B = surface.shape[0]
        n0, n1, n2, d = surface.unbind(-1)  # [B], ...
        zeros = torch.zeros(B, device=device, dtype=dtype)
        ones = torch.ones(B, device=device, dtype=dtype)

        mat = torch.stack([
            torch.stack([1 - n0.square(), -n0 * n1, -n0 * n2, -n0 * d]),
            torch.stack([-n1 * n0, 1 - n1.square(), -n1 * n2, -n1 * d]),
            torch.stack([-n2 * n0, -n2 * n1, 1 - n2.square(), -n2 * d]),
            torch.stack([zeros, zeros, zeros, ones]),
        ])  # [4, 4, B]
        # We swap dimension 1 and 2 because matmul() expects the matrix to
        # look like:
        # M = [
        #     [Rxx, Ryx, Rzx, 0],
        #     [Rxy, Ryy, Rzy, 0],
        #     [Rxz, Ryz, Rzz, 0],
        #     [Tx,  Ty,  Tz,  1],
        # ]
        self._matrix = mat.permute(2, 1, 0)  # [B, 4, 4]


class SurfaceToSurface(Transform3D):
    """A transformation that rotates a surface onto another surface."""

    def __init__(
        self,
        from_surface: torch.Tensor,
        to_surface: torch.Tensor,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
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
            device: The device to store the transformation matrix on.
            dtype: The data type of the transformation matrix.
        """
        super().__init__(device=device, dtype=dtype)

        # Perform error handling.
        if from_surface.ndim == 1:
            from_surface = from_surface.unsqueeze(0)  # [1, 4]
        if from_surface.shape[-1] != 4 or from_surface.ndim > 2:
            raise ValueError(
                "from_surface must have shape [4] or [B, 4], but got"
                f" {from_surface.shape}"
            )
        from_surface = from_surface.to(device=device, dtype=dtype)
        if to_surface.ndim == 1:
            to_surface = to_surface.unsqueeze(0)  # [1, 4]
        if to_surface.shape[-1] != 4 or to_surface.ndim > 2:
            raise ValueError(
                "to_surface must have shape [4] or [B, 4], but got"
                f" {to_surface.shape}"
            )
        to_surface = to_surface.to(device=device, dtype=dtype)
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
        from_n = from_n / from_n.norm(dim=1, keepdim=True)  # [B, 3]
        to_n = to_n / to_n.norm(dim=1, keepdim=True)  # [B, 3]

        # First translate the points from from_surface to the origin.
        # from_v is the point on from_surface closest to the origin.
        from_v = -from_n * from_d.unsqueeze(-1)  # [B, 3]
        from_v_x, from_v_y, from_v_z = from_v.unbind(1)  # [B], [B], [B]
        self._multiply(
            Translate(
                -from_v_x, -from_v_y, -from_v_z, device=device, dtype=dtype
            )
        )

        # Then rotate the points from from_n to to_n.
        rot = self.__get_matrix_rotate_vec_a_to_vec_b(
            from_n, to_n, device=device, dtype=dtype
        )  # [B, 3, 3]
        self._multiply(Rotate(rot, device=device, dtype=dtype))

        # Finally, translate the points from the origin to to_surface.
        # to_v is the point on to_surface closest to the origin.
        to_v = -to_n * to_d.unsqueeze(-1)  # [B, 3]
        to_v_x, to_v_y, to_v_z = to_v.unbind(1)  # [B], [B], [B]
        self._multiply(
            Translate(to_v_x, to_v_y, to_v_z, device=device, dtype=dtype)
        )

    @staticmethod
    def __get_matrix_rotate_vec_a_to_vec_b(
        from_n: torch.Tensor,
        to_n: torch.Tensor,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype = torch.float32,
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
            device: The device to store the rotation matrices on.
            dtype: The data type of the rotation matrices.

        Returns:
            The rotation matrices to rotate from from_n to to_n.
                Shape: [B, 3, 3]
        """
        device = device if device is not None else from_n.device
        B = len(from_n)
        zeros = torch.zeros(B, device=device, dtype=dtype)
        u = from_n.cross(to_n, dim=1)  # [B, 3]
        c = (from_n * to_n).sum(dim=1, keepdim=True).unsqueeze(2)  # [B, 1, 1]
        v_x = torch.stack([
            torch.stack([zeros, -u[:, 2], u[:, 1]]),
            torch.stack([u[:, 2], zeros, -u[:, 0]]),
            torch.stack([-u[:, 1], u[:, 0], zeros]),
        ])  # [3, 3, B]
        v_x = v_x.permute(2, 1, 0)  # [B, 3, 3]
        I = torch.eye(3, device=device, dtype=dtype)  # [3, 3]  # noqa: E741
        return torch.where(
            # Test whether the vectors are opposites.
            (from_n + to_n).abs() > 1e-7,
            # If the vectors are not opposites, perform a rotation around u.
            I + v_x + v_x.bmm(v_x) / (1 + c),
            # If the vectors are opposites, perform a 180 degree rotation
            # around any axis perpendicular to from_n.
            I - 2 * from_n.unsqueeze(2).bmm(from_n.unsqueeze(1)),
        )  # [B, 3, 3]


def points_2D_to_3D(points: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 2D points to 3D points.

    Args:
        points: The points to convert. Should represent x and y coordinates.
            The z coordinate will be set to zero.
            Shape: [..., 2]

    Returns:
        The converted points. Represents x, y, and z coordinates.
            Shape: [..., 3]
    """
    return torch.concat(
        [
            points,
            torch.zeros(
                points.shape[:-1] + (1,),
                device=points.device,
                dtype=points.dtype,
            ),
        ],
        dim=-1,
    )  # [..., 3]


def points_3D_to_2D(points: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 3D points to 2D points.

    Args:
        points: The points to convert. Should represent x, y, and z
            coordinates. The z coordinate will be ignored.
            Shape: [..., 3]

    Returns:
        The converted points. Represents x and y coordinates.
            Shape: [..., 2]
    """
    return points[..., :2]  # [..., 2]


def transform_points_3D_to_3D(
    points: torch.Tensor, transform: Transform3D
) -> torch.Tensor:
    """Transform a batch of points with a transformation.

    Args:
        points: The points to transform. Should represent x, y, and z
            coordinates. Padding could be arbitrary.
            Shape: [B, max(P_bs), 3] or [P, 3]
        transform: The transformation to apply to the points.
            Shape of .get_matrix(): [B, 4, 4] or [1, 4, 4]

    Returns:
        The transformed points. Represents x, y, and z coordinates.
        Padding could be arbitrary and is not preserved.
            Shape: [B, max(P_bs), 3] or [P, 3]
    """
    return transform.transform_points(
        points, eps=1e-6
    )  # [B, max(P_bs), 3] or [P, 3]


def transform_points_2D_to_3D(
    points: torch.Tensor, transform: Transform3D
) -> torch.Tensor:
    """Transform a batch of points with a transformation.

    Args:
        points: The points to transform. Should represent x and y coordinates.
            The z coordinate will be set to zero. Padding could be arbitrary.
            Shape: [B, max(P_bs), 2] or [P, 2]
        transform: The transformation to apply to the points.
            Shape of .get_matrix(): [B, 4, 4] or [1, 4, 4]

    Returns:
        The transformed points. Represents x, y, and z coordinates.
        Padding could be arbitrary and is not preserved.
            Shape: [B, max(P_bs), 3] or [P, 3]
    """
    # Transform the points to 3D.
    points = points_2D_to_3D(points)  # [B, max(P_bs), 3] or [P, 3]
    return transform.transform_points(
        points, eps=1e-6
    )  # [B, max(P_bs), 3] or [P, 3]


def transform_points_3D_to_2D(
    points: torch.Tensor, transform: Transform3D
) -> torch.Tensor:
    """Transform a batch of points with a transformation.

    Args:
        points: The points to transform. Should represent x, y, and z
            coordinates. Padding could be arbitrary.
            Shape: [B, max(P_bs), 3] or [P, 3]
        transform: The transformation to apply to the points.
            Shape of .get_matrix(): [B, 4, 4] or [1, 4, 4]

    Returns:
        The transformed points. Represents x and y coordinates.
        The z coordinate will be ignored.
        Padding could be arbitrary and is not preserved.
            Shape: [B, max(P_bs), 2] or [P, 2]
    """
    # Transform the points.
    points = transform.transform_points(
        points, eps=1e-6
    )  # [B, max(P_bs), 3] or [P, 3]
    return points_3D_to_2D(points)  # [B, max(P_bs), 2] or [P, 2]


def transform_points_2D_to_2D(
    points: torch.Tensor, transform: Transform3D
) -> torch.Tensor:
    """Transform a batch of points with a transformation.

    Args:
        points: The points to transform. Should represent x and y coordinates.
            The z coordinate will be set to zero. Padding could be arbitrary.
            Shape: [B, max(P_bs), 2] or [P, 2]
        transform: The transformation to apply to the points.
            Shape of .get_matrix(): [B, 4, 4] or [1, 4, 4]

    Returns:
        The transformed points. Represents x and y coordinates.
        The z coordinate will be ignored.
        Padding could be arbitrary and is not preserved.
            Shape: [B, max(P_bs), 2] or [P, 2]
    """
    # Transform the points to 3D.
    points = points_2D_to_3D(points)  # [B, max(P_bs), 3] or [P, 3]
    points = transform.transform_points(
        points, eps=1e-6
    )  # [B, max(P_bs), 3] or [P, 3]
    return points_3D_to_2D(points)  # [B, max(P_bs), 2] or [P, 2]
