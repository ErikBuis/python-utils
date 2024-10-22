import unittest
from collections.abc import Callable

import torch


try:
    from python_utils.modules.pytorch3d import get_matrix_rotate_vec_a_to_vec_b

    pytorch3d_installed = True
except ImportError:
    pytorch3d_installed = False

    get_matrix_rotate_vec_a_to_vec_b: Callable


@unittest.skipUnless(pytorch3d_installed, "PyTorch3D is not installed")
class TestGetMatrixRotateVecAToVecB(unittest.TestCase):
    DTYPE = torch.float64  # for torch.float64, the atol 1e-6 is unnecessary
    ITERATIONS = 100

    def __generate_random_unit_vec(self) -> torch.Tensor:
        v = torch.randn(3, dtype=self.DTYPE)
        return v / torch.norm(v)

    def __transform_points(
        self, R: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        """Transform a batch of 3D points with a rotation matrix.

        Args:
            rot: The rotation matrix to apply to the points.
                Shape: [B, 3, 3]
            points: The points to transform.
                Shape: [B, max(P_b), 3]

        Returns:
            The transformed points.
                Shape: [B, max(P_b), 3]
        """
        return a.bmm(R)

    def test_a_equals_b(self) -> None:
        for _ in range(self.ITERATIONS):
            from_n = self.__generate_random_unit_vec().unsqueeze(0)  # [1, 3]
            to_n = from_n.clone()  # [1, 3]
            R = get_matrix_rotate_vec_a_to_vec_b(from_n, to_n)  # [1, 3, 3]
            self.assertTrue(
                torch.allclose(
                    R, torch.eye(3, dtype=self.DTYPE).unsqueeze(0), atol=1e-6
                ),
                f"R != I\nR = {R}",
            )
            a = from_n.unsqueeze(1)  # [1, 1, 3]
            b = to_n.unsqueeze(1)  # [1, 1, 3]
            Ra = self.__transform_points(R, a)  # [1, 1, 3]
            self.assertTrue(
                torch.allclose(Ra, b), f"Ra != b\nRa = {Ra}\nb = {b}"
            )

        # Prevent floating point trickery.
        from_n = torch.tensor([1, 0, 0], dtype=self.DTYPE).unsqueeze(
            0
        )  # [1, 3]
        to_n = from_n.clone()  # [1, 3]
        R = get_matrix_rotate_vec_a_to_vec_b(from_n, to_n)  # [1, 3, 3]
        self.assertTrue(
            torch.allclose(
                R, torch.eye(3, dtype=self.DTYPE).unsqueeze(0), atol=1e-6
            ),
            f"R != I\nR = {R}",
        )
        a = from_n.unsqueeze(1)  # [1, 1, 3]
        b = to_n.unsqueeze(1)  # [1, 1, 3]
        Ra = self.__transform_points(R, a)  # [1, 1, 3]
        self.assertTrue(torch.allclose(Ra, b), f"Ra != b\nRa = {Ra}\nb = {b}")

    def test_a_opposite_b(self) -> None:
        for _ in range(self.ITERATIONS):
            from_n = self.__generate_random_unit_vec().unsqueeze(0)  # [1, 3]
            to_n = -from_n.clone()  # [1, 3]
            R = get_matrix_rotate_vec_a_to_vec_b(from_n, to_n)  # [1, 3, 3]
            a = from_n.unsqueeze(1)  # [1, 1, 3]
            b = to_n.unsqueeze(1)  # [1, 1, 3]
            Rb = self.__transform_points(R, b)  # [1, 1, 3]
            self.assertTrue(
                torch.allclose(Rb, a), f"Rb != b\nRb = {Rb}\na = {a}"
            )

        # Prevent floating point trickery.
        from_n = torch.tensor([1, 0, 0], dtype=self.DTYPE).unsqueeze(
            0
        )  # [1, 3]
        to_n = -from_n.clone()  # [1, 3]
        R = get_matrix_rotate_vec_a_to_vec_b(from_n, to_n)  # [1, 3, 3]
        a = from_n.unsqueeze(1)  # [1, 1, 3]
        b = to_n.unsqueeze(1)  # [1, 1, 3]
        Rb = self.__transform_points(R, b)  # [1, 1, 3]
        self.assertTrue(torch.allclose(Rb, a), f"Rb != b\nRb = {Rb}\na = {a}")

    def test_a_b_random(self) -> None:
        for _ in range(self.ITERATIONS):
            from_n = self.__generate_random_unit_vec().unsqueeze(0)  # [1, 3]
            to_n = self.__generate_random_unit_vec().unsqueeze(0)  # [1, 3]
            R = get_matrix_rotate_vec_a_to_vec_b(from_n, to_n)  # [1, 3, 3]
            a = from_n.unsqueeze(1)  # [1, 1, 3]
            b = to_n.unsqueeze(1)  # [1, 1, 3]
            Ra = self.__transform_points(R, a)  # [1, 1, 3]
            self.assertTrue(
                torch.allclose(Ra, b, atol=1e-6),
                f"Ra != b\nRa = {Ra}\nb = {b}",
            )
