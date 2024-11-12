import torch

from ..custom_batched.geometry import (
    distance_line_to_point_batched,
    line_intersection_batched,
)
from .torch import (
    pack_padded_batched,
    pad_packed_batched,
    sample_unique_pairs_batched,
)


def ransac_batched(
    lines: tuple[torch.Tensor, torch.Tensor],
    L_bs: torch.Tensor,
    max_iterations: torch.Tensor | int,
    inlier_threshold: torch.Tensor | float,
    min_inliers: torch.Tensor | int,
) -> torch.Tensor:
    """Determine the best intersection point of multiple lines.

    This function uses the Random Sample Consensus (RANSAC) algorithm to
    determine the best intersection point of multiple lines. The algorithm
    works by randomly selecting two lines, calculating their intersection
    point, and counting the number of lines that intersect close to that
    point. The intersection point with the most votes is returned.

    Args:
        lines: The lines to intersect. Each line is represented by a pair
            (r, theta) in Hough space as a tuple containing:
            - The values of r.
                Shape: [B, max(L_bs)]
            - The values of theta.
                Shape: [B, max(L_bs)]
        L_bs: The number of lines for each sample in the batch.
            Shape: [B]
        max_iterations: The maximum number of iterations to perform.
            Shape: [B] or [] or int
        inlier_threshold: Threshold for classifying a line as an inlier. Lines
            with a distance to the intersection point less than or equal to
            this threshold are considered inliers.
            Shape: [B] or [] or float
        min_inliers: The minimum number of inliers required to consider an
            intersection point as a "good" intersection point. If this happens
            before max_iterations is reached, the ransac algorithm stops early
            for this sample. (If all samples have found a good intersection
            point, the function itself returns early for efficiency.)
            Shape: [B] or [] or int

    Returns:
        The best intersection point of the lines.
            Shape: [B, 2]
    """
    if L_bs.min() < 2:
        raise ValueError(
            "At least two lines are required for RANSAC, but the sample at"
            f" index {L_bs.argmin().item()} contains less than two lines."
        )

    r, theta = lines
    B, max_L_b = r.shape
    dtype = theta.dtype
    device = theta.device

    if not isinstance(max_iterations, torch.Tensor):
        max_iterations = torch.full(
            (B,), max_iterations, dtype=torch.int64, device=device
        )
    elif max_iterations.ndim == 0:
        max_iterations = max_iterations.expand(B)
    if not isinstance(inlier_threshold, torch.Tensor):
        inlier_threshold = torch.full(
            (B,), inlier_threshold, dtype=dtype, device=device
        )
    elif inlier_threshold.ndim == 0:
        inlier_threshold = inlier_threshold.expand(B)
    if not isinstance(min_inliers, torch.Tensor):
        min_inliers = torch.full(
            (B,), min_inliers, dtype=torch.int64, device=device
        )
    elif min_inliers.ndim == 0:
        min_inliers = min_inliers.expand(B)

    I = int(max_iterations.max())
    P_bs = L_bs * (L_bs - 1) // 2  # [B]
    idcs_random = sample_unique_pairs_batched(L_bs, max_L_b, I)  # [B, I, 2]
    arange_B = torch.arange(B, device=device)  # [B]

    best_intersection = torch.empty(B, 2, dtype=dtype, device=device)  # [B, 2]
    best_num_inliers = torch.zeros(B, dtype=torch.int64, device=device)  # [B]

    # This variable is used to stop the algorithm early for samples that have
    # already found a good intersection point.
    is_running = torch.ones(B, dtype=torch.bool, device=device)  # [B]

    for i in range(I):
        # Only select samples that haven't yet found a good intersection point.
        if not is_running.any():
            break
        samples_running = arange_B[is_running]  # [R]
        R = len(samples_running)
        arange_R = torch.arange(R, device=device)  # [R]
        L_rs = L_bs[samples_running]  # [R]
        max_L_r = int(L_rs.max())
        P_rs = P_bs[samples_running]  # [R]
        r_running = r[samples_running, :max_L_r]  # [R, max(L_r)]
        theta_running = theta[samples_running, :max_L_r]  # [R, max(L_r)]
        idcs_random_running = idcs_random[samples_running]  # [R, I, 2]
        best_num_inliers_running = best_num_inliers[samples_running]  # [R]
        max_iterations_running = max_iterations[samples_running]  # [R]
        inlier_threshold_running = inlier_threshold[samples_running]  # [R]
        min_inliers_running = min_inliers[samples_running]  # [R]

        # Randomly select two lines.
        idcs_random_i = idcs_random_running[:, i, :]  # [R, 2]
        r1 = r_running[arange_R, idcs_random_i[:, 0]]  # [R]
        r2 = r_running[arange_R, idcs_random_i[:, 1]]  # [R]
        theta1 = theta_running[arange_R, idcs_random_i[:, 0]]  # [R]
        theta2 = theta_running[arange_R, idcs_random_i[:, 1]]  # [R]

        # Calculate the intersection point between the two lines.
        intersections = line_intersection_batched(
            (r1, theta1), (r2, theta2)
        )  # [R, 2]

        # Count the number of lines that pass close to the intersection point.
        r_packed = pack_padded_batched(r_running, L_rs)  # [sum(L_r)]
        theta_packed = pack_padded_batched(theta_running, L_rs)  # [sum(L_r)]
        intersections_packed = intersections.repeat_interleave(
            L_rs, dim=0
        )  # [sum(L_r), 2]
        distances_packed = distance_line_to_point_batched(
            (r_packed, theta_packed), intersections_packed
        )  # [sum(L_r)]
        inlier_threshold_packed = inlier_threshold_running.repeat_interleave(
            L_rs, dim=0
        )  # [sum(L_r)]
        is_inlier = distances_packed <= inlier_threshold_packed  # [sum(L_r)]
        num_inliers = pad_packed_batched(is_inlier, L_rs, max_L_r).sum(
            dim=1
        )  # [R]

        # Update the best intersection point.
        update = num_inliers > best_num_inliers_running  # [R]
        samples_update = samples_running[update]  # [U]
        best_intersection[samples_update] = intersections[update]
        best_num_inliers[samples_update] = num_inliers[update]

        # Stop the algorithm early for samples that have already found a good
        # intersection point, whose pairs have been exhausted, or that have
        # reached the maximum number of iterations.
        is_running[samples_running] = ~(
            (num_inliers >= min_inliers_running)  # [R]
            | (i >= P_rs - 1)  # [R]
            | (i >= max_iterations_running - 1)  # [R]
        )

    return best_intersection
