from collections.abc import Callable
from typing import TypeVar

from rtree import index


T = TypeVar("T")


def find_overlapping_bboxes(bboxes: list[tuple[float, float, float, float]]) -> list[list[int]]:
    """Find overlapping bounding boxes.

    Amortized complexity in most real-world scenarios: O(n log n).
    Worst-case complexity: O(n^2).

    Args:
        bboxes: The bounding boxes to find overlaps in.
            Each bounding box is a tuple containing:
            - x1: The x-coordinate of the left side of the bounding box.
            - y1: The y-coordinate of the top side of the bounding box.
            - x2: The x-coordinate of the right side of the bounding box.
            - y2: The y-coordinate of the bottom side of the bounding box.

    Returns:
        For each bounding box, a list of indices of the bounding boxes that overlap with it.
        These indices are in no particular order, and include the index of the bounding box itself.
        The output could be seen as an adjacency list of the overlap graph.
    """
    spatial_idx = index.Index()
    for u, bbox in enumerate(bboxes):
        spatial_idx.insert(u, bbox)

    overlaps = []
    for u, bbox in enumerate(bboxes):
        overlaps.append(list(spatial_idx.intersection(bbox)))
    return overlaps


def find_overlapping_geometries(
    geometries: list[T],
    bbox_func: Callable[[T], tuple[float, float, float, float]],
    binary_overlap_func: Callable[[T, T], bool],
) -> list[list[int]]:
    """Find overlapping geometries.

    Amortized complexity in most real-world scenarios: O(n log n).
    Worst-case complexity: O(n^2).

    Args:
        geometries: The geometries to find overlaps in.
        bbox_func: A function that returns the bounding box of a geometry.
        binary_overlap_func: A function that returns True if two geometries overlap.
            The function is assumed to be symmetric, i.e.
            binary_overlap_func(a, b) == binary_overlap_func(b, a).

    Returns:
        For each geometry, a list of indices of the geometries that overlap with it.
        These indices are in no particular order, and include the index of the geometry itself.
        The output could be seen as an adjacency list of the overlap graph.
    """
    overlaps = find_overlapping_bboxes([bbox_func(geometry) for geometry in geometries])
    overlaps_filtered = [[] for _ in range(len(overlaps))]
    for u, nbrs in enumerate(overlaps):
        overlaps_filtered[u].append(u)
        for v in nbrs:
            if u < v and binary_overlap_func(geometries[u], geometries[v]):
                overlaps_filtered[u].append(v)
                overlaps_filtered[v].append(u)
    return overlaps_filtered


def find_overlapping_circles(circles: list[tuple[float, float, float]]) -> list[list[int]]:
    """Find overlapping circles.

    Amortized complexity in most real-world scenarios: O(n log n).
    Worst-case complexity: O(n^2).

    Args:
        circles: The circles to find overlaps in. Each circle is a tuple containing:
            - x: The x-coordinate of the center of the circle.
            - y: The y-coordinate of the center of the circle.
            - r: The radius of the circle.

    Returns:
        For each circle, a list of indices of the circles that overlap with it.
        These indices are in no particular order, and include the index of the circle itself.
        The output could be seen as an adjacency list of the overlap graph.
    """
    return find_overlapping_geometries(
        circles,
        lambda circle: (
            circle[0] - circle[2],
            circle[1] - circle[2],
            circle[0] + circle[2],
            circle[1] + circle[2],
        ),
        lambda circle1, circle2: (
            (circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2
            < (circle1[2] + circle2[2]) ** 2
        ),
    )
