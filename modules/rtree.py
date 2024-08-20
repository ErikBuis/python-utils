from collections.abc import Callable
from typing import TypeVar

from rtree import index


T = TypeVar("T")


def __find_overlapping_geometries_bruteforce(
    geometries: list[T], binary_overlap_func: Callable[[T, T], bool]
) -> list[list[int]]:
    """Find overlapping geometries using a brute-force approach.

    Amortized complexity: O(n^2).

    Args:
        geometries: The geometries to find overlaps in.
        binary_overlap_func: A function that returns True if two geometries overlap.

    Returns:
        For each geometry, a list of indices of the geometries that overlap with it (in no particular order).
        The output could be seen as an adjacency list of the overlap graph.
    """
    overlaps = [[] for _ in geometries]
    for i, geometry in enumerate(geometries):
        for j, other_geometry in enumerate(geometries[i + 1 :], start=i + 1):
            if binary_overlap_func(geometry, other_geometry):
                overlaps[i].append(j)
                overlaps[j].append(i)
    return overlaps


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
        For each bounding box, a list of indices of the bounding boxes that overlap with it (in no particular order).
        The output could be seen as an adjacency list of the overlap graph.
    """
    # Create a spatial index.
    spatial_idx = index.Index()
    for i, bbox in enumerate(bboxes):
        spatial_idx.insert(i, bbox)

    # Find overlaps.
    overlaps = []
    for i, bbox in enumerate(bboxes):
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

    Returns:
        For each geometry, a list of indices of the geometries that overlap with it (in no particular order).
        The output could be seen as an adjacency list of the overlap graph.
    """
    return [
        group
        for group_bboxes in find_overlapping_bboxes(
            [bbox_func(geometry) for geometry in geometries]
        )
        for group in __find_overlapping_geometries_bruteforce(
            [geometries[i] for i in group_bboxes], binary_overlap_func
        )
    ]


def find_overlapping_circles(circles: list[tuple[float, float, float]]) -> list[list[int]]:
    """Find overlapping circles.

    Args:
        circles: The circles to find overlaps in.
            Each circle is a tuple containing:
            - x: The x-coordinate of the center of the circle.
            - y: The y-coordinate of the center of the circle.
            - r: The radius of the circle.

    Returns:
        For each circle, a list of indices of the circles that overlap with it (in no particular order).
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
