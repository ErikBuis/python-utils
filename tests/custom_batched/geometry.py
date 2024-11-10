import unittest

import geopandas as gpd
import numpy as np
import torch
from shapely import MultiPolygon, Polygon

from python_utils.custom.geometry import xiaolin_wu_anti_aliasing
from python_utils.custom_batched.geometry import (
    xiaolin_wu_anti_aliasing_batched,
)


def generate_random_polygon() -> Polygon:
    # Generate a random Polygon.
    # Each Polygon has a 50% chance of having holes.
    # The amount of vertices is between 3 and 50.
    exterior = torch.rand(np.random.randint(3, 51), 2).tolist()
    interiors = []
    while np.random.rand() < 0.5:
        interiors.append(torch.rand(np.random.randint(3, 51), 2).tolist())
    return Polygon(exterior, interiors)


def generate_random_multipolygon() -> MultiPolygon:
    # Generate a random MultiPolygon.
    # Each MultiPolygon contain at least 2 Polygons. Extra Polygons are
    # continuously added with a 50% chance, until the addition fails.
    polygons_list = [generate_random_polygon(), generate_random_polygon()]
    while np.random.rand() < 0.5:
        polygons_list.append(generate_random_polygon())
    return MultiPolygon(polygons_list)


def generate_random_polygon_like() -> Polygon | MultiPolygon:
    # Generate a random Polygon or MultiPolygon.
    # Each has a 50% chance of occurring.
    if np.random.rand() < 0.5:
        return generate_random_polygon()
    else:
        return generate_random_multipolygon()


def generate_random_polygons(amount: int = 64) -> gpd.GeoSeries:
    # Generate a random GeoSeries of Polygon objects.
    polygons_list = []
    for _ in range(amount):
        polygons_list.append(generate_random_polygon())
    return gpd.GeoSeries(polygons_list)  # type: ignore


def generate_random_multipolygons(amount: int = 64) -> gpd.GeoSeries:
    # Generate a random GeoSeries of MultiPolygon objects.
    multipolygons_list = []
    for _ in range(amount):
        multipolygons_list.append(generate_random_multipolygon())
    return gpd.GeoSeries(multipolygons_list)  # type: ignore


def generate_random_polygon_likes(amount: int = 64) -> gpd.GeoSeries:
    # Generate a random GeoSeries of Polygon and MultiPolygon objects.
    # Each has a 50% chance of occurring for each object.
    polygon_likes_list = []
    for _ in range(amount):
        polygon_likes_list.append(generate_random_polygon_like())
    return gpd.GeoSeries(polygon_likes_list)  # type: ignore


class XiaolinWuAntiAliasingBatched(unittest.TestCase):
    # xiaolin_wu_anti_aliasing_batched should return the same values as the
    # xiaolin_wu_anti_aliasing function, but in a batched manner.
    def test_xiaolin_wu_anti_aliasing_batched(self):
        # Generate random line segments.
        x0s = torch.concatenate([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])
        y0s = torch.concatenate([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])
        x1s = torch.concatenate([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])
        y1s = torch.concatenate([
            torch.randint(-100, 101, (50,)) / 2,
            torch.rand(50, dtype=torch.float64) * 100 - 50,
        ])

        # Get the values from the sequential function.
        pixels_x_seq = []
        pixels_y_seq = []
        vals_seq = []
        for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
            pixels_x, pixels_y, vals = xiaolin_wu_anti_aliasing(
                x0.item(), y0.item(), x1.item(), y1.item()
            )
            pixels_x_seq.append(pixels_x)
            pixels_y_seq.append(pixels_y)
            vals_seq.append(vals)
        S_bs_seq = torch.tensor(list(map(len, vals_seq)))
        pixels_x_seq = torch.nn.utils.rnn.pad_sequence(
            pixels_x_seq, batch_first=True
        )
        pixels_y_seq = torch.nn.utils.rnn.pad_sequence(
            pixels_y_seq, batch_first=True
        )
        vals_seq = torch.nn.utils.rnn.pad_sequence(vals_seq, batch_first=True)

        # Get the values from the batched function.
        pixels_x_bat, pixels_y_bat, vals_bat, S_bs_bat = (
            xiaolin_wu_anti_aliasing_batched(x0s, y0s, x1s, y1s)
        )

        # Check if the values are the same.
        self.assertEqual(pixels_x_seq.shape, pixels_x_bat.shape)
        self.assertEqual(pixels_y_seq.shape, pixels_y_bat.shape)
        self.assertEqual(vals_seq.shape, vals_bat.shape)
        self.assertTrue(torch.allclose(S_bs_seq, S_bs_bat))
        self.assertTrue(torch.allclose(pixels_x_seq, pixels_x_bat))
        self.assertTrue(torch.allclose(pixels_y_seq, pixels_y_bat))
        self.assertTrue(torch.allclose(vals_seq, vals_bat))
