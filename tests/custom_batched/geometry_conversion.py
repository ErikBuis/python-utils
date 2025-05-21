from __future__ import annotations

import unittest

import geopandas as gpd
import torch
from shapely import MultiPolygon, Polygon

from python_utils.custom_batched.geometry_conversion import (
    MultiPolygon2MultiPolygonVertices,
    MultiPolygons2MultiPolygonsVertices,
    MultiPolygonsVertices2MultiPolygons,
    MultiPolygonVertices2MultiPolygon,
    Polygon2PolygonVertices,
    PolygonLike2PolygonLikeVertices,
    PolygonLikes2PolygonLikesVertices,
    PolygonLikesVertices2PolygonLikes,
    PolygonLikeVertices2PolygonLike,
    Polygons2PolygonsVertices,
    PolygonsVertices2Polygons,
    PolygonVertices2Polygon,
    generate_random_multipolygon,
    generate_random_multipolygons,
    generate_random_polygon,
    generate_random_polygon_like,
    generate_random_polygon_likes,
    generate_random_polygons,
)


class TestGenerateRandomPolygon(unittest.TestCase):
    def test_generate_random_polygon(self) -> None:
        for _ in range(16):
            polygon = generate_random_polygon()
            self.assertIsInstance(polygon, Polygon)
            self.assertTrue(polygon.is_valid)
            self.assertTrue(polygon.is_simple)
            self.assertFalse(polygon.is_empty)


class TestGenerateRandomPolygons(unittest.TestCase):
    def test_generate_random_polygons(self) -> None:
        polygons = generate_random_polygons(amount=16)
        self.assertIsInstance(polygons, gpd.GeoSeries)
        self.assertEqual(len(polygons), 16)
        for polygon in polygons:
            self.assertIsInstance(polygon, Polygon)
            self.assertTrue(polygon.is_valid)
            self.assertTrue(polygon.is_simple)
            self.assertFalse(polygon.is_empty)


class TestGenerateRandomMultiPolygon(unittest.TestCase):
    def test_generate_random_multipolygon(self) -> None:
        for _ in range(16):
            multipolygon = generate_random_multipolygon()
            self.assertIsInstance(multipolygon, MultiPolygon)
            self.assertTrue(multipolygon.is_valid)
            self.assertTrue(multipolygon.is_simple)
            self.assertFalse(multipolygon.is_empty)


class TestGenerateRandomMultiPolygons(unittest.TestCase):
    def test_generate_random_multipolygons(self) -> None:
        multipolygons = generate_random_multipolygons(amount=16)
        self.assertIsInstance(multipolygons, gpd.GeoSeries)
        self.assertEqual(len(multipolygons), 16)
        for multipolygon in multipolygons:
            self.assertIsInstance(multipolygon, MultiPolygon)
            self.assertTrue(multipolygon.is_valid)
            self.assertTrue(multipolygon.is_simple)
            self.assertFalse(multipolygon.is_empty)


class TestPolygonVertices(unittest.TestCase):
    def test_polygon_vertices(self) -> None:
        for _ in range(16):
            polygon = generate_random_polygon()
            polygon_vertices = Polygon2PolygonVertices(
                polygon, dtype=torch.float64
            )
            polygon2 = PolygonVertices2Polygon(polygon_vertices)
            self.assertTrue(polygon.equals_exact(polygon2, 1e-6))


class TestPolygonsVertices(unittest.TestCase):
    def test_polygons_vertices(self) -> None:
        polygons = generate_random_polygons(amount=16)
        polygons_vertices = Polygons2PolygonsVertices(
            polygons, dtype=torch.float64
        )
        polygons2 = PolygonsVertices2Polygons(polygons_vertices)
        self.assertTrue(polygons.geom_equals_exact(polygons2, 1e-6).all())


class TestMultiPolygonVertices(unittest.TestCase):
    def test_multipolygon_vertices(self) -> None:
        for _ in range(16):
            multipolygon = generate_random_multipolygon()
            multipolygon_vertices = MultiPolygon2MultiPolygonVertices(
                multipolygon, dtype=torch.float64
            )
            multipolygon2 = MultiPolygonVertices2MultiPolygon(
                multipolygon_vertices
            )
            self.assertTrue(multipolygon.equals_exact(multipolygon2, 1e-6))


class TestMultiPolygonsVertices(unittest.TestCase):
    def test_multipolygons_vertices(self) -> None:
        multipolygons = generate_random_multipolygons(amount=16)
        multipolygons_vertices = MultiPolygons2MultiPolygonsVertices(
            multipolygons, dtype=torch.float64
        )
        multipolygons2 = MultiPolygonsVertices2MultiPolygons(
            multipolygons_vertices
        )
        self.assertTrue(
            multipolygons.geom_equals_exact(multipolygons2, 1e-6).all()
        )


class TestPolygonLike(unittest.TestCase):
    def test_polygon_like(self) -> None:
        for _ in range(16):
            polygon_like = generate_random_polygon_like()
            polygon_vertices = PolygonLike2PolygonLikeVertices(
                polygon_like, dtype=torch.float64
            )
            polygon_like2 = PolygonLikeVertices2PolygonLike(polygon_vertices)
            self.assertTrue(polygon_like.equals_exact(polygon_like2, 1e-6))


class TestPolygonLikes(unittest.TestCase):
    def test_polygon_likes(self) -> None:
        polygon_likes = generate_random_polygon_likes(amount=16)
        polygon_vertices = PolygonLikes2PolygonLikesVertices(
            polygon_likes, dtype=torch.float64
        )
        polygon_likes2 = PolygonLikesVertices2PolygonLikes(polygon_vertices)
        self.assertTrue(
            polygon_likes.geom_equals_exact(polygon_likes2, 1e-6).all()
        )
