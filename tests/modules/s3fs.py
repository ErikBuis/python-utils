from __future__ import annotations

import unittest

import s3fs
from typing_extensions import override

from python_utils.modules.s3fs import find


class TestFind(unittest.TestCase):
    s3: s3fs.S3FileSystem
    CROWNS_DIR = "s3://workflow-manager-ahn-workflow/SES101/polygons/"

    @classmethod
    @override
    def setUpClass(cls) -> None:
        try:
            cls.s3 = s3fs.S3FileSystem()
            cls.s3.ls(cls.CROWNS_DIR)
        except Exception as exc:
            raise unittest.SkipTest(f"S3 login failed: {exc}") from exc

    def test_find_maxdepth_none_withdirs_false_detail_false(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=None, withdirs=False, detail=False
        )
        actual = find(
            self.s3,
            self.CROWNS_DIR,
            maxdepth=None,
            withdirs=False,
            detail=False,
        )
        assert set(actual) == set(expected)

    def test_find_maxdepth_none_withdirs_true_detail_false(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=None, withdirs=True, detail=False
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=None, withdirs=True, detail=False
        )
        assert set(actual) == set(expected)

    def test_find_maxdepth_1_withdirs_false_detail_false(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=1, withdirs=False, detail=False
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=1, withdirs=False, detail=False
        )
        assert set(actual) == set(expected)

    def test_find_maxdepth_1_withdirs_true_detail_false(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=1, withdirs=True, detail=False
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=1, withdirs=True, detail=False
        )
        assert set(actual) == set(expected)

    def test_find_maxdepth_none_withdirs_false_detail_true(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=None, withdirs=False, detail=True
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=None, withdirs=False, detail=True
        )
        assert actual == expected

    def test_find_maxdepth_none_withdirs_true_detail_true(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=None, withdirs=True, detail=True
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=None, withdirs=True, detail=True
        )
        assert actual == expected

    def test_find_maxdepth_1_withdirs_false_detail_true(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=1, withdirs=False, detail=True
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=1, withdirs=False, detail=True
        )
        assert actual == expected

    def test_find_maxdepth_1_withdirs_true_detail_true(self) -> None:
        expected = self.s3.find(
            self.CROWNS_DIR, maxdepth=1, withdirs=True, detail=True
        )
        actual = find(
            self.s3, self.CROWNS_DIR, maxdepth=1, withdirs=True, detail=True
        )
        assert actual == expected
