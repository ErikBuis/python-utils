import unittest

from python_utils.modules.time import human_readable_time


class TestHumanReadableTime(unittest.TestCase):
    def test_shorter_than_one_minute(self):
        self.assertEqual(human_readable_time(1234567890), "1.23 s")
        self.assertEqual(
            human_readable_time(1234567890, significant_digits=2), "1.2 s"
        )
        self.assertEqual(
            human_readable_time(1234567890, significant_digits=1), "1 s"
        )
        self.assertEqual(human_readable_time(234567890), "235 ms")
        self.assertEqual(
            human_readable_time(234567890, significant_digits=2), "0.23 s"
        )
        self.assertEqual(
            human_readable_time(234567890, significant_digits=1), "0.2 s"
        )
        self.assertEqual(human_readable_time(34567890), "34.6 ms")
        self.assertEqual(
            human_readable_time(34567890, significant_digits=2), "35 ms"
        )
        self.assertEqual(
            human_readable_time(34567890, significant_digits=1), "0.03 s"
        )
        self.assertEqual(human_readable_time(12345678900), "12.3 s")
        self.assertEqual(
            human_readable_time(12345678900, significant_digits=2), "12 s"
        )
        self.assertEqual(
            human_readable_time(12345678900, significant_digits=1), "12 s"
        )
        self.assertEqual(human_readable_time(1295123456), "1.30 s")
        self.assertEqual(human_readable_time(1995123456), "2.00 s")
        self.assertEqual(human_readable_time(9995123456), "10.0 s")
        self.assertEqual(
            human_readable_time(1234567890, abbreviate=False), "1.23 seconds"
        )
        self.assertEqual(
            human_readable_time(234567890, abbreviate=False),
            "235 milliseconds",
        )

    def test_over_or_equal_to_one_minute(self):
        self.assertEqual(
            human_readable_time(
                (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000,
                significant_digits=3,
            ),
            "3h 14m 15s",
        )
        self.assertEqual(
            human_readable_time(
                (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000,
                significant_digits=3,
                abbreviate=False,
            ),
            "3 hours 14 minutes 15 seconds",
        )
        self.assertEqual(
            human_readable_time(
                (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000 + 23456789,
                significant_digits=5,
            ),
            "3h 14m 15s",
        )
        self.assertEqual(
            human_readable_time(
                (3 * 60 * 60 + 14 * 60 + 15) * 1_000_000_000 + 23456789,
                significant_digits=6,
            ),
            "3h 14m 15.2s",
        )
        self.assertEqual(
            human_readable_time(
                (3 * 60 * 60 + 7 * 60 + 15) * 1_000_000_000 + 23456789,
                significant_digits=6,
            ),
            "3h 7m 15.2s",
        )
        self.assertEqual(
            human_readable_time(
                4 * 365_242198790 * 24 * 60 * 60
                + (3 * 60 * 60 + 7 * 60 + 15) * 1_000_000_000
                + 23456789,
                significant_digits=10,
            ),
            "4y 3h 7m 15s",
        )
        self.assertEqual(
            human_readable_time(
                4 * 365_242198790 * 24 * 60 * 60
                + (3 * 60 * 60 + 7 * 60 + 15) * 1_000_000_000
                + 23456789,
                significant_digits=10,
                abbreviate=False,
            ),
            "4 years 3 hours 7 minutes 15 seconds",
        )
        self.assertEqual(
            human_readable_time(
                4 * 365_242198790 * 24 * 60 * 60
                + (3 * 60 * 60 + 7 * 60 + 15) * 1_000_000_000
                + 23456789,
                significant_digits=11,
            ),
            "4y 3h 7m 15.2s",
        )

    def test_significant_digits(self):
        self.assertRaises(
            ValueError, human_readable_time, 1234567890, significant_digits=0
        )
        self.assertRaises(
            ValueError, human_readable_time, 1234567890, significant_digits=-1
        )
        self.assertRaises(
            ValueError, human_readable_time, 1234567890, significant_digits=-2
        )
