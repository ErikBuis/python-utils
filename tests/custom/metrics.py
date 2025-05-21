from __future__ import annotations

import unittest

from python_utils.custom import metrics


class TestConfusionMatrixRandomModel(unittest.TestCase):
    def test_sum_to_1(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_random_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(TP + FN + FP + TN, 1)

    def test_frac_labels_pos(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_random_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(
                    (TP + FN) / (TP + FN + FP + TN), frac_labels_pos
                )

    def test_frac_preds_pos(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_random_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(
                    (TP + FP) / (TP + FN + FP + TN), frac_preds_pos
                )


class TestConfusionMatrixBestModel(unittest.TestCase):
    def test_sum_to_1(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_best_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(TP + FN + FP + TN, 1)

    def test_frac_labels_pos(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_best_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(
                    (TP + FN) / (TP + FN + FP + TN), frac_labels_pos
                )

    def test_frac_preds_pos(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_best_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(
                    (TP + FP) / (TP + FN + FP + TN), frac_preds_pos
                )


class TestConfusionMatrixWorstModel(unittest.TestCase):
    def test_sum_to_1(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_worst_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(TP + FN + FP + TN, 1)

    def test_frac_labels_pos(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_worst_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(
                    (TP + FN) / (TP + FN + FP + TN), frac_labels_pos
                )

    def test_frac_preds_pos(self) -> None:
        for i in range(100):
            for j in range(100):
                frac_labels_pos = i / 100
                frac_preds_pos = j / 100
                TP, FN, FP, TN = metrics.confusion_matrix_worst_model(
                    frac_labels_pos, frac_preds_pos
                )
                self.assertAlmostEqual(
                    (TP + FP) / (TP + FN + FP + TN), frac_preds_pos
                )
