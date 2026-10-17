"""Pruebas de las métricas comunes y de las features manuales."""

import unittest

import numpy as np
import pandas as pd

from src.bow import regression_metrics as bow_regression_metrics
from src.evaluation import regression_metrics
from src.features import manual_features


class EvaluationTests(unittest.TestCase):
    def test_mcrmse_is_mean_of_rmse(self):
        observed = np.array([[0.0, 0.0], [2.0, 4.0]])
        predicted = np.array([[1.0, 0.0], [1.0, 2.0]])
        metrics = regression_metrics(observed, predicted)
        self.assertAlmostEqual(metrics["rmse_content"], 1.0)
        self.assertAlmostEqual(metrics["rmse_wording"], np.sqrt(2.0))
        self.assertAlmostEqual(metrics["mcrmse"], (1.0 + np.sqrt(2.0)) / 2)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            regression_metrics(np.zeros((3, 2)), np.zeros((2, 2)))

    def test_bow_reexports_same_function(self):
        self.assertIs(bow_regression_metrics, regression_metrics)


class ManualFeaturesTests(unittest.TestCase):
    def test_values_and_index(self):
        summaries = pd.DataFrame(
            {"text": ["the cat sat", "a dog a dog"], "prompt_id": ["p1", "p2"]},
            index=[10, 20],
        )
        prompts = pd.DataFrame({"prompt_id": ["p1", "p2"],
                                "prompt_text": ["the cat sat on the mat", "cats only"]})
        features = manual_features(summaries, prompts)

        self.assertEqual(features.index.tolist(), [10, 20])
        self.assertEqual(features.loc[10, "token_count"], 3)
        self.assertEqual(features.loc[20, "unique_token_count"], 2)
        self.assertAlmostEqual(features.loc[20, "lexical_diversity"], 0.5)
        self.assertAlmostEqual(features.loc[10, "palabras_prompt_overlap"], 1.0)
        self.assertAlmostEqual(features.loc[20, "palabras_prompt_overlap"], 0.0)
        self.assertTrue(np.isfinite(features.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
