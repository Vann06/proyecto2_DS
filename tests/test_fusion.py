"""Pruebas de la fusión mínima empaquetada para la aplicación."""

import io
import unittest

import joblib
import numpy as np
import pandas as pd

from src.fusion import FEATURE_COLUMNS, build_features, predict_frame, predict_scores, train_fusion


class FakeDoc:
    def __init__(self, text):
        # Vector determinista según las letras del texto, para no depender del modelo de spaCy en las pruebas
        counts = np.array([text.lower().count(letter) for letter in "aeiost"], dtype=float)
        self.vector = counts + 1.0


class FakeNlp:
    def pipe(self, texts, batch_size=256):
        return (FakeDoc(text) for text in texts)


SMALL_PARAMS = {target: dict(n_estimators=5, num_leaves=3, min_child_samples=1) for target in ["content", "wording"]}


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.nlp = FakeNlp()
        self.prompts = pd.DataFrame({"prompt_id": ["p1", "p2"],
                                     "prompt_text": ["the cat sat on the mat", "dogs run in the park all day"]})
        self.summaries = pd.DataFrame({
            "prompt_id": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "text": ["the cat sat", "dogs run", "a cat on a mat", "the park", "cat", "dogs run all day in the park"],
        }, index=[10, 11, 12, 13, 14, 15])
        self.y = np.array([[0.1, 0.2], [0.3, -0.1], [0.5, 0.4], [-0.2, 0.0], [-1.0, -0.8], [0.9, 0.7]])
        features = build_features(self.summaries, self.prompts, self.nlp)
        self.bundle = train_fusion(features, self.y, SMALL_PARAMS)

    def test_features_columns_index_and_similarity(self):
        features = build_features(self.summaries, self.prompts, self.nlp)
        self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(features.index.tolist(), self.summaries.index.tolist())
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertTrue(((features["cos_sim"] >= -1) & (features["cos_sim"] <= 1 + 1e-9)).all())

    def test_serialized_bundle_preserves_predictions(self):
        buffer = io.BytesIO()
        joblib.dump(self.bundle, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        expected = predict_frame(self.bundle, self.summaries, self.prompts, self.nlp)
        actual = predict_frame(restored, self.summaries, self.prompts, self.nlp)
        pd.testing.assert_frame_equal(expected, actual)

    def test_single_summary_matches_batch(self):
        batch = predict_frame(self.bundle, self.summaries.iloc[[1]], self.prompts, self.nlp)
        single = predict_scores(self.bundle, "dogs run", "dogs run in the park all day", self.nlp)
        self.assertAlmostEqual(single["content"], batch["content"].iloc[0])
        self.assertAlmostEqual(single["wording"], batch["wording"].iloc[0])

    def test_new_prompt_needs_no_retraining(self):
        prompts = pd.DataFrame({"prompt_id": ["nuevo"], "prompt_text": ["plants use sunlight to make food"]})
        summaries = pd.DataFrame({"prompt_id": ["nuevo"], "text": ["plants make food with light"]})
        result = predict_frame(self.bundle, summaries, prompts, self.nlp)
        self.assertTrue(np.isfinite(result[["content", "wording"]].to_numpy()).all())

    def test_length_warning_outside_training_range(self):
        low, high = self.bundle["training"]["token_count_range"]
        result = predict_frame(self.bundle, self.summaries, self.prompts, self.nlp)
        self.assertEqual((low, high), (1, 7))
        self.assertTrue(result["advertencia"].isna().all())
        long_text = " ".join(["word"] * (high + 1))
        self.assertIsNotNone(predict_scores(self.bundle, long_text, "word source", self.nlp)["advertencia"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            predict_scores(self.bundle, "   ", "source text", self.nlp)
        with self.assertRaises(ValueError):
            predict_frame(self.bundle, pd.DataFrame({"prompt_id": ["p9"], "text": ["a cat"]}), self.prompts, self.nlp)
        with self.assertRaises(ValueError):
            build_features(self.summaries.drop(columns="text"), self.prompts, self.nlp)


if __name__ == "__main__":
    unittest.main()
