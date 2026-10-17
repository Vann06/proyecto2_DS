"""Pruebas de separación de vocabulario, similitud y predicción."""

import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from src.tfidf import (
    SummaryTfidfFeatures, build_tfidf_vectorizer,
    paired_cosine_similarity, predict_scores,
    transform_prompt_texts,
)


class TfidfTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "text": ["student idea idea", "summary evidence", "student summary detail"],
            "prompt_text": ["idea secretprompt", "evidence secretprompt", "detail secretprompt"],
        })

    def test_vocabulary_excludes_prompts_and_validation(self):
        transformer = clone(SummaryTfidfFeatures(include_similarity=True)).fit(self.frame)
        vocabulary = transformer.vectorizer_.vocabulary_.copy()
        held_out = pd.DataFrame({"text": ["unseenword idea"], "prompt_text": ["secretprompt idea"]})
        result = transformer.transform(held_out)
        self.assertNotIn("secretprompt", vocabulary)
        self.assertNotIn("unseenword", vocabulary)
        self.assertEqual(vocabulary, transformer.vectorizer_.vocabulary_)
        self.assertEqual(result.shape, (1, len(vocabulary) + 1))

    def test_paired_cosine_and_zero_vectors(self):
        vectorizer = build_tfidf_vectorizer().fit(self.frame["text"])
        summaries = vectorizer.transform(["idea student", "summary", "unknown"])
        prompts = vectorizer.transform(["idea", "evidence", "unknown"])
        expected = np.diag(cosine_similarity(summaries, prompts))
        actual = paired_cosine_similarity(summaries, prompts)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual[-1], 0)
        self.assertTrue(np.isfinite(actual).all())

    def test_serialized_pipeline_preserves_predictions(self):
        model = Pipeline([
            ("features", SummaryTfidfFeatures(include_similarity=True)),
            ("model", Ridge(alpha=1, solver="lsqr")),
        ]).fit(self.frame, np.array([[0, 1], [1, 0], [0.5, 0.5]]))
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_allclose(model.predict(self.frame), restored.predict(self.frame))
        scores = predict_scores(restored, "student idea", "idea detail")
        self.assertEqual(set(scores), {"content", "wording"})

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            SummaryTfidfFeatures().fit(pd.DataFrame({"text": [" "], "prompt_text": ["idea"]}))

    def test_repeated_sources_preserve_row_order(self):
        vectorizer = build_tfidf_vectorizer().fit(self.frame["text"])
        sources = pd.Series(["summary evidence", "student idea", "summary evidence", "unknown"])
        expected = vectorizer.transform(sources)
        actual = transform_prompt_texts(vectorizer, sources)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual.toarray(), expected.toarray())


if __name__ == "__main__":
    unittest.main()
