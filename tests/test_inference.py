import unittest
from unittest.mock import patch

from src import inference


class InferenceTests(unittest.TestCase):
    def tearDown(self):
        inference.get_pipeline.cache_clear()

    @patch("src.inference.pipeline")
    def test_get_pipeline_builds_model_once(self, pipeline):
        inference.get_pipeline.cache_clear()

        first = inference.get_pipeline()
        second = inference.get_pipeline()

        self.assertIs(first, second)
        pipeline.assert_called_once()


if __name__ == "__main__":
    unittest.main()
