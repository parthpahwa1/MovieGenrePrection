import unittest
from unittest.mock import patch, Mock, MagicMock
from app.models.inference import ClassificationInference
import torch
import numpy as np

class TestClassificationInference(unittest.TestCase):
    def setUp(self):
        self.ModelConfig = Mock()
        self.ModelConfig.NUM_CATEGORIES = 10
        self.ModelConfig.INPUT_SIZE = 300
        self.ModelConfig.HIDDEN_DIM = 100
        self.ModelConfig.NUM_HIDDEN_LAYER = 2
        self.ModelConfig.DROPOUT = 0.2
        self.ModelConfig.MODEL_LOC = "./tests/resources/mlp.pkl"
        self.ModelConfig.ENCODER_LOC = "./tests/resources/encoded.npy"
        self.ModelConfig.PRE_TRAINED_MODEL = "all-MiniLM-L6-v2"

    @patch.object(ClassificationInference, 'model', return_value=MagicMock())
    @patch.object(ClassificationInference, 'data_processor', return_value=MagicMock())
    def test_forward(self, mock_data_processor, mock_model):
        ci = ClassificationInference(self.ModelConfig)
        ci.data_processor = mock_data_processor        
        
        input_ = "Some text"
        
        ci.forward(input_)
        ci.data_processor.sentence_transformer.encode.assert_called_once_with([input_])

    @patch.object(ClassificationInference, 'model', return_value=MagicMock())
    @patch.object(ClassificationInference, 'data_processor', return_value=MagicMock())
    def test_get_predictions(self, mock_data_processor, mock_model):
        ci = ClassificationInference(self.ModelConfig)
        ci.data_processor = mock_data_processor
        
        mock_model.return_value.sigmoid.return_value.detach.return_value.numpy.return_value = np.ones((1,10)) * 0.5
        ci.encoder.inverse_transform = Mock()
        input_ = "Some text"
        ci.get_predictions(input_)
        mock_data_processor.sentence_transformer.encode.assert_called_once_with([input_])
        ci.encoder.inverse_transform.assert_called_once()


if __name__ == "__main__":
    unittest.main()
