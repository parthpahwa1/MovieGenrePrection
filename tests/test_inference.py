import unittest
from unittest.mock import patch, Mock, MagicMock
from app.models.inference import ClassificationInference
import torch
import numpy as np

class TestClassificationInference(unittest.TestCase):
    def setUp(self):
        self.ModelConfig = Mock()
        self.ModelConfig.NUM_CATEGORIES = 33
        self.ModelConfig.INPUT_SIZE = 384
        self.ModelConfig.HIDDEN_DIM = 256
        self.ModelConfig.NUM_HIDDEN_LAYER = 3
        self.ModelConfig.DROPOUT = 0.2
        self.ModelConfig.MODEL_LOC = "./tests/resources/mlp.pkl"
        self.ModelConfig.ENCODER_LOC = "./tests/resources/encoded.npy"
        self.ModelConfig.PRE_TRAINED_MODEL = "all-MiniLM-L6-v2"


    def test_forward(self):
        ci = ClassificationInference(self.ModelConfig)
        ci.data_processor = MagicMock()
        ci.model.forward = MagicMock()
        input_ = "Some text"
        
        ci.forward(input_)
        ci.data_processor.sentence_transformer.encode.assert_called_once_with([input_])


    def test_get_predictions(self):
        ci = ClassificationInference(self.ModelConfig)
        ci.data_processor = MagicMock()
        
        ci.model.forward = MagicMock()
        ci.model.forward.return_value = torch.FloatTensor(np.array([[0.9]]))
        ci.encoder.inverse_transform = Mock()
        input_ = "Some text"
        ci.get_predictions(input_)
        ci.data_processor.sentence_transformer.encode.assert_called_once_with([input_])
        ci.encoder.inverse_transform.assert_called_once()


if __name__ == "__main__":
    unittest.main()
