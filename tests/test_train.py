import unittest
from unittest.mock import MagicMock, patch
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from app.models.train import ClassificationTrainer

class TestClassificationTrainer(unittest.TestCase):
    def setUp(self):
        
        self.ModelConfig = MagicMock(
            NUM_CATEGORIES=3, 
            INPUT_SIZE=4, 
            HIDDEN_DIM=5, 
            NUM_HIDDEN_LAYER=2, 
            DROPOUT=0.1, 
            LR=0.01, 
            SCHEDULER_GAMMA=0.95, 
            BATCH_SIZE=32, 
            EPOCHS=2,
            SEED=42,
            MODEL_LOC='./tests/resources/mlp.pkl',
            ENCODER_LOC='./tests/resources/encoded.npy',
            VALIDATION_RESULTS_LOC='./tests/resources/val_results_loc.csv'
        )
        self.df = pd.DataFrame({
            'x': list(np.random.rand(100, 4)),
            'y': list(np.random.randint(0, 2, size=(100, 1)))
        })
        self.df_wrong = pd.DataFrame({
            'a': list(np.random.rand(100, 4)),
            'b': list(np.random.randint(0, 2, size=(100, 1)))
        })
        np.save('./tests/resources/encoded.npy', ['class1', 'class2', 'class3'])

    def test_split_data_and_train(self):
        ct = ClassificationTrainer(self.ModelConfig)
        ct.train = MagicMock()
        ct.split_data_and_train(self.df)
        ct.train.assert_called_once()

        with self.assertRaises(KeyError):
            ct.split_data_and_train(self.df_wrong)

    @patch('torch.save')
    def test_train(self, mock_save):
        ct = ClassificationTrainer(self.ModelConfig)
        ct.scheduler = MagicMock()
        ct.scheduler.step = MagicMock()
        ct.optimizer.step = MagicMock()
        
        ct._train_step = MagicMock()
        ct._train_step.return_value = (torch.randn((1, 1)), torch.randn((1, 1)))
        
        ct._loss = MagicMock()
        ct.evaluate = MagicMock()
        
        x = torch.randn((100, 4))
        y = torch.randn((100, 1))
        
        train_data = TensorDataset(x, y)
        ct.train(train_data)
        
        ct._train_step.assert_called()
        mock_save.assert_called()

    @patch('sklearn.metrics')
    def test_evaluate(self, mock_metrics):
        ct = ClassificationTrainer(self.ModelConfig)
        mock_metrics.classification_report = MagicMock()
        mock_metrics.classification_report.return_value = {'text': 'value'}
        x = torch.randn((1, 4))
        y = torch.FloatTensor(np.array([[0,1,1]]))
        val_data = TensorDataset(x, y)
        ct.evaluate(val_data)  # Just checking for runtime errors

    def test_loss(self):
        ct = ClassificationTrainer(self.ModelConfig)
        x = torch.randn((4,))
        y = torch.randn((4,))
        loss = ct._loss(x, y)
        self.assertIsInstance(loss, torch.Tensor)

    def test_train_step(self):
        ct = ClassificationTrainer(self.ModelConfig)
        x = torch.randn((4,))
        y = torch.tensor([0.0, 0 ,0])
        ct._update_weights = MagicMock()

        preds, loss = ct._train_step(x, y)
        
        self.assertIsInstance(preds, torch.Tensor)
        self.assertIsInstance(loss, torch.Tensor)
        
        ct._update_weights.assert_called_once()




