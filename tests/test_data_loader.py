# tests/test_data_loader.py

import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml
import logging
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DataLoader from src.data.data_loader
from src.data.data_loader import DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Load configuration
        with open('src/configs/config.yaml') as f:
            config = yaml.safe_load(f)

        # Initialize DataLoader
        self.data_loader = DataLoader(config)
        self.seq_length = config['data']['seq_length']
        self.target_length = config['data']['target_length']

        # Create a small dummy dataset
        dates = pd.date_range('2018-01-01', periods=100, freq='H')
        data = pd.DataFrame({'PJME_MW': np.random.rand(len(dates))}, index=dates)

        # Preprocess data
        data_scaled = self.data_loader.preprocess_data(data)

        # Create sequences
        self.X, self.y = self.data_loader.create_sequences(data_scaled)

        # Initialize TimeSeriesDataset
        self.dataset = TimeSeriesDataset(self.X, self.y)

    def test_dataset_length(self):
        """Test if dataset length matches the number of sequences."""
        self.assertEqual(len(self.dataset), len(self.X))

    def test_dataset_item_shapes(self):
        """Test if each item in the dataset has the correct shape."""
        sequence, target = self.dataset[0]
        self.assertEqual(sequence.shape, (self.seq_length, 1))
        self.assertEqual(target.shape, (self.target_length, 1))

    def test_dataset_item_types(self):
        """Test if the dataset items are torch tensors."""
        sequence, target = self.dataset[0]
        self.assertIsInstance(sequence, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)

    def test_data_loader_methods(self):
        """Test DataLoader methods individually."""
        # Test load_data method
        data = self.data_loader.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('PJME_MW', data.columns)

        # Test preprocess_data method
        data_scaled = self.data_loader.preprocess_data(data)
        self.assertIsInstance(data_scaled, pd.DataFrame)
        self.assertTrue((data_scaled.values >= 0).all() and (data_scaled.values <= 1).all())

        # Test create_sequences method
        X, y = self.data_loader.create_sequences(data_scaled)
        self.assertEqual(X.shape[1:], (self.seq_length, 1))
        self.assertEqual(y.shape[1:], (self.target_length, 1))

        # Test split_data method
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.split_data(X, y)
        total_size = len(X)
        train_size = int(self.data_loader.train_split * total_size)
        val_size = int(self.data_loader.val_split * total_size)
        self.assertEqual(len(X_train), train_size)
        self.assertEqual(len(X_val), val_size)
        self.assertEqual(len(X_test), total_size - train_size - val_size)

if __name__ == '__main__':
    unittest.main()