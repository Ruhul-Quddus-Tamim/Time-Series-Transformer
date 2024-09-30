# tests/test_transformer_model.py

import unittest
import torch
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the TimeSeriesModel from src.models.transformer_model
from src.models.transformer_model import TimeSeriesModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        # Model parameters
        self.seq_length = 24
        self.target_length = 5
        self.d_model = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1

        # Initialize the model
        self.model = TimeSeriesModel(
            seq_length=self.seq_length,
            target_length=self.target_length,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def test_model_forward_pass(self):
        """Test the forward pass of the model."""
        batch_size = 16
        src = torch.randn(batch_size, self.seq_length, 1)
        tgt = torch.randn(batch_size, self.target_length, 1)

        # Generate target mask
        tgt_mask = self.model.generate_square_subsequent_mask(self.target_length)

        # Perform forward pass
        output = self.model(src, tgt, tgt_mask=tgt_mask)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.target_length, 1))

if __name__ == '__main__':
    unittest.main()