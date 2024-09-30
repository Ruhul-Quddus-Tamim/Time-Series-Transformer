# tests/test_drift_detection.py

import unittest
import numpy as np
import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules from your project
from src.evaluation.evaluate import main as drift_detection_main
from src.utils.utils import load_config, create_tgt_mask
from src.data.data_loader import DataLoader as CustomDataLoader
from src.models.transformer_model import TimeSeriesModel
from src.utils.utils import TimeSeriesDataset

class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        # Load configuration
        self.config = load_config()
        
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        
        # Initialize DataLoader
        self.data_loader = CustomDataLoader(self.config)
        
        # Load training data
        self.X_train, self.y_train, _, _, _, _ = self.data_loader.get_data()
        
        # Flatten training data for statistical tests
        self.X_train_flat = self.X_train.reshape(-1, self.X_train.shape[-1])
        
        # Baseline MSE from previous evaluation
        self.baseline_mse = 49874.90625
        
        # Drift threshold
        self.drift_threshold = 5000

    def test_no_drift(self):
        """Test drift detection when there is no drift."""
        # Simulate new data without drift
        X_new = self.X_train.copy()
        X_new_flat = X_new.reshape(-1, X_new.shape[-1])
        
        # Data Drift Detection
        drift_detected = False
        alpha = 0.05
        for i in range(self.X_train_flat.shape[1]):
            stat, p_value = ks_2samp(self.X_train_flat[:, i], X_new_flat[:, i])
            if p_value < alpha:
                drift_detected = True
                break
        self.assertFalse(drift_detected, "Data drift incorrectly detected when there should be none.")
        
        # Concept Drift Detection
        # Evaluate model
        device = torch.device("cpu")
        model = TimeSeriesModel(
            seq_length=self.config['data']['seq_length'],
            target_length=self.config['data']['target_length'],
            d_model=self.config['model']['d_model'],
            num_heads=self.config['model']['num_heads'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(device)
        
        # Load model checkpoint
        model_load_path = os.path.join(self.config['paths']['model_save_path'], 'best_model.pth')
        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Create test dataset and loader
        test_dataset = TimeSeriesDataset(X_new, self.y_train)
        test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        
        # Evaluate model
        test_mse = self.evaluate_model(model, test_loader, device)
        performance_drop = test_mse - self.baseline_mse
        concept_drift_detected = performance_drop > self.drift_threshold
        
        self.assertFalse(concept_drift_detected, "Concept drift incorrectly detected when there should be none.")

    def test_data_drift(self):
        """Test drift detection when data drift is present."""
        # Simulate new data with data drift
        X_new = self.X_train + np.random.normal(0, 1, self.X_train.shape)
        X_new_flat = X_new.reshape(-1, X_new.shape[-1])
        
        # Data Drift Detection
        drift_detected = False
        alpha = 0.05
        for i in range(self.X_train_flat.shape[1]):
            stat, p_value = ks_2samp(self.X_train_flat[:, i], X_new_flat[:, i])
            if p_value < alpha:
                drift_detected = True
                break
        self.assertTrue(drift_detected, "Data drift not detected when it should be.")

    def test_concept_drift(self):
        """Test drift detection when concept drift is present."""
        # Simulate concept drift by altering targets
        y_new = self.y_train + np.random.normal(0, 1, self.y_train.shape)
        
        # Create test dataset and loader
        test_dataset = TimeSeriesDataset(self.X_train, y_new)
        test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        
        # Evaluate model
        device = torch.device("cpu")
        model = TimeSeriesModel(
            seq_length=self.config['data']['seq_length'],
            target_length=self.config['data']['target_length'],
            d_model=self.config['model']['d_model'],
            num_heads=self.config['model']['num_heads'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout']
        ).to(device)
        
        # Load model checkpoint
        model_load_path = os.path.join(self.config['paths']['model_save_path'], 'best_model.pth')
        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Evaluate model
        test_mse = self.evaluate_model(model, test_loader, device)
        performance_drop = test_mse - self.baseline_mse
        concept_drift_detected = performance_drop > self.drift_threshold
        
        self.assertTrue(concept_drift_detected, "Concept drift not detected when it should be.")

    def evaluate_model(self, model, data_loader, device):
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                # Prepare decoder input
                start_token = torch.zeros((targets.size(0), 1, targets.size(2)), device=device)
                decoder_inputs = torch.cat([start_token, targets[:, :-1, :]], dim=1)
                
                # Generate target mask
                tgt_mask = create_tgt_mask(self.config['data']['target_length']).to(device)
                
                # Make predictions
                outputs = model(sequences, decoder_inputs, tgt_mask=tgt_mask)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())
        
        # Concatenate all batches and reshape
        predictions = np.concatenate(predictions, axis=0).reshape(-1)
        actuals = np.concatenate(actuals, axis=0).reshape(-1)
        
        # Inverse transform the predictions and actuals to original scale
        scaler = self.data_loader.scaler
        predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # Compute evaluation metrics
        test_mse = mean_squared_error(actuals_inv, predictions_inv)
        return test_mse

if __name__ == '__main__':
    unittest.main()