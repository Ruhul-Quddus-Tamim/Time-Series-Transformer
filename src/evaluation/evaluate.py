# src/evaluation/evaluate.py

import torch
from torch.utils.data import DataLoader
from src.data.data_loader import DataLoader as CustomDataLoader
from src.models.transformer_model import TimeSeriesModel
from src.utils.utils import TimeSeriesDataset, create_tgt_mask, load_config, setup_logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # Load configuration
    config = load_config()

    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    try:
        # Set device (MPS for Apple Silicon, CUDA, or CPU)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

        # Load training data for reference
        data_loader = CustomDataLoader(config)
        X_train, y_train, _, _, X_test, y_test = data_loader.get_data()

        # Flatten training data for statistical tests
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])

        # Simulate new data (you can replace this with actual new data)
        # For demonstration, let's assume X_test is our new data
        X_new = X_test
        X_new_flat = X_new.reshape(-1, X_new.shape[-1])

        # Data Drift Detection
        logger.info("Starting data drift detection...")
        drift_detected = False
        p_values = []
        alpha = 0.05  # Significance level

        """
        1. Uses the Kolmogorov-Smirnov test to compare feature distributions.
	    2. Check whether drift is detected and the p-values for each feature.
        """
        
        for i in range(X_train_flat.shape[1]):
            stat, p_value = ks_2samp(X_train_flat[:, i], X_new_flat[:, i])
            p_values.append(p_value)
            if p_value < alpha:
                drift_detected = True
                logger.warning(f"Data drift detected in feature {i} (p-value={p_value:.4f})")
            else:
                logger.info(f"No data drift in feature {i} (p-value={p_value:.4f})")

        if drift_detected:
            logger.warning("Data drift detected!")
        else:
            logger.info("No data drift detected.")

        # Concept Drift Detection
        logger.info("Starting concept drift detection...")

        # Create test dataset and dataloader for new data
        test_dataset = TimeSeriesDataset(X_new, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        # Initialize model with parameters from config
        model = TimeSeriesModel(
            seq_length=config['data']['seq_length'],
            target_length=config['data']['target_length'],
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)

        # Load the trained model checkpoint
        model_load_path = os.path.join(config['paths']['model_save_path'], 'best_model.pth')
        if not os.path.isfile(model_load_path):
            logger.error(f"Model checkpoint not found at {model_load_path}")
            sys.exit(1)

        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {model_load_path}")

        # Set model to evaluation mode
        model.eval()
        predictions = []
        actuals = []

        # Evaluation loop
        test_bar = tqdm(test_loader, desc="Evaluating")
        with torch.no_grad():
            for sequences, targets in test_bar:
                sequences = sequences.to(device)
                targets = targets.to(device)

                # Prepare decoder input
                start_token = torch.zeros((targets.size(0), 1, targets.size(2)), device=device)
                decoder_inputs = torch.cat([start_token, targets[:, :-1, :]], dim=1)

                # Generate target mask
                tgt_mask = create_tgt_mask(config['data']['target_length']).to(device)

                # Make predictions
                outputs = model(sequences, decoder_inputs, tgt_mask=tgt_mask)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        # Concatenate all batches and reshape
        predictions = np.concatenate(predictions, axis=0).reshape(-1)
        actuals = np.concatenate(actuals, axis=0).reshape(-1)

        # Inverse transform the predictions and actuals to original scale
        scaler = data_loader.scaler
        predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        # Compute evaluation metrics
        test_mse = mean_squared_error(actuals_inv, predictions_inv)
        test_mae = mean_absolute_error(actuals_inv, predictions_inv)
        test_rmse = np.sqrt(test_mse)

        # Use the baseline MSE and drift threshold
        baseline_mse = 49874.90625  # Your baseline MSE
        drift_threshold = 5000       # Adjust this threshold as needed

        # Concept Drift Detection based on performance degradation
        performance_drop = test_mse - baseline_mse
        if performance_drop > drift_threshold:
            logger.warning(f"Concept drift detected! MSE increased by {performance_drop:.6f}")
            concept_drift_detected = True
        else:
            logger.info("No concept drift detected.")
            concept_drift_detected = False

        # Log the evaluation metrics
        logger.info(f"Test MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

        # Save drift detection results
        results_save_path = os.path.join(config['paths']['results_save_path'], 'drift_detection_results.txt')
        os.makedirs(config['paths']['results_save_path'], exist_ok=True)
        with open(results_save_path, 'w') as f:
            f.write(f"Data Drift Detected: {drift_detected}\n")
            f.write(f"Concept Drift Detected: {concept_drift_detected}\n")
            f.write(f"Baseline MSE: {baseline_mse:.6f}\n")
            f.write(f"Current MSE: {test_mse:.6f}\n")
            f.write(f"Performance Drop: {performance_drop:.6f}\n")

        logger.info(f"Drift detection results saved to {results_save_path}")

        # Plot true values vs. predictions
        plt.figure(figsize=(12, 6))
        plt.plot(actuals_inv[:100], label='True Values')
        plt.plot(predictions_inv[:100], label='Predictions')
        plt.title('True Values vs. Predictions')
        plt.xlabel('Samples')
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()

        # Save the plot to a file
        plot_save_path = os.path.join(config['paths']['plot_save_path'], 'drift_detection_plot.png')
        os.makedirs(config['paths']['plot_save_path'], exist_ok=True)
        plt.savefig(plot_save_path)
        logger.info(f"Plot saved to {plot_save_path}")
        plt.close()

    except Exception as e:
        logger.error(f"An error occurred during drift detection: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()