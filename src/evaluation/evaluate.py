import torch
from torch.utils.data import DataLoader
from src.data.data_loader import DataLoader as CustomDataLoader
from src.models.transformer_model import TimeSeriesModel
from src.utils.utils import TimeSeriesDataset, create_tgt_mask, load_config, setup_logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt

def main():
    # Load configuration
    config = load_config()

    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load data
        data_loader = CustomDataLoader(config)
        X_train, y_train, X_test, y_test = data_loader.get_data()

        # Create datasets and loaders
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        # Initialize model
        model = TimeSeriesModel(
            seq_length=config['data']['seq_length'],
            target_length=config['data']['target_length'],
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)

        # Load model checkpoint
        model_load_path = os.path.join(config['paths']['model_save_path'], 'best_model.pth')
        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {model_load_path}")

        # Set model to evaluation mode
        model.eval()
        predictions = []
        actuals = []
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

                outputs = model(sequences, decoder_inputs, tgt_mask=tgt_mask)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())

        # Concatenate all batches
        predictions = np.concatenate(predictions, axis=0).reshape(-1)
        actuals = np.concatenate(actuals, axis=0).reshape(-1)

        # Inverse transform the predictions and actuals
        scaler = data_loader.scaler
        predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_inv = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        # Compute Metrics
        test_mse = mean_squared_error(actuals, predictions)
        test_mae = mean_absolute_error(actuals, predictions)
        test_rmse = np.sqrt(test_mse)

        tolerance = config['training']['tolerance']
        error = np.abs(predictions - actuals)
        accuracy = np.mean(error <= tolerance)

        logger.info(f"Test MSE (normalized): {test_mse:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, Accuracy: {accuracy*100:.2f}%")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(actuals_inv[:100], label='True Values')
        plt.plot(predictions_inv[:100], label='Predictions')
        plt.title('True Values vs. Predictions')
        plt.xlabel('Samples')
        plt.ylabel('Energy Consumption (MW)')
        plt.legend()
        plt.show()

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
