import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.data_loader import DataLoader as CustomDataLoader
from src.models.transformer_model import TimeSeriesModel
from src.utils.utils import TimeSeriesDataset, create_tgt_mask, load_config, setup_logging
from tqdm import tqdm
import logging
import os
import sys
import numpy as np

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
        X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_data()

        # Create datasets and loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

        # Initialize model
        model = TimeSeriesModel(
            seq_length=config['data']['seq_length'],
            target_length=config['data']['target_length'],
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)
        logger.info("Model initialized")

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        # Initialize variables to track best validation loss
        best_val_loss = float('inf')
        tolerance = config['training']['tolerance']

        # Training loop
        for epoch in range(config['training']['epochs']):
            # Training phase
            model.train()
            running_loss = 0.0
            running_mae = 0.0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Training")
            for sequences, targets in train_bar:
                sequences = sequences.to(device)
                targets = targets.to(device)

                # Prepare decoder input
                start_token = torch.zeros((targets.size(0), 1, targets.size(2)), device=device)
                decoder_inputs = torch.cat([start_token, targets[:, :-1, :]], dim=1)

                # Generate target mask
                tgt_mask = create_tgt_mask(config['data']['target_length']).to(device)

                optimizer.zero_grad()
                outputs = model(sequences, decoder_inputs, tgt_mask=tgt_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * sequences.size(0)

                # Compute MAE for the batch
                mae = torch.abs(outputs - targets).mean()
                running_mae += mae.item() * sequences.size(0)

                train_bar.set_postfix(loss=loss.item(), mae=mae.item())

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_mae = running_mae / len(train_loader.dataset)
            logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Training Loss: {epoch_loss:.6f}, MAE: {epoch_mae:.6f}")

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_mae = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Validation")
                for sequences, targets in val_bar:
                    sequences = sequences.to(device)
                    targets = targets.to(device)

                    # Prepare decoder input
                    start_token = torch.zeros((targets.size(0), 1, targets.size(2)), device=device)
                    decoder_inputs = torch.cat([start_token, targets[:, :-1, :]], dim=1)

                    # Generate target mask
                    tgt_mask = create_tgt_mask(config['data']['target_length']).to(device)

                    outputs = model(sequences, decoder_inputs, tgt_mask=tgt_mask)
                    loss = criterion(outputs, targets)

                    val_running_loss += loss.item() * sequences.size(0)

                    # Compute MAE for the batch
                    mae = torch.abs(outputs - targets).mean()
                    val_running_mae += mae.item() * sequences.size(0)

                    val_bar.set_postfix(loss=loss.item(), mae=mae.item())

            val_loss = val_running_loss / len(val_loader.dataset)
            val_mae = val_running_mae / len(val_loader.dataset)
            logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}")

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(config['paths']['model_save_path'], exist_ok=True)
                model_save_path = os.path.join(config['paths']['model_save_path'], 'best_model.pth')
                torch.save({'state_dict': model.state_dict()}, model_save_path)
                logger.info(f"Model saved at {model_save_path}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
