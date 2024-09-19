import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
import logging
import os

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_path = config['data']['raw_data_path']
        self.seq_length = config['data']['seq_length']
        self.target_length = config['data']['target_length']
        self.train_split = config['data']['train_split']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            data = pd.read_csv(self.data_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
            data.sort_index(inplace=True)
            data.fillna(method='ffill', inplace=True)
            self.logger.info("Data loaded successfully")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, data):
        try:
            self.logger.info("Starting data preprocessing")
            data_scaled = self.scaler.fit_transform(data[['PJME_MW']])
            data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=['PJME_MW'])
            self.logger.info("Data preprocessing completed")
            return data_scaled
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise

    def create_sequences(self, data_scaled):
        try:
            self.logger.info("Creating sequences")
            X, y = [], []
            data_array = data_scaled.values
            total_length = len(data_scaled)
            seq_length = self.seq_length
            target_length = self.target_length

            for i in range(total_length - seq_length - target_length + 1):
                X.append(data_array[i:i + seq_length])
                y.append(data_array[i + seq_length:i + seq_length + target_length])
            self.logger.info("Sequences created successfully")
            return np.array(X), np.array(y)
        except Exception as e:
            self.logger.error(f"Error creating sequences: {e}")
            raise

    def split_data(self, X, y):
        try:
            self.logger.info("Splitting data into training and testing sets")
            train_size = int(self.train_split * len(X))
            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]
            self.logger.info("Data splitting completed")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise

    def get_data(self):
        data = self.load_data()
        data_scaled = self.preprocess_data(data)
        X, y = self.create_sequences(data_scaled)
        return self.split_data(X, y)
