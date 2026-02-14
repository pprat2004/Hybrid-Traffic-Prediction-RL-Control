"""
LSTM-Based Traffic Flow Predictor
Predicts incoming traffic volume and patterns for proactive control
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from collections import deque
import pickle


class TrafficFlowPredictor:
    def __init__(self, sequence_length=30, prediction_horizon=10):
        """
        Args:
            sequence_length: Number of past timesteps to use for prediction
            prediction_horizon: How many steps ahead to predict (in seconds)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = self._build_model()
        self.history = {
            'edge_1si': deque(maxlen=sequence_length),
            'edge_2si': deque(maxlen=sequence_length),
            'edge_3si': deque(maxlen=sequence_length),
            'edge_4si': deque(maxlen=sequence_length),
        }
        self.scaler_params = None
        
    def _build_model(self):
        """Build LSTM prediction model"""
        # Traffic sequence input (vehicles over time for each edge)
        traffic_input = Input(shape=(self.sequence_length, 4), name='traffic_sequence')
        
        # Time features (hour of day, day of week, etc.)
        time_input = Input(shape=(4,), name='time_features')
        
        # LSTM layers for temporal patterns
        x = LSTM(128, return_sequences=True)(traffic_input)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        # Time feature processing
        t = Dense(32, activation='relu')(time_input)
        t = BatchNormalization()(t)
        
        # Combine temporal and time features
        combined = Concatenate()([x, t])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        
        # Output: predicted traffic for each edge
        output = Dense(4, activation='relu', name='predictions')(combined)
        
        model = Model(inputs=[traffic_input, time_input], outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def update_history(self, edge_counts, timestep):
        """
        Update traffic history with current observations
        
        Args:
            edge_counts: Dict with keys '1si', '2si', '3si', '4si' and vehicle counts
            timestep: Current simulation timestep
        """
        self.history['edge_1si'].append(edge_counts.get('1si', 0))
        self.history['edge_2si'].append(edge_counts.get('2si', 0))
        self.history['edge_3si'].append(edge_counts.get('3si', 0))
        self.history['edge_4si'].append(edge_counts.get('4si', 0))
    
    def get_time_features(self, timestep):
        """Extract time-based features"""
        # Simulate time of day (assuming 24-hour cycle)
        hour = (timestep % 86400) / 3600.0  # Seconds to hours
        
        return np.array([
            np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
            np.cos(2 * np.pi * hour / 24),
            1.0 if 7 <= hour < 9 or 17 <= hour < 19 else 0.0,  # Rush hour flag
            1.0 if 9 <= hour < 17 else 0.0,  # Business hours flag
        ])
    
    def predict_traffic(self, timestep):
        """
        Predict future traffic flow
        
        Returns:
            dict: Predicted vehicle counts for each edge
        """
        # Need enough history to predict
        if len(self.history['edge_1si']) < self.sequence_length:
            # Return current state if not enough history
            return {
                '1si': self.history['edge_1si'][-1] if self.history['edge_1si'] else 0,
                '2si': self.history['edge_2si'][-1] if self.history['edge_2si'] else 0,
                '3si': self.history['edge_3si'][-1] if self.history['edge_3si'] else 0,
                '4si': self.history['edge_4si'][-1] if self.history['edge_4si'] else 0,
            }
        
        # Prepare input sequence
        sequence = np.array([
            list(self.history['edge_1si']),
            list(self.history['edge_2si']),
            list(self.history['edge_3si']),
            list(self.history['edge_4si']),
        ]).T  # Shape: (sequence_length, 4)
        
        # Normalize
        if self.scaler_params is None:
            mean = np.mean(sequence, axis=0)
            std = np.std(sequence, axis=0) + 1e-8
            self.scaler_params = {'mean': mean, 'std': std}
        else:
            mean = self.scaler_params['mean']
            std = self.scaler_params['std']
        
        sequence_normalized = (sequence - mean) / std
        sequence_batch = sequence_normalized.reshape(1, self.sequence_length, 4)
        
        # Get time features
        time_features = self.get_time_features(timestep).reshape(1, -1)
        
        # Predict
        prediction_normalized = self.model.predict(
            [sequence_batch, time_features], 
            verbose=0
        )[0]
        
        # Denormalize
        prediction = prediction_normalized * std + mean
        prediction = np.maximum(prediction, 0)  # Ensure non-negative
        
        return {
            '1si': float(prediction[0]),
            '2si': float(prediction[1]),
            '3si': float(prediction[2]),
            '4si': float(prediction[3]),
        }
    
    def train_on_batch(self, sequences, time_features, targets):
        """
        Train predictor on a batch of historical data
        
        Args:
            sequences: Shape (batch, sequence_length, 4)
            time_features: Shape (batch, 4)
            targets: Shape (batch, 4)
        """
        return self.model.train_on_batch(
            [sequences, time_features],
            targets
        )
    
    def save(self, filepath):
        """Save model and history"""
        self.model.save(f"{filepath}_model.h5")
        with open(f"{filepath}_history.pkl", 'wb') as f:
            pickle.dump({
                'history': dict(self.history),
                'scaler_params': self.scaler_params
            }, f)
    
    def load(self, filepath):
        """Load model and history"""
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        try:
            with open(f"{filepath}_history.pkl", 'rb') as f:
                data = pickle.load(f)
                self.history = {k: deque(v, maxlen=self.sequence_length) 
                              for k, v in data['history'].items()}
                self.scaler_params = data['scaler_params']
        except FileNotFoundError:
            print("No history file found, starting fresh")


class PredictionBuffer:
    """Buffer to collect data for predictor training"""
    def __init__(self, max_size=10000):
        self.sequences = []
        self.time_features = []
        self.targets = []
        self.max_size = max_size
    
    def add(self, sequence, time_feat, target):
        """Add a training sample"""
        self.sequences.append(sequence)
        self.time_features.append(time_feat)
        self.targets.append(target)
        
        # Keep buffer from growing too large
        if len(self.sequences) > self.max_size:
            self.sequences.pop(0)
            self.time_features.pop(0)
            self.targets.pop(0)
    
    def get_batch(self, batch_size=32):
        """Get random batch for training"""
        if len(self.sequences) < batch_size:
            return None
        
        indices = np.random.choice(len(self.sequences), batch_size, replace=False)
        
        return (
            np.array([self.sequences[i] for i in indices]),
            np.array([self.time_features[i] for i in indices]),
            np.array([self.targets[i] for i in indices])
        )
    
    def clear(self):
        """Clear buffer"""
        self.sequences = []
        self.time_features = []
        self.targets = []


if __name__ == "__main__":
    # Test predictor
    predictor = TrafficFlowPredictor()
    
    # Simulate some traffic data
    for t in range(100):
        edge_counts = {
            '1si': np.random.randint(0, 20),
            '2si': np.random.randint(0, 20),
            '3si': np.random.randint(0, 20),
            '4si': np.random.randint(0, 20),
        }
        predictor.update_history(edge_counts, t)
        
        if t > 30:
            prediction = predictor.predict_traffic(t)
            print(f"Step {t}: Predicted traffic: {prediction}")
