"""
Emergency-Aware Deep Q-Network Agent
Multi-objective optimization: efficiency, fairness, emergency response
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from collections import deque
import random


class EmergencyAwareDQNAgent:
    def __init__(self, emergency_priority=True):
        """
        Args:
            emergency_priority: Whether to prioritize emergency vehicles
        """
        self.gamma = 0.95
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.memory = deque(maxlen=10000)
        self.emergency_memory = deque(maxlen=2000)  # Separate buffer for emergency scenarios
        self.action_size = 2
        self.emergency_priority = emergency_priority
        
        # Build dual models for stable learning
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Performance tracking
        self.metrics = {
            'avg_waiting_time': [],
            'emergency_response_time': [],
            'throughput': [],
            'fairness_index': []
        }
    
    def _build_model(self):
        """Build enhanced neural network with emergency awareness"""
        # Position matrix input (12x12 grid showing vehicle positions)
        input_position = Input(shape=(12, 12, 1), name='position')
        x1 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(input_position)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Flatten()(x1)
        
        # Velocity matrix input
        input_velocity = Input(shape=(12, 12, 1), name='velocity')
        x2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(input_velocity)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(64, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Flatten()(x2)
        
        # Traffic light state
        input_light = Input(shape=(2, 1), name='light_state')
        x3 = Flatten()(input_light)
        
        # Queue lengths per edge
        input_queue = Input(shape=(4,), name='queue_lengths')
        
        # Average speeds per edge
        input_speed = Input(shape=(4,), name='avg_speeds')
        
        # Emergency vehicle flags per edge
        input_emergency = Input(shape=(4,), name='emergency_flags')
        
        # Predicted future traffic (from predictor)
        input_prediction = Input(shape=(4,), name='predicted_traffic')
        
        # Waiting time per edge
        input_waiting = Input(shape=(4,), name='waiting_times')
        
        # Concatenate all features
        x = Concatenate()([x1, x2, x3, input_queue, input_speed, 
                          input_emergency, input_prediction, input_waiting])
        
        # Deep network with residual connections
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        residual = x
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = keras.layers.add([x, residual])  # Residual connection
        x = BatchNormalization()(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Dueling DQN architecture
        # Value stream
        value = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear', name='value')(value)
        
        # Advantage stream
        advantage = Dense(64, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear', name='advantage')(advantage)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = keras.layers.Lambda(
            lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
            name='q_values'
        )([value, advantage])
        
        model = Model(
            inputs=[input_position, input_velocity, input_light, input_queue, 
                   input_speed, input_emergency, input_prediction, input_waiting],
            outputs=q_values
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber'  # More robust than MSE
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done, is_emergency=False):
        """Store experience in replay memory"""
        experience = (state, action, reward, next_state, done)
        
        if is_emergency and self.emergency_priority:
            self.emergency_memory.append(experience)
        else:
            self.memory.append(experience)
    
    def act(self, state, emergency_present=False):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state (list of numpy arrays)
            emergency_present: Whether emergency vehicle is present
            
        Returns:
            action: 0 or 1 (traffic light phase choice)
        """
        # Force emergency-friendly action if emergency vehicle present
        if emergency_present and self.emergency_priority:
            # Predict Q-values
            q_values = self.model.predict(state, verbose=0)[0]
            
            # Get emergency flags from state
            emergency_flags = state[5][0]  # Shape: (4,)
            
            # If emergency on edges 1si or 2si, prefer action 0 (horizontal green)
            # If emergency on edges 3si or 4si, prefer action 1 (vertical green)
            if emergency_flags[0] > 0 or emergency_flags[1] > 0:
                return 0  # Horizontal green
            elif emergency_flags[2] > 0 or emergency_flags[3] > 0:
                return 1  # Vertical green
        
        # Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploit: choose best action
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=64):
        """Train on batch from experience replay"""
        # Sample from both normal and emergency memories
        normal_size = int(batch_size * 0.7)
        emergency_size = batch_size - normal_size
        
        minibatch = []
        
        # Sample from normal memory
        if len(self.memory) >= normal_size:
            minibatch.extend(random.sample(self.memory, normal_size))
        elif len(self.memory) > 0:
            minibatch.extend(random.sample(self.memory, len(self.memory)))
        
        # Sample from emergency memory (prioritized)
        if len(self.emergency_memory) >= emergency_size:
            minibatch.extend(random.sample(self.emergency_memory, emergency_size))
        elif len(self.emergency_memory) > 0:
            minibatch.extend(random.sample(self.emergency_memory, len(self.emergency_memory)))
        
        if len(minibatch) == 0:
            return
        
        # Prepare batch
        states = [[] for _ in range(8)]  # 8 inputs
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            # Calculate target Q-value using Double DQN
            if done:
                target = reward
            else:
                # Use online network to select action
                next_q_online = self.model.predict(next_state, verbose=0)[0]
                best_action = np.argmax(next_q_online)
                
                # Use target network to evaluate action
                next_q_target = self.target_model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * next_q_target[best_action]
            
            # Get current Q-values
            target_f = self.model.predict(state, verbose=0)[0]
            target_f[action] = target
            
            # Append to batch
            for i in range(8):
                states[i].append(state[i][0])
            targets.append(target_f)
        
        # Convert to numpy arrays
        states = [np.array(s) for s in states]
        targets = np.array(targets)
        
        # Train
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=len(minibatch))
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def calculate_multi_objective_reward(self, metrics):
        """
        Calculate reward balancing multiple objectives
        
        Args:
            metrics: Dict with keys:
                - throughput: Number of vehicles that passed
                - waiting_time: Total waiting time
                - emergency_delay: Delay for emergency vehicles
                - queue_balance: Standard deviation of queue lengths
                
        Returns:
            reward: Scalar reward value
        """
        # Weights for different objectives
        w_throughput = 0.3
        w_waiting = 0.25
        w_emergency = 0.35
        w_fairness = 0.1
        
        # Normalize and combine
        reward = (
            w_throughput * metrics.get('throughput', 0) -
            w_waiting * metrics.get('waiting_time', 0) / 100.0 -
            w_emergency * metrics.get('emergency_delay', 0) * 10.0 -  # High penalty
            w_fairness * metrics.get('queue_balance', 0)
        )
        
        return reward
    
    def save(self, filepath):
        """Save model weights"""
        self.model.save_weights(f"{filepath}_online.h5")
        self.target_model.save_weights(f"{filepath}_target.h5")
        
        # Save metrics
        np.save(f"{filepath}_metrics.npy", self.metrics)
    
    def load(self, filepath):
        """Load model weights"""
        try:
            self.model.load_weights(f"{filepath}_online.h5")
            self.target_model.load_weights(f"{filepath}_target.h5")
            self.metrics = np.load(f"{filepath}_metrics.npy", allow_pickle=True).item()
            print(f"Models loaded from {filepath}")
        except Exception as e:
            print(f"Could not load models: {e}")


if __name__ == "__main__":
    # Test agent
    agent = EmergencyAwareDQNAgent()
    
    # Create dummy state
    state = [
        np.random.rand(1, 12, 12, 1),  # position
        np.random.rand(1, 12, 12, 1),  # velocity
        np.array([[0, 1]]).reshape(1, 2, 1),  # light
        np.random.rand(1, 4),  # queue
        np.random.rand(1, 4),  # speed
        np.array([[1, 0, 0, 0]]),  # emergency (on edge 1si)
        np.random.rand(1, 4),  # prediction
        np.random.rand(1, 4),  # waiting
    ]
    
    action = agent.act(state, emergency_present=True)
    print(f"Chosen action: {action}")
    print(f"Expected: 0 (horizontal green) because emergency on edge 1si")
