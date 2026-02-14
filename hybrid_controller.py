"""
Hybrid Traffic Controller
Integrates LSTM prediction with Emergency-Aware DQN
"""

import numpy as np
import traci
from traffic_predictor import TrafficFlowPredictor, PredictionBuffer
from emergency_aware_agent import EmergencyAwareDQNAgent


class HybridTrafficController:
    def __init__(self, junction_id='0'):
        """
        Args:
            junction_id: SUMO junction ID to control
        """
        self.junction_id = junction_id
        self.predictor = TrafficFlowPredictor(sequence_length=30, prediction_horizon=10)
        self.agent = EmergencyAwareDQNAgent(emergency_priority=True)
        self.prediction_buffer = PredictionBuffer(max_size=10000)
        
        # Performance tracking
        self.episode_metrics = {
            'total_waiting_time': 0,
            'emergency_response_times': [],
            'throughput': 0,
            'phase_changes': 0,
            'emergency_count': 0
        }
        
        # State tracking
        self.current_phase = 0
        self.phase_duration = 0
        self.min_phase_duration = 5
        self.max_phase_duration = 30
        
        # Emergency vehicle tracking
        self.emergency_vehicles = {}
        self.emergency_types = ['ambulance', 'firetruck', 'police']
    
    def reset_episode_metrics(self):
        """Reset metrics for new episode"""
        self.episode_metrics = {
            'total_waiting_time': 0,
            'emergency_response_times': [],
            'throughput': 0,
            'phase_changes': 0,
            'emergency_count': 0
        }
    
    def get_state(self, timestep):
        """
        Get comprehensive state representation
        
        Returns:
            list: State components for agent
        """
        # Grid-based position and velocity matrices (from original code)
        positionMatrix = np.zeros((12, 12))
        velocityMatrix = np.zeros((12, 12))
        
        cellLength = 7
        offset = 11
        speedLimit = 14
        
        junctionPosition = traci.junction.getPosition(self.junction_id)[0]
        
        edges = ['1si', '2si', '3si', '4si']
        edge_data = {}
        
        for edge_idx, edge in enumerate(edges):
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            edge_data[edge] = {
                'count': len(vehicles),
                'halting': 0,
                'avg_speed': 0,
                'has_emergency': False,
                'waiting_time': 0
            }
            
            speeds = []
            for v in vehicles:
                # Check if emergency vehicle
                vtype = traci.vehicle.getTypeID(v)
                if vtype in self.emergency_types:
                    edge_data[edge]['has_emergency'] = True
                    
                    # Track emergency vehicle
                    if v not in self.emergency_vehicles:
                        depart_time = traci.vehicle.getDeparture(v)
                        self.emergency_vehicles[v] = {
                            'spawn_time': depart_time if depart_time >= 0 else timestep,
                            'edge': edge
                        }
                
                # Position in grid
                pos = traci.vehicle.getPosition(v)
                if edge in ['1si', '2si']:
                    ind = int(abs((junctionPosition - pos[0] - offset)) / cellLength)
                    lane_offset = [2, 1, 0][traci.vehicle.getLaneIndex(v)] if edge == '1si' else [11, 10, 9][traci.vehicle.getLaneIndex(v)]
                else:
                    ind = int(abs((junctionPosition - pos[1] - offset)) / cellLength)
                    lane_offset = [6, 7, 8][traci.vehicle.getLaneIndex(v)] if edge == '3si' else [3, 4, 5][traci.vehicle.getLaneIndex(v)]
                
                if ind < 12:
                    if edge == '1si':
                        positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                        velocityMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
                    elif edge == '2si':
                        positionMatrix[11 - (2 - traci.vehicle.getLaneIndex(v))][ind] = 1
                        velocityMatrix[11 - (2 - traci.vehicle.getLaneIndex(v))][ind] = traci.vehicle.getSpeed(v) / speedLimit
                    elif edge == '3si':
                        positionMatrix[ind][2 - traci.vehicle.getLaneIndex(v)] = 1
                        velocityMatrix[ind][2 - traci.vehicle.getLaneIndex(v)] = traci.vehicle.getSpeed(v) / speedLimit
                    elif edge == '4si':
                        positionMatrix[11 - ind][11 - (2 - traci.vehicle.getLaneIndex(v))] = 1
                        velocityMatrix[11 - ind][11 - (2 - traci.vehicle.getLaneIndex(v))] = traci.vehicle.getSpeed(v) / speedLimit
                
                # Speed and waiting time
                speed = traci.vehicle.getSpeed(v)
                speeds.append(speed)
                
                if speed < 0.1:
                    edge_data[edge]['halting'] += 1
                
                edge_data[edge]['waiting_time'] += traci.vehicle.getWaitingTime(v)
            
            edge_data[edge]['avg_speed'] = np.mean(speeds) if speeds else 0
        
        # Update predictor history
        edge_counts = {edge: edge_data[edge]['count'] for edge in edges}
        self.predictor.update_history(edge_counts, timestep)
        
        # Get prediction
        prediction = self.predictor.predict_traffic(timestep)
        
        # Current traffic light state
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        light_state = [1, 0] if current_phase in [0, 2, 4] else [0, 1]
        
        # Prepare state for agent
        state = [
            positionMatrix.reshape(1, 12, 12, 1),  # Position
            velocityMatrix.reshape(1, 12, 12, 1),  # Velocity
            np.array(light_state).reshape(1, 2, 1),  # Light state
            np.array([edge_data[e]['count'] for e in edges]).reshape(1, 4),  # Queue lengths
            np.array([edge_data[e]['avg_speed'] for e in edges]).reshape(1, 4),  # Avg speeds
            np.array([1.0 if edge_data[e]['has_emergency'] else 0.0 for e in edges]).reshape(1, 4),  # Emergency flags
            np.array([prediction[e] for e in edges]).reshape(1, 4),  # Predicted traffic
            np.array([edge_data[e]['waiting_time'] for e in edges]).reshape(1, 4),  # Waiting times
        ]
        
        return state, edge_data
    
    def calculate_reward(self, edge_data, action, emergency_present, current_timestep):
        """
        Calculate multi-objective reward
        
        Args:
            edge_data: Dictionary with edge statistics
            action: Action taken (0 or 1)
            emergency_present: Whether emergency vehicle is present
            
        Returns:
            reward: Scalar reward value
        """
        edges = ['1si', '2si', '3si', '4si']
        
        # Throughput: vehicles moving through
        throughput = sum([edge_data[e]['count'] for e in edges])
        
        # Waiting time penalty
        total_waiting = sum([edge_data[e]['waiting_time'] for e in edges])
        
        # Emergency delay penalty (very high cost)
        emergency_delay = 0
        if emergency_present:
            for edge in edges:
                if edge_data[edge]['has_emergency']:
                    # Penalize if emergency vehicle is waiting
                    emergency_delay += edge_data[edge]['halting'] * 100  # High penalty
        
        # Queue balance (fairness)
        queue_lengths = [edge_data[e]['count'] for e in edges]
        queue_balance = np.std(queue_lengths) if len(queue_lengths) > 0 else 0
        
        # Calculate composite reward
        metrics = {
            'throughput': throughput,
            'waiting_time': total_waiting,
            'emergency_delay': emergency_delay,
            'queue_balance': queue_balance
        }
        
        reward = self.agent.calculate_multi_objective_reward(metrics)
        
        # Bonus for completing emergency vehicle passage
        for vid, info in list(self.emergency_vehicles.items()):
            if vid not in traci.vehicle.getIDList():
                # Emergency vehicle completed journey
                response_time = current_timestep - info['spawn_time']
                self.episode_metrics['emergency_response_times'].append(response_time)
                reward += 50  # Bonus for completing emergency vehicle
                del self.emergency_vehicles[vid]
        
        return reward, metrics
    
    def execute_action(self, action, current_light_state):
        """
        Execute traffic light action with proper phase transitions
        
        Args:
            action: 0 (horizontal green) or 1 (vertical green)
            current_light_state: Current light state [h, v]
        """
        current_phase = traci.trafficlight.getPhase(self.junction_id)
        
        # Action 0: Horizontal green (phases 0, 2, 4)
        # Action 1: Vertical green (phases 6, 0)
        
        if action == 0 and current_light_state[0] == 0:  # Want horizontal, currently vertical
            # Transition: yellow -> all red -> horizontal green
            for _ in range(6):  # Yellow phase
                traci.trafficlight.setPhase(self.junction_id, 1)
                yield 1
            for _ in range(10):  # All red
                traci.trafficlight.setPhase(self.junction_id, 2)
                yield 1
            for _ in range(6):  # Yellow other direction
                traci.trafficlight.setPhase(self.junction_id, 3)
                yield 1
            # Now set horizontal green
            traci.trafficlight.setPhase(self.junction_id, 4)
            self.phase_duration = 0
            self.episode_metrics['phase_changes'] += 1
            
        elif action == 0 and current_light_state[0] == 1:  # Want horizontal, already horizontal
            # Just continue
            traci.trafficlight.setPhase(self.junction_id, 4)
            
        elif action == 1 and current_light_state[0] == 0:  # Want vertical, already vertical
            # Just continue
            traci.trafficlight.setPhase(self.junction_id, 0)
            
        elif action == 1 and current_light_state[0] == 1:  # Want vertical, currently horizontal
            # Transition: yellow -> all red -> vertical green
            for _ in range(6):  # Yellow phase
                traci.trafficlight.setPhase(self.junction_id, 5)
                yield 1
            for _ in range(10):  # All red
                traci.trafficlight.setPhase(self.junction_id, 6)
                yield 1
            for _ in range(6):  # Yellow other direction
                traci.trafficlight.setPhase(self.junction_id, 7)
                yield 1
            # Now set vertical green
            traci.trafficlight.setPhase(self.junction_id, 0)
            self.phase_duration = 0
            self.episode_metrics['phase_changes'] += 1
    
    def step(self, timestep):
        """
        Execute one control step
        
        Returns:
            dict: Step information
        """
        # Get current state
        state, edge_data = self.get_state(timestep)
        
        # Check for emergency vehicles
        emergency_present = any([edge_data[e]['has_emergency'] for e in ['1si', '2si', '3si', '4si']])
        if emergency_present:
            self.episode_metrics['emergency_count'] += 1
        
        # Decide action
        action = self.agent.act(state, emergency_present)
        
        # Execute action (this advances simulation)
        current_light_state = state[2][0].flatten()[:2]  # [h, v]
        transition_generator = self.execute_action(action, current_light_state)
        
        steps_taken = 0
        for step in transition_generator:
            traci.simulationStep()
            steps_taken += step
            
            # Update metrics during transition
            for edge in ['1si', '2si', '3si', '4si']:
                self.episode_metrics['total_waiting_time'] += traci.edge.getLastStepHaltingNumber(edge)
        
        # Maintain minimum phase duration
        for _ in range(self.min_phase_duration):
            traci.simulationStep()
            steps_taken += 1
            for edge in ['1si', '2si', '3si', '4si']:
                self.episode_metrics['total_waiting_time'] += traci.edge.getLastStepHaltingNumber(edge)
        
        # Get next state
        next_state, next_edge_data = self.get_state(timestep + steps_taken)
        
        # Calculate reward
        reward, metrics = self.calculate_reward(next_edge_data, action, emergency_present, timestep + steps_taken)
        
        # Store experience
        done = traci.simulation.getMinExpectedNumber() == 0
        self.agent.remember(state, action, reward, next_state, done, is_emergency=emergency_present)
        
        return {
            'state': next_state,
            'reward': reward,
            'done': done,
            'steps': steps_taken,
            'metrics': metrics,
            'emergency_present': emergency_present
        }
    
    def train_predictor(self, batch_size=32):
        """Train traffic predictor on collected data"""
        batch = self.prediction_buffer.get_batch(batch_size)
        if batch is not None:
            sequences, time_features, targets = batch
            loss = self.predictor.train_on_batch(sequences, time_features, targets)
            return loss
        return None
    
    def save_models(self, filepath_prefix):
        """Save all models"""
        self.agent.save(f"{filepath_prefix}_agent")
        self.predictor.save(f"{filepath_prefix}_predictor")
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix):
        """Load all models"""
        self.agent.load(f"{filepath_prefix}_agent")
        self.predictor.load(f"{filepath_prefix}_predictor")
        print(f"Models loaded with prefix: {filepath_prefix}")


if __name__ == "__main__":
    print("Hybrid Traffic Controller module loaded successfully")
    print("Use main.py to run the full simulation")
