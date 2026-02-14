"""
Main Training Script for Hybrid Emergency-Aware Traffic Control
"""

import tensorflow as tf

# GPU Configuration - MUST BE FIRST
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        print(f"  Using: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠ GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - will use CPU (training will be slower)")

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traci
from sumolib import checkBinary

# Import our modules
from generate_routes import EnhancedRouteGenerator
from hybrid_controller import HybridTrafficController


class TrainingManager:
    def __init__(self, config_file="cross3ltl.sumocfg", gui=False):
        """
        Args:
            config_file: Path to SUMO configuration file
            gui: Whether to use SUMO GUI
        """
        self.config_file = config_file
        self.gui = gui
        
        # Setup directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Training history
        self.training_history = {
            'episode': [],
            'total_waiting_time': [],
            'avg_waiting_time': [],
            'emergency_response_time': [],
            'throughput': [],
            'phase_changes': [],
            'loss': [],
            'epsilon': [],
            'emergency_count': []
        }
        
    def run_episode(self, controller, episode_num, emergency_rate=0.05):
     """Run a single training episode"""

     try:
         # Generate routes with emergency vehicles
         route_gen = EnhancedRouteGenerator(
             emergency_rate=emergency_rate,
             rush_hour=True
         )
         total_vehicles, emergency_vehicles = route_gen.generate_routes(
             "input_routes.rou.xml",
             duration=3600
         )

         # Start SUMO
         sumo_binary = checkBinary('sumo-gui') if self.gui else checkBinary('sumo')
         traci.start([sumo_binary, "-c", self.config_file, '--start', '--quit-on-end'])

         # Initialize
         controller.reset_episode_metrics()
         traci.trafficlight.setPhase("0", 0)
         traci.trafficlight.setPhaseDuration("0", 200)

         timestep = 0
         total_reward = 0
         losses = []

         print(f"\nEpisode {episode_num}: {total_vehicles} vehicles ({emergency_vehicles} emergency)")

         # Main simulation loop
         while traci.simulation.getMinExpectedNumber() > 0 and timestep < 7200:
             # Controller step
             step_info = controller.step(timestep)

             total_reward += step_info['reward']
             timestep += step_info['steps']

             # Train agent every 32 steps
             if timestep % 32 == 0 and len(controller.agent.memory) > 128:
                 loss = controller.agent.replay(batch_size=128)
                 if loss is not None:
                     losses.append(loss)

             # Update target network every 100 steps
             if timestep % 100 == 0:
                 controller.agent.update_target_model()

             # Train predictor every 50 steps
             if timestep % 50 == 0:
                 controller.train_predictor(batch_size=32)

             # Progress indicator
             if timestep % 500 == 0:
                 print(f"  Step {timestep}, Reward: {total_reward:.1f}, "
                       f"Epsilon: {controller.agent.epsilon:.3f}, "
                       f"Emergency: {controller.episode_metrics['emergency_count']}")

         # Close SUMO
         try:
             if traci.isLoaded():
                 traci.close(wait=False)
         except:
             pass
            
         # Calculate statistics
         metrics = controller.episode_metrics
         avg_emergency_response = (np.mean(metrics['emergency_response_times']) 
                                   if metrics['emergency_response_times'] else 0)

         episode_stats = {
             'total_waiting_time': metrics['total_waiting_time'],
             'avg_waiting_time': metrics['total_waiting_time'] / max(total_vehicles, 1),
             'emergency_response_time': avg_emergency_response,
             'throughput': total_vehicles,
             'phase_changes': metrics['phase_changes'],
             'loss': np.mean(losses) if losses else 0,
             'epsilon': controller.agent.epsilon,
             'emergency_count': metrics['emergency_count']
         }

         return episode_stats

     except Exception as e:
         print(f"⚠ Episode {episode_num} crashed: {e}")
         print(f"  Continuing to next episode...")

         # Try to close SUMO if still running
         try:
             if traci.isLoaded():
                 traci.close(wait=False)
         except:
             pass
            
         # Return dummy stats so training can continue
         return {
             'total_waiting_time': 999999,
             'avg_waiting_time': 999,
             'emergency_response_time': 999,
             'throughput': 0,
             'phase_changes': 0,
             'loss': 0,
             'epsilon': controller.agent.epsilon,
             'emergency_count': 0
         }
    
    def train(self, episodes=500, emergency_rate=0.05, save_interval=50):
        """
        Main training loop
        
        Args:
            episodes: Number of episodes to train
            emergency_rate: Emergency vehicle generation rate
            save_interval: Save models every N episodes
        """
        controller = HybridTrafficController(junction_id='0')
        
        # Try to load existing models
        try:
            controller.load_models("models/hybrid_controller")
            print("Loaded existing models, continuing training...")
        except:
            print("Starting training from scratch...")
        
        print(f"\n{'='*60}")
        print(f"Training Hybrid Emergency-Aware Traffic Controller")
        print(f"Episodes: {episodes}, Emergency Rate: {emergency_rate*100}%")
        print(f"{'='*60}\n")
        
        for episode in range(episodes):
            # Run episode
            stats = self.run_episode(controller, episode, emergency_rate)
            
            # Store history
            self.training_history['episode'].append(episode)
            for key, value in stats.items():
                self.training_history[key].append(value)

            
            # Print summary
            print(f"\nEpisode {episode} Summary:")
            print(f"  Total Waiting Time: {stats['total_waiting_time']:,.0f}")
            print(f"  Avg Waiting Time: {stats['avg_waiting_time']:.2f}")
            print(f"  Avg Emergency Response: {stats['emergency_response_time']:.2f}s")
            print(f"  Phase Changes: {stats['phase_changes']}")
            print(f"  Loss: {stats['loss']:.4f}")
            
            # Save models periodically
            if (episode + 1) % save_interval == 0:
                controller.save_models(f"models/hybrid_controller_ep{episode}")
                self.save_training_history()
                self.plot_training_progress()
                print(f"  → Models saved at episode {episode}")
        
        # Save final models
        controller.save_models("models/hybrid_controller_final")
        self.save_training_history()
        self.plot_training_progress()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}\n")
    
    def save_training_history(self):
        """Save training history to CSV"""
        df = pd.DataFrame(self.training_history)
        df.to_csv("logs/training_history.csv", index=False)
    
    def plot_training_progress(self):
        """Create training progress plots"""
        if len(self.training_history['episode']) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress - Hybrid Emergency-Aware Traffic Control', 
                     fontsize=16, fontweight='bold')
        
        # Waiting time
        axes[0, 0].plot(self.training_history['episode'], 
                       self.training_history['total_waiting_time'], 
                       linewidth=0.5, alpha=0.5)
        axes[0, 0].plot(self.training_history['episode'], 
                       pd.Series(self.training_history['total_waiting_time']).rolling(20).mean(),
                       linewidth=2, color='red', label='Moving Avg (20)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Waiting Time')
        axes[0, 0].set_title('Total Waiting Time Over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Emergency response time
        axes[0, 1].plot(self.training_history['episode'], 
                       self.training_history['emergency_response_time'],
                       linewidth=0.5, alpha=0.5)
        axes[0, 1].plot(self.training_history['episode'], 
                       pd.Series(self.training_history['emergency_response_time']).rolling(20).mean(),
                       linewidth=2, color='green', label='Moving Avg (20)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Avg Response Time (s)')
        axes[0, 1].set_title('Emergency Vehicle Response Time')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Throughput
        axes[0, 2].plot(self.training_history['episode'], 
                       self.training_history['throughput'],
                       linewidth=0.5, alpha=0.5)
        axes[0, 2].plot(self.training_history['episode'], 
                       pd.Series(self.training_history['throughput']).rolling(20).mean(),
                       linewidth=2, color='blue', label='Moving Avg (20)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Vehicles')
        axes[0, 2].set_title('Throughput (Total Vehicles)')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # Loss
        axes[1, 0].plot(self.training_history['episode'], 
                       self.training_history['loss'],
                       linewidth=0.5, alpha=0.5)
        axes[1, 0].plot(self.training_history['episode'], 
                       pd.Series(self.training_history['loss']).rolling(20).mean(),
                       linewidth=2, color='purple', label='Moving Avg (20)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Epsilon
        axes[1, 1].plot(self.training_history['episode'], 
                       self.training_history['epsilon'],
                       linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].grid(alpha=0.3)
        
        # Phase changes
        axes[1, 2].plot(self.training_history['episode'], 
                       self.training_history['phase_changes'],
                       linewidth=0.5, alpha=0.5)
        axes[1, 2].plot(self.training_history['episode'], 
                       pd.Series(self.training_history['phase_changes']).rolling(20).mean(),
                       linewidth=2, color='brown', label='Moving Avg (20)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Phase Changes')
        axes[1, 2].set_title('Traffic Light Phase Changes')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  → Training plots saved to plots/training_progress.png")
    
    def test(self, episodes=10, emergency_rate=0.05, visualize=True):
        """
        Test trained controller
        
        Args:
            episodes: Number of test episodes
            emergency_rate: Emergency vehicle rate
            visualize: Whether to use GUI
        """
        self.gui = visualize
        controller = HybridTrafficController(junction_id='0')
        
        # Load trained models
        try:
            controller.load_models("models/hybrid_controller_final")
            print("Loaded trained models for testing")
        except:
            print("No trained models found!")
            return
        
        # Disable exploration
        controller.agent.epsilon = 0.0
        
        test_results = []
        
        for episode in range(episodes):
            stats = self.run_episode(controller, episode, emergency_rate)
            test_results.append(stats)
            
            print(f"\nTest Episode {episode}:")
            print(f"  Waiting Time: {stats['total_waiting_time']:,.0f}")
            print(f"  Emergency Response: {stats['emergency_response_time']:.2f}s")
        
        # Summary statistics
        df = pd.DataFrame(test_results)
        print(f"\n{'='*60}")
        print(f"Test Results Summary ({episodes} episodes)")
        print(f"{'='*60}")
        print(df.describe())
        
        # Save results
        df.to_csv("logs/test_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='Hybrid Emergency-Aware Traffic Control')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'emergency-test'],
                       help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=500, 
                       help='Number of episodes')
    parser.add_argument('--emergency-rate', type=float, default=0.05,
                       help='Emergency vehicle rate (0.05 = 5%%)')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI')
    parser.add_argument('--config', type=str, default='cross3ltl.sumocfg',
                       help='SUMO configuration file')
    
    args = parser.parse_args()
    
    # Setup SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("Please set SUMO_HOME environment variable")
        sys.exit(1)
    
    # Create manager
    config_path = os.path.join(os.getcwd(), 'cross3ltl.sumocfg')
    manager = TrainingManager(config_file=args.config, gui=args.gui)
    
    if args.mode == 'train':
        manager.train(episodes=args.episodes, emergency_rate=args.emergency_rate)
    elif args.mode == 'test':
        manager.test(episodes=args.episodes, emergency_rate=args.emergency_rate, 
                    visualize=args.gui)
    elif args.mode == 'emergency-test':
        # Test with high emergency rate
        manager.test(episodes=args.episodes, emergency_rate=0.15, visualize=args.gui)


if __name__ == "__main__":
    main()
