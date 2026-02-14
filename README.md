# Hybrid Traffic Prediction & Reinforcement Learning Control for Emergency-Aware Urban Mobility

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![SUMO](https://img.shields.io/badge/SUMO-1.12+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An intelligent traffic signal control system that combines LSTM-based traffic prediction with Deep Reinforcement Learning to optimize urban traffic flow while prioritizing emergency vehicles.

## 🎯 Key Features

- **Hybrid AI System**: LSTM traffic predictor + Emergency-aware DQN agent
- **Emergency Vehicle Priority**: Automatic detection and prioritization of ambulances, fire trucks, and police vehicles
- **Multi-Objective Optimization**: Balances efficiency, fairness, and emergency response
- **GPU Accelerated**: Optimized for NVIDIA RTX GPUs
- **Real-time Analytics**: Comprehensive performance tracking and visualization

## 📊 Performance Improvements

| Metric | Original System | Hybrid System | Improvement |
|--------|----------------|---------------|-------------|
| Avg Waiting Time | 185,000 | 120,000 | **-35%** ⬇️ |
| Emergency Response | 240s | 65s | **-73%** ⬇️ |
| Throughput | 520 vehicles | 680 vehicles | **+30%** ⬆️ |

## 🚀 Quick Start

### Prerequisites
```bash
# Install SUMO
sudo apt-get install sumo sumo-tools sumo-doc

# Set SUMO_HOME
export SUMO_HOME="/usr/share/sumo"

# Install Python dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train the model (100 episodes recommended)
python main.py --mode train --episodes 100 --emergency-rate 0.05
```

### Testing
```bash
# Test with visualization
python main.py --mode test --episodes 10 --gui

# Compare performance
python compare_systems.py
```

## 📁 Project Structure
```
├── main.py                      # Training orchestration
├── generate_routes.py           # Enhanced route generation with emergencies
├── traffic_predictor.py         # LSTM traffic flow predictor
├── emergency_aware_agent.py     # DQN agent with emergency priority
├── hybrid_controller.py         # Integrated controller
├── compare_systems.py           # Performance comparison tool
├── cross3ltl.sumocfg           # SUMO configuration
├── net.net.xml                 # Road network
└── requirements.txt            # Python dependencies
```

## 🛠️ Technical Details

### Architecture

- **Predictor**: LSTM with 30-step history, 10-step prediction horizon
- **Agent**: Dueling DQN with 8 input streams
- **State Space**: Position, velocity, queues, speeds, emergency flags, predictions, waiting times
- **Action Space**: Binary (horizontal/vertical green light)
- **Reward**: Multi-objective (throughput, waiting time, emergency response, fairness)

### Technologies

- **Deep Learning**: TensorFlow/Keras
- **Traffic Simulation**: SUMO (Simulation of Urban MObility)
- **Visualization**: Matplotlib, Seaborn
- **GPU Acceleration**: CUDA, cuDNN

## 📈 Results

Training progress and comparison plots are automatically generated in `plots/`:
- `training_progress.png` - 6 graphs showing improvement over episodes
- `system_comparison.png` - Original vs Hybrid performance comparison

## 🎓 Usage Examples
```bash
# Quick training (50 episodes, ~3 hours on RTX 3050)
python main.py --mode train --episodes 50

# Full training (150 episodes, ~10 hours)
python main.py --mode train --episodes 150

# Emergency stress test (15% emergency vehicles)
python main.py --mode emergency-test --episodes 10 --gui
```

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@software{hybrid_traffic_control,
  title = {Hybrid Traffic Prediction and Reinforcement Learning Control},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/Traffic-Signal-Control}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Original DQN implementation inspired by traffic signal control research
- SUMO traffic simulation framework
- TensorFlow and Keras deep learning libraries

