#!/bin/bash
# Quick Start Script for Hybrid Emergency-Aware Traffic Control

echo "=============================================================="
echo "Hybrid Traffic Prediction & RL Control Setup"
echo "=============================================================="
echo ""

# Check SUMO installation
if [ -z "$SUMO_HOME" ]; then
    echo "ERROR: SUMO_HOME not set!"
    echo "Please install SUMO and set SUMO_HOME environment variable"
    echo "Example: export SUMO_HOME=/usr/share/sumo"
    exit 1
fi

echo "✓ SUMO_HOME: $SUMO_HOME"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# Create directories
echo "Creating project directories..."
mkdir -p models logs plots
echo "✓ Directories created"
echo ""

# Rename network files to match configuration
echo "Setting up SUMO network files..."
if [ -f "net_net.xml" ]; then
    cp net_net.xml net.net.xml
    echo "✓ Network file ready"
fi

if [ -f "input_routes_rou.xml" ]; then
    cp input_routes_rou.xml input_routes.rou.xml
    echo "✓ Route file ready"
fi
echo ""

echo "=============================================================="
echo "Setup Complete! You can now run:"
echo "=============================================================="
echo ""
echo "1. Train the model:"
echo "   python main.py --mode train --episodes 500 --emergency-rate 0.05"
echo ""
echo "2. Test the model:"
echo "   python main.py --mode test --episodes 10 --gui"
echo ""
echo "3. Compare systems:"
echo "   python compare_systems.py"
echo ""
echo "4. Emergency scenario test:"
echo "   python main.py --mode emergency-test --gui"
echo ""
echo "=============================================================="
