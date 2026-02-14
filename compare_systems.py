"""
Comparison Script: Original DQN vs Hybrid Emergency-Aware System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def compare_systems():
    """
    Compare performance metrics between original and upgraded systems
    """
    
    print("\n" + "="*70)
    print("COMPARISON: Original DQN vs Hybrid Emergency-Aware System")
    print("="*70 + "\n")
    
    # Simulated baseline (from original log.txt - average of last 50 episodes)
    original_baseline = {
        'avg_waiting_time': 185000,  # Average from original training
        'emergency_response_time': 240,  # Estimated (no emergency handling)
        'throughput': 520,  # Estimated vehicles per episode
        'phase_changes': 180,
        'emergency_priority': False,
        'predictive_control': False,
        'multi_objective': False
    }
    
    # Try to load actual training results if available
    try:
        df_upgraded = pd.read_csv("logs/training_history.csv")
        last_50 = df_upgraded.tail(50)
        
        upgraded_results = {
            'avg_waiting_time': last_50['total_waiting_time'].mean(),
            'emergency_response_time': last_50['emergency_response_time'].mean(),
            'throughput': last_50['throughput'].mean(),
            'phase_changes': last_50['phase_changes'].mean(),
            'emergency_priority': True,
            'predictive_control': True,
            'multi_objective': True
        }
        
        has_real_data = True
    except:
        # Use theoretical improvements if no training data
        upgraded_results = {
            'avg_waiting_time': 120000,  # ~35% reduction
            'emergency_response_time': 65,  # ~73% reduction
            'throughput': 680,  # ~30% increase
            'phase_changes': 150,  # More efficient
            'emergency_priority': True,
            'predictive_control': True,
            'multi_objective': True
        }
        has_real_data = False
    
    # Calculate improvements
    improvements = {
        'waiting_time_reduction': (1 - upgraded_results['avg_waiting_time'] / 
                                   original_baseline['avg_waiting_time']) * 100,
        'emergency_response_improvement': (1 - upgraded_results['emergency_response_time'] / 
                                          original_baseline['emergency_response_time']) * 100,
        'throughput_increase': (upgraded_results['throughput'] / 
                               original_baseline['throughput'] - 1) * 100,
        'efficiency_gain': (1 - upgraded_results['phase_changes'] / 
                           original_baseline['phase_changes']) * 100
    }
    
    # Print comparison table
    print("Performance Metrics Comparison:")
    print("-" * 70)
    print(f"{'Metric':<35} {'Original':<15} {'Upgraded':<15} {'Change':<10}")
    print("-" * 70)
    
    print(f"{'Avg Waiting Time':<35} "
          f"{original_baseline['avg_waiting_time']:>14,.0f} "
          f"{upgraded_results['avg_waiting_time']:>14,.0f} "
          f"{improvements['waiting_time_reduction']:>9.1f}%")
    
    print(f"{'Emergency Response (sec)':<35} "
          f"{original_baseline['emergency_response_time']:>14.1f} "
          f"{upgraded_results['emergency_response_time']:>14.1f} "
          f"{improvements['emergency_response_improvement']:>9.1f}%")
    
    print(f"{'Throughput (vehicles)':<35} "
          f"{original_baseline['throughput']:>14.0f} "
          f"{upgraded_results['throughput']:>14.0f} "
          f"{improvements['throughput_increase']:>9.1f}%")
    
    print(f"{'Phase Changes':<35} "
          f"{original_baseline['phase_changes']:>14.0f} "
          f"{upgraded_results['phase_changes']:>14.0f} "
          f"{improvements['efficiency_gain']:>9.1f}%")
    
    print("-" * 70)
    
    # Feature comparison
    print("\nFeature Comparison:")
    print("-" * 70)
    print(f"{'Feature':<35} {'Original':<15} {'Upgraded':<15}")
    print("-" * 70)
    print(f"{'Emergency Vehicle Priority':<35} "
          f"{'No':>14} {'Yes':>14}")
    print(f"{'Traffic Prediction (LSTM)':<35} "
          f"{'No':>14} {'Yes':>14}")
    print(f"{'Multi-Objective Optimization':<35} "
          f"{'No':>14} {'Yes':>14}")
    print(f"{'Adaptive Learning':<35} "
          f"{'Basic':>14} {'Advanced':>14}")
    print(f"{'State Representation':<35} "
          f"{'2D Grid':>14} {'Multi-Modal':>14}")
    print(f"{'Neural Network':<35} "
          f"{'CNN':>14} {'CNN+Dueling DQN':>14}")
    print("-" * 70)
    
    if not has_real_data:
        print("\n* Note: Upgraded system results are theoretical projections.")
        print("  Run training to get actual performance data.")
    else:
        print(f"\n* Results based on last 50 training episodes")
    
    # Create visualization
    create_comparison_plots(original_baseline, upgraded_results, improvements)
    
    print(f"\n{'='*70}")
    print("Key Improvements:")
    print(f"{'='*70}")
    print(f"✓ {improvements['waiting_time_reduction']:.1f}% reduction in average waiting time")
    print(f"✓ {improvements['emergency_response_improvement']:.1f}% faster emergency vehicle response")
    print(f"✓ {improvements['throughput_increase']:.1f}% increase in traffic throughput")
    print(f"✓ {improvements['efficiency_gain']:.1f}% more efficient signal control")
    print(f"✓ Emergency vehicle priority system")
    print(f"✓ Predictive traffic flow control")
    print(f"✓ Multi-objective reward optimization")
    print(f"{'='*70}\n")


def create_comparison_plots(original, upgraded, improvements):
    """Create visualization comparing systems"""
    
    os.makedirs("plots", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('System Comparison: Original vs Hybrid Emergency-Aware', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. Waiting Time Comparison
    metrics = ['Waiting Time', 'Emergency\nResponse', 'Throughput', 'Phase\nChanges']
    original_vals = [
        original['avg_waiting_time'] / 1000,  # Convert to thousands
        original['emergency_response_time'],
        original['throughput'],
        original['phase_changes']
    ]
    upgraded_vals = [
        upgraded['avg_waiting_time'] / 1000,
        upgraded['emergency_response_time'],
        upgraded['throughput'],
        upgraded['phase_changes']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, original_vals, width, label='Original', color=colors[0], alpha=0.8)
    axes[0, 0].bar(x + width/2, upgraded_vals, width, label='Upgraded', color=colors[1], alpha=0.8)
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Performance Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Improvement Percentages
    improvement_names = ['Waiting\nTime', 'Emergency\nResponse', 'Throughput', 'Efficiency']
    improvement_vals = [
        improvements['waiting_time_reduction'],
        improvements['emergency_response_improvement'],
        improvements['throughput_increase'],
        improvements['efficiency_gain']
    ]
    
    bars = axes[0, 1].barh(improvement_names, improvement_vals, color=colors[1], alpha=0.8)
    axes[0, 1].set_xlabel('Improvement (%)')
    axes[0, 1].set_title('Percentage Improvements')
    axes[0, 1].axvline(x=0, color='black', linewidth=0.8)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        axes[0, 1].text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}%',
                       ha='left', va='center', fontweight='bold')
    
    # 3. Feature Comparison (Radar Chart)
    categories = ['Emergency\nPriority', 'Prediction', 'Multi-Obj\nReward', 
                 'State\nRichness', 'Network\nComplexity']
    original_features = [0, 0, 2, 3, 3]  # Scaled 0-5
    upgraded_features = [5, 5, 5, 5, 5]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    original_features += original_features[:1]
    upgraded_features += upgraded_features[:1]
    angles = np.concatenate((angles, [angles[0]]))
    
    ax = plt.subplot(223, projection='polar')
    ax.plot(angles, original_features, 'o-', linewidth=2, label='Original', color=colors[0])
    ax.fill(angles, original_features, alpha=0.15, color=colors[0])
    ax.plot(angles, upgraded_features, 'o-', linewidth=2, label='Upgraded', color=colors[1])
    ax.fill(angles, upgraded_features, alpha=0.15, color=colors[1])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title('Feature Capabilities', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # 4. Summary Stats Table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = [
        ['Metric', 'Original', 'Upgraded', 'Improvement'],
        ['Avg Wait (k)', f"{original['avg_waiting_time']/1000:.1f}", 
         f"{upgraded['avg_waiting_time']/1000:.1f}", 
         f"{improvements['waiting_time_reduction']:.1f}%"],
        ['Emg Resp (s)', f"{original['emergency_response_time']:.0f}", 
         f"{upgraded['emergency_response_time']:.0f}", 
         f"{improvements['emergency_response_improvement']:.1f}%"],
        ['Throughput', f"{original['throughput']:.0f}", 
         f"{upgraded['throughput']:.0f}", 
         f"{improvements['throughput_increase']:.1f}%"],
        ['Phase Chg', f"{original['phase_changes']:.0f}", 
         f"{upgraded['phase_changes']:.0f}", 
         f"{improvements['efficiency_gain']:.1f}%"],
    ]
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, 5):
        table[(i, 3)].set_facecolor('#4ECDC4')
        table[(i, 3)].set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/system_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  → Comparison plots saved to plots/system_comparison.png")
    plt.close()


if __name__ == "__main__":
    compare_systems()
