"""
MLflow Utility Functions
Helper functions for analyzing MLflow experiments
"""

from mlflow_config import get_mlflow_manager
import pandas as pd
import matplotlib.pyplot as plt


def compare_all_runs():
    """
    Compare all runs in the experiment
    """
    mlflow = get_mlflow_manager()
    
    runs = mlflow.client.search_runs(
        experiment_ids=[mlflow.experiment_id],
        order_by=["start_time DESC"]
    )
    
    if not runs:
        print("No runs found!")
        return None
    
    print(f"\n{'='*80}")
    print(f"Found {len(runs)} runs in experiment: {mlflow.experiment_name}")
    print(f"{'='*80}\n")
    
    data = []
    for run in runs:
        data.append({
            'Run ID': run.info.run_id[:8],
            'Name': run.data.tags.get('mlflow.runName', 'Unknown'),
            'Episodes': run.data.params.get('episodes', 'N/A'),
            'Avg Waiting': f"{run.data.metrics.get('final/avg_waiting_time', 0):.0f}",
            'Emg Response': f"{run.data.metrics.get('final/avg_emergency_response', 0):.2f}s",
            'Throughput': f"{run.data.metrics.get('final/avg_throughput', 0):.0f}",
            'Status': run.info.status
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    return df


def plot_run_comparison(run_ids, metric='waiting_time/total'):
    """
    Plot comparison of specific metric across multiple runs
    
    Args:
        run_ids: List of run IDs to compare
        metric: Metric to plot
    """
    mlflow = get_mlflow_manager()
    
    plt.figure(figsize=(12, 6))
    
    for run_id in run_ids:
        # Get metric history
        history = mlflow.client.get_metric_history(run_id, metric)
        
        if history:
            steps = [h.step for h in history]
            values = [h.value for h in history]
            
            run = mlflow.client.get_run(run_id)
            run_name = run.data.tags.get('mlflow.runName', run_id[:8])
            
            plt.plot(steps, values, label=run_name, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Comparison: {metric}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/comparison_{metric.replace("/", "_")}.png', dpi=150)
    plt.show()
    
    print(f"✓ Comparison plot saved to plots/comparison_{metric.replace('/', '_')}.png")


def get_best_model_info():
    """
    Get information about the best performing model
    """
    mlflow = get_mlflow_manager()
    
    best_run = mlflow.get_best_run(metric_name="waiting_time/total", ascending=True)
    
    if best_run:
        print(f"\n{'='*80}")
        print(f"BEST MODEL INFORMATION")
        print(f"{'='*80}\n")
        print(f"Run ID: {best_run.info.run_id}")
        print(f"Run Name: {best_run.data.tags.get('mlflow.runName', 'Unknown')}")
        print(f"\nParameters:")
        for key, value in best_run.data.params.items():
            print(f"  {key}: {value}")
        print(f"\nFinal Metrics:")
        for key, value in best_run.data.metrics.items():
            if key.startswith('final/'):
                print(f"  {key}: {value}")
        print(f"\n{'='*80}\n")
        
        return best_run
    else:
        print("No runs found!")
        return None


if __name__ == "__main__":
    # Example usage
    print("MLflow Experiment Analysis")
    print("="*80)
    
    # Compare all runs
    compare_all_runs()
    
    # Get best model info
    get_best_model_info()