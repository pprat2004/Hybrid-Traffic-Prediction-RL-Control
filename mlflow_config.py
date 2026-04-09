"""
MLflow Configuration and Utilities
Centralized MLflow setup for the traffic control project
"""

import mlflow
import mlflow.keras
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
import os
from datetime import datetime


class MLflowManager:
    """
    Manages all MLflow operations for the traffic control project
    """
    
    def __init__(self, experiment_name="Traffic-Control-Hybrid", tracking_uri="./mlruns"):
        """
        Initialize MLflow manager
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Local directory for MLflow tracking
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set tracking URI (local directory)
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": "Minor Project",
                    "team": "Section C",
                    "description": "Hybrid LSTM-DQN for Emergency-Aware Traffic Control"
                }
            )
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self.run = None
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        print(f"✓ MLflow initialized")
        print(f"  Experiment: {experiment_name}")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment ID: {self.experiment_id}")
    
    def start_run(self, run_name=None, nested=False):
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
        
        Returns:
            MLflow run object
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested
        )
        
        print(f"✓ MLflow run started: {run_name}")
        print(f"  Run ID: {self.run.info.run_id}")
        
        return self.run
    
    def log_params(self, params):
        """
        Log hyperparameters
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        print(f"✓ Logged {len(params)} parameters")
    
    def log_metric(self, key, value, step=None):
        """
        Log a single metric
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (episode number)
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics, step=None):
        """
        Log multiple metrics at once
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number (episode number)
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path, artifact_path=None):
        """
        Log a file as an artifact
        
        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
    
    def log_model(self, model, artifact_path, registered_model_name=None):
        """
        Log a Keras model
        
        Args:
            model: Keras model object
            artifact_path: Name for the model artifact
            registered_model_name: Optional name to register in Model Registry
        """
        mlflow.keras.log_model(
            model, 
            artifact_path,
            registered_model_name=registered_model_name
        )
        print(f"✓ Model logged: {artifact_path}")
    
    def set_tags(self, tags):
        """
        Set tags for the current run
        
        Args:
            tags: Dictionary of tags
        """
        mlflow.set_tags(tags)
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        print("✓ MLflow run ended")
    
    def get_best_run(self, metric_name="waiting_time/total", ascending=True):
        """
        Get the best run based on a metric
        
        Args:
            metric_name: Metric to optimize
            ascending: True for minimization, False for maximization
        
        Returns:
            Best run info
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if runs:
            best_run = runs[0]
            print(f"\n✓ Best run found:")
            print(f"  Run ID: {best_run.info.run_id}")
            print(f"  {metric_name}: {best_run.data.metrics.get(metric_name, 'N/A')}")
            return best_run
        else:
            print("No runs found")
            return None
    
    def compare_runs(self, run_ids, metrics=None):
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare (None = all)
        
        Returns:
            Comparison dataframe
        """
        import pandas as pd
        
        data = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'start_time': run.info.start_time
            }
            row.update(run.data.metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if metrics:
            cols = ['run_id', 'run_name', 'start_time'] + metrics
            df = df[cols]
        
        return df


# Global instance
mlflow_manager = None


def get_mlflow_manager(experiment_name="Traffic-Control-Hybrid"):
    """
    Get or create global MLflow manager instance
    
    Args:
        experiment_name: Name of the experiment
    
    Returns:
        MLflowManager instance
    """
    global mlflow_manager
    if mlflow_manager is None:
        mlflow_manager = MLflowManager(experiment_name=experiment_name)
    return mlflow_manager