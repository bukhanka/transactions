import optuna
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from test_solution import main as run_simulation
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.results_history = []
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        # Suggest values for our hyperparameters
        params = {
            'penalty_weight': trial.suggest_float('penalty_weight', 0.5, 2.0),
            'balance_factor': trial.suggest_float('balance_factor', 0.1, 0.5),
            'conversion_weight': trial.suggest_float('conversion_weight', 0.5, 2.0),
            'time_weight': trial.suggest_float('time_weight', 0.2, 1.0),
            'utilization_boost': trial.suggest_float('utilization_boost', 0.1, 0.4)
        }
        
        # Run simulation with these parameters
        try:
            results = self.run_evaluation(params)
            
            # Calculate objective score
            objective_score = self.calculate_objective(results, params)
            
            # Store results
            self.results_history.append({
                'params': params,
                'results': results,
                'score': objective_score
            })
            
            return objective_score
            
        except Exception as e:
            print(f"Error in trial: {str(e)}")
            return float('-inf')
    
    def calculate_objective(self, results, params):
        """Calculate combined objective score from results"""
        if not results:
            return float('-inf')
            
        # Extract metrics
        profit = results.get('profit', 0)
        conversion = results.get('conversion_rate', 0)
        penalties = results.get('penalties', 0)
        avg_time = results.get('avg_time', 0)
        
        # Normalize metrics to similar scales
        normalized_profit = profit / 1_000_000  # Assume profits in millions
        normalized_conversion = conversion * 100  # Convert to percentage
        normalized_penalties = penalties / 10_000  # Scale penalties
        normalized_time = min(avg_time / 100, 1)  # Cap normalized time at 1
        
        # Combined score with parameter weights
        score = (
            normalized_profit * 1.0 +  # Base profit weight
            normalized_conversion * params['conversion_weight'] -
            normalized_penalties * params['penalty_weight'] -
            normalized_time * params['time_weight']
        )
        
        return score
    
    def run_evaluation(self, params):
        """Run simulation with given parameters and return metrics"""
        # Inject parameters into global config or pass directly to simulation
        results = run_simulation(
            force_reprocess=True,
            hyperparams=params
        )
        
        return results
    
    def optimize(self):
        """Run optimization process"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.study = study
        self.best_params = study.best_params
        
        return study.best_params
    
    def save_results(self, filepath='tuning_results.joblib'):
        """Save optimization results"""
        results = {
            'best_params': self.best_params,
            'study': self.study,
            'history': self.results_history
        }
        joblib.dump(results, filepath)
    
    def load_results(self, filepath='tuning_results.joblib'):
        """Load previous optimization results"""
        results = joblib.load(filepath)
        self.best_params = results['best_params']
        self.study = results['study']
        self.results_history = results['history']
    
    def plot_optimization_history(self):
        """Plot optimization history using Optuna's visualization"""
        try:
            import plotly
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.show()
            
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.show()
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
    
    def log_tuning_results(self, trial):
        """Log detailed results of each trial"""
        results_path = Path('results/tuning')
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Create detailed results log
        trial_results = {
            'trial_number': trial.number,
            'params': trial.params,
            'metrics': {
                'profit': trial.value,
                'conversion_rate': self.last_results.get('conversion_rate', 0),
                'avg_chain_length': self.last_results.get('avg_chain_length', 0),
                'avg_processing_time': self.last_results.get('avg_time', 0),
                'total_penalties': self.last_results.get('penalties', 0)
            }
        }
        
        # Save to CSV for easy analysis
        results_df = pd.DataFrame([trial_results])
        csv_path = results_path / 'tuning_history.csv'
        
        if csv_path.exists():
            results_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_path, index=False)

def main():
    # Initialize tuner
    tuner = HyperparameterTuner(n_trials=20)  # Adjust trials as needed
    
    # Run optimization
    print("Starting hyperparameter optimization...")
    best_params = tuner.optimize()
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value:.4f}")
    
    # Save results
    tuner.save_results()
    
    # Plot results
    print("\nGenerating visualization...")
    tuner.plot_optimization_history()

if __name__ == "__main__":
    main() 