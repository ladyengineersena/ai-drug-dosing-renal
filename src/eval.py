"""
Evaluation scripts for MIPD models
Metrics: AUC error, time-to-target, target achievement rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.hybrid import HybridDosePredictor
from src.poppk.bayesian_forecasting import BayesianForecaster, OneCompartmentPKModel
from src.models.ml_regressor import MLDosePredictor
from src.utils import calculate_auc


class MIPDEvaluator:
    """Evaluator for MIPD models"""
    
    def __init__(self, drug_type: str = "vancomycin"):
        """
        Initialize evaluator
        
        Args:
            drug_type: 'vancomycin' or 'amikacin'
        """
        self.drug_type = drug_type
        
        if drug_type == "vancomycin":
            self.target_auc = 400  # mg*h/L
            self.target_trough_min = 10  # mg/L
            self.target_trough_max = 20  # mg/L
        elif drug_type == "amikacin":
            self.target_auc = 200  # mg*h/L
            self.target_trough_min = 2  # mg/L
            self.target_trough_max = 5  # mg/L
        else:
            raise ValueError(f"Unknown drug type: {drug_type}")
    
    def evaluate_concentration_prediction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate concentration predictions
        
        Args:
            y_true: True concentrations
            y_pred: Predicted concentrations
        
        Returns:
            Evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Percentage within 20% of true value
        percent_error = np.abs((y_true - y_pred) / y_true) * 100
        within_20_percent = (percent_error <= 20).mean() * 100
        
        # Bias (mean prediction error)
        bias = np.mean(y_pred - y_true)
        
        # Precision (SD of prediction error)
        precision = np.std(y_pred - y_true)
        
        return {
            'mae_mg_per_L': mae,
            'rmse_mg_per_L': rmse,
            'r2': r2,
            'bias_mg_per_L': bias,
            'precision_mg_per_L': precision,
            'within_20_percent': within_20_percent
        }
    
    def evaluate_auc_prediction(
        self,
        true_aucs: np.ndarray,
        pred_aucs: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate AUC predictions
        
        Args:
            true_aucs: True AUCs
            pred_aucs: Predicted AUCs
        
        Returns:
            Evaluation metrics
        """
        mae = mean_absolute_error(true_aucs, pred_aucs)
        rmse = np.sqrt(mean_squared_error(true_aucs, pred_aucs))
        r2 = r2_score(true_aucs, pred_aucs)
        
        # Percentage within target AUC ± 20%
        target_achieved = np.abs(pred_aucs - self.target_auc) <= (self.target_auc * 0.2)
        target_achievement_rate = target_achieved.mean() * 100
        
        # Bias
        bias = np.mean(pred_aucs - true_aucs)
        
        return {
            'mae_mg_h_per_L': mae,
            'rmse_mg_h_per_L': rmse,
            'r2': r2,
            'bias_mg_h_per_L': bias,
            'target_achievement_rate_percent': target_achievement_rate
        }
    
    def evaluate_target_achievement(
        self,
        df: pd.DataFrame,
        concentration_column: str = 'concentration_mg_per_L',
        time_column: str = 'time_since_dose_hours'
    ) -> Dict[str, float]:
        """
        Evaluate target achievement rate
        
        Args:
            df: DataFrame with concentration data
            concentration_column: Column name for concentrations
            time_column: Column name for time
        
        Returns:
            Target achievement metrics
        """
        results = []
        
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id].copy()
            
            # Get trough concentrations (typically at end of interval)
            if 'dose_number' in patient_df.columns:
                # Steady state troughs
                ss_df = patient_df[patient_df['dose_number'] >= 3]
                if len(ss_df) > 0:
                    # Assume trough is near end of interval
                    max_time = ss_df[time_column].max()
                    trough_df = ss_df[ss_df[time_column] >= max_time * 0.8]
                    if len(trough_df) > 0:
                        trough_conc = trough_df[concentration_column].mean()
                        
                        # Check if target achieved
                        target_achieved = (
                            self.target_trough_min <= trough_conc <= self.target_trough_max
                        ) if self.drug_type == "vancomycin" else (trough_conc <= self.target_trough_max)
                        
                        results.append({
                            'patient_id': patient_id,
                            'trough_concentration': trough_conc,
                            'target_achieved': target_achieved,
                            'time_to_target_hours': None  # Could be calculated if we track when target was first achieved
                        })
            
            # If no steady state data, use last measurement
            if len(results) == 0 or results[-1]['patient_id'] != patient_id:
                last_conc = patient_df[concentration_column].iloc[-1]
                target_achieved = (
                    self.target_trough_min <= last_conc <= self.target_trough_max
                ) if self.drug_type == "vancomycin" else (last_conc <= self.target_trough_max)
                
                results.append({
                    'patient_id': patient_id,
                    'trough_concentration': last_conc,
                    'target_achieved': target_achieved,
                    'time_to_target_hours': None
                })
        
        results_df = pd.DataFrame(results)
        
        target_achievement_rate = results_df['target_achieved'].mean() * 100
        
        return {
            'target_achievement_rate_percent': target_achievement_rate,
            'n_patients': len(results_df),
            'n_achieved': results_df['target_achieved'].sum(),
            'mean_trough_concentration': results_df['trough_concentration'].mean(),
            'median_trough_concentration': results_df['trough_concentration'].median()
        }
    
    def evaluate_time_to_target(
        self,
        df: pd.DataFrame,
        concentration_column: str = 'concentration_mg_per_L',
        time_column: str = 'time_since_dose_hours'
    ) -> Dict[str, float]:
        """
        Evaluate time to reach target concentration
        
        Args:
            df: DataFrame with concentration data
            concentration_column: Column name for concentrations
            time_column: Column name for time
        
        Returns:
            Time to target metrics
        """
        times_to_target = []
        
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id].copy()
            patient_df = patient_df.sort_values(time_column)
            
            # Find first time target is achieved
            if self.drug_type == "vancomycin":
                target_achieved = (patient_df[concentration_column] >= self.target_trough_min) & \
                                 (patient_df[concentration_column] <= self.target_trough_max)
            else:
                target_achieved = patient_df[concentration_column] <= self.target_trough_max
            
            if target_achieved.any():
                first_achieved_idx = target_achieved.idxmax()
                time_to_target = patient_df.loc[first_achieved_idx, time_column]
                times_to_target.append(time_to_target)
        
        if len(times_to_target) == 0:
            return {
                'mean_time_to_target_hours': None,
                'median_time_to_target_hours': None,
                'n_achieved': 0
            }
        
        return {
            'mean_time_to_target_hours': np.mean(times_to_target),
            'median_time_to_target_hours': np.median(times_to_target),
            'min_time_to_target_hours': np.min(times_to_target),
            'max_time_to_target_hours': np.max(times_to_target),
            'n_achieved': len(times_to_target)
        }
    
    def compare_models(
        self,
        df: pd.DataFrame,
        models: Dict[str, any],
        patient_data_column: str = 'patient_data'
    ) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            df: Test DataFrame
            models: Dictionary of model_name -> model_instance
            patient_data_column: Column name for patient data dictionaries
        
        Returns:
            Comparison results DataFrame
        """
        results = []
        
        for model_name, model in models.items():
            # Get predictions for each patient
            predictions = []
            true_values = []
            
            for _, row in df.iterrows():
                patient_data = row[patient_data_column] if patient_data_column in row else row.to_dict()
                
                # Predict concentration (simplified: use first TDM time)
                if 'time_since_dose_hours' in row:
                    time = row['time_since_dose_hours']
                else:
                    time = 12.0  # Default
                
                if 'dose_mg' in row:
                    dose = row['dose_mg']
                else:
                    dose = 15 * patient_data.get('weight_kg', 70)  # Default
                
                # Get prediction
                if hasattr(model, 'predict'):
                    pred = model.predict(
                        patient_data=patient_data,
                        dose=dose,
                        time=time,
                        interval=row.get('interval_hours', 12.0),
                        n_doses=row.get('dose_number', 1)
                    )
                    
                    if isinstance(pred, dict):
                        pred_conc = pred.get('predicted_concentration_mg_per_L', 0.0)
                    else:
                        pred_conc = pred
                else:
                    # Use model's predict method directly
                    pred_conc = model.predict([[time, dose]])[0]
                
                predictions.append(pred_conc)
                
                if 'concentration_mg_per_L' in row:
                    true_values.append(row['concentration_mg_per_L'])
            
            if len(true_values) > 0:
                # Evaluate
                metrics = self.evaluate_concentration_prediction(
                    np.array(true_values),
                    np.array(predictions)
                )
                
                results.append({
                    'model': model_name,
                    **metrics
                })
        
        return pd.DataFrame(results)
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """
        Plot predictions vs true values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name for plot title
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Concentration (mg/L)')
        axes[0].set_ylabel('Predicted Concentration (mg/L)')
        axes[0].set_title(f'{model_name}: Predictions vs True Values')
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Concentration (mg/L)')
        axes[1].set_ylabel('Residuals (mg/L)')
        axes[1].set_title(f'{model_name}: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def evaluate_dataset(
    df: pd.DataFrame,
    model,
    drug_type: str = "vancomycin"
) -> Dict:
    """
    Evaluate model on dataset
    
    Args:
        df: Test DataFrame
        model: Trained model
        drug_type: Drug type
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = MIPDEvaluator(drug_type=drug_type)
    
    # Prepare predictions
    predictions = []
    true_concentrations = []
    
    for _, row in df.iterrows():
        patient_data = {
            'age': row['age'],
            'sex': row['sex'],
            'weight_kg': row['weight_kg'],
            'height_cm': row.get('height_cm', 170),
            'creatinine_mg_dL': row['creatinine_mg_dL'],
            'egfr': row.get('egfr', 100)
        }
        
        # Predict
        if hasattr(model, 'predict'):
            pred = model.predict(
                patient_data=patient_data,
                dose=row['dose_mg'],
                time=row['time_since_dose_hours'],
                interval=row.get('interval_hours', 12.0),
                n_doses=row.get('dose_number', 1)
            )
            
            if isinstance(pred, dict):
                pred_conc = pred.get('predicted_concentration_mg_per_L', 0.0)
            else:
                pred_conc = pred
        else:
            pred_conc = 0.0
        
        predictions.append(pred_conc)
        true_concentrations.append(row['concentration_mg_per_L'])
    
    # Evaluate
    conc_metrics = evaluator.evaluate_concentration_prediction(
        np.array(true_concentrations),
        np.array(predictions)
    )
    
    target_metrics = evaluator.evaluate_target_achievement(df)
    time_metrics = evaluator.evaluate_time_to_target(df)
    
    return {
        'concentration_prediction': conc_metrics,
        'target_achievement': target_metrics,
        'time_to_target': time_metrics
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MIPD models')
    parser.add_argument('--data', type=str, required=True, help='Test data CSV file')
    parser.add_argument('--model', type=str, help='Trained model file')
    parser.add_argument('--drug', type=str, default='vancomycin', choices=['vancomycin', 'amikacin'])
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Load model (if provided)
    # This would need to be adapted based on model type
    # For now, just evaluate on data
    evaluator = MIPDEvaluator(drug_type=args.drug)
    
    # Evaluate target achievement
    target_metrics = evaluator.evaluate_target_achievement(df)
    time_metrics = evaluator.evaluate_time_to_target(df)
    
    print("Evaluation Results:")
    print(f"Target Achievement Rate: {target_metrics['target_achievement_rate_percent']:.2f}%")
    print(f"Mean Trough Concentration: {target_metrics['mean_trough_concentration']:.2f} mg/L")
    
    if time_metrics['mean_time_to_target_hours']:
        print(f"Mean Time to Target: {time_metrics['mean_time_to_target_hours']:.2f} hours")

