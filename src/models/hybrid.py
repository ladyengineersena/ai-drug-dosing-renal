"""
Hybrid model combining popPK/Bayesian forecasting with ML correction
Uses residual learning approach
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.poppk.bayesian_forecasting import BayesianForecaster, OneCompartmentPKModel
from src.models.ml_regressor import MLDosePredictor
from src.utils import calculate_auc


class HybridDosePredictor:
    """
    Hybrid model: popPK/Bayesian + ML correction
    """
    
    def __init__(
        self,
        drug_type: str = "vancomycin",
        ml_model_type: str = "xgboost",
        use_residual_learning: bool = True
    ):
        """
        Initialize hybrid predictor
        
        Args:
            drug_type: 'vancomycin' or 'amikacin'
            ml_model_type: ML model type ('xgboost', 'lightgbm', 'random_forest', 'neural_network')
            use_residual_learning: If True, ML corrects popPK predictions; if False, ML selects best popPK model
        """
        self.drug_type = drug_type
        self.ml_model_type = ml_model_type
        self.use_residual_learning = use_residual_learning
        
        # Initialize popPK model
        self.pk_model = OneCompartmentPKModel(drug_type=drug_type)
        self.bayesian_forecaster = BayesianForecaster(self.pk_model)
        
        # Initialize ML model (for residual correction or model selection)
        self.ml_predictor = None
        self.is_trained = False
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'concentration_mg_per_L'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for hybrid model
        
        Args:
            df: DataFrame with patient and dosing data
            target_column: Target column name
        
        Returns:
            Features, popPK predictions, and residuals
        """
        # Prepare ML features
        ml_predictor_temp = MLDosePredictor(model_type=self.ml_model_type)
        X, feature_names = ml_predictor_temp.prepare_features(df)
        
        # Get popPK predictions for each patient
        poppk_predictions = []
        residuals = []
        
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id].copy()
            patient_data = patient_df.iloc[0]
            
            # Estimate PK parameters using popPK
            weight = patient_data['weight_kg']
            egfr = patient_data.get('egfr', 100)
            
            # Clear popPK forecaster
            self.bayesian_forecaster.observed_concentrations = []
            self.bayesian_forecaster.observed_times = []
            self.bayesian_forecaster.doses = []
            self.bayesian_forecaster.dose_times = []
            
            # Add any existing TDM observations
            if 'concentration_mg_per_L' in patient_df.columns:
                for idx, row in patient_df.iterrows():
                    if pd.notna(row.get('concentration_mg_per_L')):
                        self.bayesian_forecaster.add_observation(
                            row['concentration_mg_per_L'],
                            row['time_since_dose_hours'],
                            row['dose_mg'],
                            dose_time=0.0
                        )
            
            # Estimate parameters
            pk_params = self.bayesian_forecaster.estimate_parameters(
                weight=weight,
                egfr=egfr
            )
            
            # Predict concentrations using popPK
            for idx, row in patient_df.iterrows():
                pred_conc = self.pk_model.predict_concentration(
                    time=row['time_since_dose_hours'],
                    dose=row['dose_mg'],
                    clearance=pk_params['clearance_L_per_h'],
                    volume=pk_params['volume_L'],
                    n_doses=row.get('dose_number', 1),
                    interval=row.get('interval_hours', 12.0)
                )
                
                poppk_predictions.append(pred_conc)
                
                # Calculate residual
                if pd.notna(row.get(target_column)):
                    residual = row[target_column] - pred_conc
                else:
                    residual = 0.0
                
                residuals.append(residual)
        
        poppk_predictions = np.array(poppk_predictions)
        residuals = np.array(residuals)
        
        # Add popPK prediction as feature
        X_with_poppk = np.column_stack([X, poppk_predictions])
        feature_names.append('poppk_prediction')
        
        return X_with_poppk, poppk_predictions, residuals, feature_names
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'concentration_mg_per_L',
        residual_target: bool = True
    ):
        """
        Train hybrid model
        
        Args:
            df: Training DataFrame
            target_column: Target column name
            residual_target: If True, train ML on residuals; if False, train ML on absolute values
        """
        # Prepare training data
        X, poppk_predictions, residuals, feature_names = self.prepare_training_data(df, target_column)
        
        # Initialize ML predictor
        self.ml_predictor = MLDosePredictor(model_type=self.ml_model_type)
        self.ml_predictor.feature_names = feature_names
        
        # Train ML model
        if self.use_residual_learning and residual_target:
            # Train on residuals
            y_ml = residuals
        else:
            # Train on absolute values (ML will learn to combine with popPK)
            y_ml = df[target_column].values - poppk_predictions
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_ml, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train ML model
        self.ml_predictor.train(X_train, y_train, X_val, y_val)
        self.is_trained = True
    
    def predict(
        self,
        patient_data: Dict,
        dose: float,
        time: float,
        interval: float = 12.0,
        n_doses: int = 1,
        tdm_history: Optional[List[Dict]] = None,
        return_components: bool = False
    ) -> Dict:
        """
        Predict concentration using hybrid model
        
        Args:
            patient_data: Patient demographics and lab values
            dose: Dose (mg)
            time: Time since dose (hours)
            interval: Dosing interval (hours)
            n_doses: Number of doses
            tdm_history: Previous TDM measurements (optional)
            return_components: If True, return popPK and ML components separately
        
        Returns:
            Prediction dictionary
        """
        # Step 1: Get popPK prediction
        weight = patient_data['weight_kg']
        egfr = patient_data.get('egfr', 100)
        
        # Clear and update Bayesian forecaster with TDM history
        self.bayesian_forecaster.observed_concentrations = []
        self.bayesian_forecaster.observed_times = []
        self.bayesian_forecaster.doses = []
        self.bayesian_forecaster.dose_times = []
        
        if tdm_history:
            for tdm in tdm_history:
                self.bayesian_forecaster.add_observation(
                    tdm['concentration'],
                    tdm['time'],
                    tdm.get('dose', dose),
                    dose_time=tdm.get('dose_time', 0.0)
                )
        
        # Estimate PK parameters
        pk_params = self.bayesian_forecaster.estimate_parameters(
            weight=weight,
            egfr=egfr
        )
        
        # Predict using popPK
        poppk_pred = self.pk_model.predict_concentration(
            time=time,
            dose=dose,
            clearance=pk_params['clearance_L_per_h'],
            volume=pk_params['volume_L'],
            n_doses=n_doses,
            interval=interval
        )
        
        if not self.is_trained:
            # Return popPK prediction only
            return {
                'predicted_concentration_mg_per_L': poppk_pred,
                'poppk_prediction': poppk_pred,
                'ml_correction': 0.0,
                'method': 'poppk_only'
            }
        
        # Step 2: Get ML correction
        # Prepare features
        feature_dict = {
            'age': patient_data.get('age', 50),
            'sex': patient_data.get('sex', 'M'),
            'weight_kg': weight,
            'height_cm': patient_data.get('height_cm', 170),
            'creatinine_mg_dL': patient_data.get('creatinine_mg_dL', 1.0),
            'egfr': egfr,
            'dose_mg': dose,
            'dose_mg_per_kg': dose / weight,
            'interval_hours': interval,
            'dose_number': n_doses,
            'time_since_dose_hours': time,
            'prev_concentration_mg_per_L': tdm_history[-1]['concentration'] if tdm_history else 0.0,
            'bmi': weight / ((patient_data.get('height_cm', 170) / 100) ** 2),
            'total_time_hours': time + (n_doses - 1) * interval if n_doses > 1 else time
        }
        
        # Create DataFrame for feature preparation
        feature_df = pd.DataFrame([feature_dict])
        X, _ = self.ml_predictor.prepare_features(feature_df, include_tdm=len(tdm_history) > 0)
        
        # Add popPK prediction
        X_with_poppk = np.column_stack([X, [poppk_pred]])
        
        # Get ML correction
        ml_correction = self.ml_predictor.predict(X_with_poppk)[0]
        
        # Combine predictions
        if self.use_residual_learning:
            hybrid_pred = poppk_pred + ml_correction
        else:
            # ML predicts absolute value, but we still combine with popPK
            hybrid_pred = 0.7 * poppk_pred + 0.3 * (poppk_pred + ml_correction)
        
        result = {
            'predicted_concentration_mg_per_L': max(hybrid_pred, 0.0),
            'poppk_prediction': poppk_pred,
            'ml_correction': ml_correction,
            'method': 'hybrid'
        }
        
        if return_components:
            result['components'] = {
                'poppk': poppk_pred,
                'ml_correction': ml_correction,
                'hybrid': hybrid_pred
            }
        
        return result
    
    def recommend_dose(
        self,
        patient_data: Dict,
        target_auc: float,
        interval: float = 12.0,
        tdm_history: Optional[List[Dict]] = None,
        dose_range: Tuple[float, float] = (5.0, 25.0)  # mg/kg
    ) -> Dict:
        """
        Recommend dose using hybrid model
        
        Args:
            patient_data: Patient demographics and lab values
            target_auc: Target AUC (mg*h/L)
            interval: Dosing interval (hours)
            tdm_history: Previous TDM measurements (optional)
            dose_range: Dose range to search (mg/kg)
        
        Returns:
            Dose recommendation dictionary
        """
        weight = patient_data['weight_kg']
        egfr = patient_data.get('egfr', 100)
        
        # Start with popPK recommendation
        pk_params = self.bayesian_forecaster.estimate_parameters(
            weight=weight,
            egfr=egfr
        )
        
        poppk_recommendation = self.bayesian_forecaster.recommend_dose(
            target_auc=target_auc,
            weight=weight,
            interval=interval,
            pk_params=pk_params,
            egfr=egfr
        )
        
        initial_dose_per_kg = poppk_recommendation['recommended_dose_mg_per_kg']
        
        # Refine using hybrid model (binary search)
        dose_lower = max(dose_range[0], initial_dose_per_kg * 0.7)
        dose_upper = min(dose_range[1], initial_dose_per_kg * 1.3)
        
        best_dose = initial_dose_per_kg
        best_auc_error = float('inf')
        
        # Simple grid search
        for dose_per_kg in np.linspace(dose_lower, dose_upper, 20):
            dose = dose_per_kg * weight
            
            # Predict AUC using hybrid model
            # Sample multiple time points to estimate AUC
            times = np.linspace(0, interval, 10)
            concentrations = []
            
            for t in times:
                pred = self.predict(
                    patient_data=patient_data,
                    dose=dose,
                    time=t,
                    interval=interval,
                    n_doses=3,  # Steady state
                    tdm_history=tdm_history
                )
                concentrations.append(pred['predicted_concentration_mg_per_L'])
            
            # Estimate AUC using trapezoidal rule
            predicted_auc = calculate_auc(times, np.array(concentrations))
            
            # Calculate error
            auc_error = abs(predicted_auc - target_auc)
            
            if auc_error < best_auc_error:
                best_auc_error = auc_error
                best_dose = dose_per_kg
        
        return {
            'recommended_dose_mg_per_kg': best_dose,
            'recommended_dose_mg': best_dose * weight,
            'recommended_interval_hours': interval,
            'target_auc_mg_h_per_L': target_auc,
            'predicted_auc_mg_h_per_L': target_auc - best_auc_error,  # Approximate
            'poppk_recommendation': poppk_recommendation,
            'method': 'hybrid'
        }
    
    def save(self, filepath: str):
        """Save hybrid model"""
        import joblib
        
        # Save ML predictor
        ml_filepath = filepath.replace('.pkl', '_ml.pkl')
        if self.ml_predictor is not None:
            self.ml_predictor.save(ml_filepath)
        
        # Save hybrid model metadata
        joblib.dump({
            'drug_type': self.drug_type,
            'ml_model_type': self.ml_model_type,
            'use_residual_learning': self.use_residual_learning,
            'is_trained': self.is_trained,
            'ml_filepath': ml_filepath
        }, filepath)
    
    def load(self, filepath: str):
        """Load hybrid model"""
        import joblib
        
        data = joblib.load(filepath)
        self.drug_type = data['drug_type']
        self.ml_model_type = data['ml_model_type']
        self.use_residual_learning = data['use_residual_learning']
        self.is_trained = data['is_trained']
        
        # Reload PK model
        self.pk_model = OneCompartmentPKModel(drug_type=self.drug_type)
        self.bayesian_forecaster = BayesianForecaster(self.pk_model)
        
        # Load ML predictor
        if self.is_trained:
            ml_filepath = data.get('ml_filepath', filepath.replace('.pkl', '_ml.pkl'))
            self.ml_predictor = MLDosePredictor(model_type=self.ml_model_type)
            self.ml_predictor.load(ml_filepath)

