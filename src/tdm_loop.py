"""
TDM (Therapeutic Drug Monitoring) loop simulation
Online learning and dose adjustment based on TDM measurements
"""

import numpy as np
import pandas as pd
import argparse
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.hybrid import HybridDosePredictor
from src.poppk.bayesian_forecasting import BayesianForecaster, OneCompartmentPKModel
from src.utils import format_dose_recommendation, calculate_auc


class TDMLoop:
    """
    TDM loop for online dose adjustment
    """
    
    def __init__(
        self,
        drug_type: str = "vancomycin",
        model_type: str = "hybrid",
        target_auc: Optional[float] = None,
        target_trough_min: Optional[float] = None,
        target_trough_max: Optional[float] = None
    ):
        """
        Initialize TDM loop
        
        Args:
            drug_type: 'vancomycin' or 'amikacin'
            model_type: 'hybrid', 'poppk', or 'ml'
            target_auc: Target AUC (mg*h/L)
            target_trough_min: Minimum target trough (mg/L)
            target_trough_max: Maximum target trough (mg/L)
        """
        self.drug_type = drug_type
        self.model_type = model_type
        
        # Set targets
        if drug_type == "vancomycin":
            self.target_auc = target_auc if target_auc else 400  # mg*h/L
            self.target_trough_min = target_trough_min if target_trough_min else 10  # mg/L
            self.target_trough_max = target_trough_max if target_trough_max else 20  # mg/L
        elif drug_type == "amikacin":
            self.target_auc = target_auc if target_auc else 200  # mg*h/L
            self.target_trough_min = target_trough_min if target_trough_min else 2  # mg/L
            self.target_trough_max = target_trough_max if target_trough_max else 5  # mg/L
        else:
            raise ValueError(f"Unknown drug type: {drug_type}")
        
        # Initialize predictor
        if model_type == "hybrid":
            self.predictor = HybridDosePredictor(drug_type=drug_type)
        elif model_type == "poppk":
            self.predictor = None
            self.pk_model = OneCompartmentPKModel(drug_type=drug_type)
            self.bayesian_forecaster = BayesianForecaster(self.pk_model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # TDM history
        self.tdm_history = []
        self.dose_history = []
        self.recommendations_history = []
    
    def add_tdm_measurement(
        self,
        concentration: float,
        time: float,
        dose: float,
        dose_time: float = 0.0
    ):
        """
        Add TDM measurement to history
        
        Args:
            concentration: Measured concentration (mg/L)
            time: Time of measurement (hours since first dose)
            dose: Dose administered (mg)
            dose_time: Time of dose administration (hours since first dose)
        """
        self.tdm_history.append({
            'concentration': concentration,
            'time': time,
            'dose': dose,
            'dose_time': dose_time,
            'timestamp': datetime.now()
        })
    
    def add_dose(
        self,
        dose: float,
        time: float,
        interval: float = 12.0
    ):
        """
        Add dose to history
        
        Args:
            dose: Dose administered (mg)
            time: Time of administration (hours since first dose)
            interval: Dosing interval (hours)
        """
        self.dose_history.append({
            'dose': dose,
            'time': time,
            'interval': interval,
            'timestamp': datetime.now()
        })
    
    def get_current_recommendation(
        self,
        patient_data: Dict,
        interval: float = 12.0
    ) -> Dict:
        """
        Get current dose recommendation
        
        Args:
            patient_data: Patient demographics and lab values
            interval: Dosing interval (hours)
        
        Returns:
            Dose recommendation dictionary
        """
        if self.model_type == "hybrid":
            if not self.predictor.is_trained:
                # Use popPK only if ML not trained
                return self._get_poppk_recommendation(patient_data, interval)
            
            recommendation = self.predictor.recommend_dose(
                patient_data=patient_data,
                target_auc=self.target_auc,
                interval=interval,
                tdm_history=self.tdm_history
            )
        else:
            # popPK only
            recommendation = self._get_poppk_recommendation(patient_data, interval)
        
        # Add explanation
        recommendation['explanation'] = self._generate_explanation(
            recommendation,
            patient_data,
            len(self.tdm_history)
        )
        
        # Add risk assessment
        recommendation['risk_assessment'] = self._assess_risk(recommendation, patient_data)
        
        # Store recommendation
        self.recommendations_history.append({
            'recommendation': recommendation,
            'timestamp': datetime.now(),
            'tdm_count': len(self.tdm_history)
        })
        
        return recommendation
    
    def _get_poppk_recommendation(
        self,
        patient_data: Dict,
        interval: float
    ) -> Dict:
        """Get popPK-only recommendation"""
        weight = patient_data['weight_kg']
        egfr = patient_data.get('egfr', 100)
        
        # Update Bayesian forecaster with TDM history
        self.bayesian_forecaster.observed_concentrations = []
        self.bayesian_forecaster.observed_times = []
        self.bayesian_forecaster.doses = []
        self.bayesian_forecaster.dose_times = []
        
        for tdm in self.tdm_history:
            self.bayesian_forecaster.add_observation(
                tdm['concentration'],
                tdm['time'],
                tdm['dose'],
                dose_time=tdm['dose_time']
            )
        
        # Estimate parameters and recommend dose
        recommendation = self.bayesian_forecaster.recommend_dose(
            target_auc=self.target_auc,
            weight=weight,
            interval=interval,
            egfr=egfr
        )
        
        return recommendation
    
    def _generate_explanation(
        self,
        recommendation: Dict,
        patient_data: Dict,
        n_tdm: int
    ) -> str:
        """Generate explanation for recommendation"""
        dose_per_kg = recommendation['recommended_dose_mg_per_kg']
        egfr = patient_data.get('egfr', 100)
        age = patient_data.get('age', 50)
        
        explanation_parts = []
        
        if n_tdm == 0:
            explanation_parts.append("Initial dose recommendation based on population pharmacokinetics.")
        else:
            explanation_parts.append(f"Recommendation updated based on {n_tdm} TDM measurement(s).")
        
        if egfr < 30:
            explanation_parts.append("Patient has severe renal impairment; reduced clearance expected.")
        elif egfr < 60:
            explanation_parts.append("Patient has moderate renal impairment; clearance moderately reduced.")
        
        if dose_per_kg > 20:
            explanation_parts.append("Higher than typical dose recommended due to patient characteristics.")
        elif dose_per_kg < 10:
            explanation_parts.append("Lower than typical dose recommended due to renal function.")
        
        explanation_parts.append(f"Target AUC: {self.target_auc} mg*h/L.")
        
        return " ".join(explanation_parts)
    
    def _assess_risk(self, recommendation: Dict, patient_data: Dict) -> Dict:
        """Assess risk of recommendation"""
        dose_per_kg = recommendation['recommended_dose_mg_per_kg']
        egfr = patient_data.get('egfr', 100)
        
        risk_factors = []
        risk_level = "low"
        
        # High dose risk
        if dose_per_kg > 25:
            risk_factors.append("Very high dose")
            risk_level = "high"
        elif dose_per_kg > 20:
            risk_factors.append("High dose")
            risk_level = "moderate"
        
        # Renal impairment risk
        if egfr < 15:
            risk_factors.append("Severe renal impairment")
            risk_level = "high"
        elif egfr < 30:
            risk_factors.append("Moderate-severe renal impairment")
            if risk_level == "low":
                risk_level = "moderate"
        
        # Uncertainty risk
        ci_lower, ci_upper = recommendation.get('dose_confidence_interval', (dose_per_kg * 0.8, dose_per_kg * 1.2))
        ci_width = (ci_upper - ci_lower) / dose_per_kg
        
        if ci_width > 0.5:
            risk_factors.append("High uncertainty in prediction")
            if risk_level == "low":
                risk_level = "moderate"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'expert_review_required': risk_level == "high",
            'confidence_interval_width': ci_width
        }
    
    def simulate_tdm_cycle(
        self,
        patient_data: Dict,
        initial_dose: float,
        interval: float = 12.0,
        n_cycles: int = 3
    ) -> pd.DataFrame:
        """
        Simulate TDM cycle with dose adjustments
        
        Args:
            patient_data: Patient demographics and lab values
            initial_dose: Initial dose (mg)
            interval: Dosing interval (hours)
            n_cycles: Number of TDM cycles
        
        Returns:
            DataFrame with cycle results
        """
        results = []
        current_dose = initial_dose
        current_time = 0.0
        
        for cycle in range(n_cycles):
            # Administer dose
            self.add_dose(current_dose, current_time, interval)
            
            # Simulate TDM measurement (after first dose or at steady state)
            if cycle == 0:
                # First TDM: 1-2 hours post-infusion
                tdm_time = current_time + 1.0 + np.random.uniform(0, 1)
            else:
                # Steady state TDM: trough (just before next dose)
                tdm_time = current_time + interval - 0.5
            
            # Predict concentration (simplified: use popPK for simulation)
            if self.model_type == "hybrid" and self.predictor.is_trained:
                pred = self.predictor.predict(
                    patient_data=patient_data,
                    dose=current_dose,
                    time=tdm_time - current_time,
                    interval=interval,
                    n_doses=cycle + 1,
                    tdm_history=self.tdm_history
                )
                simulated_conc = pred['predicted_concentration_mg_per_L']
            else:
                # Use popPK for simulation
                pk_params = self.bayesian_forecaster.estimate_parameters(
                    weight=patient_data['weight_kg'],
                    egfr=patient_data.get('egfr', 100)
                )
                simulated_conc = self.pk_model.predict_concentration(
                    time=tdm_time - current_time,
                    dose=current_dose,
                    clearance=pk_params['clearance_L_per_h'],
                    volume=pk_params['volume_L'],
                    n_doses=cycle + 1,
                    interval=interval
                )
            
            # Add measurement noise
            measured_conc = simulated_conc * np.random.lognormal(0, 0.1)
            
            # Add TDM measurement
            self.add_tdm_measurement(
                concentration=measured_conc,
                time=tdm_time,
                dose=current_dose,
                dose_time=current_time
            )
            
            # Get new recommendation
            recommendation = self.get_current_recommendation(patient_data, interval)
            
            # Update dose for next cycle
            new_dose_per_kg = recommendation['recommended_dose_mg_per_kg']
            new_dose = new_dose_per_kg * patient_data['weight_kg']
            
            # Check if target achieved
            target_achieved = (
                self.target_trough_min <= measured_conc <= self.target_trough_max
            ) if self.drug_type == "vancomycin" else (measured_conc <= self.target_trough_max)
            
            results.append({
                'cycle': cycle + 1,
                'time_hours': tdm_time,
                'dose_mg': current_dose,
                'dose_mg_per_kg': current_dose / patient_data['weight_kg'],
                'measured_conc_mg_per_L': measured_conc,
                'predicted_conc_mg_per_L': simulated_conc,
                'target_trough_min': self.target_trough_min,
                'target_trough_max': self.target_trough_max,
                'target_achieved': target_achieved,
                'recommended_dose_mg_per_kg': new_dose_per_kg,
                'recommended_dose_mg': new_dose,
                'dose_change_percent': ((new_dose - current_dose) / current_dose) * 100,
                'risk_level': recommendation['risk_assessment']['risk_level'],
                'expert_review_required': recommendation['risk_assessment']['expert_review_required']
            })
            
            # Update for next cycle
            current_dose = new_dose
            current_time += interval
        
        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Simulate TDM loop')
    parser.add_argument('--patient', type=str, required=True, help='Patient data JSON file')
    parser.add_argument('--model', type=str, default='hybrid', choices=['hybrid', 'poppk'])
    parser.add_argument('--drug', type=str, default='vancomycin', choices=['vancomycin', 'amikacin'])
    parser.add_argument('--model_file', type=str, help='Trained model file (for hybrid)')
    parser.add_argument('--output', type=str, default='outputs/tdm_loop_results.csv')
    parser.add_argument('--n_cycles', type=int, default=3, help='Number of TDM cycles')
    
    args = parser.parse_args()
    
    # Load patient data
    with open(args.patient, 'r') as f:
        patient_data = json.load(f)
    
    # Initialize TDM loop
    tdm_loop = TDMLoop(
        drug_type=args.drug,
        model_type=args.model
    )
    
    # Load trained model if provided
    if args.model == 'hybrid' and args.model_file:
        tdm_loop.predictor.load(args.model_file)
        print(f"Loaded trained model from {args.model_file}")
    
    # Get initial recommendation
    initial_recommendation = tdm_loop.get_current_recommendation(patient_data)
    initial_dose = initial_recommendation['recommended_dose_mg']
    interval = initial_recommendation['recommended_interval_hours']
    
    print(f"Initial recommendation: {initial_recommendation['recommended_dose_mg_per_kg']:.2f} mg/kg")
    print(f"Explanation: {initial_recommendation['explanation']}")
    print(f"Risk level: {initial_recommendation['risk_assessment']['risk_level']}")
    
    # Simulate TDM cycles
    print(f"\nSimulating {args.n_cycles} TDM cycles...")
    results = tdm_loop.simulate_tdm_cycle(
        patient_data=patient_data,
        initial_dose=initial_dose,
        interval=interval,
        n_cycles=args.n_cycles
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\nTDM Cycle Summary:")
    print(results[['cycle', 'dose_mg_per_kg', 'measured_conc_mg_per_L', 
                   'target_achieved', 'recommended_dose_mg_per_kg', 'risk_level']].to_string())
    
    target_achievement_rate = results['target_achieved'].mean()
    print(f"\nTarget achievement rate: {target_achievement_rate:.2%}")


if __name__ == '__main__':
    main()

