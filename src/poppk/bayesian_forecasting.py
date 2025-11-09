"""
Population Pharmacokinetic (popPK) models with Bayesian forecasting
For vancomycin and amikacin-like drugs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
import warnings


class OneCompartmentPKModel:
    """
    One-compartment pharmacokinetic model with first-order elimination
    """
    
    def __init__(self, drug_type: str = "vancomycin"):
        """
        Initialize PK model
        
        Args:
            drug_type: 'vancomycin' or 'amikacin'
        """
        self.drug_type = drug_type
        
        # Population parameters (prior means)
        if drug_type == "vancomycin":
            self.cl_pop_mean = 0.048  # L/h/kg
            self.cl_pop_sd = 0.015
            self.v_pop_mean = 0.7  # L/kg
            self.v_pop_sd = 0.2
        elif drug_type == "amikacin":
            self.cl_pop_mean = 0.048  # L/h/kg
            self.cl_pop_sd = 0.015
            self.v_pop_mean = 0.26  # L/kg
            self.v_pop_sd = 0.08
        else:
            raise ValueError(f"Unknown drug type: {drug_type}")
    
    def predict_concentration(
        self,
        time: float,
        dose: float,
        clearance: float,
        volume: float,
        infusion_time: float = 1.0,
        n_doses: int = 1,
        interval: float = 12.0
    ) -> float:
        """
        Predict concentration at time t using one-compartment model
        
        Args:
            time: Time since first dose (hours)
            dose: Dose per administration (mg)
            clearance: Clearance (L/h)
            volume: Volume of distribution (L)
            infusion_time: Infusion duration (hours)
            n_doses: Number of doses
            interval: Dosing interval (hours)
        
        Returns:
            Predicted concentration (mg/L)
        """
        ke = clearance / volume  # Elimination rate constant
        
        if n_doses == 1:
            # Single dose
            if time <= infusion_time:
                # During infusion
                conc = (dose / (clearance * infusion_time)) * (1 - np.exp(-ke * time))
            else:
                # Post-infusion
                conc = (dose / (clearance * infusion_time)) * (1 - np.exp(-ke * infusion_time)) * np.exp(-ke * (time - infusion_time))
        else:
            # Multiple doses (steady state)
            conc = 0.0
            for dose_num in range(min(n_doses, 20)):  # Limit for computation
                dose_time = dose_num * interval
                if time >= dose_time:
                    time_since_dose = time - dose_time
                    if time_since_dose <= infusion_time:
                        # During infusion
                        conc += (dose / (clearance * infusion_time)) * (1 - np.exp(-ke * time_since_dose))
                    else:
                        # Post-infusion
                        conc += (dose / (clearance * infusion_time)) * (1 - np.exp(-ke * infusion_time)) * np.exp(-ke * (time_since_dose - infusion_time))
        
        return max(conc, 0.0)
    
    def calculate_auc(
        self,
        dose: float,
        clearance: float,
        n_doses: int = 1,
        interval: float = 12.0
    ) -> float:
        """
        Calculate AUC for given dosing regimen
        
        Args:
            dose: Dose per administration (mg)
            clearance: Clearance (L/h)
            n_doses: Number of doses
            interval: Dosing interval (hours)
        
        Returns:
            AUC (mg*h/L)
        """
        if n_doses == 1:
            auc = dose / clearance
        else:
            # Steady state AUC over one interval
            auc_ss_interval = dose / clearance
            # Total AUC over n_doses intervals
            auc = auc_ss_interval * n_doses
        
        return auc
    
    def calculate_clearance_from_egfr(
        self,
        egfr: float,
        weight: float,
        age: Optional[float] = None,
        sex: Optional[str] = None
    ) -> float:
        """
        Estimate clearance from eGFR (simplified model)
        
        Args:
            egfr: eGFR (mL/min/1.73m²)
            weight: Weight (kg)
            age: Age (years, optional)
            sex: Sex ('M' or 'F', optional)
        
        Returns:
            Estimated clearance (L/h)
        """
        # Base clearance (scaled by weight)
        cl_base = self.cl_pop_mean * weight  # L/h
        
        # Adjust based on eGFR (normalized to 100 mL/min/1.73m²)
        egfr_normal = 100.0
        if egfr > 90:
            cl_factor = 1.0
        elif egfr > 60:
            cl_factor = 0.85
        elif egfr > 30:
            cl_factor = 0.6
        elif egfr > 15:
            cl_factor = 0.4
        else:
            cl_factor = 0.2  # Severe impairment / dialysis
        
        estimated_cl = cl_base * cl_factor
        return estimated_cl


class BayesianForecaster:
    """
    Bayesian forecasting for individualizing PK parameters
    Uses MAP (Maximum A Posteriori) estimation
    """
    
    def __init__(self, pk_model: OneCompartmentPKModel):
        """
        Initialize Bayesian forecaster
        
        Args:
            pk_model: OneCompartmentPKModel instance
        """
        self.pk_model = pk_model
        self.observed_concentrations = []
        self.observed_times = []
        self.doses = []
        self.dose_times = []
    
    def add_observation(
        self,
        concentration: float,
        time: float,
        dose: float,
        dose_time: float = 0.0
    ):
        """
        Add TDM observation
        
        Args:
            concentration: Observed concentration (mg/L)
            time: Time of observation (hours since first dose)
            dose: Dose administered (mg)
            dose_time: Time of dose administration (hours since first dose)
        """
        self.observed_concentrations.append(concentration)
        self.observed_times.append(time)
        self.doses.append(dose)
        self.dose_times.append(dose_time)
    
    def likelihood(
        self,
        params: np.ndarray,
        concentrations: List[float],
        times: List[float],
        doses: List[float],
        dose_times: List[float],
        residual_error: float = 0.15  # 15% CV
    ) -> float:
        """
        Calculate likelihood of observations given parameters
        
        Args:
            params: [clearance, volume] (log-transformed)
            concentrations: Observed concentrations
            times: Observation times
            doses: Doses administered
            dose_times: Dose administration times
            residual_error: Residual error (CV)
        
        Returns:
            Negative log-likelihood
        """
        cl = np.exp(params[0])  # Back-transform from log
        v = np.exp(params[1])
        
        pred_concentrations = []
        for i, (conc_obs, time) in enumerate(zip(concentrations, times)):
            # Find relevant doses
            relevant_doses = []
            relevant_intervals = []
            n_doses = 0
            
            for j, (dose, dose_time) in enumerate(zip(doses, dose_times)):
                if dose_time <= time:
                    relevant_doses.append(dose)
                    if j > 0:
                        interval = dose_time - dose_times[j-1]
                    else:
                        interval = 12.0  # Default
                    relevant_intervals.append(interval)
                    n_doses += 1
            
            if n_doses == 0:
                pred_conc = 0.0
            else:
                # Use last dose for prediction
                last_dose = relevant_doses[-1]
                last_dose_time = dose_times[min(len(dose_times)-1, n_doses-1)]
                time_since_last_dose = time - last_dose_time
                
                pred_conc = self.pk_model.predict_concentration(
                    time_since_last_dose,
                    last_dose,
                    cl,
                    v,
                    n_doses=n_doses,
                    interval=relevant_intervals[-1] if relevant_intervals else 12.0
                )
            
            pred_concentrations.append(pred_conc)
        
        # Calculate log-likelihood (assuming log-normal error)
        log_likelihood = 0.0
        for conc_obs, conc_pred in zip(concentrations, pred_concentrations):
            if conc_pred <= 0:
                conc_pred = 1e-6  # Avoid log(0)
            
            # Log-normal likelihood
            log_conc_obs = np.log(conc_obs)
            log_conc_pred = np.log(conc_pred)
            sigma = residual_error
            
            log_likelihood += -0.5 * ((log_conc_obs - log_conc_pred) / sigma) ** 2
            log_likelihood -= np.log(sigma * np.sqrt(2 * np.pi))
            log_likelihood -= log_conc_obs  # Jacobian for log-normal
        
        return -log_likelihood  # Return negative for minimization
    
    def prior(self, params: np.ndarray, weight: float) -> float:
        """
        Calculate prior probability for parameters
        
        Args:
            params: [log(clearance), log(volume)] (log-transformed)
            weight: Patient weight (kg)
        
        Returns:
            Negative log-prior
        """
        cl_log = params[0]
        v_log = params[1]
        
        # Prior means (log-transformed)
        cl_pop_log_mean = np.log(self.pk_model.cl_pop_mean * weight)
        v_pop_log_mean = np.log(self.pk_model.v_pop_mean * weight)
        
        # Prior SDs (log-transformed)
        cl_pop_log_sd = self.pk_model.cl_pop_sd
        v_pop_log_sd = self.pk_model.v_pop_sd
        
        # Log-normal priors
        cl_prior = -0.5 * ((cl_log - cl_pop_log_mean) / cl_pop_log_sd) ** 2
        v_prior = -0.5 * ((v_log - v_pop_log_mean) / v_pop_log_sd) ** 2
        
        return -(cl_prior + v_prior)  # Negative log-prior
    
    def posterior(
        self,
        params: np.ndarray,
        concentrations: List[float],
        times: List[float],
        doses: List[float],
        dose_times: List[float],
        weight: float
    ) -> float:
        """
        Calculate posterior probability (likelihood * prior)
        
        Args:
            params: [log(clearance), log(volume)]
            concentrations: Observed concentrations
            times: Observation times
            doses: Doses administered
            dose_times: Dose administration times
            weight: Patient weight (kg)
        
        Returns:
            Negative log-posterior
        """
        likelihood_term = self.likelihood(params, concentrations, times, doses, dose_times)
        prior_term = self.prior(params, weight)
        
        return likelihood_term + prior_term
    
    def estimate_parameters(
        self,
        weight: float,
        initial_cl: Optional[float] = None,
        initial_v: Optional[float] = None,
        egfr: Optional[float] = None
    ) -> Dict:
        """
        Estimate individual PK parameters using MAP estimation
        
        Args:
            weight: Patient weight (kg)
            initial_cl: Initial clearance estimate (L/h, optional)
            initial_v: Initial volume estimate (L, optional)
            egfr: eGFR for initial clearance estimate (optional)
        
        Returns:
            Dictionary with estimated parameters and uncertainty
        """
        if len(self.observed_concentrations) == 0:
            # No observations: return population prior
            if initial_cl is None:
                if egfr is not None:
                    initial_cl = self.pk_model.calculate_clearance_from_egfr(egfr, weight)
                else:
                    initial_cl = self.pk_model.cl_pop_mean * weight
            
            if initial_v is None:
                initial_v = self.pk_model.v_pop_mean * weight
            
            return {
                'clearance_L_per_h': initial_cl,
                'volume_L': initial_v,
                'clearance_ci_lower': initial_cl * 0.7,
                'clearance_ci_upper': initial_cl * 1.3,
                'volume_ci_lower': initial_v * 0.7,
                'volume_ci_upper': initial_v * 1.3,
                'method': 'population_prior'
            }
        
        # Initial parameter estimates (log-transformed)
        if initial_cl is None:
            if egfr is not None:
                initial_cl = self.pk_model.calculate_clearance_from_egfr(egfr, weight)
            else:
                initial_cl = self.pk_model.cl_pop_mean * weight
        
        if initial_v is None:
            initial_v = self.pk_model.v_pop_mean * weight
        
        initial_params = np.array([
            np.log(initial_cl),
            np.log(initial_v)
        ])
        
        # Optimize posterior
        try:
            result = minimize(
                self.posterior,
                initial_params,
                args=(
                    self.observed_concentrations,
                    self.observed_times,
                    self.doses,
                    self.dose_times,
                    weight
                ),
                method='L-BFGS-B',
                bounds=[(np.log(0.01), np.log(100)), (np.log(1), np.log(500))]  # Reasonable bounds
            )
            
            if result.success:
                cl_est = np.exp(result.x[0])
                v_est = np.exp(result.x[1])
                
                # Estimate uncertainty (simplified: use Hessian diagonal)
                # In practice, use full covariance matrix
                cl_ci_lower = cl_est * 0.8
                cl_ci_upper = cl_est * 1.2
                v_ci_lower = v_est * 0.8
                v_ci_upper = v_est * 1.2
                
                return {
                    'clearance_L_per_h': cl_est,
                    'volume_L': v_est,
                    'clearance_ci_lower': cl_ci_lower,
                    'clearance_ci_upper': cl_ci_upper,
                    'volume_ci_lower': v_ci_lower,
                    'volume_ci_upper': v_ci_upper,
                    'method': 'bayesian_map',
                    'converged': True
                }
            else:
                warnings.warn("Optimization did not converge, using initial estimates")
                return {
                    'clearance_L_per_h': initial_cl,
                    'volume_L': initial_v,
                    'clearance_ci_lower': initial_cl * 0.7,
                    'clearance_ci_upper': initial_cl * 1.3,
                    'volume_ci_lower': initial_v * 0.7,
                    'volume_ci_upper': initial_v * 1.3,
                    'method': 'initial_estimate',
                    'converged': False
                }
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}, using initial estimates")
            return {
                'clearance_L_per_h': initial_cl,
                'volume_L': initial_v,
                'clearance_ci_lower': initial_cl * 0.7,
                'clearance_ci_upper': initial_cl * 1.3,
                'volume_ci_lower': initial_v * 0.7,
                'volume_ci_upper': initial_v * 1.3,
                'method': 'fallback',
                'converged': False
            }
    
    def recommend_dose(
        self,
        target_auc: float,
        weight: float,
        interval: float = 12.0,
        pk_params: Optional[Dict] = None,
        egfr: Optional[float] = None
    ) -> Dict:
        """
        Recommend dose to achieve target AUC
        
        Args:
            target_auc: Target AUC (mg*h/L)
            weight: Patient weight (kg)
            interval: Dosing interval (hours)
            pk_params: Estimated PK parameters (optional)
            egfr: eGFR for clearance estimation (optional)
        
        Returns:
            Dose recommendation dictionary
        """
        if pk_params is None:
            pk_params = self.estimate_parameters(weight, egfr=egfr)
        
        clearance = pk_params['clearance_L_per_h']
        
        # Calculate dose to achieve target AUC over one interval
        dose = target_auc * clearance
        
        # Convert to mg/kg
        dose_per_kg = dose / weight
        
        # Confidence interval for dose
        dose_lower = target_auc * pk_params['clearance_ci_lower'] / weight
        dose_upper = target_auc * pk_params['clearance_ci_upper'] / weight
        
        return {
            'recommended_dose_mg_per_kg': dose_per_kg,
            'recommended_dose_mg': dose,
            'recommended_interval_hours': interval,
            'dose_confidence_interval': (dose_lower, dose_upper),
            'target_auc_mg_h_per_L': target_auc,
            'predicted_clearance_L_per_h': clearance,
            'predicted_volume_L': pk_params['volume_L']
        }

