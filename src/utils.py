"""
Utility functions for MIPD system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def calculate_egfr(creatinine: float, age: int, sex: str, race: str = "non-black") -> float:
    """
    Calculate eGFR using CKD-EPI equation
    
    Args:
        creatinine: Serum creatinine (mg/dL)
        age: Age in years
        sex: 'M' or 'F'
        race: 'black' or 'non-black'
    
    Returns:
        eGFR in mL/min/1.73m²
    """
    if sex.upper() == 'F':
        if race.lower() == 'black':
            k = 0.7
            alpha = -0.329
            min_val = 0.993 ** age
        else:
            k = 0.7
            alpha = -0.329
            min_val = 0.993 ** age
    else:  # Male
        if race.lower() == 'black':
            k = 0.9
            alpha = -0.411
            min_val = 1.0
        else:
            k = 0.9
            alpha = -0.411
            min_val = 1.0
    
    egfr = 141 * (min_val) * (creatinine / k) ** alpha
    if sex.upper() == 'F':
        egfr *= 1.018
    
    if race.lower() == 'black':
        egfr *= 1.159
    
    return max(egfr, 0)  # Ensure non-negative


def calculate_auc(times: np.ndarray, concentrations: np.ndarray) -> float:
    """
    Calculate AUC using trapezoidal rule
    
    Args:
        times: Time points (hours)
        concentrations: Drug concentrations (mg/L)
    
    Returns:
        AUC in mg*h/L
    """
    if len(times) < 2:
        return 0.0
    
    # Sort by time
    sorted_indices = np.argsort(times)
    times_sorted = times[sorted_indices]
    conc_sorted = concentrations[sorted_indices]
    
    # Calculate trapezoidal AUC
    auc = np.trapz(conc_sorted, times_sorted)
    return max(auc, 0.0)


def calculate_clearance_from_auc(dose: float, auc: float) -> float:
    """
    Calculate clearance from dose and AUC
    
    Args:
        dose: Total dose administered (mg)
        auc: Area under the curve (mg*h/L)
    
    Returns:
        Clearance in L/h
    """
    if auc <= 0:
        return np.nan
    return dose / auc


def predict_trough_concentration(
    dose: float,
    interval: float,
    clearance: float,
    volume: float,
    ka: float = 0.0,  # Absorption rate (0 for IV)
    n_doses: int = 1
) -> float:
    """
    Predict trough concentration at steady state
    
    Args:
        dose: Dose per administration (mg)
        interval: Dosing interval (hours)
        clearance: Clearance (L/h)
        volume: Volume of distribution (L)
        ka: Absorption rate constant (1/h, 0 for IV)
        n_doses: Number of doses
    
    Returns:
        Predicted trough concentration (mg/L)
    """
    ke = clearance / volume  # Elimination rate constant
    
    if ka > 0:
        # Oral administration
        c_min = (dose * ka / (volume * (ka - ke))) * (
            (np.exp(-ke * interval) / (1 - np.exp(-ke * interval))) -
            (np.exp(-ka * interval) / (1 - np.exp(-ka * interval)))
        )
    else:
        # IV administration
        if n_doses == 1:
            c_min = 0.0
        else:
            c_min = (dose / volume) * (np.exp(-ke * interval) / (1 - np.exp(-ke * interval)))
    
    return max(c_min, 0.0)


def calculate_creatinine_clearance(
    creatinine: float,
    age: int,
    weight: float,
    sex: str
) -> float:
    """
    Calculate creatinine clearance using Cockcroft-Gault equation
    
    Args:
        creatinine: Serum creatinine (mg/dL)
        age: Age in years
        weight: Weight in kg
        sex: 'M' or 'F'
    
    Returns:
        Creatinine clearance in mL/min
    """
    if creatinine <= 0:
        return np.nan
    
    factor = 1.0 if sex.upper() == 'M' else 0.85
    crcl = ((140 - age) * weight * factor) / (72 * creatinine)
    return max(crcl, 0)


def format_dose_recommendation(
    recommended_dose: float,
    interval: float,
    confidence_interval: Tuple[float, float],
    target_auc: Optional[float] = None,
    predicted_auc: Optional[float] = None,
    explanation: Optional[str] = None
) -> Dict:
    """
    Format dose recommendation output
    
    Args:
        recommended_dose: Recommended dose (mg/kg)
        interval: Recommended interval (hours)
        confidence_interval: (lower, upper) bounds for dose
        target_auc: Target AUC (mg*h/L)
        predicted_auc: Predicted AUC (mg*h/L)
        explanation: Explanation of recommendation
    
    Returns:
        Formatted recommendation dictionary
    """
    return {
        "recommended_dose_mg_per_kg": recommended_dose,
        "recommended_interval_hours": interval,
        "dose_confidence_interval": {
            "lower": confidence_interval[0],
            "upper": confidence_interval[1]
        },
        "target_auc_mg_h_per_L": target_auc,
        "predicted_auc_mg_h_per_L": predicted_auc,
        "explanation": explanation,
        "timestamp": datetime.now().isoformat()
    }
