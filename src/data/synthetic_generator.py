"""
Synthetic TDM data generator for vancomycin/amikacin-like drugs
Simulates patient data with renal impairment scenarios
"""

import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import calculate_egfr, calculate_auc


class SyntheticTDMGenerator:
    """
    Generate synthetic TDM data for vancomycin/amikacin-like drugs
    """
    
    def __init__(self, drug_type: str = "vancomycin", seed: int = 42):
        """
        Initialize generator
        
        Args:
            drug_type: 'vancomycin' or 'amikacin'
            seed: Random seed
        """
        self.drug_type = drug_type
        np.random.seed(seed)
        
        # Drug-specific parameters (based on literature)
        if drug_type == "vancomycin":
            self.v_pop = 0.7  # L/kg - population mean volume
            self.cl_pop = 0.048  # L/h/kg - population mean clearance
            self.v_sd = 0.2  # SD for volume
            self.cl_sd = 0.015  # SD for clearance
            self.target_auc = 400  # mg*h/L (typical target)
            self.target_trough_min = 10  # mg/L
            self.target_trough_max = 20  # mg/L
        elif drug_type == "amikacin":
            self.v_pop = 0.26  # L/kg
            self.cl_pop = 0.048  # L/h/kg
            self.v_sd = 0.08
            self.cl_sd = 0.015
            self.target_auc = 200  # mg*h/L
            self.target_trough_min = 2  # mg/L
            self.target_trough_max = 5  # mg/L
        else:
            raise ValueError(f"Unknown drug type: {drug_type}")
    
    def generate_patient_demographics(self, n_patients: int) -> pd.DataFrame:
        """Generate patient demographics"""
        patients = []
        
        for i in range(n_patients):
            age = np.random.randint(18, 90)
            sex = np.random.choice(['M', 'F'], p=[0.6, 0.4])
            weight = np.random.normal(70, 15)  # kg
            weight = max(40, min(weight, 150))  # Clip to reasonable range
            height = np.random.normal(170, 10) if sex == 'M' else np.random.normal(160, 8)
            height = max(140, min(height, 200))
            
            # Renal impairment status
            renal_status = np.random.choice(
                ['normal', 'mild', 'moderate', 'severe', 'dialysis'],
                p=[0.3, 0.25, 0.25, 0.15, 0.05]
            )
            
            patients.append({
                'patient_id': f'PAT_{i+1:04d}',
                'age': age,
                'sex': sex,
                'weight_kg': weight,
                'height_cm': height,
                'renal_status': renal_status
            })
        
        return pd.DataFrame(patients)
    
    def generate_creatinine_timeseries(
        self,
        patient: Dict,
        n_days: int = 14,
        n_measurements: int = 10
    ) -> pd.DataFrame:
        """Generate creatinine time series"""
        base_cr = {
            'normal': np.random.normal(0.9, 0.2),
            'mild': np.random.normal(1.3, 0.3),
            'moderate': np.random.normal(2.0, 0.5),
            'severe': np.random.normal(4.0, 1.0),
            'dialysis': np.random.normal(6.0, 1.5)
        }
        
        cr_mean = base_cr[patient['renal_status']]
        cr_values = []
        times = []
        
        start_time = datetime.now() - timedelta(days=n_days)
        
        for i in range(n_measurements):
            time_offset = timedelta(days=i * (n_days / n_measurements))
            time = start_time + time_offset
            
            # Add some temporal variation
            cr = cr_mean + np.random.normal(0, cr_mean * 0.1)
            cr = max(0.5, cr)  # Minimum realistic value
            
            cr_values.append(cr)
            times.append(time)
        
        df = pd.DataFrame({
            'patient_id': patient['patient_id'],
            'timestamp': times,
            'creatinine_mg_dL': cr_values
        })
        
        # Calculate eGFR
        df['egfr'] = df.apply(
            lambda row: calculate_egfr(
                row['creatinine_mg_dL'],
                patient['age'],
                patient['sex']
            ),
            axis=1
        )
        
        return df
    
    def generate_pk_parameters(self, patient: Dict, creatinine: float) -> Dict:
        """
        Generate patient-specific PK parameters
        Clearance depends on renal function
        """
        # Base clearance (scaled by weight)
        cl_base = self.cl_pop * patient['weight_kg']
        
        # Adjust clearance based on eGFR
        egfr = calculate_egfr(creatinine, patient['age'], patient['sex'])
        egfr_normal = 100  # Reference eGFR
        
        # Clearance scaling factor (simplified model)
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
        
        # Add inter-individual variability
        cl_individual = cl_base * cl_factor * np.random.lognormal(0, self.cl_sd)
        
        # Volume of distribution (scaled by weight, less affected by renal function)
        v_individual = self.v_pop * patient['weight_kg'] * np.random.lognormal(0, self.v_sd)
        
        return {
            'clearance_L_per_h': cl_individual,
            'volume_L': v_individual,
            'ke_1_per_h': cl_individual / v_individual,  # Elimination rate constant
            'half_life_h': np.log(2) / (cl_individual / v_individual)
        }
    
    def simulate_concentration_profile(
        self,
        dose: float,
        pk_params: Dict,
        infusion_time: float = 1.0,  # hours
        sampling_times: Optional[np.ndarray] = None,
        n_doses: int = 1,
        interval: float = 12.0
    ) -> pd.DataFrame:
        """
        Simulate drug concentration profile using one-compartment model
        """
        if sampling_times is None:
            # Default sampling: pre-dose, 1h, 2h, 6h, 12h, 24h
            sampling_times = np.array([0, 1, 2, 6, 12, 24])
        
        cl = pk_params['clearance_L_per_h']
        v = pk_params['volume_L']
        ke = pk_params['ke_1_per_h']
        
        concentrations = []
        
        for time in sampling_times:
            conc = 0.0
            
            if n_doses == 1:
                # Single dose
                if time <= infusion_time:
                    # During infusion
                    conc = (dose / (cl * infusion_time)) * (1 - np.exp(-ke * time))
                else:
                    # Post-infusion
                    conc = (dose / (cl * infusion_time)) * (1 - np.exp(-ke * infusion_time)) * np.exp(-ke * (time - infusion_time))
            else:
                # Multiple doses (steady state approximation)
                for dose_num in range(min(n_doses, 10)):  # Limit to 10 doses for computation
                    dose_time = dose_num * interval
                    if time >= dose_time:
                        if time <= dose_time + infusion_time:
                            # During infusion
                            conc += (dose / (cl * infusion_time)) * (1 - np.exp(-ke * (time - dose_time)))
                        else:
                            # Post-infusion
                            time_since_dose = time - dose_time
                            conc += (dose / (cl * infusion_time)) * (1 - np.exp(-ke * infusion_time)) * np.exp(-ke * (time_since_dose - infusion_time))
            
            # Add measurement noise (10% CV)
            conc_with_noise = conc * np.random.lognormal(0, 0.1)
            concentrations.append(max(conc_with_noise, 0.01))  # Minimum detectable
        
        df = pd.DataFrame({
            'time_hours': sampling_times,
            'concentration_mg_per_L': concentrations
        })
        
        return df
    
    def generate_patient_tdm_data(
        self,
        patient: Dict,
        n_days: int = 7,
        initial_dose: Optional[float] = None
    ) -> Dict:
        """Generate complete TDM data for a patient"""
        
        # Generate creatinine time series
        cr_data = self.generate_creatinine_timeseries(patient, n_days=n_days)
        
        # Use average creatinine for PK parameter estimation
        avg_cr = cr_data['creatinine_mg_dL'].mean()
        avg_egfr = cr_data['egfr'].mean()
        
        # Generate PK parameters
        pk_params = self.generate_pk_parameters(patient, avg_cr)
        
        # Determine initial dose (if not provided)
        if initial_dose is None:
            # Simple dose calculation based on weight and renal function
            if avg_egfr > 60:
                dose_mg_per_kg = 15 if self.drug_type == "vancomycin" else 7.5
            elif avg_egfr > 30:
                dose_mg_per_kg = 10 if self.drug_type == "vancomycin" else 5
            else:
                dose_mg_per_kg = 7.5 if self.drug_type == "vancomycin" else 3.5
            
            initial_dose = dose_mg_per_kg * patient['weight_kg']
        
        # Determine dosing interval based on renal function
        if avg_egfr > 60:
            interval = 12.0
        elif avg_egfr > 30:
            interval = 24.0
        else:
            interval = 48.0
        
        # Simulate TDM measurements
        n_doses = max(1, int(n_days * 24 / interval))
        
        # First TDM: after first dose (random timing: 1-4 hours post-infusion)
        first_tdm_time = 1.0 + np.random.uniform(0, 3)
        sampling_times_dose1 = np.array([0, first_tdm_time, interval])
        
        conc_profile_1 = self.simulate_concentration_profile(
            initial_dose,
            pk_params,
            sampling_times=sampling_times_dose1,
            n_doses=1,
            interval=interval
        )
        
        # Second TDM: steady state (trough before next dose)
        trough_time = interval - 0.5  # 30 min before next dose
        sampling_times_ss = np.array([0, interval/4, interval/2, trough_time, interval])
        
        conc_profile_ss = self.simulate_concentration_profile(
            initial_dose,
            pk_params,
            sampling_times=sampling_times_ss,
            n_doses=3,  # Steady state
            interval=interval
        )
        
        # Combine TDM data
        tdm_data = []
        for _, row in conc_profile_1.iterrows():
            if row['time_hours'] > 0:  # Skip pre-dose
                tdm_data.append({
                    'patient_id': patient['patient_id'],
                    'dose_number': 1,
                    'time_since_dose_hours': row['time_hours'],
                    'concentration_mg_per_L': row['concentration_mg_per_L'],
                    'timestamp': datetime.now() + timedelta(hours=row['time_hours'])
                })
        
        for _, row in conc_profile_ss.iterrows():
            if row['time_hours'] > 0 and row['time_hours'] < interval:
                tdm_data.append({
                    'patient_id': patient['patient_id'],
                    'dose_number': 3,
                    'time_since_dose_hours': row['time_hours'],
                    'concentration_mg_per_L': row['concentration_mg_per_L'],
                    'timestamp': datetime.now() + timedelta(days=2, hours=row['time_hours'])
                })
        
        # Calculate AUC (simplified: using trapezoidal rule on full profile)
        full_times = np.concatenate([conc_profile_1['time_hours'].values, 
                                     conc_profile_ss['time_hours'].values + interval * 2])
        full_conc = np.concatenate([conc_profile_1['concentration_mg_per_L'].values,
                                    conc_profile_ss['concentration_mg_per_L'].values])
        observed_auc = calculate_auc(full_times, full_conc)
        
        # Determine if target was achieved
        trough_conc = conc_profile_ss.loc[conc_profile_ss['time_hours'].idxmax(), 'concentration_mg_per_L']
        target_achieved = (
            self.target_trough_min <= trough_conc <= self.target_trough_max
        ) if self.drug_type == "vancomycin" else (trough_conc <= self.target_trough_max)
        
        return {
            'patient': patient,
            'creatinine_data': cr_data,
            'pk_parameters': pk_params,
            'dosing': {
                'initial_dose_mg': initial_dose,
                'dose_mg_per_kg': initial_dose / patient['weight_kg'],
                'interval_hours': interval,
                'n_doses': n_doses
            },
            'tdm_measurements': pd.DataFrame(tdm_data),
            'observed_auc': observed_auc,
            'target_auc': self.target_auc,
            'trough_concentration': trough_conc,
            'target_achieved': target_achieved,
            'avg_egfr': avg_egfr
        }
    
    def generate_dataset(self, n_patients: int = 200, n_days: int = 7) -> pd.DataFrame:
        """Generate complete dataset"""
        patients_df = self.generate_patient_demographics(n_patients)
        
        all_data = []
        
        for _, patient_row in patients_df.iterrows():
            patient_dict = patient_row.to_dict()
            tdm_data = self.generate_patient_tdm_data(patient_dict, n_days=n_days)
            
            # Flatten data for DataFrame
            for _, tdm_row in tdm_data['tdm_measurements'].iterrows():
                all_data.append({
                    'patient_id': patient_dict['patient_id'],
                    'age': patient_dict['age'],
                    'sex': patient_dict['sex'],
                    'weight_kg': patient_dict['weight_kg'],
                    'height_cm': patient_dict['height_cm'],
                    'renal_status': patient_dict['renal_status'],
                    'creatinine_mg_dL': tdm_data['creatinine_data']['creatinine_mg_dL'].iloc[-1],
                    'egfr': tdm_data['avg_egfr'],
                    'dose_mg': tdm_data['dosing']['initial_dose_mg'],
                    'dose_mg_per_kg': tdm_data['dosing']['dose_mg_per_kg'],
                    'interval_hours': tdm_data['dosing']['interval_hours'],
                    'dose_number': tdm_row['dose_number'],
                    'time_since_dose_hours': tdm_row['time_since_dose_hours'],
                    'concentration_mg_per_L': tdm_row['concentration_mg_per_L'],
                    'timestamp': tdm_row['timestamp'],
                    'observed_auc': tdm_data['observed_auc'],
                    'target_auc': tdm_data['target_auc'],
                    'trough_concentration': tdm_data['trough_concentration'],
                    'target_achieved': tdm_data['target_achieved'],
                    'clearance_L_per_h': tdm_data['pk_parameters']['clearance_L_per_h'],
                    'volume_L': tdm_data['pk_parameters']['volume_L'],
                    'half_life_h': tdm_data['pk_parameters']['half_life_h']
                })
        
        return pd.DataFrame(all_data)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic TDM data')
    parser.add_argument('--out', type=str, default='data/synthetic', help='Output directory')
    parser.add_argument('--npatients', type=int, default=200, help='Number of patients')
    parser.add_argument('--drug', type=str, default='vancomycin', choices=['vancomycin', 'amikacin'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ndays', type=int, default=7, help='Number of days to simulate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Generate data
    print(f"Generating synthetic {args.drug} data for {args.npatients} patients...")
    generator = SyntheticTDMGenerator(drug_type=args.drug, seed=args.seed)
    dataset = generator.generate_dataset(n_patients=args.npatients, n_days=args.ndays)
    
    # Save dataset
    output_file = os.path.join(args.out, f'synthetic_tdm_{args.drug}_{args.npatients}patients.csv')
    dataset.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Target achievement rate: {dataset['target_achieved'].mean():.2%}")
    
    # Save metadata
    metadata = {
        'drug_type': args.drug,
        'n_patients': args.npatients,
        'n_days': args.ndays,
        'seed': args.seed,
        'generated_at': datetime.now().isoformat(),
        'columns': list(dataset.columns)
    }
    
    metadata_file = os.path.join(args.out, f'metadata_{args.drug}.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}")


if __name__ == '__main__':
    main()

