"""
Quick start script for MIPD Renal Dose Optimizer
Generates synthetic data and demonstrates basic functionality
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("MIPD Renal Dose Optimizer - Quick Start")
    print("=" * 60)
    print()
    
    # Check if data directory exists
    data_dir = "data/synthetic"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic data
    print("Step 1: Generating synthetic data...")
    print("-" * 60)
    try:
        from src.data.synthetic_generator import SyntheticTDMGenerator
        
        generator = SyntheticTDMGenerator(drug_type="vancomycin", seed=42)
        dataset = generator.generate_dataset(n_patients=50, n_days=7)
        
        output_file = os.path.join(data_dir, "synthetic_tdm_vancomycin_50patients.csv")
        dataset.to_csv(output_file, index=False)
        print(f"✓ Generated synthetic data: {output_file}")
        print(f"  - Patients: {dataset['patient_id'].nunique()}")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Target achievement rate: {dataset['target_achieved'].mean():.2%}")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return
    
    print()
    print("Step 2: Testing popPK/Bayesian model...")
    print("-" * 60)
    try:
        from src.poppk.bayesian_forecasting import OneCompartmentPKModel, BayesianForecaster
        
        pk_model = OneCompartmentPKModel(drug_type="vancomycin")
        forecaster = BayesianForecaster(pk_model)
        
        # Example patient
        patient_data = {
            'age': 65,
            'sex': 'M',
            'weight_kg': 70,
            'egfr': 25
        }
        
        # Estimate parameters
        pk_params = forecaster.estimate_parameters(
            weight=patient_data['weight_kg'],
            egfr=patient_data['egfr']
        )
        
        # Recommend dose
        recommendation = forecaster.recommend_dose(
            target_auc=400,
            weight=patient_data['weight_kg'],
            interval=24.0,
            egfr=patient_data['egfr']
        )
        
        print(f"✓ popPK model test successful")
        print(f"  - Recommended dose: {recommendation['recommended_dose_mg_per_kg']:.2f} mg/kg")
        print(f"  - Estimated clearance: {pk_params['clearance_L_per_h']:.2f} L/h")
    except Exception as e:
        print(f"✗ Error testing popPK model: {e}")
        return
    
    print()
    print("Step 3: Testing ML model...")
    print("-" * 60)
    try:
        from src.models.ml_regressor import train_ml_model
        
        # Train ML model on small subset
        df_subset = dataset.head(100)  # Use first 100 samples for quick test
        model, metrics = train_ml_model(
            df_subset,
            target_column='concentration_mg_per_L',
            model_type='xgboost',
            test_size=0.2,
            random_state=42
        )
        
        print(f"✓ ML model test successful")
        print(f"  - Test MAE: {metrics['test_mae']:.2f} mg/L")
        print(f"  - Test R²: {metrics['test_r2']:.3f}")
    except Exception as e:
        print(f"✗ Error testing ML model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 60)
    print("Quick start completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Explore data: Open notebooks/01_exploratory.ipynb")
    print("2. Run popPK demo: Open notebooks/02_popPK_bayesian_demo.ipynb")
    print("3. Run ML hybrid demo: Open notebooks/03_ml_hybrid_demo.ipynb")
    print("4. Train full models: See README.md for instructions")
    print()
    print("For more information, see README.md")


if __name__ == '__main__':
    main()

