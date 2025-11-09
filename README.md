# Model-Informed Precision Dosing (MIPD) System for Renal Impairment

A hybrid machine learning and pharmacokinetic modeling system for personalized dose optimization in patients with renal impairment (acute or chronic kidney disease). This system provides decision-support level dose recommendations for vancomycin and amikacin-like drugs.

## ⚠️ Important Notice

**RESEARCH USE ONLY - NOT FOR CLINICAL USE**

This software is provided for research purposes only. It is not approved for clinical use and should not be used to make clinical decisions. All dose recommendations are for decision-support purposes only and require expert clinical review.

## Overview

This MIPD system combines:
- **Population Pharmacokinetic (popPK) models** with Bayesian forecasting
- **Machine Learning models** (XGBoost, LightGBM, Random Forest, Neural Networks)
- **Hybrid approach** using residual learning to correct popPK predictions

The system provides:
- Personalized dose recommendations (mg/kg) and dosing intervals
- Expected PK targets (AUC, trough concentrations) with uncertainty estimates
- Online learning with TDM (Therapeutic Drug Monitoring) loop updates

## Features

- **Hybrid Modeling**: Combines popPK/Bayesian forecasting with ML correction
- **Renal Function Integration**: Accounts for eGFR and creatinine changes over time
- **TDM Loop**: Online dose adjustment based on TDM measurements
- **Uncertainty Quantification**: Provides confidence intervals for recommendations
- **Multiple Models**: Supports XGBoost, LightGBM, Random Forest, and Neural Networks
- **Synthetic Data Generation**: Includes synthetic TDM data generator for prototyping

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ladyengineersena/ai-drug-dosing-renal.git
cd ai-drug-dosing-renal
```

2. Create a virtual environment:
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data

```bash
python src/data/synthetic_generator.py --out data/synthetic --npatients 200 --drug vancomycin
```

### 2. Explore Data

Open `notebooks/01_exploratory.ipynb` to explore the synthetic dataset.

### 3. Train Models

#### Train ML Model:
```bash
python -c "
import pandas as pd
from src.models.ml_regressor import train_ml_model

df = pd.read_csv('data/synthetic/synthetic_tdm_vancomycin_200patients.csv')
model, metrics = train_ml_model(df, model_type='xgboost')
model.save('outputs/ml_model.pkl')
print(metrics)
"
```

#### Train Hybrid Model:
```python
from src.models.hybrid import HybridDosePredictor
import pandas as pd

df = pd.read_csv('data/synthetic/synthetic_tdm_vancomycin_200patients.csv')
hybrid = HybridDosePredictor(drug_type='vancomycin', ml_model_type='xgboost')
hybrid.train(df, target_column='concentration_mg_per_L')
hybrid.save('outputs/hybrid_model.pkl')
```

### 4. Use TDM Loop

Create a patient JSON file (`sample_patient.json`):
```json
{
    "age": 65,
    "sex": "M",
    "weight_kg": 70,
    "height_cm": 175,
    "creatinine_mg_dL": 2.5,
    "egfr": 25
}
```

Run TDM loop simulation:
```bash
python src/tdm_loop.py --patient sample_patient.json --model hybrid --drug vancomycin --model_file outputs/hybrid_model.pkl --n_cycles 3
```

## Project Structure

```
mipd-renal-dose-optimizer/
├── data/
│   ├── synthetic/                     # Synthetic TDM data
│   └── README_DATA.md
├── notebooks/
│   ├── 01_exploratory.ipynb          # Data exploration
│   ├── 02_popPK_bayesian_demo.ipynb  # popPK/Bayesian demo
│   └── 03_ml_hybrid_demo.ipynb       # ML hybrid demo
├── src/
│   ├── data/
│   │   └── synthetic_generator.py    # Synthetic data generator
│   ├── poppk/
│   │   └── bayesian_forecasting.py   # popPK/Bayesian models
│   ├── models/
│   │   ├── ml_regressor.py           # ML models
│   │   └── hybrid.py                 # Hybrid model
│   ├── tdm_loop.py                   # TDM loop simulation
│   ├── eval.py                       # Evaluation metrics
│   └── utils.py                      # Utility functions
├── outputs/                           # Model outputs
├── requirements.txt
├── README.md
├── ETHICS.md
└── references.md
```

## Usage Examples

### popPK/Bayesian Forecasting

```python
from src.poppk.bayesian_forecasting import OneCompartmentPKModel, BayesianForecaster

# Initialize model
pk_model = OneCompartmentPKModel(drug_type='vancomycin')
forecaster = BayesianForecaster(pk_model)

# Patient data
patient_data = {
    'age': 65,
    'sex': 'M',
    'weight_kg': 70,
    'egfr': 25
}

# Add TDM observation
forecaster.add_observation(concentration=25.0, time=2.0, dose=1000, dose_time=0.0)

# Estimate parameters
pk_params = forecaster.estimate_parameters(weight=70, egfr=25)

# Recommend dose
recommendation = forecaster.recommend_dose(
    target_auc=400,
    weight=70,
    interval=24.0,
    egfr=25
)
```

### Hybrid Model

```python
from src.models.hybrid import HybridDosePredictor

# Initialize and train
hybrid = HybridDosePredictor(drug_type='vancomycin', ml_model_type='xgboost')
hybrid.train(df_train, target_column='concentration_mg_per_L')

# Predict concentration
prediction = hybrid.predict(
    patient_data=patient_data,
    dose=1000,
    time=12.0,
    interval=24.0,
    n_doses=3
)

# Recommend dose
recommendation = hybrid.recommend_dose(
    patient_data=patient_data,
    target_auc=400,
    interval=24.0
)
```

## Evaluation Metrics

The system evaluates models using:
- **Mean Absolute Error (MAE)** for concentration predictions
- **Root Mean Squared Error (RMSE)**
- **R² score**
- **Target achievement rate** (% of patients achieving target AUC/trough)
- **Time to target** (hours to reach target concentration)

## Model Performance

Expected performance (on synthetic data):
- **popPK only**: Baseline performance, clinically acceptable
- **ML only**: May outperform popPK in some scenarios
- **Hybrid**: Combines strengths of both approaches, typically best performance

## Data Requirements

### Minimum Required Data:
- Patient demographics: age, sex, weight, height
- Renal function: serum creatinine, eGFR (time-series)
- Drug administration: dose(s), timestamps, infusion rate
- TDM measurements: serum concentrations with timestamps

### Optional Data:
- Concurrent medications
- CRRT/HD parameters
- Laboratory values: albumin, bilirubin
- Genetic data (CYP, transporters)

## Limitations

1. **Synthetic Data**: Current implementation uses synthetic data only. Real clinical data requires IRB approval.
2. **Model Validation**: Models need external validation on independent datasets.
3. **Clinical Approval**: Not approved for clinical use - research only.
4. **Drug Coverage**: Currently supports vancomycin and amikacin-like drugs.
5. **Population**: Trained on synthetic data - may not generalize to all populations.

## Ethical Considerations

See [ETHICS.md](ETHICS.md) for detailed ethical considerations, including:
- Data privacy and PHI protection
- IRB requirements for real data
- Clinical use limitations
- Fairness and bias considerations
- Audit trails and explainability

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
MIPD Renal Dose Optimizer. (2024). Model-Informed Precision Dosing System for Renal Impairment.
```

## References

See [references.md](references.md) for detailed literature references.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- ASHP/IDSA vancomycin TDM guidelines
- Population pharmacokinetic modeling community
- Machine learning for precision dosing researchers

