# Data Directory

## Overview

This directory contains synthetic/simulated TDM (Therapeutic Drug Monitoring) data for model development and testing.

## ⚠️ Important Notice

**NO REAL PATIENT DATA**: This directory contains ONLY synthetic/simulated data. No real patient data or Protected Health Information (PHI) is stored in this repository.

## Synthetic Data

### Generated Data Files

- `synthetic_tdm_vancomycin_200patients.csv`: Synthetic vancomycin TDM data for 200 patients
- `synthetic_tdm_amikacin_200patients.csv`: Synthetic amikacin TDM data for 200 patients (if generated)
- `metadata_vancomycin.json`: Metadata for vancomycin dataset
- `metadata_amikacin.json`: Metadata for amikacin dataset (if generated)

### Data Generation

To generate synthetic data:

```bash
python src/data/synthetic_generator.py --out data/synthetic --npatients 200 --drug vancomycin
```

### Data Structure

The synthetic data includes:

#### Patient Demographics
- `patient_id`: Unique patient identifier
- `age`: Age in years
- `sex`: Sex ('M' or 'F')
- `weight_kg`: Weight in kilograms
- `height_cm`: Height in centimeters
- `renal_status`: Renal status ('normal', 'mild', 'moderate', 'severe', 'dialysis')

#### Renal Function
- `creatinine_mg_dL`: Serum creatinine (mg/dL)
- `egfr`: Estimated glomerular filtration rate (mL/min/1.73m²)

#### Dosing Information
- `dose_mg`: Dose administered (mg)
- `dose_mg_per_kg`: Dose per kilogram (mg/kg)
- `interval_hours`: Dosing interval (hours)
- `dose_number`: Dose number (1, 2, 3, etc.)

#### TDM Measurements
- `time_since_dose_hours`: Time since dose administration (hours)
- `concentration_mg_per_L`: Measured concentration (mg/L)
- `timestamp`: Timestamp of measurement

#### Pharmacokinetic Parameters
- `clearance_L_per_h`: Clearance (L/h)
- `volume_L`: Volume of distribution (L)
- `half_life_h`: Half-life (hours)

#### Outcomes
- `observed_auc`: Observed AUC (mg*h/L)
- `target_auc`: Target AUC (mg*h/L)
- `trough_concentration`: Trough concentration (mg/L)
- `target_achieved`: Whether target was achieved (boolean)

## Real Data Requirements

If using real clinical data:

### IRB Approval
- Institutional Review Board (IRB) approval is REQUIRED
- Follow your institution's IRB protocols
- Obtain appropriate consent if required

### Data Protection
- De-identify all data according to HIPAA guidelines
- Remove all 18 HIPAA identifiers
- Implement access controls
- Encrypt data at rest and in transit
- Maintain audit trails

### Data Use Agreements
- Establish Data Use Agreements (DUA) with data providers
- Specify data use limitations
- Define data retention policies
- Establish data sharing protocols

### Data Storage
- Store data in secure, encrypted locations
- Limit access to authorized personnel only
- Implement backup and recovery procedures
- Follow institutional data retention policies

## Data Preprocessing

### Recommended Steps

1. **Data Cleaning**
   - Remove duplicates
   - Handle missing values
   - Identify and handle outliers
   - Validate data ranges

2. **Data Transformation**
   - Normalize units
   - Calculate derived variables (e.g., eGFR, BMI)
   - Create time-series features
   - Encode categorical variables

3. **Data Validation**
   - Validate clinical ranges
   - Check for data inconsistencies
   - Verify data completeness
   - Validate relationships between variables

## Data Splitting

### Recommended Splits

- **Training Set**: 60-70% of data
- **Validation Set**: 10-20% of data
- **Test Set**: 20% of data

### Splitting Strategy

- **Patient-level splitting**: Split by patient ID to avoid data leakage
- **Temporal splitting**: Use temporal splits if data has time dependencies
- **Stratified splitting**: Stratify by important variables (e.g., renal status)

## Data Quality

### Quality Metrics

- **Completeness**: Percentage of missing values
- **Accuracy**: Comparison with gold standard (if available)
- **Consistency**: Internal consistency checks
- **Timeliness**: Data freshness and relevance

### Quality Assurance

- Regular data quality checks
- Data validation rules
- Anomaly detection
- Data quality reports

## Data Sharing

### Sharing Synthetic Data

Synthetic data can be freely shared for research purposes.

### Sharing Real Data

If sharing real data:
- Ensure proper de-identification
- Use Data Use Agreements (DUA)
- Use secure sharing channels
- Follow institutional policies
- Comply with regulations (HIPAA, GDPR, etc.)

## Data Citation

If using this data in publications, please cite:

```
MIPD Renal Dose Optimizer. (2024). Synthetic TDM Dataset for Vancomycin/Amikacin.
```

## Contact

For questions about data, please open an issue on GitHub.

## Version History

- **v0.1.0** (2024): Initial synthetic data generator and dataset

## License

See main LICENSE file for data licensing information.

