"""
Script to copy files from workspace to physical location
This script reads files from workspace and writes them to physical location
"""
import os
import sys

# Files to copy from workspace
files_to_copy = [
    # Source files
    ('src/data/synthetic_generator.py', 'src/data/synthetic_generator.py'),
    ('src/poppk/bayesian_forecasting.py', 'src/poppk/bayesian_forecasting.py'),
    ('src/models/ml_regressor.py', 'src/models/ml_regressor.py'),
    ('src/models/hybrid.py', 'src/models/hybrid.py'),
    ('src/tdm_loop.py', 'src/tdm_loop.py'),
    ('src/eval.py', 'src/eval.py'),
    
    # Documentation
    ('ETHICS.md', 'ETHICS.md'),
    ('references.md', 'references.md'),
    ('data/README_DATA.md', 'data/README_DATA.md'),
    
    # Other files
    ('sample_patient.json', 'sample_patient.json'),
    ('quick_start.py', 'quick_start.py'),
]

# Base directory
base_dir = r'C:\Users\i\mipd-renal-dose-optimizer'

# Create directories
for src, dst in files_to_copy:
    dst_path = os.path.join(base_dir, dst)
    dst_dir = os.path.dirname(dst_path)
    if dst_dir and not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Created directory: {dst_dir}")

print(f"Directories created. Now copy files manually or use workspace read_file + write tools.")

