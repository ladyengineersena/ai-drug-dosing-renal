"""
Machine Learning models for concentration/AUC prediction
XGBoost, RandomForest, and Neural Network implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import warnings


class TabularDataset(Dataset):
    """PyTorch dataset for tabular data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    """Feedforward neural network for regression"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.2):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class MLDosePredictor:
    """Machine Learning-based dose prediction"""
    
    def __init__(self, model_type: str = "xgboost", **model_kwargs):
        """
        Initialize ML predictor
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'random_forest', or 'neural_network'
            **model_kwargs: Additional arguments for model initialization
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                **model_kwargs
            )
        elif model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                **model_kwargs
            )
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **model_kwargs
            )
        elif model_type == "neural_network":
            self.model = None  # Will be initialized after knowing input_dim
            self.nn_config = {
                'hidden_dims': model_kwargs.get('hidden_dims', [64, 32, 16]),
                'dropout': model_kwargs.get('dropout', 0.2),
                'lr': model_kwargs.get('lr', 0.001),
                'batch_size': model_kwargs.get('batch_size', 32),
                'epochs': model_kwargs.get('epochs', 100)
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, include_tdm: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features from patient data
        
        Args:
            df: DataFrame with patient and dosing data
            include_tdm: Whether to include TDM measurements as features
        
        Returns:
            Feature matrix and feature names
        """
        features = []
        feature_names = []
        
        # Demographics
        features.append(df['age'].values)
        feature_names.append('age')
        
        features.append((df['sex'] == 'M').astype(int).values)
        feature_names.append('sex_male')
        
        features.append(df['weight_kg'].values)
        feature_names.append('weight_kg')
        
        features.append(df['height_cm'].values)
        feature_names.append('height_cm')
        
        # Renal function
        features.append(df['creatinine_mg_dL'].values)
        feature_names.append('creatinine_mg_dL')
        
        features.append(df['egfr'].values)
        feature_names.append('egfr')
        
        # Dosing history
        features.append(df['dose_mg'].values)
        feature_names.append('dose_mg')
        
        features.append(df['dose_mg_per_kg'].values)
        feature_names.append('dose_mg_per_kg')
        
        features.append(df['interval_hours'].values)
        feature_names.append('interval_hours')
        
        features.append(df['dose_number'].values)
        feature_names.append('dose_number')
        
        features.append(df['time_since_dose_hours'].values)
        feature_names.append('time_since_dose_hours')
        
        # TDM measurements (if available)
        if include_tdm and 'concentration_mg_per_L' in df.columns:
            # Use previous concentration if available
            features.append(df['concentration_mg_per_L'].fillna(0).values)
            feature_names.append('prev_concentration_mg_per_L')
        else:
            features.append(np.zeros(len(df)))
            feature_names.append('prev_concentration_mg_per_L')
        
        # Additional derived features
        bmi = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        features.append(bmi.values)
        feature_names.append('bmi')
        
        # Time since first dose
        if 'time_since_dose_hours' in df.columns:
            total_time = df.groupby('patient_id')['time_since_dose_hours'].cumsum()
            features.append(total_time.values)
            feature_names.append('total_time_hours')
        else:
            features.append(np.zeros(len(df)))
            feature_names.append('total_time_hours')
        
        X = np.column_stack(features)
        return X, feature_names
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        if self.model_type == "neural_network":
            self._train_neural_network(X_train, y_train, X_val, y_val)
        else:
            # Scale features for tree-based models (optional, but helps with NN)
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                
                if self.model_type == "xgboost":
                    self.model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                elif self.model_type == "lightgbm":
                    self.model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                else:
                    self.model.fit(X_train_scaled, y_train)
            else:
                self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
    
    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train neural network"""
        input_dim = X_train.shape[1]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model
        self.model = NeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.nn_config['hidden_dims'],
            dropout=self.nn_config['dropout']
        )
        
        # Create datasets
        train_dataset = TabularDataset(X_train_scaled, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.nn_config['batch_size'],
            shuffle=True
        )
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = TabularDataset(X_val_scaled, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.nn_config['batch_size'], shuffle=False)
        else:
            val_loader = None
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.nn_config['lr'])
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.nn_config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = self.model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """
        Predict target values
        
        Args:
            X: Feature matrix
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            Predictions (and optionally uncertainty)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == "neural_network":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions = self.model(X_tensor).numpy()
        else:
            predictions = self.model.predict(X_scaled)
        
        if return_uncertainty:
            # Simple uncertainty estimation (could be improved with MC dropout, ensembles, etc.)
            if self.model_type == "random_forest":
                # Use tree predictions for uncertainty
                tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                uncertainty = np.std(tree_predictions, axis=0)
            else:
                # Default: use prediction variance (simplified)
                uncertainty = np.abs(predictions) * 0.15  # 15% CV
            return predictions, uncertainty
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (for tree-based models)"""
        if self.model_type == "neural_network":
            return {"message": "Feature importance not available for neural networks"}
        
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.model_type in ["xgboost", "lightgbm"]:
            importance = self.model.feature_importances_
        elif self.model_type == "random_forest":
            importance = self.model.feature_importances_
        else:
            return {"message": "Feature importance not available for this model type"}
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        else:
            feature_names = self.feature_names
        
        return dict(zip(feature_names, importance))
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model_type == "neural_network":
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'nn_config': self.nn_config,
                'input_dim': self.model.network[0].in_features
            }, filepath)
        else:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }, filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        if self.model_type == "neural_network":
            checkpoint = torch.load(filepath)
            input_dim = checkpoint['input_dim']
            self.model = NeuralNetwork(
                input_dim=input_dim,
                hidden_dims=checkpoint['nn_config']['hidden_dims'],
                dropout=checkpoint['nn_config']['dropout']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.feature_names = checkpoint['feature_names']
            self.nn_config = checkpoint['nn_config']
        else:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.model_type = data['model_type']
        
        self.is_trained = True


def train_ml_model(
    df: pd.DataFrame,
    target_column: str = 'concentration_mg_per_L',
    model_type: str = 'xgboost',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MLDosePredictor, Dict[str, float]]:
    """
    Train ML model on dataset
    
    Args:
        df: DataFrame with patient and dosing data
        target_column: Target column name
        model_type: Model type ('xgboost', 'lightgbm', 'random_forest', 'neural_network')
        test_size: Test set size
        random_state: Random seed
    
    Returns:
        Trained model and evaluation metrics
    """
    # Prepare features
    predictor = MLDosePredictor(model_type=model_type)
    X, feature_names = predictor.prepare_features(df)
    predictor.feature_names = feature_names
    
    y = df[target_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    # Train model
    predictor.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_pred_train = predictor.predict(X_train)
    y_pred_test = predictor.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    return predictor, metrics

