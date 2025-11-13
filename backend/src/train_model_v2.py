"""
Simplified NFL Prediction Model Training Pipeline V2

This is a complete rewrite focusing on:
1. Proper feature preparation and alignment
2. Simple, effective model architecture
3. Proper validation and evaluation
4. No unnecessary complexity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from prepare_enhanced_features import prepare_enhanced_features
from data_collection import get_current_season

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class SimpleNFLModel:
    """
    Simplified NFL prediction model focused on reliability.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        
    def train(self, X, y, X_val=None, y_val=None, feature_list=None):
        """Train the model."""
        print("\nTraining model...")
        
        # Clean data
        X, y = self._prepare_data(X, y, feature_list=feature_list)
        if X_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val, feature_list=X.columns.tolist())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Train LightGBM model with good defaults
        print("  Training LightGBM...")
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True
        )
        
        if X_val is not None:
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(X_scaled, y)
        
        print(f"  Model trained with {len(self.feature_names)} features")
        
        # Evaluate on validation set if available
        if X_val is not None:
            y_proba = self.predict_proba(X_val)
            y_pred = (y_proba >= self.threshold).astype(int)
            
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            ll = log_loss(y_val, y_proba)
            
            print(f"  Validation - Acc: {acc:.4f}, AUC: {auc:.4f}, LogLoss: {ll:.4f}")
            
            # Tune threshold
            self.threshold = self._tune_threshold(y_val, y_proba)
            print(f"  Optimal threshold: {self.threshold:.3f}")
    
    def _prepare_data(self, X, y, feature_list=None):
        """Clean and prepare data."""
        # Remove rows with invalid targets
        valid_mask = ~(pd.isna(y) | np.isinf(y))
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        # Ensure y is binary
        y = y.astype(int)
        if not y.isin([0, 1]).all():
            y = (y > 0.5).astype(int)
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Use provided feature list if available, otherwise remove constant features
        if feature_list is not None:
            # Align to feature list
            missing_features = [f for f in feature_list if f not in X.columns]
            for f in missing_features:
                X[f] = 0
            X = X[feature_list]
        else:
            # Remove constant features
            constant_features = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_features:
                X = X.drop(columns=constant_features)
        
        return X, y
    
    def _tune_threshold(self, y_true, y_proba, threshold_range=np.arange(0.40, 0.60, 0.01)):
        """Tune decision threshold."""
        best_threshold = 0.5
        best_score = 0
        
        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)
            score = f1_score(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def predict_proba(self, X):
        """Get probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Align features
        X = self._align_features(X)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X):
        """Get binary predictions."""
        y_proba = self.predict_proba(X)
        return (y_proba >= self.threshold).astype(int)
    
    def _align_features(self, X):
        """Align features with training features."""
        X = X.copy()
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        if self.feature_names is None:
            return X
        
        # Add missing features
        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            for f in missing_features:
                X[f] = 0
        
        # Select only training features in correct order
        X = X[self.feature_names]
        
        return X
    
    def save(self, filepath):
        """Save model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'random_state': self.random_state
        }
        pickle.dump(model_data, open(filepath, 'wb'))
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model."""
        model_data = pickle.load(open(filepath, 'rb'))
        model = cls(random_state=model_data.get('random_state', 42))
        model.model = model_data['model']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.threshold = model_data.get('threshold', 0.5)
        return model


def forward_chaining_cv(X, y, n_splits=5):
    """Forward-chaining cross-validation."""
    print("\nPerforming Forward-Chaining Cross-Validation...")
    
    # First, identify all features that will be used (remove constant features from full dataset)
    # This ensures consistent features across folds
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    all_features = [col for col in X.columns if col not in constant_features]
    # Remove 'season' from features if it exists (it's metadata, not a feature)
    if 'season' in all_features:
        all_features.remove('season')
    
    # Group by season if available
    if 'season' in X.columns:
        seasons = sorted(X['season'].unique())
        if len(seasons) > n_splits:
            # Use season-based splits
            accuracies = []
            loglosses = []
            aucs = []
            thresholds = []
            
            for i in range(len(seasons) - n_splits):
                train_seasons = seasons[:i + n_splits]
                val_season = seasons[i + n_splits]
                
                train_mask = X['season'].isin(train_seasons)
                val_mask = X['season'] == val_season
                
                X_train, X_val = X[train_mask].copy(), X[val_mask].copy()
                y_train, y_val = y[train_mask].copy(), y[val_mask].copy()
                
                if len(X_train) == 0 or len(X_val) == 0:
                    continue
                
                # Ensure both have the same features
                common_features = [f for f in all_features if f in X_train.columns and f in X_val.columns]
                
                # Train model with consistent feature list
                model = SimpleNFLModel()
                model.train(X_train, y_train, X_val, y_val, feature_list=common_features)
                
                # Evaluate
                y_proba = model.predict_proba(X_val)
                y_pred = model.predict(X_val)
                
                acc = accuracy_score(y_val, y_pred)
                ll = log_loss(y_val, y_proba)
                auc = roc_auc_score(y_val, y_proba)
                
                accuracies.append(acc)
                loglosses.append(ll)
                aucs.append(auc)
                thresholds.append(model.threshold)
                
                print(f"  Season {val_season}: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")
            
            optimal_threshold = np.mean(thresholds)
            
            print(f"\nCV Results:")
            print(f"  Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
            print(f"  Log Loss: {np.mean(loglosses):.4f} (+/- {np.std(loglosses):.4f})")
            print(f"  ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
            print(f"  Optimal Threshold: {optimal_threshold:.3f}")
            
            return optimal_threshold, {
                'accuracies': accuracies,
                'loglosses': loglosses,
                'aucs': aucs,
                'thresholds': thresholds
            }
    
    # Fallback to TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies = []
    loglosses = []
    aucs = []
    thresholds = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        
        # Ensure both have the same features
        common_features = [f for f in all_features if f in X_train.columns and f in X_val.columns]
        
        # Train model with consistent feature list
        model = SimpleNFLModel()
        model.train(X_train, y_train, X_val, y_val, feature_list=common_features)
        
        # Evaluate
        y_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, y_proba)
        auc = roc_auc_score(y_val, y_proba)
        
        accuracies.append(acc)
        loglosses.append(ll)
        aucs.append(auc)
        thresholds.append(model.threshold)
        
        print(f"  Fold {fold+1}: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")
    
    optimal_threshold = np.mean(thresholds)
    
    print(f"\nCV Results:")
    print(f"  Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"  Log Loss: {np.mean(loglosses):.4f} (+/- {np.std(loglosses):.4f})")
    print(f"  ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    
    return optimal_threshold, {
        'accuracies': accuracies,
        'loglosses': loglosses,
        'aucs': aucs,
        'thresholds': thresholds
    }


def main():
    print("=" * 70)
    print("NFL Model Training V2 (Simplified)")
    print("=" * 70)
    
    # Prepare features
    print("\n[1/3] Preparing features...")
    current_season = get_current_season()
    min_season = max(2018, current_season - 6)
    
    X, y = prepare_enhanced_features(min_season=min_season, max_season=current_season)
    
    if X is None or y is None or len(X) == 0:
        print("Error: Failed to prepare features or no data available.")
        return
    
    print(f"Training on {len(X)} games with {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Cross-validation
    print("\n[2/3] Performing cross-validation...")
    optimal_threshold, cv_results = forward_chaining_cv(X, y, n_splits=5)
    
    # Train final model
    print("\n[3/3] Training final model on all data...")
    
    # Remove constant features and 'season' if present
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if 'season' in X.columns:
        constant_features.append('season')
    X_final = X.drop(columns=constant_features, errors='ignore')
    
    model = SimpleNFLModel()
    model.train(X_final, y)
    model.threshold = optimal_threshold
    
    # Save model
    model_path = MODELS_DIR / "model_v2.pkl"
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Average CV Accuracy: {np.mean(cv_results['accuracies']):.4f}")
    print(f"Average CV AUC: {np.mean(cv_results['aucs']):.4f}")


if __name__ == "__main__":
    main()

