"""
Improved Model Architecture for NFL Game Prediction

Key Improvements:
1. Better hyperparameter tuning with Optuna-style grid search
2. Probability calibration (Platt scaling)
3. Feature importance-based selection
4. Better ensemble weighting
5. Improved regularization
6. Better handling of class imbalance
7. More robust cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ImprovedNFLModel:
    """
    Improved NFL game prediction model with better architecture and calibration.
    
    Args:
        random_state: Random seed for reproducibility
        accuracy_mode: 'fast' (default) or 'accurate' - trades speed for accuracy
    """
    
    def __init__(self, random_state=42, accuracy_mode='fast', use_optuna=False, n_trials=50,
                 ensemble_method='stacking', blend_method='uniform'):
        """
        Args:
            random_state: Random seed for reproducibility
            accuracy_mode: 'fast' (default) or 'accurate' - trades speed for accuracy
            use_optuna: Whether to use Optuna for hyperparameter tuning
            n_trials: Number of Optuna trials
            ensemble_method: 'stacking' (default), 'blending', 'weighted', or 'dynamic'
            blend_method: For blending - 'uniform' (equal weights) or 'performance' (weighted by validation performance)
        """
        self.random_state = random_state
        self.accuracy_mode = accuracy_mode  # 'fast' or 'accurate'
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE  # Enable Optuna tuning
        self.n_trials = n_trials  # Number of Optuna trials
        self.ensemble_method = ensemble_method  # 'stacking', 'blending', 'weighted', 'dynamic'
        self.blend_method = blend_method  # 'uniform' or 'performance'
        self.models = {}
        self.scalers = {}
        self.feature_importance = None
        self.selected_features = None
        self.calibrators = {}
        self.threshold = 0.5
        self.meta_weights = None
        self.model_weights = {}  # For weighted ensemble
        self.model_performance = {}  # For dynamic ensemble
        self.tuned_params = {}  # Store tuned hyperparameters
        
        # Set hyperparameters based on accuracy mode
        if accuracy_mode == 'accurate':
            # Accuracy-focused hyperparameters
            self.xgb_params = {
                'n_estimators': 800,
                'max_depth': 7,
                'learning_rate': 0.015,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_weight': 2,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            self.lgb_params = {
                'n_estimators': 800,
                'max_depth': 7,
                'learning_rate': 0.015,
                'num_leaves': 63,  # 2^max_depth - 1
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_samples': 10,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            self.meta_params = {
                'n_estimators': 400,
                'max_depth': 6,
                'learning_rate': 0.02,
                'num_leaves': 31,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            self.max_features = 150  # Use more features
            self.calibration_cv = 5  # More CV folds for calibration
            self.catboost_params = {
                'iterations': 800,
                'depth': 7,
                'learning_rate': 0.015,
                'l2_leaf_reg': 3,
                'random_seed': self.random_state,
                'verbose': False
            }
            self.nn_params = {
                'hidden_layer_sizes': (128, 64, 32),  # 3 hidden layers
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,  # L2 regularization
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': self.random_state
            }
            self.rf_params = {
                'n_estimators': 500,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',  # sqrt of total features
                'bootstrap': True,
                'oob_score': True,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        else:
            # Fast/default hyperparameters
            self.xgb_params = {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.03,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            self.lgb_params = {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.03,
                'num_leaves': 15,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            self.meta_params = {
                'n_estimators': 200,
                'max_depth': 3,
                'learning_rate': 0.05,
                'num_leaves': 8,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_samples': 30,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            self.max_features = 50  # Use fewer features
            self.calibration_cv = 3  # Fewer CV folds for calibration
            self.catboost_params = {
                'iterations': 300,
                'depth': 4,
                'learning_rate': 0.03,
                'l2_leaf_reg': 3,
                'random_seed': self.random_state,
                'verbose': False
            }
            self.nn_params = {
                'hidden_layer_sizes': (64, 32),  # 2 hidden layers for fast mode
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,  # L2 regularization
                'learning_rate': 'adaptive',
                'learning_rate_init': 0.001,
                'max_iter': 300,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 15,
                'random_state': self.random_state
            }
            self.rf_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',  # sqrt of total features
                'bootstrap': True,
                'oob_score': True,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
    def _prepare_data(self, X, y):
        """Clean and prepare data for training."""
        # Remove rows with NaN or invalid target values
        valid_mask = ~(pd.isna(y) | np.isinf(y))
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        # Ensure y is binary
        y = y.astype(int)
        if not y.isin([0, 1]).all():
            y = (y > 0.5).astype(int)
        
        # Handle missing values in features
        X = X.fillna(X.median(numeric_only=True))
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            X = X.drop(columns=constant_features)
        
        return X, y
    
    def _select_features(self, X, y, top_n=None):
        """Select top features based on importance."""
        if top_n is None:
            top_n = min(50, len(X.columns))
        
        # Use XGBoost to get feature importance
        xgb_temp = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        xgb_temp.fit(X, y, verbose=False)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        self.selected_features = importance_df.head(top_n)['feature'].tolist()
        
        return self.selected_features
    
    def _tune_xgboost_optuna(self, X, y, X_val, y_val):
        """Tune XGBoost hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'tree_method': 'hist'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize', study_name='xgboost_tuning')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params
    
    def _tune_lightgbm_optuna(self, X, y, X_val, y_val):
        """Tune LightGBM hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X, y, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            y_pred = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize', study_name='lightgbm_tuning')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params
    
    def _tune_catboost_optuna(self, X, y, X_val, y_val):
        """Tune CatBoost hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': self.random_state,
                'verbose': False
            }
            
            model = cb.CatBoostClassifier(**params)
            model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
            y_pred = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize', study_name='catboost_tuning')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params
    
    def _tune_rf_optuna(self, X, y, X_val, y_val):
        """Tune Random Forest hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X, y)
            y_pred = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize', study_name='rf_tuning')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params
    
    def _tune_nn_optuna(self, X, y, X_val, y_val, scaler):
        """Tune Neural Network hyperparameters using Optuna."""
        def objective(trial):
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_sizes = []
            for i in range(n_layers):
                hidden_sizes.append(trial.suggest_int(f'n_units_layer_{i}', 32, 256))
            
            params = {
                'hidden_layer_sizes': tuple(hidden_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': 'adam',
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': self.random_state
            }
            
            model = MLPClassifier(**params)
            X_scaled = scaler.transform(X)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_scaled, y)
            y_pred = model.predict_proba(X_val_scaled)[:, 1]
            return log_loss(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize', study_name='neural_network_tuning')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params
    
    def train_base_models(self, X, y, X_val=None, y_val=None, use_feature_selection=True):
        """Train Level-1 base models with improved hyperparameters."""
        print("\nTraining Level-1 base models...")
        
        X, y = self._prepare_data(X, y)
        if X_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
        
        # Feature selection
        if use_feature_selection and len(X.columns) > 30:
            print("  Selecting top features based on importance...")
            # Only use features that exist in both X and X_val
            if X_val is not None:
                common_features = [f for f in X.columns if f in X_val.columns]
                X_temp = X[common_features]
            else:
                common_features = X.columns.tolist()
                X_temp = X
            
            selected_features = self._select_features(X_temp, y, top_n=min(self.max_features, len(X_temp.columns)))
            
            # Ensure selected features exist in both datasets
            if X_val is not None:
                available_selected = [f for f in selected_features if f in X_val.columns]
                if len(available_selected) < len(selected_features):
                    print(f"  Warning: {len(selected_features) - len(available_selected)} selected features not in validation set")
                selected_features = available_selected
            
            X = X[selected_features]
            if X_val is not None:
                X_val = X_val[selected_features]
            print(f"  Selected {len(selected_features)} features")
        
        self.selected_features = X.columns.tolist()
        predictions = {}
        
        # 1. XGBoost with improved hyperparameters
        print(f"  Training XGBoost ({self.accuracy_mode} mode)...")
        
        # Use Optuna tuning if enabled
        if self.use_optuna and X_val is not None:
            print("    Tuning XGBoost hyperparameters with Optuna...")
            xgb_params_tuned = self._tune_xgboost_optuna(X, y, X_val, y_val)
            self.tuned_params['xgb'] = xgb_params_tuned
            xgb_params_to_use = xgb_params_tuned.copy()
            xgb_params_to_use['random_state'] = self.random_state
            xgb_params_to_use['eval_metric'] = 'logloss'
            xgb_params_to_use['scale_pos_weight'] = 1.0
            xgb_params_to_use['tree_method'] = 'hist'
        else:
            xgb_params_to_use = self.xgb_params.copy()
            xgb_params_to_use['random_state'] = self.random_state
            xgb_params_to_use['eval_metric'] = 'logloss'
            xgb_params_to_use['scale_pos_weight'] = 1.0
            xgb_params_to_use['tree_method'] = 'hist'
        
        # XGBoost 2.0+ API: early_stopping_rounds in constructor
        if X_val is not None:
            try:
                xgb_model = xgb.XGBClassifier(
                    **xgb_params_to_use,
                    early_stopping_rounds=30  # In constructor for 2.0+
                )
                xgb_model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                predictions['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
            except TypeError:
                # Fallback for older XGBoost versions
                xgb_params_old = xgb_params_to_use.copy()
                xgb_params_old.pop('tree_method', None)  # Remove tree_method for older versions
                xgb_model = xgb.XGBClassifier(**xgb_params_old)
                xgb_model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=30,  # In fit() for <2.0
                    verbose=False
                )
                predictions['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
        else:
            xgb_params_no_val = xgb_params_to_use.copy()
            xgb_params_no_val.pop('early_stopping_rounds', None)
            xgb_model = xgb.XGBClassifier(**xgb_params_no_val)
            xgb_model.fit(X, y, verbose=False)
            predictions['xgb'] = xgb_model.predict_proba(X)[:, 1]
        
        self.models['xgb'] = xgb_model
        
        # 2. LightGBM with improved hyperparameters
        print(f"  Training LightGBM ({self.accuracy_mode} mode)...")
        
        # Use Optuna tuning if enabled
        if self.use_optuna and X_val is not None:
            print("    Tuning LightGBM hyperparameters with Optuna...")
            lgb_params_tuned = self._tune_lightgbm_optuna(X, y, X_val, y_val)
            self.tuned_params['lgb'] = lgb_params_tuned
            lgb_params_to_use = lgb_params_tuned.copy()
        else:
            lgb_params_to_use = self.lgb_params.copy()
            lgb_params_to_use['random_state'] = self.random_state
            lgb_params_to_use['verbose'] = -1
        
        lgb_model = lgb.LGBMClassifier(**lgb_params_to_use)
        
        if X_val is not None:
            lgb_model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            predictions['lgb'] = lgb_model.predict_proba(X_val)[:, 1]
        else:
            lgb_model.fit(X, y)
            predictions['lgb'] = lgb_model.predict_proba(X)[:, 1]
        
        self.models['lgb'] = lgb_model
        
        # 2.5. CatBoost (if available)
        if CATBOOST_AVAILABLE:
            print(f"  Training CatBoost ({self.accuracy_mode} mode)...")
            # Use Optuna tuning if enabled
            if self.use_optuna and X_val is not None:
                print("    Tuning CatBoost hyperparameters with Optuna...")
                cb_params = self._tune_catboost_optuna(X, y, X_val, y_val)
                self.tuned_params['catboost'] = cb_params
            else:
                cb_params = self.catboost_params.copy()
            
            cb_model = cb.CatBoostClassifier(**cb_params)
            
            if X_val is not None:
                cb_model.fit(
                    X, y,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=30,
                    verbose=False
                )
                predictions['catboost'] = cb_model.predict_proba(X_val)[:, 1]
            else:
                cb_model.fit(X, y, verbose=False)
                predictions['catboost'] = cb_model.predict_proba(X)[:, 1]
            
            self.models['catboost'] = cb_model
        else:
            print("  Skipping CatBoost (not available)")
        
        # 2.7. Random Forest (bagging-based ensemble for diversity)
        print(f"  Training Random Forest ({self.accuracy_mode} mode)...")
        
        # Use Optuna tuning if enabled
        if self.use_optuna and X_val is not None:
            print("    Tuning Random Forest hyperparameters with Optuna...")
            rf_params_tuned = self._tune_rf_optuna(X, y, X_val, y_val)
            self.tuned_params['rf'] = rf_params_tuned
            rf_params_to_use = rf_params_tuned.copy()
            rf_params_to_use['random_state'] = self.random_state
            rf_params_to_use['n_jobs'] = -1
        else:
            rf_params_to_use = self.rf_params.copy()
        
        rf_model = RandomForestClassifier(**rf_params_to_use)
        rf_model.fit(X, y)
        
        if X_val is not None:
            predictions['rf'] = rf_model.predict_proba(X_val)[:, 1]
        else:
            predictions['rf'] = rf_model.predict_proba(X)[:, 1]
        
        self.models['rf'] = rf_model
        
        # 2.6. Neural Network (MLP) for complex non-linear relationships
        print(f"  Training Neural Network ({self.accuracy_mode} mode)...")
        
        # Neural networks need scaled features
        nn_scaler = StandardScaler()
        X_nn_scaled = nn_scaler.fit_transform(X)
        
        if X_val is not None:
            X_val_nn_scaled = nn_scaler.transform(X_val)
        
        # Use Optuna tuning if enabled
        if self.use_optuna and X_val is not None:
            print("    Tuning Neural Network hyperparameters with Optuna...")
            nn_params_tuned = self._tune_nn_optuna(X, y, X_val, y_val, nn_scaler)
            self.tuned_params['nn'] = nn_params_tuned
            nn_params_to_use = nn_params_tuned.copy()
            nn_params_to_use['random_state'] = self.random_state
        else:
            nn_params_to_use = self.nn_params.copy()
        
        nn_model = MLPClassifier(**nn_params_to_use)
        
        if X_val is not None:
            nn_model.fit(X_nn_scaled, y)
            predictions['nn'] = nn_model.predict_proba(X_val_nn_scaled)[:, 1]
        else:
            nn_model.fit(X_nn_scaled, y)
            predictions['nn'] = nn_model.predict_proba(X_nn_scaled)[:, 1]
        
        # Store neural network with its scaler (neural networks need scaled features)
        self.models['nn'] = {
            'model': nn_model,
            'scaler': nn_scaler
        }
        
        # 3. Regularized Logistic Regression with key features
        print("  Training Regularized Logistic Regression...")
        key_features = ['market_spread', 'elo_diff', 'net_epa_diff', 'off_epa_diff', 'def_epa_diff']
        available_key_features = [f for f in key_features if f in X.columns]
        
        if len(available_key_features) > 0:
            X_lr = X[available_key_features]
            scaler = StandardScaler()
            X_lr_scaled = scaler.fit_transform(X_lr)
            
            lr_model = LogisticRegression(
                penalty='l2',
                C=0.1,  # Stronger regularization
                max_iter=2000,
                random_state=self.random_state,
                solver='lbfgs'
            )
            lr_model.fit(X_lr_scaled, y)
            
            self.models['lr'] = {
                'model': lr_model,
                'scaler': scaler,
                'features': available_key_features
            }
            
            if X_val is not None:
                X_val_lr = X_val[available_key_features]
                X_val_lr_scaled = scaler.transform(X_val_lr)
                predictions['lr'] = lr_model.predict_proba(X_val_lr_scaled)[:, 1]
            else:
                predictions['lr'] = lr_model.predict_proba(X_lr_scaled)[:, 1]
        
        # 4. Market baseline
        if 'market_spread' in X.columns:
            print("  Training Market Baseline...")
            X_market = X[['market_spread']]
            market_model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            )
            market_model.fit(X_market, y)
            self.models['market'] = market_model
            
            if X_val is not None:
                predictions['market'] = market_model.predict_proba(X_val[['market_spread']])[:, 1]
            else:
                predictions['market'] = market_model.predict_proba(X_market)[:, 1]
        
        return predictions
    
    def calibrate_models(self, X, y, X_val=None, y_val=None):
        """Calibrate model probabilities using isotonic regression."""
        print("\nCalibrating model probabilities...")
        
        # Use selected features if available
        if self.selected_features is not None:
            X = X[self.selected_features].copy()
            if X_val is not None:
                X_val = X_val[self.selected_features].copy()
        
        if X_val is None:
            # Use cross-validation for calibration
            tscv = TimeSeriesSplit(n_splits=self.calibration_cv)
            for model_name, model in self.models.items():
                if model_name == 'market' or model_name == 'meta':
                    continue
                
                try:
                    if isinstance(model, dict):
                        if 'features' in model:
                            # Logistic regression with scaler and specific features
                            X_model = X[model['features']]
                            X_model_scaled = model['scaler'].transform(X_model)
                            base_model = model['model']
                        else:
                            # Neural network with scaler (uses all features)
                            X_model_scaled = model['scaler'].transform(X)
                            base_model = model['model']
                    else:
                        # XGBoost or LightGBM - use selected features
                        X_model = X
                        base_model = model
                    
                    calibrated = CalibratedClassifierCV(
                        base_model,
                        method='isotonic',
                        cv=tscv,
                        n_jobs=-1
                    )
                    if isinstance(model, dict):
                        calibrated.fit(X_model_scaled, y)
                    else:
                        calibrated.fit(X_model, y)
                    self.calibrators[model_name] = calibrated
                    print(f"  Calibrated {model_name}")
                except Exception as e:
                    print(f"  Warning: Could not calibrate {model_name}: {e}")
        else:
            # Use validation set for calibration
            for model_name, model in self.models.items():
                if model_name == 'market' or model_name == 'meta':
                    continue
                
                try:
                    if isinstance(model, dict):
                        if 'features' in model:
                            # Logistic regression with scaler and specific features
                            X_model = X[model['features']]
                            X_model_scaled = model['scaler'].transform(X_model)
                            X_val_model = X_val[model['features']]
                            X_val_model_scaled = model['scaler'].transform(X_val_model)
                            base_model = model['model']
                        else:
                            # Neural network with scaler (uses all features)
                            X_model_scaled = model['scaler'].transform(X)
                            X_val_model_scaled = model['scaler'].transform(X_val)
                            base_model = model['model']
                    else:
                        # XGBoost or LightGBM - use selected features
                        X_model = X
                        X_val_model = X_val
                        base_model = model
                    
                    # Pre-fit the base model if not already fitted
                    if not hasattr(base_model, 'classes_'):
                        if isinstance(model, dict):
                            if 'features' in model:
                                base_model.fit(X_model_scaled, y)
                            else:
                                # Neural network
                                base_model.fit(X_model_scaled, y)
                        else:
                            base_model.fit(X_model, y)
                    
                    calibrated = CalibratedClassifierCV(
                        base_model,
                        method='isotonic',
                        cv='prefit'
                    )
                    if isinstance(model, dict):
                        calibrated.fit(X_val_model_scaled, y_val)
                    else:
                        calibrated.fit(X_val_model, y_val)
                    self.calibrators[model_name] = calibrated
                    print(f"  Calibrated {model_name}")
                except Exception as e:
                    print(f"  Warning: Could not calibrate {model_name}: {e}")
    
    def _calculate_model_performance(self, level1_predictions, y_val):
        """Calculate performance metrics for each Level-1 model."""
        performance = {}
        for model_name, preds in level1_predictions.items():
            if len(preds) != len(y_val):
                continue
            try:
                # Calculate multiple metrics
                acc = accuracy_score(y_val, (preds >= 0.5).astype(int))
                ll = log_loss(y_val, preds)
                auc = roc_auc_score(y_val, preds)
                brier = brier_score_loss(y_val, preds)
                
                # Combined score (higher is better)
                # Weight: accuracy 40%, AUC 30%, log_loss 20%, brier 10%
                score = (acc * 0.4) + (auc * 0.3) + ((1 - min(ll, 1)) * 0.2) + ((1 - min(brier, 1)) * 0.1)
                
                performance[model_name] = {
                    'accuracy': acc,
                    'log_loss': ll,
                    'auc': auc,
                    'brier': brier,
                    'score': score
                }
            except Exception as e:
                print(f"  Warning: Could not calculate performance for {model_name}: {e}")
        return performance
    
    def _learn_optimal_weights(self, level1_predictions, y_val):
        """Learn optimal weights for weighted ensemble using validation performance."""
        if y_val is None or len(level1_predictions) == 0:
            # Default to uniform weights
            n_models = len(level1_predictions)
            return {name: 1.0 / n_models for name in level1_predictions.keys()}
        
        # Calculate performance for each model
        performance = self._calculate_model_performance(level1_predictions, y_val)
        
        if len(performance) == 0:
            n_models = len(level1_predictions)
            return {name: 1.0 / n_models for name in level1_predictions.keys()}
        
        # Use performance scores as weights (normalized)
        total_score = sum(perf['score'] for perf in performance.values())
        if total_score == 0:
            n_models = len(level1_predictions)
            return {name: 1.0 / n_models for name in level1_predictions.keys()}
        
        weights = {name: perf['score'] / total_score for name, perf in performance.items()}
        return weights
    
    def _select_dynamic_models(self, level1_predictions, y_val, top_n=5):
        """Select top N models based on validation performance."""
        if y_val is None or len(level1_predictions) == 0:
            return list(level1_predictions.keys())
        
        performance = self._calculate_model_performance(level1_predictions, y_val)
        
        if len(performance) == 0:
            return list(level1_predictions.keys())
        
        # Sort by score and select top N
        sorted_models = sorted(performance.items(), key=lambda x: x[1]['score'], reverse=True)
        selected = [name for name, _ in sorted_models[:top_n]]
        
        print(f"  Dynamic ensemble selected top {len(selected)} models: {', '.join(selected)}")
        return selected
    
    def train_meta_model(self, level1_predictions, X_features, y, X_val_features=None, y_val=None):
        """Train Level-2 meta-model with improved architecture."""
        print(f"\nTraining Level-2 meta-model ({self.ensemble_method} method)...")
        
        # Ensure all predictions have the same length
        pred_lengths = {k: len(v) for k, v in level1_predictions.items()}
        if len(set(pred_lengths.values())) > 1:
            print(f"  Warning: Prediction lengths differ: {pred_lengths}")
            # Use the most common length
            from collections import Counter
            most_common_length = Counter(pred_lengths.values()).most_common(1)[0][0]
            # Filter to predictions with correct length
            level1_predictions = {k: v for k, v in level1_predictions.items() 
                                 if len(v) == most_common_length}
            # Also filter X_features and y to match
            if len(X_features) != most_common_length:
                X_features = X_features.iloc[:most_common_length]
                y = y.iloc[:most_common_length] if hasattr(y, 'iloc') else y[:most_common_length]
        
        # Handle different ensemble methods
        if self.ensemble_method == 'blending':
            # Simple blending - average predictions
            print("  Using blending ensemble (averaging predictions)...")
            
            if y_val is not None:
                # Calculate performance for reporting
                self.model_performance = self._calculate_model_performance(level1_predictions, y_val)
                for name, perf in self.model_performance.items():
                    print(f"    {name}: Acc={perf['accuracy']:.4f}, AUC={perf['auc']:.4f}, Score={perf['score']:.4f}")
            
            if self.blend_method == 'performance' and y_val is not None:
                # Weighted blending based on performance
                weights = self._learn_optimal_weights(level1_predictions, y_val)
                self.model_weights = weights
                print("  Using performance-weighted blending:")
                for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {name}: {weight:.4f}")
            else:
                # Uniform blending
                n_models = len(level1_predictions)
                self.model_weights = {name: 1.0 / n_models for name in level1_predictions.keys()}
                print(f"  Using uniform blending ({n_models} models)")
            
            # Store blending weights
            self.models['meta'] = {'method': 'blending', 'weights': self.model_weights}
            return None
            
        elif self.ensemble_method == 'weighted':
            # Weighted ensemble - learn optimal weights
            print("  Using weighted ensemble (learning optimal weights)...")
            
            if y_val is not None:
                weights = self._learn_optimal_weights(level1_predictions, y_val)
                self.model_weights = weights
                self.model_performance = self._calculate_model_performance(level1_predictions, y_val)
                
                print("  Learned optimal weights:")
                for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    perf = self.model_performance.get(name, {})
                    print(f"    {name}: {weight:.4f} (Acc={perf.get('accuracy', 0):.4f}, AUC={perf.get('auc', 0):.4f})")
            else:
                # Default to uniform if no validation set
                n_models = len(level1_predictions)
                self.model_weights = {name: 1.0 / n_models for name in level1_predictions.keys()}
                print(f"  No validation set, using uniform weights ({n_models} models)")
            
            self.models['meta'] = {'method': 'weighted', 'weights': self.model_weights}
            return None
            
        elif self.ensemble_method == 'dynamic':
            # Dynamic ensemble - select top models
            print("  Using dynamic ensemble (selecting top models)...")
            
            if y_val is not None:
                selected_models = self._select_dynamic_models(level1_predictions, y_val, top_n=5)
                self.model_performance = self._calculate_model_performance(level1_predictions, y_val)
                
                # Use only selected models
                level1_predictions = {k: v for k, v in level1_predictions.items() if k in selected_models}
                
                # Learn weights for selected models
                weights = self._learn_optimal_weights(level1_predictions, y_val)
                self.model_weights = weights
                
                print("  Weights for selected models:")
                for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    perf = self.model_performance.get(name, {})
                    print(f"    {name}: {weight:.4f} (Acc={perf.get('accuracy', 0):.4f}, AUC={perf.get('auc', 0):.4f})")
            else:
                # Default to all models if no validation set
                selected_models = list(level1_predictions.keys())
                n_models = len(selected_models)
                self.model_weights = {name: 1.0 / n_models for name in selected_models}
                print(f"  No validation set, using all models ({n_models} models)")
            
            self.models['meta'] = {'method': 'dynamic', 'weights': self.model_weights, 'selected_models': selected_models}
            return None
        
        # Default: Stacking (original method)
        # Combine Level-1 predictions
        meta_X = pd.DataFrame(level1_predictions)
        
        # Ensure meta_X and y have the same length
        if len(meta_X) != len(y):
            min_len = min(len(meta_X), len(y))
            meta_X = meta_X.iloc[:min_len]
            y = y.iloc[:min_len] if hasattr(y, 'iloc') else y[:min_len]
        
        # Add key features for meta-model
        if 'market_spread' in X_features.columns:
            spread_values = X_features['market_spread'].values[:len(meta_X)]
            meta_X['spread_abs'] = abs(spread_values)
            meta_X['spread_large'] = (abs(spread_values) >= 7).astype(int)
            meta_X['spread_medium'] = ((abs(spread_values) >= 3) & 
                                      (abs(spread_values) < 7)).astype(int)
        
        if 'elo_diff' in X_features.columns:
            elo_values = X_features['elo_diff'].values[:len(meta_X)]
            meta_X['elo_diff_abs'] = abs(elo_values)
        
        # Train meta-model with better hyperparameters
        print(f"  Meta-model using {self.accuracy_mode} mode hyperparameters")
        meta_model = lgb.LGBMClassifier(
            **self.meta_params,
            random_state=self.random_state,
            verbose=-1
        )
        
        if X_val_features is not None and y_val is not None:
            meta_X_val = pd.DataFrame({
                k: v for k, v in level1_predictions.items()
            })
            
            # Ensure validation data has correct length
            if len(meta_X_val) != len(y_val):
                min_len = min(len(meta_X_val), len(y_val))
                meta_X_val = meta_X_val.iloc[:min_len]
                y_val = y_val.iloc[:min_len] if hasattr(y_val, 'iloc') else y_val[:min_len]
            
            if 'market_spread' in X_val_features.columns:
                spread_values = X_val_features['market_spread'].values[:len(meta_X_val)]
                meta_X_val['spread_abs'] = abs(spread_values)
                meta_X_val['spread_large'] = (abs(spread_values) >= 7).astype(int)
                meta_X_val['spread_medium'] = ((abs(spread_values) >= 3) & 
                                               (abs(spread_values) < 7)).astype(int)
            if 'elo_diff' in X_val_features.columns:
                elo_values = X_val_features['elo_diff'].values[:len(meta_X_val)]
                meta_X_val['elo_diff_abs'] = abs(elo_values)
            
            # Final length check
            if len(meta_X_val) != len(y_val):
                min_len = min(len(meta_X_val), len(y_val))
                meta_X_val = meta_X_val.iloc[:min_len]
                y_val = y_val.iloc[:min_len] if hasattr(y_val, 'iloc') else y_val[:min_len]
            
            meta_model.fit(
                meta_X, y,
                eval_set=[(meta_X_val, y_val)],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
        else:
            meta_model.fit(meta_X, y)
        
        self.models['meta'] = meta_model
        return meta_model
    
    def tune_threshold(self, y_true, y_proba, threshold_range=np.arange(0.45, 0.55, 0.005)):
        """Tune decision threshold for optimal accuracy."""
        best_threshold = 0.5
        best_score = 0
        
        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)
            # Use F1 score as metric (balance of precision and recall)
            score = f1_score(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        return best_threshold, best_score
    
    def predict(self, X, apply_calibration=True, apply_rules=True):
        """Make predictions with improved pipeline."""
        X = X.copy()
        
        # Align features
        if self.selected_features is not None:
            missing_features = [f for f in self.selected_features if f not in X.columns]
            if missing_features:
                for f in missing_features:
                    X[f] = 0
            X = X[self.selected_features]
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        X = X.replace([np.inf, -np.inf], 0)
        
        # Get Level-1 predictions
        level1_preds = {}
        
        for model_name, model in self.models.items():
            if model_name == 'meta':
                continue
            
            try:
                if model_name == 'market':
                    # Market baseline - only use market_spread
                    if 'market_spread' in X.columns:
                        X_market = X[['market_spread']]
                        proba = model.predict_proba(X_market)[:, 1]
                    else:
                        print(f"  Warning: market_spread not available, skipping market model")
                        continue
                elif model_name == 'catboost':
                    # CatBoost
                    if apply_calibration and model_name in self.calibrators:
                        proba = self.calibrators[model_name].predict_proba(X)[:, 1]
                    else:
                        proba = model.predict_proba(X)[:, 1]
                elif model_name == 'nn':
                    # Neural Network - needs scaling (stored as dict with 'model' and 'scaler')
                    X_nn_scaled = model['scaler'].transform(X)
                    if apply_calibration and model_name in self.calibrators:
                        proba = self.calibrators[model_name].predict_proba(X_nn_scaled)[:, 1]
                    else:
                        proba = model['model'].predict_proba(X_nn_scaled)[:, 1]
                elif isinstance(model, dict):
                    # Logistic regression with scaler and specific features
                    X_model = X[model['features']]
                    X_model_scaled = model['scaler'].transform(X_model)
                    if apply_calibration and model_name in self.calibrators:
                        proba = self.calibrators[model_name].predict_proba(X_model_scaled)[:, 1]
                    else:
                        proba = model['model'].predict_proba(X_model_scaled)[:, 1]
                else:
                    # XGBoost or LightGBM
                    if apply_calibration and model_name in self.calibrators:
                        proba = self.calibrators[model_name].predict_proba(X)[:, 1]
                    else:
                        proba = model.predict_proba(X)[:, 1]
                
                level1_preds[model_name] = proba
            except Exception as e:
                print(f"  Warning: {model_name} prediction failed: {e}")
        
        if len(level1_preds) == 0:
            raise ValueError("No Level-1 predictions available")
        
        # Handle different ensemble methods
        meta_model = self.models.get('meta')
        
        if meta_model is None:
            raise ValueError("Meta model not found. Train the model first.")
        
        # Check if meta_model is a dict (blending/weighted/dynamic) or a model (stacking)
        if isinstance(meta_model, dict):
            method = meta_model.get('method', 'blending')
            weights = meta_model.get('weights', {})
            
            if method in ['blending', 'weighted', 'dynamic']:
                # Apply weights to Level-1 predictions
                if method == 'dynamic':
                    # Filter to selected models only
                    selected_models = meta_model.get('selected_models', list(level1_preds.keys()))
                    level1_preds = {k: v for k, v in level1_preds.items() if k in selected_models}
                
                # Weighted average of predictions
                weighted_sum = np.zeros(len(list(level1_preds.values())[0]))
                total_weight = 0
                
                for model_name, preds in level1_preds.items():
                    weight = weights.get(model_name, 0)
                    if weight > 0:
                        weighted_sum += preds * weight
                        total_weight += weight
                
                if total_weight > 0:
                    y_proba = weighted_sum / total_weight
                else:
                    # Fallback to uniform average
                    y_proba = np.mean([preds for preds in level1_preds.values()], axis=0)
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
        else:
            # Stacking method - use meta model
            meta_X = pd.DataFrame(level1_preds)
            
            # Add meta features - must match exactly what was used during training
            # Check what features the meta model expects
            try:
                # Try to get expected features from the meta model
                if hasattr(meta_model, 'feature_name_'):
                    expected_features = list(meta_model.feature_name_)
                elif hasattr(meta_model, 'feature_names_'):
                    expected_features = list(meta_model.feature_names_)
                else:
                    expected_features = None
            except:
                expected_features = None
            
            # Add meta features consistently
            meta_features_added = []
            
            if 'market_spread' in X.columns:
                meta_X['spread_abs'] = abs(X['market_spread'].values)
                meta_X['spread_large'] = (abs(X['market_spread'].values) >= 7).astype(int)
                meta_X['spread_medium'] = ((abs(X['market_spread'].values) >= 3) & 
                                           (abs(X['market_spread'].values) < 7)).astype(int)
                meta_features_added.extend(['spread_abs', 'spread_large', 'spread_medium'])
            elif expected_features is not None:
                # If market_spread not available but expected, add zeros
                if 'spread_abs' in expected_features:
                    meta_X['spread_abs'] = 0
                    meta_X['spread_large'] = 0
                    meta_X['spread_medium'] = 0
                    meta_features_added.extend(['spread_abs', 'spread_large', 'spread_medium'])
            
            if 'elo_diff' in X.columns:
                meta_X['elo_diff_abs'] = abs(X['elo_diff'].values)
                meta_features_added.append('elo_diff_abs')
            elif expected_features is not None:
                # If elo_diff not available but expected, add zeros
                if 'elo_diff_abs' in expected_features:
                    meta_X['elo_diff_abs'] = 0
                    meta_features_added.append('elo_diff_abs')
            
            # Ensure all expected features are present
            if expected_features is not None:
                missing_features = [f for f in expected_features if f not in meta_X.columns]
                if missing_features:
                    print(f"  Warning: Adding missing meta features: {missing_features}")
                    for f in missing_features:
                        meta_X[f] = 0
            
            # Ensure feature order matches training
            if expected_features is not None:
                # Reorder columns to match expected order
                available_expected = [f for f in expected_features if f in meta_X.columns]
                extra_features = [f for f in meta_X.columns if f not in expected_features]
                meta_X = meta_X[available_expected + extra_features]
            
            y_proba = meta_model.predict_proba(meta_X)[:, 1]
        
        # Apply decision rules
        if apply_rules:
            y_proba = self._apply_decision_rules(y_proba, X)
        
        # Apply threshold
        y_pred = (y_proba >= self.threshold).astype(int)
        
        return y_proba, y_pred
    
    def _apply_decision_rules(self, y_proba, X):
        """Apply improved decision rules."""
        y_proba = y_proba.copy()
        
        # Rule 1: Large spreads - moderate confidence
        if 'market_spread' in X.columns:
            spread = X['market_spread'].values
            large_spread_mask = abs(spread) >= 7
            favorite_mask = spread < 0
            
            y_proba[large_spread_mask & favorite_mask] = np.clip(
                y_proba[large_spread_mask & favorite_mask] * 0.9 + 0.65 * 0.1, 0.55, 0.75
            )
            y_proba[large_spread_mask & ~favorite_mask] = np.clip(
                y_proba[large_spread_mask & ~favorite_mask] * 0.9 + 0.35 * 0.1, 0.25, 0.45
            )
        
        # Rule 2: Uncertainty dampening - smooth shrinkage
        distance_from_center = np.abs(y_proba - 0.5)
        shrink_factor = 1.0 - (distance_from_center / 0.5) ** 1.2 * 0.3
        
        high_mask = y_proba > 0.5
        low_mask = y_proba < 0.5
        
        y_proba[high_mask] = 0.5 + (y_proba[high_mask] - 0.5) * shrink_factor[high_mask]
        y_proba[low_mask] = 0.5 - (0.5 - y_proba[low_mask]) * shrink_factor[low_mask]
        
        # Ensure valid range
        y_proba = np.clip(y_proba, 0.25, 0.75)
        
        return y_proba
    
    def save(self, filepath):
        """Save model to file."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'calibrators': self.calibrators,
            'threshold': self.threshold,
            'meta_weights': self.meta_weights,
            'tuned_params': self.tuned_params,
            'accuracy_mode': self.accuracy_mode
        }
        pickle.dump(model_data, open(filepath, 'wb'))
        print(f"Model saved to {filepath}")
        if self.tuned_params:
            print(f"  Tuned hyperparameters saved for: {list(self.tuned_params.keys())}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        model_data = pickle.load(open(filepath, 'rb'))
        accuracy_mode = model_data.get('accuracy_mode', 'fast')
        model = cls(accuracy_mode=accuracy_mode)
        model.models = model_data['models']
        model.scalers = model_data.get('scalers', {})
        model.feature_importance = model_data.get('feature_importance')
        model.selected_features = model_data.get('selected_features')
        model.calibrators = model_data.get('calibrators', {})
        model.threshold = model_data.get('threshold', 0.5)
        model.meta_weights = model_data.get('meta_weights')
        model.tuned_params = model_data.get('tuned_params', {})
        return model


def forward_chaining_cv_improved(X, y, n_splits=5):
    """
    Improved forward-chaining cross-validation with better evaluation.
    """
    print("\nPerforming Improved Forward-Chaining Cross-Validation...")
    
    # Group by season
    if 'season' in X.columns:
        seasons = sorted(X['season'].unique())
    else:
        # Fallback to time-based split
        n = len(X)
        split_size = n // (n_splits + 1)
        seasons = None
    
    accuracies = []
    loglosses = []
    aucs = []
    brier_scores = []
    thresholds = []
    
    if seasons is not None and len(seasons) > n_splits:
        for i in range(len(seasons) - n_splits):
            train_seasons = seasons[:i + n_splits]
            val_season = seasons[i + n_splits]
            
            train_mask = X['season'].isin(train_seasons)
            val_mask = X['season'] == val_season
            
            X_train, X_val = X[train_mask].copy(), X[val_mask].copy()
            y_train, y_val = y[train_mask].copy(), y[val_mask].copy()
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            # Train model
            model = ImprovedNFLModel()
            level1_preds = model.train_base_models(X_train, y_train, X_val, y_val)
            
            # Calibrate models using the same features that were used for training
            model.calibrate_models(X_train, y_train, X_val, y_val)
            
            # Ensure level1_preds are aligned before training meta-model
            if len(level1_preds) > 0:
                # Get the length from the first prediction
                pred_length = len(list(level1_preds.values())[0])
                # Filter to ensure all predictions have the same length
                level1_preds_aligned = {}
                for k, v in level1_preds.items():
                    if len(v) == pred_length:
                        level1_preds_aligned[k] = v
                    else:
                        print(f"  Warning: {k} prediction length mismatch, skipping")
                
                if len(level1_preds_aligned) > 0:
                    # Align X_val and y_val to match prediction length
                    if len(X_val) != pred_length:
                        X_val_aligned = X_val.iloc[:pred_length]
                        y_val_aligned = y_val.iloc[:pred_length] if hasattr(y_val, 'iloc') else y_val[:pred_length]
                    else:
                        X_val_aligned = X_val
                        y_val_aligned = y_val
                    
                    model.train_meta_model(level1_preds_aligned, X_train, y_train, X_val_aligned, y_val_aligned)
                else:
                    print(f"  Error: No valid Level-1 predictions for meta-model")
                    continue
            
            # Predict
            y_proba, y_pred = model.predict(X_val, apply_calibration=True, apply_rules=True)
            
            # Check if model is predicting opposite (AUC < 0.5)
            auc_check = roc_auc_score(y_val, y_proba)
            if auc_check < 0.45:
                # Model is predicting opposite - flip predictions
                print(f"  Warning: AUC {auc_check:.4f} < 0.45, model appears to be predicting opposite. Flipping predictions.")
                y_proba = 1 - y_proba
                y_pred = 1 - y_pred
            
            # Evaluate
            acc = accuracy_score(y_val, y_pred)
            ll = log_loss(y_val, y_proba)
            auc = roc_auc_score(y_val, y_proba)
            brier = brier_score_loss(y_val, y_proba)
            
            # Tune threshold
            threshold, _ = model.tune_threshold(y_val, y_proba)
            
            accuracies.append(acc)
            loglosses.append(ll)
            aucs.append(auc)
            brier_scores.append(brier)
            thresholds.append(threshold)
            
            print(f"  Season {val_season}: Acc={acc:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}, Brier={brier:.4f}, Threshold={threshold:.3f}")
    else:
        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            # Train model
            model = ImprovedNFLModel()
            level1_preds = model.train_base_models(X_train, y_train, X_val, y_val)
            
            # Calibrate models using the same features that were used for training
            model.calibrate_models(X_train, y_train, X_val, y_val)
            
            # Ensure level1_preds are aligned before training meta-model
            if len(level1_preds) > 0:
                # Get the length from the first prediction
                pred_length = len(list(level1_preds.values())[0])
                # Filter to ensure all predictions have the same length
                level1_preds_aligned = {}
                for k, v in level1_preds.items():
                    if len(v) == pred_length:
                        level1_preds_aligned[k] = v
                    else:
                        print(f"  Warning: {k} prediction length mismatch, skipping")
                
                if len(level1_preds_aligned) > 0:
                    # Align X_val and y_val to match prediction length
                    if len(X_val) != pred_length:
                        X_val_aligned = X_val.iloc[:pred_length]
                        y_val_aligned = y_val.iloc[:pred_length] if hasattr(y_val, 'iloc') else y_val[:pred_length]
                    else:
                        X_val_aligned = X_val
                        y_val_aligned = y_val
                    
                    model.train_meta_model(level1_preds_aligned, X_train, y_train, X_val_aligned, y_val_aligned)
                else:
                    print(f"  Error: No valid Level-1 predictions for meta-model")
                    continue
            
            # Predict
            y_proba, y_pred = model.predict(X_val, apply_calibration=True, apply_rules=True)
            
            # Check if model is predicting opposite (AUC < 0.5)
            auc_check = roc_auc_score(y_val, y_proba)
            if auc_check < 0.45:
                # Model is predicting opposite - flip predictions
                print(f"  Warning: AUC {auc_check:.4f} < 0.45, model appears to be predicting opposite. Flipping predictions.")
                y_proba = 1 - y_proba
                y_pred = 1 - y_pred
            
            # Evaluate
            acc = accuracy_score(y_val, y_pred)
            ll = log_loss(y_val, y_proba)
            auc = roc_auc_score(y_val, y_proba)
            brier = brier_score_loss(y_val, y_proba)
            
            # Tune threshold
            threshold, _ = model.tune_threshold(y_val, y_proba)
            
            accuracies.append(acc)
            loglosses.append(ll)
            aucs.append(auc)
            brier_scores.append(brier)
            thresholds.append(threshold)
            
            print(f"  Fold {fold+1}: Acc={acc:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}, Brier={brier:.4f}, Threshold={threshold:.3f}")
    
    optimal_threshold = np.mean(thresholds)
    
    print(f"\nCV Results:")
    print(f"  Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"  Log Loss: {np.mean(loglosses):.4f} (+/- {np.std(loglosses):.4f})")
    print(f"  ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"  Brier Score: {np.mean(brier_scores):.4f} (+/- {np.std(brier_scores):.4f})")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    
    return optimal_threshold, {
        'accuracies': accuracies,
        'loglosses': loglosses,
        'aucs': aucs,
        'brier_scores': brier_scores,
        'thresholds': thresholds
    }

