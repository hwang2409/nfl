"""
Enhanced NFL Prediction Model Training Pipeline V2

Enhanced with:
1. Multiple base models (Logistic Regression, XGBoost, LightGBM, CatBoost, Random Forest, Neural Networks)
2. Ensemble methods (stacking, blending, weighted)
3. Monte Carlo simulations for uncertainty quantification
4. Reinforcement learning for adaptive learning from outcomes
5. Proper feature preparation and alignment
6. Time-aware validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try to import CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

sys.path.append(str(Path(__file__).parent))

from prepare_enhanced_features import prepare_enhanced_features
from data_collection import get_current_season

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ReinforcementLearner:
    """
    Reinforcement Learning system that adapts model weights based on prediction outcomes.
    Rewards accurate predictions and penalizes inaccurate ones.
    """
    
    def __init__(self, learning_rate=0.01, decay=0.95):
        """
        Args:
            learning_rate: How quickly to adapt weights
            decay: Decay factor for historical performance
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.model_performance = {}  # Track performance per model
        self.prediction_history = []  # Store predictions and outcomes
        
    def update(self, model_name, predicted_prob, actual_outcome, confidence):
        """
        Update model performance based on prediction outcome.
        
        Args:
            model_name: Name of the model
            predicted_prob: Predicted probability
            actual_outcome: Actual outcome (0 or 1)
            confidence: Confidence level of prediction
        """
        # Calculate reward: positive for correct predictions, negative for incorrect
        predicted_outcome = 1 if predicted_prob >= 0.5 else 0
        correct = 1 if predicted_outcome == actual_outcome else 0
        
        # Reward is higher for confident correct predictions
        reward = (2 * correct - 1) * confidence
        
        # Update performance tracking
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'total_reward': 0.0,
                'count': 0,
                'weight': 1.0
            }
        
        # Decay old performance
        self.model_performance[model_name]['total_reward'] *= self.decay
        self.model_performance[model_name]['total_reward'] += reward
        self.model_performance[model_name]['count'] += 1
        
        # Update weight based on cumulative reward
        avg_reward = self.model_performance[model_name]['total_reward'] / max(
            self.model_performance[model_name]['count'], 1
        )
        self.model_performance[model_name]['weight'] = max(
            0.1,  # Minimum weight
            1.0 + self.learning_rate * avg_reward
        )
        
        # Store history
        self.prediction_history.append({
            'model': model_name,
            'predicted_prob': predicted_prob,
            'actual': actual_outcome,
            'reward': reward
        })
    
    def get_weights(self):
        """Get current model weights."""
        weights = {}
        total_weight = sum(p['weight'] for p in self.model_performance.values())
        
        if total_weight > 0:
            for model_name, perf in self.model_performance.items():
                weights[model_name] = perf['weight'] / total_weight
        else:
            # Equal weights if no history
            n_models = len(self.model_performance)
            if n_models > 0:
                weights = {name: 1.0 / n_models for name in self.model_performance.keys()}
        
        return weights
    
    def reset(self):
        """Reset learning history."""
        self.model_performance = {}
        self.prediction_history = []


class SimpleNFLModel:
    """
    Enhanced NFL prediction model with multiple base models, ensemble methods,
    Monte Carlo simulations, and reinforcement learning.
    """
    
    def __init__(self, random_state=42, use_ensemble=True, ensemble_method='weighted',
                 use_monte_carlo=False, n_simulations=1000, use_rl=False):
        """
        Args:
            random_state: Random seed
            use_ensemble: Whether to use ensemble of multiple models
            ensemble_method: 'weighted', 'stacking', or 'blending'
            use_monte_carlo: Whether to use Monte Carlo for uncertainty quantification
            n_simulations: Number of Monte Carlo simulations
            use_rl: Whether to use reinforcement learning
        """
        self.random_state = random_state
        self.use_ensemble = use_ensemble
        self.ensemble_method = ensemble_method
        self.use_monte_carlo = use_monte_carlo
        self.n_simulations = n_simulations
        self.use_rl = use_rl
        
        # Model storage
        self.models = {}  # Store all base models
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        
        # Ensemble components
        self.meta_model = None  # For stacking
        self.model_weights = {}  # For weighted ensemble
        
        # RL system
        self.rl_learner = ReinforcementLearner() if use_rl else None
        
        # Monte Carlo storage
        self.mc_predictions = None
        
    def train(self, X, y, X_val=None, y_val=None, feature_list=None):
        """Train all base models and ensemble."""
        print("\nTraining Enhanced Model V2...")
        print(f"  Ensemble: {self.use_ensemble}, Method: {self.ensemble_method}")
        print(f"  Monte Carlo: {self.use_monte_carlo}, RL: {self.use_rl}")
        
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
        
        # Train all base models
        base_predictions = {}
        base_predictions_val = {}
        
        print("\nTraining Base Models:")
        
        # 1. Logistic Regression
        print("  1. Logistic Regression...")
        lr_model = LogisticRegression(
            C=1.0,
            penalty='l2',
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        lr_model.fit(X_scaled, y)
        self.models['lr'] = lr_model
        base_predictions['lr'] = lr_model.predict_proba(X_scaled)[:, 1]
        if X_val is not None:
            base_predictions_val['lr'] = lr_model.predict_proba(X_val_scaled)[:, 1]
        
        # 2. XGBoost
        print("  2. XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        if X_val is not None:
            # XGBoost early stopping API varies by version - use eval_set without early stopping
            # Early stopping is handled by n_estimators being set appropriately
            try:
                xgb_model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
            except Exception as e:
                # If eval_set fails, just fit without it
                print(f"  Warning: XGBoost eval_set failed: {e}")
                xgb_model.fit(X_scaled, y, verbose=False)
        else:
            xgb_model.fit(X_scaled, y, verbose=False)
        self.models['xgb'] = xgb_model
        base_predictions['xgb'] = xgb_model.predict_proba(X_scaled)[:, 1]
        if X_val is not None:
            base_predictions_val['xgb'] = xgb_model.predict_proba(X_val_scaled)[:, 1]
        
        # 3. LightGBM
        print("  3. LightGBM...")
        lgb_model = lgb.LGBMClassifier(
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
            lgb_model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            lgb_model.fit(X_scaled, y)
        self.models['lgb'] = lgb_model
        base_predictions['lgb'] = lgb_model.predict_proba(X_scaled)[:, 1]
        if X_val is not None:
            base_predictions_val['lgb'] = lgb_model.predict_proba(X_val_scaled)[:, 1]
        
        # 4. CatBoost (if available)
        if CATBOOST_AVAILABLE:
            print("  4. CatBoost...")
            cat_model = cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=False
            )
            if X_val is not None:
                cat_model.fit(
                    X_scaled, y,
                    eval_set=(X_val_scaled, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                cat_model.fit(X_scaled, y, verbose=False)
            self.models['cat'] = cat_model
            base_predictions['cat'] = cat_model.predict_proba(X_scaled)[:, 1]
            if X_val is not None:
                base_predictions_val['cat'] = cat_model.predict_proba(X_val_scaled)[:, 1]
        
        # 5. Random Forest
        print("  5. Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y)
        self.models['rf'] = rf_model
        base_predictions['rf'] = rf_model.predict_proba(X_scaled)[:, 1]
        if X_val is not None:
            base_predictions_val['rf'] = rf_model.predict_proba(X_val_scaled)[:, 1]
        
        # 6. Neural Network (MLP)
        print("  6. Neural Network (MLP)...")
        # MLP early stopping requires validation_fraction > 0, so disable it when we have external validation
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state,
            early_stopping=(X_val is None),  # Only use early stopping if no external validation
            validation_fraction=0.1 if X_val is None else 0.0
        )
        nn_model.fit(X_scaled, y)
        self.models['nn'] = nn_model
        base_predictions['nn'] = nn_model.predict_proba(X_scaled)[:, 1]
        if X_val is not None:
            base_predictions_val['nn'] = nn_model.predict_proba(X_val_scaled)[:, 1]
        
        print(f"\n  Trained {len(self.models)} base models with {len(self.feature_names)} features")
        
        # Train ensemble
        if self.use_ensemble:
            print(f"\nTraining Ensemble ({self.ensemble_method})...")
            if self.ensemble_method == 'stacking':
                self._train_stacking_ensemble(base_predictions, y, base_predictions_val, y_val)
            elif self.ensemble_method == 'weighted':
                self._train_weighted_ensemble(base_predictions_val, y_val)
            elif self.ensemble_method == 'blending':
                self._train_blending_ensemble(base_predictions_val, y_val)
        else:
            # Use LightGBM as default single model
            self.model = self.models['lgb']
        
        # Evaluate on validation set if available
        if X_val is not None:
            y_proba = self.predict_proba(X_val)
            y_pred = (y_proba >= self.threshold).astype(int)
            
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            ll = log_loss(y_val, y_proba)
            
            print(f"\nValidation Results:")
            print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, LogLoss: {ll:.4f}")
            
            # Tune threshold
            self.threshold = self._tune_threshold(y_val, y_proba)
            print(f"  Optimal threshold: {self.threshold:.3f}")
    
    def _train_stacking_ensemble(self, base_predictions, y, base_predictions_val=None, y_val=None):
        """Train stacking meta-model."""
        # Prepare meta-features
        meta_X = pd.DataFrame(base_predictions)
        
        if base_predictions_val is not None and y_val is not None:
            meta_X_val = pd.DataFrame(base_predictions_val)
            # Train meta-model on validation predictions
            self.meta_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            )
            self.meta_model.fit(meta_X_val, y_val)
        else:
            # Use simple averaging if no validation set
            self.meta_model = None
    
    def _train_weighted_ensemble(self, base_predictions_val, y_val):
        """Train weighted ensemble based on validation performance."""
        if base_predictions_val is None or y_val is None:
            # Equal weights if no validation
            n_models = len(self.models)
            self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}
            return
        
        # Calculate performance for each model
        performances = {}
        for model_name, preds in base_predictions_val.items():
            acc = accuracy_score(y_val, (preds >= 0.5).astype(int))
            auc = roc_auc_score(y_val, preds)
            ll = log_loss(y_val, preds)
            # Combined score (higher is better)
            score = (acc * 0.4) + (auc * 0.3) + ((1 - min(ll, 1)) * 0.3)
            performances[model_name] = score
        
        # Convert to weights (softmax-like)
        total_score = sum(performances.values())
        if total_score > 0:
            self.model_weights = {name: score / total_score for name, score in performances.items()}
        else:
            n_models = len(self.models)
            self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}
        
        print(f"  Model weights: {self.model_weights}")
    
    def _train_blending_ensemble(self, base_predictions_val, y_val):
        """Train blending ensemble (simple averaging)."""
        # Equal weights for blending
        n_models = len(self.models)
        self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}
    
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
    
    def predict_proba(self, X, use_monte_carlo=None):
        """
        Get probability predictions with optional Monte Carlo uncertainty quantification.
        
        Args:
            X: Feature matrix
            use_monte_carlo: Override instance setting for Monte Carlo
        
        Returns:
            Predicted probabilities (or mean if Monte Carlo)
        """
        if len(self.models) == 0:
            raise ValueError("Model not trained yet")
        
        use_mc = use_monte_carlo if use_monte_carlo is not None else self.use_monte_carlo
        
        if use_mc:
            return self._predict_monte_carlo(X)
        
        # Align features
        X = self._align_features(X)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all base models
        base_predictions = {}
        for model_name, model in self.models.items():
            try:
                if model_name == 'nn':
                    # Neural network already uses scaled features
                    base_predictions[model_name] = model.predict_proba(X_scaled)[:, 1]
                else:
                    base_predictions[model_name] = model.predict_proba(X_scaled)[:, 1]
            except Exception as e:
                print(f"Warning: Error predicting with {model_name}: {e}")
                continue
        
        if len(base_predictions) == 0:
            raise ValueError("No valid predictions from base models")
        
        # Combine predictions based on ensemble method
        if self.use_ensemble:
            if self.ensemble_method == 'stacking' and self.meta_model is not None:
                # Use meta-model
                meta_X = pd.DataFrame(base_predictions)
                return self.meta_model.predict_proba(meta_X)[:, 1]
            elif self.ensemble_method in ['weighted', 'blending']:
                # Weighted average
                if self.use_rl and self.rl_learner is not None:
                    # Use RL-learned weights
                    weights = self.rl_learner.get_weights()
                    # Fallback to trained weights if RL has no history
                    if not weights:
                        weights = self.model_weights
                else:
                    weights = self.model_weights
                
                # Weighted average
                weighted_sum = np.zeros(len(X))
                total_weight = 0
                for model_name, preds in base_predictions.items():
                    weight = weights.get(model_name, 1.0 / len(base_predictions))
                    weighted_sum += preds * weight
                    total_weight += weight
                
                if total_weight > 0:
                    return weighted_sum / total_weight
                else:
                    return np.mean(list(base_predictions.values()), axis=0)
            else:
                # Simple average
                return np.mean(list(base_predictions.values()), axis=0)
        else:
            # Use single model (LightGBM) - backward compatibility
            if hasattr(self, 'model') and self.model is not None:
                # Old format - single model
                return self.model.predict_proba(X_scaled)[:, 1]
            else:
                # Use LightGBM as default
                return base_predictions.get('lgb', list(base_predictions.values())[0])
    
    def _predict_monte_carlo(self, X, n_simulations=None):
        """
        Monte Carlo simulation for uncertainty quantification.
        Simulates many possible outcomes to get probability distribution.
        """
        n_sim = n_simulations or self.n_simulations
        
        # Get base predictions
        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        # Collect predictions from all models with some randomness
        all_predictions = []
        
        for _ in range(n_sim):
            # Sample from base models with some noise
            sim_predictions = []
            
            for model_name, model in self.models.items():
                try:
                    pred = model.predict_proba(X_scaled)[:, 1]
                    # Add small random noise to simulate uncertainty
                    noise = np.random.normal(0, 0.02, len(pred))
                    pred_noisy = np.clip(pred + noise, 0, 1)
                    sim_predictions.append(pred_noisy)
                except:
                    continue
            
            if sim_predictions:
                # Average across models for this simulation
                sim_avg = np.mean(sim_predictions, axis=0)
                all_predictions.append(sim_avg)
        
        if not all_predictions:
            # Fallback to regular prediction
            return self.predict_proba(X, use_monte_carlo=False)
        
        # Return mean of all simulations
        mc_mean = np.mean(all_predictions, axis=0)
        self.mc_predictions = np.array(all_predictions)  # Store for uncertainty analysis
        
        return mc_mean
    
    def get_uncertainty(self, X):
        """
        Get uncertainty estimates from Monte Carlo simulations.
        Must call predict_proba with Monte Carlo first.
        """
        if self.mc_predictions is None:
            # Run Monte Carlo
            self.predict_proba(X, use_monte_carlo=True)
        
        if self.mc_predictions is None:
            return None
        
        # Calculate standard deviation across simulations
        return np.std(self.mc_predictions, axis=0)
    
    def predict(self, X):
        """Get binary predictions."""
        y_proba = self.predict_proba(X)
        return (y_proba >= self.threshold).astype(int)
    
    def update_rl(self, X, y_true):
        """
        Update reinforcement learning system with actual outcomes.
        
        Args:
            X: Feature matrix
            y_true: Actual outcomes
        """
        if not self.use_rl or self.rl_learner is None:
            return
        
        # Get predictions from all models
        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        for model_name, model in self.models.items():
            try:
                preds = model.predict_proba(X_scaled)[:, 1]
                confidences = np.abs(preds - 0.5) * 2  # Confidence metric
                
                # Update RL for each prediction
                for i in range(len(preds)):
                    self.rl_learner.update(
                        model_name,
                        preds[i],
                        y_true.iloc[i] if isinstance(y_true, pd.Series) else y_true[i],
                        confidences[i]
                    )
            except Exception as e:
                print(f"Warning: Error updating RL for {model_name}: {e}")
                continue
    
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
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'use_ensemble': self.use_ensemble,
            'ensemble_method': self.ensemble_method,
            'model_weights': self.model_weights,
            'meta_model': self.meta_model,
            'use_monte_carlo': self.use_monte_carlo,
            'n_simulations': self.n_simulations,
            'use_rl': self.use_rl,
            'rl_learner': self.rl_learner
        }
        pickle.dump(model_data, open(filepath, 'wb'))
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model."""
        # Use custom unpickler to handle ReinforcementLearner class lookup
        import sys
        import importlib.util
        
        # Create a custom unpickler that knows where to find ReinforcementLearner
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # If looking for ReinforcementLearner, handle both __main__ and train_model_v2 module names
                if name == 'ReinforcementLearner':
                    # Check if it's in the current module (when loaded from train_model_v2)
                    if module == '__main__' or module == 'train_model_v2':
                        # Import the module if not already imported
                        if 'train_model_v2' not in sys.modules:
                            # Try to import it
                            try:
                                import train_model_v2
                                sys.modules['train_model_v2'] = train_model_v2
                            except:
                                # If import fails, load it directly
                                spec = importlib.util.spec_from_file_location(
                                    'train_model_v2',
                                    str(Path(__file__).parent / 'train_model_v2.py')
                                )
                                train_module = importlib.util.module_from_spec(spec)
                                sys.modules['train_model_v2'] = train_module
                                spec.loader.exec_module(train_module)
                        return getattr(sys.modules['train_model_v2'], 'ReinforcementLearner')
                # For other classes, use default lookup
                return super().find_class(module, name)
        
        try:
            with open(filepath, 'rb') as f:
                unpickler = CustomUnpickler(f)
                model_data = unpickler.load()
        except Exception as e:
            # Fallback: try regular pickle with import workaround
            try:
                # Ensure ReinforcementLearner is in sys.modules
                if 'train_model_v2' not in sys.modules:
                    import train_model_v2
                    sys.modules['train_model_v2'] = train_model_v2
                # Also add to current module's namespace
                if 'ReinforcementLearner' not in globals():
                    from train_model_v2 import ReinforcementLearner
                    globals()['ReinforcementLearner'] = ReinforcementLearner
                
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            except Exception as e2:
                # Last resort: try loading without RL object
                print(f"Warning: Could not load RL learner: {e2}")
                print("Attempting to load model without RL learner...")
                with open(filepath, 'rb') as f:
                    # Load and skip RL learner if it fails
                    model_data = pickle.load(f)
                # Remove RL learner if it causes issues
                if 'rl_learner' in model_data:
                    try:
                        _ = model_data['rl_learner']
                    except:
                        model_data['rl_learner'] = None
        
        # Backward compatibility: check if it's old format
        if 'model' in model_data and 'models' not in model_data:
            # Old format - single model
            model = cls(use_ensemble=False)
            model.model = model_data['model']
            model.scaler = model_data['scaler']
            model.feature_names = model_data['feature_names']
            model.threshold = model_data.get('threshold', 0.5)
            return model
        else:
            # New format
            model = cls(
                random_state=model_data.get('random_state', 42),
                use_ensemble=model_data.get('use_ensemble', True),
                ensemble_method=model_data.get('ensemble_method', 'weighted'),
                use_monte_carlo=model_data.get('use_monte_carlo', False),
                n_simulations=model_data.get('n_simulations', 1000),
                use_rl=model_data.get('use_rl', False)
            )
            model.models = model_data.get('models', {})
            model.model_weights = model_data.get('model_weights', {})
            model.meta_model = model_data.get('meta_model', None)
            
            # Handle RL learner - reconstruct if needed
            rl_learner_data = model_data.get('rl_learner', None)
            if rl_learner_data is not None:
                try:
                    model.rl_learner = rl_learner_data
                except:
                    # If RL learner can't be loaded, create a new one
                    if model.use_rl:
                        model.rl_learner = ReinforcementLearner()
                    else:
                        model.rl_learner = None
            else:
                model.rl_learner = None
        
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.threshold = model_data.get('threshold', 0.5)
        
        return model


def forward_chaining_cv(X, y, n_splits=5, use_ensemble=True, ensemble_method='weighted',
                       use_monte_carlo=False, use_rl=False):
    """Forward-chaining cross-validation with enhanced models."""
    print("\nPerforming Forward-Chaining Cross-Validation...")
    
    # First, identify all features that will be used
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    all_features = [col for col in X.columns if col not in constant_features]
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
                model = SimpleNFLModel(
                    use_ensemble=use_ensemble,
                    ensemble_method=ensemble_method,
                    use_monte_carlo=use_monte_carlo,
                    use_rl=use_rl
                )
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
        model = SimpleNFLModel(
            use_ensemble=use_ensemble,
            ensemble_method=ensemble_method,
            use_monte_carlo=use_monte_carlo,
            use_rl=use_rl
        )
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
    print("NFL Model Training V2 (Enhanced)")
    print("=" * 70)
    
    # Check command line arguments
    use_ensemble = '--no-ensemble' not in sys.argv
    ensemble_method = 'weighted'
    if '--stacking' in sys.argv:
        ensemble_method = 'stacking'
    elif '--blending' in sys.argv:
        ensemble_method = 'blending'
    
    use_monte_carlo = '--monte-carlo' in sys.argv
    use_rl = '--rl' in sys.argv or '--reinforcement' in sys.argv
    
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
    optimal_threshold, cv_results = forward_chaining_cv(
        X, y, n_splits=5,
        use_ensemble=use_ensemble,
        ensemble_method=ensemble_method,
        use_monte_carlo=use_monte_carlo,
        use_rl=use_rl
    )
    
    # Train final model
    print("\n[3/3] Training final model on all data...")
    
    # Remove constant features and 'season' if present
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if 'season' in X.columns:
        constant_features.append('season')
    X_final = X.drop(columns=constant_features, errors='ignore')
    
    model = SimpleNFLModel(
        use_ensemble=use_ensemble,
        ensemble_method=ensemble_method,
        use_monte_carlo=use_monte_carlo,
        use_rl=use_rl
    )
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
    print(f"\nModel Configuration:")
    print(f"  Ensemble: {use_ensemble}, Method: {ensemble_method}")
    print(f"  Monte Carlo: {use_monte_carlo}")
    print(f"  Reinforcement Learning: {use_rl}")


if __name__ == "__main__":
    main()
