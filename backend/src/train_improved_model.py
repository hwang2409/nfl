"""
Train Improved NFL Prediction Model

This is the main training script for the improved model architecture.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from prepare_enhanced_features import prepare_enhanced_features
from improved_model import ImprovedNFLModel, forward_chaining_cv_improved
from data_collection import get_current_season
from sklearn.metrics import roc_auc_score

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("NFL Improved Model Training")
    print("=" * 70)
    
    # 1. Prepare enhanced features
    print("\n[1/3] Preparing enhanced features...")
    current_season = get_current_season()
    import sys
    accuracy_mode = 'accurate' if '--accurate' in sys.argv else 'fast'
    # Use more historical data for accuracy mode
    if accuracy_mode == 'accurate':
        min_season = max(2010, current_season - 10)  # Use last 10+ seasons
        print(f"  Using more historical data: seasons {min_season}-{current_season}")
    else:
        min_season = max(2018, current_season - 6)  # Use last 6-7 seasons
    
    X, y = prepare_enhanced_features(min_season=min_season, max_season=current_season)
    
    if X is None or y is None or len(X) == 0:
        print("Error: Failed to prepare enhanced features or no data available.")
        return
    
    print(f"Training on {len(X)} games with {len(X.columns)} features from seasons {min_season}-{current_season}")
    
    # 2. Perform improved forward-chaining cross-validation
    print("\n[2/3] Performing improved forward-chaining cross-validation...")
    n_splits = 7 if accuracy_mode == 'accurate' else 5  # More CV folds for accuracy mode
    optimal_threshold, cv_results = forward_chaining_cv_improved(X, y, n_splits=n_splits)
    
    # 3. Train final improved model on all data
    print("\n[3/3] Training final improved model on all data...")
    # Use 'accurate' mode for maximum accuracy (trades speed for accuracy)
    use_optuna = '--optuna' in sys.argv
    n_trials = 100 if '--optuna-trials' in sys.argv else 50
    if '--optuna-trials' in sys.argv:
        try:
            idx = sys.argv.index('--optuna-trials')
            n_trials = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass
    
    if accuracy_mode == 'accurate':
        print("  Using ACCURACY mode (slower but more accurate)")
    else:
        print("  Using FAST mode (faster but less accurate)")
        print("  Use --accurate flag for maximum accuracy")
    
    if use_optuna:
        print(f"  Using Optuna hyperparameter tuning ({n_trials} trials)")
        print("  This will significantly increase training time but improve accuracy")
    else:
        print("  Use --optuna flag to enable hyperparameter tuning")
    
    model = ImprovedNFLModel(accuracy_mode=accuracy_mode, use_optuna=use_optuna, n_trials=n_trials)
    
    # Train base models
    level1_preds = model.train_base_models(X, y, use_feature_selection=True)
    
    # Calibrate models
    model.calibrate_models(X, y)
    
    # Train meta-model
    model.train_meta_model(level1_preds, X, y)
    
    # Check if model needs prediction flipping
    # Test on a sample to see if predictions are inverted
    if len(X) > 100:
        test_sample = X.sample(min(100, len(X)), random_state=42)
        y_sample = y.loc[test_sample.index]
        y_proba_test, _ = model.predict(test_sample, apply_calibration=True, apply_rules=True)
        test_auc = roc_auc_score(y_sample, y_proba_test)
        if test_auc < 0.45:
            print(f"\n  Warning: Model appears to be predicting opposite (AUC: {test_auc:.4f})")
            print(f"  This will be automatically corrected during predictions.")
    
    # Set optimal threshold
    model.threshold = optimal_threshold
    
    # Save model
    model_path = MODELS_DIR / "improved_model.pkl"
    model.save(model_path)
    
    # Print tuned hyperparameters if Optuna was used
    if model.tuned_params:
        print("\n" + "=" * 70)
        print("Optuna-Tuned Hyperparameters:")
        print("=" * 70)
        for model_name, params in model.tuned_params.items():
            print(f"\n{model_name.upper()}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    
    # Print feature importance if available
    if model.feature_importance is not None:
        print("\n" + "=" * 70)
        print("Top 15 Most Important Features:")
        print("=" * 70)
        print(model.feature_importance.head(15).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Average CV Accuracy: {sum(cv_results['accuracies']) / len(cv_results['accuracies']):.4f}")
    print(f"Average CV Log Loss: {sum(cv_results['loglosses']) / len(cv_results['loglosses']):.4f}")
    print(f"Average CV ROC-AUC: {sum(cv_results['aucs']) / len(cv_results['aucs']):.4f}")
    print(f"Average CV Brier Score: {sum(cv_results['brier_scores']) / len(cv_results['brier_scores']):.4f}")


if __name__ == "__main__":
    main()

