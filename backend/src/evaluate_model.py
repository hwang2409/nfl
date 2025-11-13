"""
Model Evaluation and Analysis

Evaluates model accuracy on historical games and provides detailed analysis.
Uses the improved model as the default.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from prepare_enhanced_features import prepare_enhanced_features
from improved_model import ImprovedNFLModel
from data_collection import get_current_season
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    log_loss, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score
)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


def evaluate_model(season=None, week=None, min_confidence=0.0):
    """
    Evaluate model accuracy on historical games.
    
    Args:
        season: Specific season to evaluate (None = most recent complete season)
        week: Specific week to evaluate (None = all weeks)
        min_confidence: Minimum confidence threshold to include
    
    Returns:
        Dictionary with evaluation metrics and analysis
    """
    print("=" * 70)
    print("Model Evaluation and Analysis")
    print("=" * 70)
    
    # Load model
    model_path = MODELS_DIR / "improved_model.pkl"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python src/train_improved_model.py")
        return None
    
    try:
        model = ImprovedNFLModel.load(model_path)
        print(f"Loaded improved model (threshold: {model.threshold:.3f})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Prepare features
    # IMPORTANT: The model was trained on ALL available data (seasons 2018-2025 or 2010-2025)
    # This means evaluating on any of those seasons will show inflated accuracy due to train/test overlap
    # 
    # For proper evaluation, we need to use time-series cross-validation or hold out recent data.
    # Since the model is already trained on all data, we'll evaluate on the most recent season
    # but warn the user that this is still somewhat optimistic.
    #
    # For truly unbiased evaluation, retrain the model holding out the test season.
    current_season = get_current_season()
    
    if season is None:
        # Default: evaluate on the most recent complete season
        # WARNING: This may still overlap with training data if model was trained on all seasons
        test_season = current_season - 1  # Most recent complete season
        print(f"\n⚠️  WARNING: Model was likely trained on seasons including {test_season}.")
        print(f"   This evaluation may show inflated accuracy due to train/test overlap.")
        print(f"   For unbiased evaluation, retrain holding out season {test_season}.")
        print(f"   Evaluating on season {test_season}...")
        min_season = test_season
        max_season = test_season
    else:
        # User specified a season, use that
        print(f"\n⚠️  WARNING: If model was trained on season {season}, accuracy will be inflated.")
        min_season = season
        max_season = season
    
    X, y = prepare_enhanced_features(min_season=min_season, max_season=max_season)
    
    if X is None or y is None or len(X) == 0:
        print("Error: Could not prepare features")
        return None
    
    # Filter by week if specified
    if week is not None:
        if hasattr(X, '_metadata') and X._metadata is not None:
            metadata = X._metadata
            if 'week' in metadata.columns:
                mask = metadata['week'] == week
                X = X[mask].copy()
                y = y[mask].copy()
        elif 'week' in X.columns:
            mask = X['week'] == week
            X = X[mask].copy()
            y = y[mask].copy()
    
    if len(X) == 0:
        print("No data available for evaluation")
        return None
    
    print(f"\nEvaluating on {len(X)} games...")
    
    # Make predictions
    try:
        y_proba, y_pred = model.predict(X, apply_calibration=True, apply_rules=True)
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_proba)
    auc = roc_auc_score(y, y_proba)
    brier = brier_score_loss(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # Calculate confidence
    confidence = np.abs(y_proba - 0.5) * 2
    
    # Filter by confidence if specified
    if min_confidence > 0:
        conf_mask = confidence >= min_confidence
        X_filtered = X[conf_mask]
        y_filtered = y[conf_mask]
        y_pred_filtered = y_pred[conf_mask]
        y_proba_filtered = y_proba[conf_mask]
        
        accuracy_filtered = accuracy_score(y_filtered, y_pred_filtered)
        logloss_filtered = log_loss(y_filtered, y_proba_filtered)
        auc_filtered = roc_auc_score(y_filtered, y_proba_filtered)
        brier_filtered = brier_score_loss(y_filtered, y_proba_filtered)
        
        print(f"\nOverall Performance ({len(X)} games):")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print(f"\nFiltered Performance (confidence >= {min_confidence}, {len(X_filtered)} games):")
        print(f"  Accuracy: {accuracy_filtered:.4f} ({accuracy_filtered*100:.2f}%)")
        print(f"  Log Loss: {logloss_filtered:.4f}")
        print(f"  ROC-AUC: {auc_filtered:.4f}")
        print(f"  Brier Score: {brier_filtered:.4f}")
        print(f"  Coverage: {len(X_filtered) / len(X) * 100:.1f}% of all games")
    else:
        print(f"\nPerformance Metrics ({len(X)} games):")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Confusion matrix
    print("\n" + "=" * 70)
    print("Confusion Matrix")
    print("=" * 70)
    cm = confusion_matrix(y, y_pred)
    print(f"                Predicted")
    print(f"              Home  Away")
    print(f"Actual Home   {cm[1,1]:4d}  {cm[1,0]:4d}")
    print(f"       Away   {cm[0,1]:4d}  {cm[0,0]:4d}")
    
    # Accuracy by confidence level
    print("\n" + "=" * 70)
    print("Accuracy by Confidence Level")
    print("=" * 70)
    
    conf_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(conf_bins) - 1):
        conf_low = conf_bins[i]
        conf_high = conf_bins[i + 1]
        mask = (confidence >= conf_low) & (confidence < conf_high)
        
        if mask.sum() > 0:
            acc = accuracy_score(y[mask], y_pred[mask])
            count = mask.sum()
            print(f"  {conf_low:.1f}-{conf_high:.1f}: {acc:.4f} ({acc*100:.2f}%) - {count} games")
    
    # Classification report
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(classification_report(y, y_pred, target_names=['Away Win', 'Home Win']))
    
    # Analysis by season
    if hasattr(X, '_metadata') and X._metadata is not None and 'season' in X._metadata.columns:
        print("\n" + "=" * 70)
        print("Accuracy by Season")
        print("=" * 70)
        
        seasons = sorted(X._metadata['season'].unique())
        season_results = {}
        
        for s in seasons:
            season_mask = X._metadata['season'] == s
            if season_mask.sum() > 0:
                season_accuracy = accuracy_score(y[season_mask], y_pred[season_mask])
                season_count = season_mask.sum()
                print(f"{s}: {season_accuracy:.4f} ({season_count} games)")
                season_results[s] = {
                    'accuracy': season_accuracy,
                    'count': int(season_count)
                }
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Games: {len(X)}")
    print(f"Correct: {int(np.sum(y == y_pred))}")
    print(f"Incorrect: {int(np.sum(y != y_pred))}")
    
    results = {
        'overall': {
            'accuracy': accuracy,
            'total_games': len(X),
            'correct': int(np.sum(y == y_pred)),
            'incorrect': int(np.sum(y != y_pred))
        },
        'metrics': {
            'logloss': logloss,
            'auc': auc,
            'brier': brier,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'confusion_matrix': cm,
        'predictions': {
            'probabilities': y_proba,
            'predictions': y_pred,
            'actual': y.values,
            'confidence': confidence
        }
    }
    
    if min_confidence > 0:
        results['filtered'] = {
            'accuracy': accuracy_filtered,
            'total_games': len(X_filtered),
            'correct': int(np.sum(y_filtered == y_pred_filtered)),
            'incorrect': int(np.sum(y_filtered != y_pred_filtered)),
            'min_confidence': min_confidence
        }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--season', type=int, default=None, help='Specific season to evaluate')
    parser.add_argument('--week', type=int, default=None, help='Specific week to evaluate')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        season=args.season,
        week=args.week,
        min_confidence=args.min_confidence
    )
    
    if results:
        # Save results
        results_path = DATA_DIR / "evaluation_results.pkl"
        import pickle
        pickle.dump(results, open(results_path, 'wb'))
        print(f"\nResults saved to {results_path}")
