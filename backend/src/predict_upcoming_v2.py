"""
Predict Upcoming NFL Games Using Simplified Model V2

Uses the same feature preparation pipeline as training for consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from prepare_enhanced_features import prepare_enhanced_features
from train_model_v2 import SimpleNFLModel
from data_collection import load_game_data, get_current_season

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


def predict_upcoming_v2(season=None, week=None, min_confidence=0.0):
    """
    Predict upcoming games using simplified model V2.
    
    Args:
        season: Specific season (None = current)
        week: Specific week (None = next week)
        min_confidence: Minimum confidence threshold
    """
    print("=" * 70)
    print("Predicting Upcoming Games (Model V2)")
    print("=" * 70)
    
    # Load model
    model_path = MODELS_DIR / "model_v2.pkl"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python src/train_model_v2.py")
        return None
    
    try:
        model = SimpleNFLModel.load(model_path)
        print(f"Loaded model (threshold: {model.threshold:.3f})")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Get current season/week if not specified
    current_season = get_current_season()
    if season is None:
        season = current_season
    
    # Load games
    pbp, games = load_game_data()
    if games is None:
        print("Error: Could not load game data")
        return None
    
    # Find upcoming games
    if week is None:
        # Find next week with games
        date_col = None
        for col in ['game_date', 'gameday', 'date', 'game_time']:
            if col in games.columns:
                date_col = col
                break
        
        if date_col:
            current_date = datetime.now().strftime('%Y-%m-%d')
            upcoming_games = games[
                (games['season'] == season) &
                (games[date_col] >= current_date)
            ].sort_values(date_col)
        else:
            current_week = datetime.now().isocalendar()[1]
            upcoming_games = games[
                (games['season'] == season) &
                (games['week'] >= current_week)
            ].sort_values('week')
    else:
        upcoming_games = games[
            (games['season'] == season) &
            (games['week'] == week)
        ]
    
    if len(upcoming_games) == 0:
        print(f"No upcoming games found for season {season}")
        return None
    
    print(f"\nFound {len(upcoming_games)} upcoming games")
    
    # Use the SAME feature preparation pipeline as training
    # This ensures features are built consistently
    print("\nBuilding features using training pipeline...")
    
    # Prepare features for all games (including upcoming)
    # The prepare_enhanced_features function will include upcoming games
    # and return None for y when games don't have results
    X_all, y_all = prepare_enhanced_features(
        min_season=min(2018, season - 6),
        max_season=season
    )
    
    if X_all is None:
        print("Error: Could not prepare features")
        return None
    
    # Get metadata to identify upcoming games
    metadata = None
    if hasattr(X_all, '_metadata'):
        metadata = X_all._metadata
    else:
        # Try to reconstruct metadata from X_all if it has game_id, season, week, etc.
        metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team']
        available_cols = [col for col in metadata_cols if col in X_all.columns]
        if available_cols:
            metadata = X_all[available_cols].copy()
    
    # Strategy 1: Match by game_id if available
    if metadata is not None and 'game_id' in metadata.columns:
        upcoming_game_ids = set()
        if 'game_id' in upcoming_games.columns:
            upcoming_game_ids = set(upcoming_games['game_id'].dropna())
        
        if len(upcoming_game_ids) > 0:
            upcoming_mask = metadata['game_id'].isin(upcoming_game_ids)
            X_upcoming = X_all[upcoming_mask].copy()
            print(f"  Matched {len(X_upcoming)} games by game_id")
        else:
            # Strategy 2: Match by season, week, home_team, away_team
            if all(col in metadata.columns for col in ['season', 'week', 'home_team', 'away_team']):
                upcoming_mask = pd.Series(False, index=X_all.index)
                for _, game in upcoming_games.iterrows():
                    home = game.get('home_team')
                    away = game.get('away_team')
                    s = game.get('season', season)
                    w = game.get('week')
                    if pd.notna(home) and pd.notna(away) and pd.notna(s) and pd.notna(w):
                        match_mask = (
                            (metadata['home_team'] == home) &
                            (metadata['away_team'] == away) &
                            (metadata['season'] == s) &
                            (metadata['week'] == w)
                        )
                        upcoming_mask |= match_mask
                X_upcoming = X_all[upcoming_mask].copy()
                print(f"  Matched {len(X_upcoming)} games by season/week/teams")
            else:
                # Strategy 3: Use target-based filtering
                if y_all is not None:
                    X_upcoming = X_all[y_all.isna()].copy()
                    print(f"  Using target-based filtering: {len(X_upcoming)} games without results")
                else:
                    # Strategy 4: Filter by season/week if metadata has those
                    if metadata is not None and 'season' in metadata.columns and 'week' in metadata.columns:
                        if week is not None:
                            upcoming_mask = (
                                (metadata['season'] == season) &
                                (metadata['week'] == week)
                            )
                        else:
                            # Use all games from the specified season
                            upcoming_mask = (metadata['season'] == season)
                        X_upcoming = X_all[upcoming_mask].copy()
                        print(f"  Using season/week filtering: {len(X_upcoming)} games")
                    else:
                        # Last resort: use all games (not ideal)
                        X_upcoming = X_all.copy()
                        print(f"  Warning: Using all games as fallback: {len(X_upcoming)} games")
    else:
        # No metadata available - use target-based or season filtering
        print("  Warning: Metadata not available, using fallback methods")
        if y_all is not None:
            X_upcoming = X_all[y_all.isna()].copy()
            print(f"  Using target-based filtering: {len(X_upcoming)} games without results")
        elif 'season' in X_all.columns and 'week' in X_all.columns:
            if week is not None:
                X_upcoming = X_all[
                    (X_all['season'] == season) &
                    (X_all['week'] == week)
                ].copy()
            else:
                X_upcoming = X_all[X_all['season'] == season].copy()
            print(f"  Using season/week columns: {len(X_upcoming)} games")
        else:
            X_upcoming = X_all.copy()
            print(f"  Warning: Using all games as last resort: {len(X_upcoming)} games")
    
    if X_upcoming is None or len(X_upcoming) == 0:
        print("Error: Could not identify upcoming games in feature matrix")
        print(f"  Total games in feature matrix: {len(X_all)}")
        print(f"  Upcoming games found: {len(upcoming_games)}")
        if metadata is not None:
            print(f"  Metadata columns: {list(metadata.columns)}")
        print("  This may indicate a mismatch between game data and feature preparation")
        return None
    
    print(f"Built features for {len(X_upcoming)} upcoming games")
    
    # Debug: check feature statistics
    print(f"\nFeature statistics:")
    print(f"  Total features: {len(X_upcoming.columns)}")
    print(f"  Features with all zeros: {(X_upcoming == 0).all().sum()}")
    print(f"  Features with any NaN: {X_upcoming.isna().any().sum()}")
    
    # Make predictions
    print(f"\nMaking predictions...")
    try:
        y_proba = model.predict_proba(X_upcoming)
        y_pred = model.predict(X_upcoming)
        
        print(f"  Probability range: [{y_proba.min():.3f}, {y_proba.max():.3f}]")
        print(f"  Mean probability: {y_proba.mean():.3f}")
        print(f"  Std probability: {y_proba.std():.3f}")
        
        # Check if predictions are too uniform
        if y_proba.std() < 0.01:
            print(f"  ⚠️  WARNING: All probabilities are nearly identical!")
            print(f"     This suggests features may not be differentiating games properly.")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate confidence
    confidence = np.abs(y_proba - 0.5) * 2
    
    # Create results
    predictions = []
    
    # Try to get game info from metadata or reconstruct it
    for i in range(len(X_upcoming)):
        if i < len(upcoming_games):
            game = upcoming_games.iloc[i]
            home_team = game.get('home_team', 'UNK')
            away_team = game.get('away_team', 'UNK')
            game_season = game.get('season', season)
            game_week = game.get('week', 0)
            game_date = game.get('game_date') or game.get('gameday')
            gametime = game.get('gametime')
        else:
            # Fallback if we can't match
            home_team = 'UNK'
            away_team = 'UNK'
            game_season = season
            game_week = 0
            game_date = None
            gametime = None
        
        conf = confidence[i] if i < len(confidence) else 0
        
        if conf < min_confidence:
            continue
        
        home_prob = y_proba[i] if i < len(y_proba) else 0.5
        pred_winner = home_team if (y_pred[i] == 1 if i < len(y_pred) else True) else away_team
        
        predictions.append({
            'season': game_season,
            'week': game_week,
            'gameday': game_date,
            'gametime': gametime,
            'away_team': away_team,
            'home_team': home_team,
            'home_win_probability': float(home_prob),
            'away_win_probability': float(1 - home_prob),
            'predicted_winner': pred_winner,
            'confidence': float(conf),
            'prediction': 'Home' if (y_pred[i] == 1 if i < len(y_pred) else True) else 'Away'
        })
    
    if len(predictions) == 0:
        print(f"No predictions meet confidence threshold of {min_confidence}")
        return None
    
    # Display results
    results_df = pd.DataFrame(predictions)
    
    # Sort by week, date, time
    sort_cols = ['week']
    if 'gameday' in results_df.columns:
        results_df['_sort_date'] = pd.to_datetime(results_df['gameday'], errors='coerce')
        sort_cols.append('_sort_date')
    if 'gametime' in results_df.columns:
        def parse_time(time_str):
            if pd.isna(time_str) or time_str == '':
                return pd.NaT
            try:
                time_parts = str(time_str).split(':')
                if len(time_parts) >= 2:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    return pd.Timestamp(2000, 1, 1, hours, minutes)
            except:
                pass
            return pd.NaT
        results_df['_sort_time'] = results_df['gametime'].apply(parse_time)
        sort_cols.append('_sort_time')
    sort_cols.append('confidence')
    
    results_df = results_df.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False])
    results_df = results_df.drop(columns=['_sort_date', '_sort_time'], errors='ignore')
    
    print("\n" + "=" * 70)
    print("Upcoming Game Predictions")
    print("=" * 70)
    
    # Display results
    current_week = None
    for _, row in results_df.iterrows():
        week = row['week']
        
        if week != current_week:
            if current_week is not None:
                print()
            print(f"\n{'=' * 70}")
            print(f"Week {int(week)} - Season {int(row.get('season', season))}")
            print(f"{'=' * 70}")
            current_week = week
        
        winner = row['predicted_winner']
        home_prob = row['home_win_probability']
        away_prob = row['away_win_probability']
        conf = row['confidence']
        winner_prob = home_prob if row['prediction'] == 'Home' else away_prob
        
        game_date = row.get('gameday', '')
        gametime = row.get('gametime', '')
        date_str = f" ({game_date})" if game_date and pd.notna(game_date) else ""
        time_str = f" {gametime}" if gametime and pd.notna(gametime) else ""
        
        print(f"  {row['away_team']:3s} @ {row['home_team']:3s}: {winner:3s} wins ({winner_prob:.1%}, home: {home_prob:.1%}, away: {away_prob:.1%}, conf: {conf:.1%}){date_str}{time_str}")
    
    # Save predictions
    try:
        from database import save_predictions, init_db
        init_db(create_tables=False)
        save_predictions(results_df, model_version='v2')
        print(f"\n✓ Predictions saved to database")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save to database: {e}")
        predictions_path = DATA_DIR / "predictions_v2.csv"
        results_df.to_csv(predictions_path, index=False)
        print(f"  Predictions saved to {predictions_path}")
    
    return results_df




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict upcoming NFL games using model V2')
    parser.add_argument('--season', type=int, default=None, help='Season year (default: current)')
    parser.add_argument('--week', type=int, default=None, help='Week number (default: next week)')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    predict_upcoming_v2(
        season=args.season,
        week=args.week,
        min_confidence=args.min_confidence
    )

