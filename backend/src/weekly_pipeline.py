"""
Automated Weekly Prediction Pipeline

Implements real-time updates and weekly retraining as per "Strategies for Improving Accuracy".
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

from data_collection import collect_game_data, get_current_season
from train_improved_model import main as train_model
from predict_upcoming_improved import predict_upcoming_improved
from evaluate_model import evaluate_model
from roster_tracking import track_roster_changes, update_player_database_from_changes

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def weekly_pipeline(season=None, week=None, snapshot_time='friday_6pm'):
    """
    Automated weekly pipeline for NFL predictions.
    
    Implements:
    - Weekly retraining with new data
    - Feature updates with most recent games
    - Prediction generation at snapshot time
    - Performance tracking
    
    Args:
        season: Season year (default: current)
        week: Week number (default: upcoming)
        snapshot_time: When to lock features ('friday_6pm', 'thursday_8pm', etc.)
    """
    print("=" * 60)
    print("Weekly NFL Prediction Pipeline")
    print("=" * 60)
    print(f"Snapshot time: {snapshot_time}")
    print(f"Run time: {datetime.now()}")
    
    if season is None:
        season = get_current_season()
    
    # Step 1: Collect latest data
    print("\n[1/5] Collecting latest game data...")
    try:
        collect_game_data(start_year=season-1, end_year=season)
        print("✓ Data collection complete")
    except Exception as e:
        print(f"✗ Error collecting data: {e}")
        return
    
    # Step 1.5: Track roster changes
    print("\n[1.5/5] Tracking roster changes...")
    try:
        changes = track_roster_changes(season=season, week=week, save_changes=True)
        if len(changes) > 0:
            print(f"✓ Found {len(changes)} roster changes")
            # Update player database if there are significant changes
            if len(changes) > 0:
                update_player_database_from_changes(season=season)
        else:
            print("✓ No roster changes detected")
    except Exception as e:
        print(f"✗ Error tracking roster changes: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 2: Retrain model with new data
    print("\n[2/5] Retraining improved model...")
    try:
        train_model()
        print("✓ Model retraining complete")
    except Exception as e:
        print(f"✗ Error retraining model: {e}")
        return
    
    # Step 3: Evaluate previous week's predictions (if available)
    if week is not None and week > 1:
        print(f"\n[3/5] Evaluating Week {week-1} predictions...")
        try:
            results = evaluate_model(season=season, week=week-1)
            if results:
                print(f"✓ Week {week-1} accuracy: {results['overall']['accuracy']:.4f}")
        except Exception as e:
            print(f"✗ Error evaluating: {e}")
    else:
        print("\n[3/5] Skipping evaluation (Week 1 or no previous week)")
    
    # Step 4: Generate predictions for upcoming week
    print(f"\n[4/5] Generating predictions for {season} Week {week or 'upcoming'}...")
    try:
        predictions = predict_upcoming_improved(season=season, week=week, min_confidence=0.0)
        if predictions is not None and len(predictions) > 0:
            # Display predictions (already done in predict_upcoming_improved)
            
            # Save predictions with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = DATA_DIR / f"predictions_{season}_week{week or 'upcoming'}_{timestamp}.csv"
            predictions.to_csv(output_path, index=False)
            print(f"\n✓ Predictions saved to {output_path}")
        else:
            print("✗ No predictions generated")
    except Exception as e:
        print(f"✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Weekly pipeline complete!")
    print("=" * 60)


def track_predictions(season, week):
    """
    Track prediction performance over time.
    
    Args:
        season: Season year
        week: Week number
    """
    print(f"\nTracking predictions for {season} Week {week}...")
    
    # Try multiple file patterns
    prediction_files = list(DATA_DIR.glob(f"predictions_{season}_week{week}_*.csv"))
    prediction_files += list(DATA_DIR.glob(f"predictions_improved.csv"))
    prediction_files = sorted(set(prediction_files))  # Remove duplicates
    
    if len(prediction_files) == 0:
        print(f"\nNo prediction files found for {season} Week {week}")
        print("\nTo generate predictions, run:")
        print(f"  python src/predict_upcoming_improved.py --season {season} --week {week}")
        print("\nOr use the weekly pipeline:")
        print(f"  python src/weekly_pipeline.py --week {week}")
        return
    
    # Load most recent predictions
    latest_predictions = pd.read_csv(prediction_files[-1])
    print(f"Loaded predictions from: {prediction_files[-1].name}")
    print(f"Total predictions: {len(latest_predictions)}")
    
    # Load actual results
    from data_collection import load_game_data
    pbp, games = load_game_data()
    
    if games is None:
        print("Could not load game results. Run data_collection.py first.")
        return
    
    week_games = games[(games['season'] == season) & (games['week'] == week)]
    
    if len(week_games) == 0:
        print(f"No games found for {season} Week {week} in game data.")
        print("Games may not have been played yet, or data needs to be collected.")
        return
    
    # Match predictions to results
    results = []
    matched_games = 0
    
    for idx, pred in latest_predictions.iterrows():
        home_team = pred.get('home_team')
        away_team = pred.get('away_team')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        game = week_games[
            (week_games['home_team'] == home_team) &
            (week_games['away_team'] == away_team)
        ]
        
        if len(game) == 0:
            # Try reverse order
            game = week_games[
                (week_games['home_team'] == away_team) &
                (week_games['away_team'] == home_team)
            ]
        
        if len(game) > 0:
            matched_games += 1
            game_row = game.iloc[0]
            
            if 'result' in game_row and not pd.isna(game_row['result']):
                actual_winner = home_team if game_row['result'] > 0 else away_team
                predicted_winner = pred.get('predicted_winner')
                correct = 1 if actual_winner == predicted_winner else 0
                
                home_score = game_row.get('home_score', 0) if 'home_score' in game_row else 0
                away_score = game_row.get('away_score', 0) if 'away_score' in game_row else 0
                
                results.append({
                    'game': f"{away_team} @ {home_team}",
                    'predicted': predicted_winner,
                    'actual': actual_winner,
                    'score': f"{away_score}-{home_score}",
                    'correct': correct,
                    'confidence': pred.get('confidence', 0),
                    'home_prob': pred.get('home_win_probability', 0)
                })
            else:
                # Game found but no result yet
                results.append({
                    'game': f"{away_team} @ {home_team}",
                    'predicted': pred.get('predicted_winner', 'N/A'),
                    'actual': 'Not played yet',
                    'score': 'N/A',
                    'correct': 'N/A',
                    'confidence': pred.get('confidence', 0),
                    'home_prob': pred.get('home_win_probability', 0)
                })
    
    if len(results) == 0:
        print(f"\nNo games matched between predictions and game data.")
        print(f"Predictions had {len(latest_predictions)} games")
        print(f"Game data had {len(week_games)} games")
        if len(week_games) > 0:
            print(f"\nGames in data: {week_games[['home_team', 'away_team']].to_string()}")
        return
    
    results_df = pd.DataFrame(results)
    
    # Filter to completed games
    completed = results_df[results_df['actual'] != 'Not played yet']
    
    if len(completed) > 0:
        accuracy = completed['correct'].mean()
        
        print(f"\n{'='*60}")
        print(f"Week {week} Tracking Results")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"Correct: {completed['correct'].sum()}/{len(completed)}")
        print(f"Games tracked: {len(completed)}/{len(results_df)}")
        print(f"\nGame-by-game results:")
        print(completed[['game', 'predicted', 'actual', 'score', 'correct', 'confidence']].to_string(index=False))
        
        # Save tracking results
        tracking_path = DATA_DIR / f"tracking_{season}_week{week}.csv"
        results_df.to_csv(tracking_path, index=False)
        print(f"\nTracking saved to {tracking_path}")
    else:
        print(f"\nNo completed games found for {season} Week {week}.")
        print("Games may not have been played yet.")
        print(f"\nPredictions made for:")
        print(results_df[['game', 'predicted', 'confidence']].to_string(index=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Weekly NFL prediction pipeline")
    parser.add_argument("--season", type=int, help="Season year (default: current)")
    parser.add_argument("--week", type=int, help="Week number (default: upcoming)")
    parser.add_argument("--snapshot", type=str, default="friday_6pm",
                       help="Snapshot time (default: friday_6pm)")
    parser.add_argument("--track", action="store_true",
                       help="Track previous week's predictions")
    parser.add_argument("--track-rosters", action="store_true",
                       help="Track roster changes only")
    
    args = parser.parse_args()
    
    if args.track and args.season and args.week:
        track_predictions(args.season, args.week)
    elif args.track_rosters:
        changes = track_roster_changes(season=args.season, week=args.week)
        if len(changes) > 0:
            print("\nRoster Changes:")
            print(changes[['type', 'team', 'position', 'player', 'previous_player']].to_string(index=False))
    else:
        weekly_pipeline(season=args.season, week=args.week, snapshot_time=args.snapshot)

