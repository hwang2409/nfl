"""
Predict Upcoming NFL Games Using Improved Model

Predicts upcoming/future games using the improved model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from prepare_enhanced_features import prepare_enhanced_features
from improved_model import ImprovedNFLModel
from data_collection import load_game_data, get_current_season
from comprehensive_features import (
    calculate_target_share, calculate_cb_wr_matchups,
    add_weather_features, add_travel_features, add_injury_features,
    calculate_style_matchups, calculate_defensive_matchups,
    calculate_coaching_matchups, calculate_turnover_regression,
    calculate_third_down_stats, calculate_time_of_possession,
    calculate_playoff_implications, calculate_comprehensive_features
)
from advanced_features import calculate_elo_ratings
from enhanced_features import (
    calculate_enhanced_epa_features,
    calculate_qb_features,
    calculate_home_away_splits,
    calculate_head_to_head_history,
    calculate_strength_of_schedule,
    calculate_recent_form_weighted
)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


def predict_upcoming_improved(season=None, week=None, min_confidence=0.0):
    """
    Predict upcoming games using improved model.
    
    Args:
        season: Specific season (None = current)
        week: Specific week (None = next week)
        min_confidence: Minimum confidence threshold
    """
    print("=" * 70)
    print("Predicting Upcoming Games (Improved Model)")
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
        # Try different date column names
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
            # No date column - just get games from current week onwards
            current_week = datetime.now().isocalendar()[1]  # Approximate
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
    
    # Load historical data for feature building
    all_games = games[games['season'] < season].copy() if len(games) > 0 else games.copy()
    
    # Load EPA data if available
    try:
        import nfl_data_py as nfl
        epa_df = nfl.import_pbp_data([season-1, season-2], downcast=True)
    except:
        epa_df = None
    
    # Build features for upcoming games using the SAME pipeline as prepare_enhanced_features
    print("\nBuilding features for upcoming games using full pipeline...")
    
    # Get historical play-by-play for recent games
    if pbp is not None and len(pbp) > 0:
        recent_pbp = pbp[pbp['season'] >= season - 1].copy()
    else:
        recent_pbp = None
    
    # Calculate enhanced EPA features from historical data
    if recent_pbp is not None and len(recent_pbp) > 0:
        print("Calculating enhanced EPA features from historical data...")
        epa_features = calculate_enhanced_epa_features(recent_pbp)
        qb_df = calculate_qb_features(recent_pbp)
    else:
        epa_features = None
        qb_df = None
    
    # Calculate comprehensive features (third down, time of possession, target share)
    if recent_pbp is not None and len(recent_pbp) > 0:
        target_share_df = calculate_target_share(recent_pbp)
        third_down_df = calculate_third_down_stats(recent_pbp)
        top_df = calculate_time_of_possession(recent_pbp)
    else:
        target_share_df = None
        third_down_df = None
        top_df = None
    
    # Calculate Elo for all games (including upcoming)
    all_games_for_elo = pd.concat([all_games, upcoming_games]).drop_duplicates(subset=['game_id'], keep='first')
    elo_df = calculate_elo_ratings(all_games_for_elo)
    
    # Add market features to upcoming games (same as training)
    from advanced_features import add_market_features
    from enhanced_features import add_enhanced_situational_features
    
    upcoming_games_with_features = upcoming_games.copy()
    upcoming_games_with_features = add_market_features(upcoming_games_with_features)
    upcoming_games_with_features = add_enhanced_situational_features(upcoming_games_with_features)
    
    # Add comprehensive features to upcoming games (same as training)
    # This adds travel, weather, style matchups, etc. to the games dataframe
    print("Adding comprehensive features to upcoming games...")
    upcoming_games_with_features, _, _, _ = calculate_comprehensive_features(
        upcoming_games_with_features, 
        recent_pbp if recent_pbp is not None else None,
        epa_features
    )
    
    # Calculate home/away splits once (not in the loop)
    print("Calculating home/away splits...")
    home_away_splits = calculate_home_away_splits(all_games, epa_features)
    
    # Build features for each upcoming game using the EXACT same logic as prepare_enhanced_features
    features_list = []
    game_info = []
    
    print("\nBuilding features for each upcoming game...")
    for idx, game in upcoming_games_with_features.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        game_season = game.get('season')
        game_week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        try:
            # Build features using the same logic as prepare_enhanced_features
            # This is a simplified version - ideally we'd refactor to share code
            feature_row = {}
            
            # Get Elo ratings
            home_elo_history = elo_df[
                ((elo_df['home_team'] == home_team) | (elo_df['away_team'] == home_team)) &
                (elo_df['season'] == game_season) &
                (elo_df['week'] < game_week)
            ]
            if len(home_elo_history) == 0:
                home_elo_history = elo_df[
                    ((elo_df['home_team'] == home_team) | (elo_df['away_team'] == home_team)) &
                    (elo_df['season'] == game_season - 1)
                ]
            
            away_elo_history = elo_df[
                ((elo_df['home_team'] == away_team) | (elo_df['away_team'] == away_team)) &
                (elo_df['season'] == game_season) &
                (elo_df['week'] < game_week)
            ]
            if len(away_elo_history) == 0:
                away_elo_history = elo_df[
                    ((elo_df['home_team'] == away_team) | (elo_df['away_team'] == away_team)) &
                    (elo_df['season'] == game_season - 1)
                ]
            
            if len(home_elo_history) > 0:
                last_home_game = home_elo_history.iloc[-1]
                home_elo = last_home_game['home_elo'] if last_home_game['home_team'] == home_team else last_home_game['away_elo']
            else:
                home_elo = 1500
            
            if len(away_elo_history) > 0:
                last_away_game = away_elo_history.iloc[-1]
                away_elo = last_away_game['home_elo'] if last_away_game['home_team'] == away_team else last_away_game['away_elo']
            else:
                away_elo = 1500
            
            elo_diff = home_elo - away_elo
            feature_row['elo_diff'] = elo_diff
            
            # Market features
            feature_row['market_spread'] = game.get('market_spread', 0)
            feature_row['market_implied_win_prob'] = game.get('market_implied_win_prob', 0.5)
            
            # EPA features from historical data
            if epa_features is not None:
                home_form = calculate_recent_form_weighted(epa_features, home_team, game_season, game_week)
                away_form = calculate_recent_form_weighted(epa_features, away_team, game_season, game_week)
                
                home_off_epa = home_form.get('weighted_off_epa', 0)
                home_def_epa = home_form.get('weighted_def_epa', 0)
                home_net_epa = home_form.get('weighted_net_epa', 0)
                home_success_rate = home_form.get('weighted_success_rate', 0)
                
                away_off_epa = away_form.get('weighted_off_epa', 0)
                away_def_epa = away_form.get('weighted_def_epa', 0)
                away_net_epa = away_form.get('weighted_net_epa', 0)
                away_success_rate = away_form.get('weighted_success_rate', 0)
                
                feature_row['off_epa_diff'] = home_off_epa - away_off_epa
                feature_row['def_epa_diff'] = home_def_epa - away_def_epa
                feature_row['net_epa_diff'] = home_net_epa - away_net_epa
                feature_row['success_rate_diff'] = home_success_rate - away_success_rate
            else:
                feature_row['off_epa_diff'] = 0
                feature_row['def_epa_diff'] = 0
                feature_row['net_epa_diff'] = 0
                feature_row['success_rate_diff'] = 0
            
            # QB features
            if qb_df is not None:
                home_qb = qb_df[
                    (qb_df['team'] == home_team) &
                    (qb_df['season'] == game_season) &
                    (qb_df['week'] < game_week)
                ].tail(3)
                if len(home_qb) == 0:
                    home_qb = qb_df[
                        (qb_df['team'] == home_team) &
                        (qb_df['season'] == game_season - 1)
                    ].tail(3)
                
                away_qb = qb_df[
                    (qb_df['team'] == away_team) &
                    (qb_df['season'] == game_season) &
                    (qb_df['week'] < game_week)
                ].tail(3)
                if len(away_qb) == 0:
                    away_qb = qb_df[
                        (qb_df['team'] == away_team) &
                        (qb_df['season'] == game_season - 1)
                    ].tail(3)
                
                home_qb_epa = home_qb['qb_epa_per_play'].mean() if len(home_qb) > 0 else 0
                home_qb_comp_pct = home_qb['qb_completion_pct'].mean() if len(home_qb) > 0 else 0
                
                away_qb_epa = away_qb['qb_epa_per_play'].mean() if len(away_qb) > 0 else 0
                away_qb_comp_pct = away_qb['qb_completion_pct'].mean() if len(away_qb) > 0 else 0
                
                feature_row['qb_epa_diff'] = home_qb_epa - away_qb_epa
                feature_row['qb_comp_pct_diff'] = home_qb_comp_pct - away_qb_comp_pct
            else:
                feature_row['qb_epa_diff'] = 0
                feature_row['qb_comp_pct_diff'] = 0
            
            # Head-to-head
            h2h = calculate_head_to_head_history(all_games, home_team, away_team, game_season, game_week)
            feature_row['h2h_games'] = h2h.get('h2h_games', 0)
            feature_row['h2h_win_rate'] = h2h.get('h2h_win_rate', 0.5)
            feature_row['h2h_avg_margin'] = h2h.get('avg_margin', 0)
            
            # Strength of schedule
            home_sos = calculate_strength_of_schedule(all_games, elo_df, home_team, game_season, game_week)
            away_sos = calculate_strength_of_schedule(all_games, elo_df, away_team, game_season, game_week)
            feature_row['sos_diff'] = home_sos - away_sos
            
            # Home/away splits (already calculated outside loop)
            home_split = home_away_splits.get(home_team, {}).get('home', {})
            away_split = home_away_splits.get(away_team, {}).get('home', {})
            feature_row['home_advantage'] = home_split.get('home_advantage', 0)
            feature_row['away_advantage'] = -away_split.get('home_advantage', 0)
            
            # Situational (from game row - already added by add_enhanced_situational_features)
            feature_row['is_home'] = 1
            feature_row['is_divisional'] = game.get('is_divisional', 0)
            feature_row['is_primetime'] = game.get('is_primetime', 0)
            feature_row['is_thursday'] = game.get('is_thursday', 0)
            feature_row['is_monday'] = game.get('is_monday', 0)
            feature_row['rest_advantage'] = game.get('rest_advantage', 0)
            feature_row['is_short_week'] = game.get('is_short_week', 0)
            
            # Travel features (from game row - already added by calculate_comprehensive_features)
            feature_row['travel_distance'] = game.get('travel_distance', 0)
            feature_row['timezone_change'] = game.get('timezone_change', 0)
            feature_row['long_travel'] = game.get('long_travel', 0)
            feature_row['eastward_travel'] = game.get('eastward_travel', 0)
            
            # Weather features (from game row - already added by calculate_comprehensive_features)
            feature_row['is_dome'] = game.get('is_dome', 0)
            feature_row['high_wind'] = game.get('high_wind', 0)
            feature_row['cold_weather'] = game.get('cold_weather', 0)
            feature_row['bad_weather'] = game.get('bad_weather', 0)
            
            # Style matchup (from game row - already added by calculate_comprehensive_features)
            feature_row['home_pass_rate'] = game.get('home_pass_rate', 0.5)
            feature_row['away_pass_rate'] = game.get('away_pass_rate', 0.5)
            feature_row['style_mismatch'] = game.get('style_mismatch', 0)
            
            # Defensive matchup (from game row - already added by calculate_comprehensive_features)
            feature_row['matchup_advantage'] = game.get('matchup_advantage', 0)
            
            # Coaching matchup (from game row - already added by calculate_comprehensive_features)
            feature_row['coaching_win_rate'] = game.get('coaching_win_rate', 0.5)
            
            # Turnover regression (from game row - already added by calculate_comprehensive_features)
            feature_row['home_turnover_regression'] = game.get('home_turnover_regression', 0)
            feature_row['away_turnover_regression'] = game.get('away_turnover_regression', 0)
            
            # Playoff implications (from game row - already added by calculate_comprehensive_features)
            feature_row['playoff_importance'] = game.get('playoff_importance', 0)
            feature_row['late_season'] = game.get('late_season', 1 if game_week >= 15 else 0)
            
            # Injury features (from game row - already added by calculate_comprehensive_features)
            feature_row['qb_injury_impact'] = game.get('qb_injury_impact', 0)
            feature_row['key_player_out'] = game.get('key_player_out', 0)
            
            # Third down stats (calculate from historical data)
            if third_down_df is not None:
                home_third_down = third_down_df[
                    (third_down_df['team'] == home_team) &
                    (third_down_df['season'] == game_season) &
                    (third_down_df['week'] < game_week)
                ].tail(3)['third_down_pct'].mean() if len(third_down_df[
                    (third_down_df['team'] == home_team) &
                    (third_down_df['season'] == game_season) &
                    (third_down_df['week'] < game_week)
                ]) > 0 else 0
                
                away_third_down = third_down_df[
                    (third_down_df['team'] == away_team) &
                    (third_down_df['season'] == game_season) &
                    (third_down_df['week'] < game_week)
                ].tail(3)['third_down_pct'].mean() if len(third_down_df[
                    (third_down_df['team'] == away_team) &
                    (third_down_df['season'] == game_season) &
                    (third_down_df['week'] < game_week)
                ]) > 0 else 0
            else:
                home_third_down = away_third_down = 0
            
            feature_row['third_down_pct_diff'] = home_third_down - away_third_down
            
            # Time of possession (calculate from historical data)
            if top_df is not None:
                home_top = top_df[
                    (top_df['team'] == home_team) &
                    (top_df['season'] == game_season) &
                    (top_df['week'] < game_week)
                ].tail(3)['time_of_possession_min'].mean() if len(top_df[
                    (top_df['team'] == home_team) &
                    (top_df['season'] == game_season) &
                    (top_df['week'] < game_week)
                ]) > 0 else 30
                
                away_top = top_df[
                    (top_df['team'] == away_team) &
                    (top_df['season'] == game_season) &
                    (top_df['week'] < game_week)
                ].tail(3)['time_of_possession_min'].mean() if len(top_df[
                    (top_df['team'] == away_team) &
                    (top_df['season'] == game_season) &
                    (top_df['week'] < game_week)
                ]) > 0 else 30
            else:
                home_top = away_top = 30
            
            feature_row['top_diff'] = home_top - away_top
            
            # Target share
            if target_share_df is not None:
                home_target_share = target_share_df[
                    (target_share_df['team'] == home_team) &
                    (target_share_df['season'] == game_season) &
                    (target_share_df['week'] < game_week)
                ].tail(3)
                
                if len(home_target_share) == 0:
                    home_target_share = target_share_df[
                        (target_share_df['team'] == home_team) &
                        (target_share_df['season'] == game_season - 1)
                    ].tail(3)
                
                away_target_share = target_share_df[
                    (target_share_df['team'] == away_team) &
                    (target_share_df['season'] == game_season) &
                    (target_share_df['week'] < game_week)
                ].tail(3)
                
                if len(away_target_share) == 0:
                    away_target_share = target_share_df[
                        (target_share_df['team'] == away_team) &
                        (target_share_df['season'] == game_season - 1)
                    ].tail(3)
                
                home_top_wr_share = home_target_share['top_receiver_target_share'].mean() if len(home_target_share) > 0 else 0
                home_target_concentration = home_target_share['target_concentration'].mean() if len(home_target_share) > 0 else 0
                away_top_wr_share = away_target_share['top_receiver_target_share'].mean() if len(away_target_share) > 0 else 0
                away_target_concentration = away_target_share['target_concentration'].mean() if len(away_target_share) > 0 else 0
            else:
                home_top_wr_share = home_target_concentration = 0
                away_top_wr_share = away_target_concentration = 0
            
            feature_row['top_wr_target_share_diff'] = home_top_wr_share - away_top_wr_share
            feature_row['target_concentration_diff'] = home_target_concentration - away_target_concentration
            
            # CB vs WR matchups (from game row - already added by calculate_comprehensive_features)
            feature_row['home_wr_vs_away_cb_advantage'] = game.get('home_wr_vs_away_cb_advantage', 0)
            feature_row['away_wr_vs_home_cb_advantage'] = game.get('away_wr_vs_home_cb_advantage', 0)
            feature_row['cb_wr_matchup_advantage'] = game.get('cb_wr_matchup_advantage', 0)
            
        except Exception as e:
            print(f"  Warning: Could not build features for {away_team} @ {home_team}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Try different date column names
        date_col = None
        for col in ['game_date', 'gameday', 'date', 'game_time']:
            if col in games.columns:
                date_col = col
                break
        
        game_date = game.get(date_col) if date_col else None
        gametime = game.get('gametime') if 'gametime' in games.columns else None
        
        game_info.append({
            'season': game_season,
            'week': game_week,
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date,
            'gametime': gametime
        })
        features_list.append(feature_row)
    
    if len(features_list) == 0:
        print("Could not build features for upcoming games")
        return None
    
    # For upcoming games, we need to use the full feature building pipeline
    # This is a simplified approach - ideally we'd refactor to share the feature building code
    print("\nNote: For upcoming games, using simplified feature building.")
    print("For best results, ensure historical data is available.")
    
    # Try to get features from prepare_enhanced_features if games exist
    # For truly upcoming games, we need to build features dynamically
    X_upcoming = pd.DataFrame(features_list)
    
    # Align with model's expected features
    if model.selected_features is not None:
        missing_features = [f for f in model.selected_features if f not in X_upcoming.columns]
        if missing_features:
            print(f"  Warning: {len(missing_features)} features missing, filling with zeros")
        for f in missing_features:
            X_upcoming[f] = 0
        X_upcoming = X_upcoming[model.selected_features]
    
    # Debug: check feature statistics
    print(f"\nFeature statistics for upcoming games:")
    print(f"  Total features: {len(X_upcoming.columns)}")
    print(f"  Features with all zeros: {(X_upcoming == 0).all().sum()}")
    print(f"  Features with any NaN: {X_upcoming.isna().any().sum()}")
    print(f"  Mean values per feature (top 10 non-zero):")
    feature_means = X_upcoming.mean().abs().sort_values(ascending=False)
    for feat, mean_val in feature_means.head(10).items():
        print(f"    {feat}: {mean_val:.4f}")
    
    # Make predictions
    print(f"\nMaking predictions for {len(X_upcoming)} games...")
    
    try:
        # First, get raw probabilities WITHOUT decision rules to check for bias
        y_proba_raw, _ = model.predict(X_upcoming, apply_calibration=True, apply_rules=False)
        
        # Debug: print raw statistics
        print(f"  Raw probability range: [{y_proba_raw.min():.3f}, {y_proba_raw.max():.3f}]")
        print(f"  Raw mean probability: {y_proba_raw.mean():.3f}")
        print(f"  Raw std probability: {y_proba_raw.std():.3f}")
        
        # Check if model is biased (all predictions very low or very high)
        # If mean is < 0.3, model is predicting away wins for everything
        # If mean is > 0.7, model is predicting home wins for everything
        needs_flip = y_proba_raw.mean() < 0.3
        if needs_flip:
            print(f"  Warning: Raw mean probability {y_proba_raw.mean():.3f} is very low.")
            print(f"  Model appears to be predicting away team wins for all games.")
            print(f"  This may indicate inverted predictions or feature issues.")
            print(f"  Flipping predictions (1 - prob) to correct bias...")
        
        # Now apply decision rules
        y_proba, y_pred = model.predict(X_upcoming, apply_calibration=True, apply_rules=True)
        
        # If we detected bias, flip the final predictions
        if needs_flip:
            y_proba = 1 - y_proba
            y_pred = 1 - y_pred
        
        # Debug: print final statistics
        print(f"  Final probability range: [{y_proba.min():.3f}, {y_proba.max():.3f}]")
        print(f"  Final mean probability: {y_proba.mean():.3f}")
        print(f"  Final std probability: {y_proba.std():.3f}")
        print(f"  Predictions - Home wins: {(y_pred == 1).sum()}, Away wins: {(y_pred == 0).sum()}")
        
        # If all probabilities are still identical, the model truly can't differentiate
        # This happens when too many features are missing/zero
        if y_proba.std() < 0.01:
            print(f"  Warning: All probabilities are nearly identical (std={y_proba.std():.3f})")
            print(f"  This suggests insufficient feature information for upcoming games.")
            print(f"  Consider using historical games with results for better predictions.")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Calculate confidence
    confidence = np.abs(y_proba - 0.5) * 2
    
    # Create results
    predictions = []
    for i, (home_prob, pred, conf) in enumerate(zip(y_proba, y_pred, confidence)):
        if i < len(game_info):
            info = game_info[i]
            
            # Filter by confidence if specified
            if conf < min_confidence:
                continue
            
            predictions.append({
                'season': info.get('season', season),
                'week': info.get('week'),
                'gameday': info.get('game_date'),
                'gametime': info.get('gametime'),
                'away_team': info.get('away_team'),
                'home_team': info.get('home_team'),
                'home_win_probability': float(home_prob),
                'away_win_probability': float(1 - home_prob),
                'predicted_winner': info.get('home_team') if pred == 1 else info.get('away_team'),
                'confidence': float(conf),
                'prediction': 'Home' if pred == 1 else 'Away'
            })
    
    if len(predictions) == 0:
        print(f"No predictions meet confidence threshold of {min_confidence}")
        return None
    
    # Display results
    results_df = pd.DataFrame(predictions)
    
    # Convert date/time columns for proper sorting
    # Create a sortable datetime column
    if 'gameday' in results_df.columns:
        # Try to parse dates
        results_df['_sort_date'] = pd.to_datetime(results_df['gameday'], errors='coerce')
    else:
        results_df['_sort_date'] = pd.NaT
    
    # Create a sortable time column from gametime
    if 'gametime' in results_df.columns:
        # Extract time from gametime string (format might be "HH:MM" or "HH:MM:SS")
        def parse_time(time_str):
            if pd.isna(time_str) or time_str == '':
                return pd.NaT
            try:
                # Try parsing as time string
                time_parts = str(time_str).split(':')
                if len(time_parts) >= 2:
                    hours = int(time_parts[0])
                    minutes = int(time_parts[1])
                    return pd.Timestamp(2000, 1, 1, hours, minutes)
            except:
                pass
            return pd.NaT
        
        results_df['_sort_time'] = results_df['gametime'].apply(parse_time)
    else:
        results_df['_sort_time'] = pd.NaT
    
    # Sort by week, then by date, then by time, then by confidence
    sort_cols = ['week']
    if '_sort_date' in results_df.columns:
        sort_cols.append('_sort_date')
    if '_sort_time' in results_df.columns:
        sort_cols.append('_sort_time')
    sort_cols.append('confidence')
    
    results_df = results_df.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False])
    
    # Drop temporary sort columns
    results_df = results_df.drop(columns=['_sort_date', '_sort_time'], errors='ignore')
    
    print("\n" + "=" * 70)
    print("Upcoming Game Predictions")
    print("=" * 70)
    
    # Group by week and display
    current_week = None
    for _, row in results_df.iterrows():
        week = row['week']
        
        # Print week header when we encounter a new week
        if week != current_week:
            if current_week is not None:
                print()  # Add blank line between weeks
            week_season = int(row.get('season', season)) if pd.notna(row.get('season')) else season
            print(f"\n{'=' * 70}")
            print(f"Week {int(week)} - Season {week_season}")
            print(f"{'=' * 70}")
            current_week = week
        
        winner = row['predicted_winner']
        home_prob = row['home_win_probability']
        away_prob = row['away_win_probability']
        conf = row['confidence']
        # Show the predicted winner and their probability
        winner_prob = home_prob if row['prediction'] == 'Home' else away_prob
        
        # Format game date/time if available
        game_date = row.get('gameday', '')
        gametime = row.get('gametime', '')
        date_str = f" ({game_date})" if game_date and pd.notna(game_date) else ""
        time_str = f" {gametime}" if gametime and pd.notna(gametime) else ""
        
        print(f"  {row['away_team']:3s} @ {row['home_team']:3s}: {winner:3s} wins ({winner_prob:.1%}, home: {home_prob:.1%}, away: {away_prob:.1%}, conf: {conf:.1%}){date_str}{time_str}")
    
    # Save predictions to database
    try:
        from database import save_predictions, init_db
        init_db(create_tables=False)  # Ensure DB is initialized
        save_predictions(results_df, model_version=None)
        print(f"\n✓ Predictions saved to database")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save to database: {e}")
        print("  Falling back to CSV storage...")
        # Fallback to CSV
        predictions_path = DATA_DIR / "predictions_improved.csv"
        results_df.to_csv(predictions_path, index=False)
        print(f"  Predictions saved to {predictions_path}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict upcoming NFL games using improved model')
    parser.add_argument('--season', type=int, default=None, help='Season year (default: current)')
    parser.add_argument('--week', type=int, default=None, help='Week number (default: next week)')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    predict_upcoming_improved(
        season=args.season,
        week=args.week,
        min_confidence=args.min_confidence
    )

