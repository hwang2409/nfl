"""
Prepare Enhanced Features for Training

Integrates all enhanced features from enhanced_features.py into the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from data_collection import load_game_data, get_current_season
from advanced_features import calculate_elo_ratings, add_market_features
from enhanced_features import (
    calculate_enhanced_epa_features,
    calculate_qb_features,
    add_enhanced_situational_features,
    calculate_home_away_splits,
    calculate_head_to_head_history,
    calculate_strength_of_schedule,
    calculate_recent_form_weighted
)
from comprehensive_features import (
    add_weather_features,
    add_travel_features,
    add_injury_features,
    calculate_style_matchups,
    calculate_defensive_matchups,
    calculate_coaching_matchups,
    calculate_turnover_regression,
    calculate_third_down_stats,
    calculate_time_of_possession,
    calculate_playoff_implications,
    calculate_comprehensive_features
)
from pfr_integration import calculate_pfr_features_for_games
from pfr_player_features import calculate_player_features_for_games


def prepare_enhanced_features(min_season=None, max_season=None, keep_games_without_targets=False):
    """
    Prepare all enhanced features for training.
    
    This integrates:
    - Enhanced EPA metrics (explosive plays, red zone, success rate)
    - QB performance metrics
    - Home/away splits
    - Divisional games
    - Rest days
    - Head-to-head history
    - Recent form (weighted)
    - Strength of schedule
    
    Args:
        min_season: Minimum season to include
        max_season: Maximum season to include
    
    Returns:
        X: Feature matrix
        y: Target variable
    """
    print("=" * 60)
    print("Preparing Enhanced Features")
    print("=" * 60)
    
    # Load data
    pbp, games = load_game_data()
    
    if games is None:
        print("No game data found.")
        return None, None
    
    # Filter to recent seasons
    if min_season is not None:
        games = games[games['season'] >= min_season]
    if max_season is not None:
        games = games[games['season'] <= max_season]
    
    print(f"Processing {len(games)} games from seasons {min_season} to {max_season}")
    
    # Calculate Elo ratings
    elo_df = calculate_elo_ratings(games)
    
    # Calculate enhanced EPA features
    if pbp is not None:
        epa_df = calculate_enhanced_epa_features(pbp)
        qb_df = calculate_qb_features(pbp)
    else:
        epa_df = None
        qb_df = None
    
    # Add market features
    games = add_market_features(games)
    
    # Add enhanced situational features (divisional, rest days, primetime)
    games = add_enhanced_situational_features(games)
    
    # Add comprehensive features (weather, travel, injuries, style matchups, etc.)
    games, third_down_df, top_df, target_share_df = calculate_comprehensive_features(games, pbp, epa_df)
    
    # Add Pro Football Reference features
    games = calculate_pfr_features_for_games(games, min_season=min_season, max_season=max_season)
    
    # Add Pro Football Reference player-level features
    games = calculate_player_features_for_games(games, pbp_data=pbp, min_season=min_season, max_season=max_season)
    
    # Calculate home/away splits
    home_away_splits = calculate_home_away_splits(games, epa_df)
    
    # Build features
    features_list = []
    
    print("\nBuilding features for each game...")
    for idx, game in games.iterrows():
        game_id = game.get('game_id')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Get Elo ratings
        home_elo_history = elo_df[
            ((elo_df['home_team'] == home_team) | (elo_df['away_team'] == home_team)) &
            (elo_df['season'] == season) &
            (elo_df['week'] < week)
        ]
        if len(home_elo_history) == 0:
            home_elo_history = elo_df[
                ((elo_df['home_team'] == home_team) | (elo_df['away_team'] == home_team)) &
                (elo_df['season'] == season - 1)
            ]
        
        away_elo_history = elo_df[
            ((elo_df['home_team'] == away_team) | (elo_df['away_team'] == away_team)) &
            (elo_df['season'] == season) &
            (elo_df['week'] < week)
        ]
        if len(away_elo_history) == 0:
            away_elo_history = elo_df[
                ((elo_df['home_team'] == away_team) | (elo_df['away_team'] == away_team)) &
                (elo_df['season'] == season - 1)
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
        
        # Get enhanced EPA features (weighted recent form)
        if epa_df is not None:
            home_form = calculate_recent_form_weighted(epa_df, home_team, season, week)
            away_form = calculate_recent_form_weighted(epa_df, away_team, season, week)
            
            home_off_epa = home_form['weighted_off_epa']
            home_def_epa = home_form['weighted_def_epa']
            home_net_epa = home_form['weighted_net_epa']
            home_success_rate = home_form['weighted_success_rate']
            
            away_off_epa = away_form['weighted_off_epa']
            away_def_epa = away_form['weighted_def_epa']
            away_net_epa = away_form['weighted_net_epa']
            away_success_rate = away_form['weighted_success_rate']
        else:
            home_off_epa = home_def_epa = home_net_epa = home_success_rate = 0
            away_off_epa = away_def_epa = away_net_epa = away_success_rate = 0
        
        # Get QB features
        if qb_df is not None:
            home_qb = qb_df[
                (qb_df['team'] == home_team) &
                (qb_df['season'] == season) &
                (qb_df['week'] < week)
            ].tail(3)
            if len(home_qb) == 0:
                home_qb = qb_df[
                    (qb_df['team'] == home_team) &
                    (qb_df['season'] == season - 1)
                ].tail(3)
            
            away_qb = qb_df[
                (qb_df['team'] == away_team) &
                (qb_df['season'] == season) &
                (qb_df['week'] < week)
            ].tail(3)
            if len(away_qb) == 0:
                away_qb = qb_df[
                    (qb_df['team'] == away_team) &
                    (qb_df['season'] == season - 1)
                ].tail(3)
            
            home_qb_epa = home_qb['qb_epa_per_play'].mean() if len(home_qb) > 0 else 0
            home_qb_comp_pct = home_qb['qb_completion_pct'].mean() if len(home_qb) > 0 else 0
            
            away_qb_epa = away_qb['qb_epa_per_play'].mean() if len(away_qb) > 0 else 0
            away_qb_comp_pct = away_qb['qb_completion_pct'].mean() if len(away_qb) > 0 else 0
        else:
            home_qb_epa = home_qb_comp_pct = away_qb_epa = away_qb_comp_pct = 0
        
        # Head-to-head history
        h2h = calculate_head_to_head_history(games, home_team, away_team, season, week)
        
        # Strength of schedule
        home_sos = calculate_strength_of_schedule(games, elo_df, home_team, season, week)
        away_sos = calculate_strength_of_schedule(games, elo_df, away_team, season, week)
        
        # Home/away splits
        home_split = home_away_splits.get(home_team, {'home_advantage': 0})
        away_split = home_away_splits.get(away_team, {'home_advantage': 0})
        
        # Get game row for comprehensive features
        game_row = games[
            (games['home_team'] == home_team) &
            (games['away_team'] == away_team) &
            (games['season'] == season) &
            (games['week'] == week)
        ]
        
        if len(game_row) > 0:
            game_row = game_row.iloc[0]
            
            # Travel features
            travel_distance = game_row.get('travel_distance', 0)
            timezone_change = game_row.get('timezone_change', 0)
            long_travel = game_row.get('long_travel', 0)
            eastward_travel = game_row.get('eastward_travel', 0)
            
            # Weather features
            is_dome = game_row.get('is_dome', 0)
            high_wind = game_row.get('high_wind', 0)
            cold_weather = game_row.get('cold_weather', 0)
            bad_weather = game_row.get('bad_weather', 0)
            
            # Style matchup
            home_pass_rate = game_row.get('home_pass_rate', 0.5)
            away_pass_rate = game_row.get('away_pass_rate', 0.5)
            style_mismatch = game_row.get('style_mismatch', 0)
            
            # Defensive matchup
            matchup_advantage = game_row.get('matchup_advantage', 0)
            
            # Coaching matchup
            coaching_win_rate = game_row.get('coaching_win_rate', 0.5)
            
            # Turnover regression
            home_turnover_reg = game_row.get('home_turnover_regression', 0)
            away_turnover_reg = game_row.get('away_turnover_regression', 0)
            
            # Playoff implications
            playoff_importance = game_row.get('playoff_importance', 0)
            late_season = game_row.get('late_season', 0)
            
            # Injury features
            qb_injury_impact = game_row.get('qb_injury_impact', 0)
            key_player_out = game_row.get('key_player_out', 0)
            
            # Third down stats
            if third_down_df is not None:
                home_third_down = third_down_df[
                    (third_down_df['team'] == home_team) &
                    (third_down_df['season'] == season) &
                    (third_down_df['week'] < week)
                ].tail(3)['third_down_pct'].mean() if len(third_down_df[
                    (third_down_df['team'] == home_team) &
                    (third_down_df['season'] == season) &
                    (third_down_df['week'] < week)
                ]) > 0 else 0
                
                away_third_down = third_down_df[
                    (third_down_df['team'] == away_team) &
                    (third_down_df['season'] == season) &
                    (third_down_df['week'] < week)
                ].tail(3)['third_down_pct'].mean() if len(third_down_df[
                    (third_down_df['team'] == away_team) &
                    (third_down_df['season'] == season) &
                    (third_down_df['week'] < week)
                ]) > 0 else 0
            else:
                home_third_down = away_third_down = 0
            
            # Target share features
            if target_share_df is not None:
                home_target_share = target_share_df[
                    (target_share_df['team'] == home_team) &
                    (target_share_df['season'] == season) &
                    (target_share_df['week'] < week)
                ].tail(3)
                if len(home_target_share) == 0:
                    home_target_share = target_share_df[
                        (target_share_df['team'] == home_team) &
                        (target_share_df['season'] == season - 1)
                    ].tail(3)
                
                away_target_share = target_share_df[
                    (target_share_df['team'] == away_team) &
                    (target_share_df['season'] == season) &
                    (target_share_df['week'] < week)
                ].tail(3)
                if len(away_target_share) == 0:
                    away_target_share = target_share_df[
                        (target_share_df['team'] == away_team) &
                        (target_share_df['season'] == season - 1)
                    ].tail(3)
                
                home_top_wr_share = home_target_share['top_receiver_target_share'].mean() if len(home_target_share) > 0 else 0
                home_target_concentration = home_target_share['target_concentration'].mean() if len(home_target_share) > 0 else 0
                
                away_top_wr_share = away_target_share['top_receiver_target_share'].mean() if len(away_target_share) > 0 else 0
                away_target_concentration = away_target_share['target_concentration'].mean() if len(away_target_share) > 0 else 0
            else:
                home_top_wr_share = home_target_concentration = 0
                away_top_wr_share = away_target_concentration = 0
            
            # CB vs WR matchup features (from comprehensive features)
            home_wr_vs_away_cb = game_row.get('home_wr_vs_away_cb_advantage', 0)
            away_wr_vs_home_cb = game_row.get('away_wr_vs_home_cb_advantage', 0)
            cb_wr_matchup_advantage = game_row.get('cb_wr_matchup_advantage', 0)
            
            # Time of possession
            if top_df is not None:
                home_top = top_df[
                    (top_df['team'] == home_team) &
                    (top_df['season'] == season) &
                    (top_df['week'] < week)
                ].tail(3)['time_of_possession_min'].mean() if len(top_df[
                    (top_df['team'] == home_team) &
                    (top_df['season'] == season) &
                    (top_df['week'] < week)
                ]) > 0 else 30
                
                away_top = top_df[
                    (top_df['team'] == away_team) &
                    (top_df['season'] == season) &
                    (top_df['week'] < week)
                ].tail(3)['time_of_possession_min'].mean() if len(top_df[
                    (top_df['team'] == away_team) &
                    (top_df['season'] == season) &
                    (top_df['week'] < week)
                ]) > 0 else 30
            else:
                home_top = away_top = 30
        else:
            # Defaults if game row not found
            travel_distance = timezone_change = long_travel = eastward_travel = 0
            is_dome = high_wind = cold_weather = bad_weather = 0
            home_pass_rate = away_pass_rate = 0.5
            style_mismatch = matchup_advantage = coaching_win_rate = 0
            home_turnover_reg = away_turnover_reg = 0
            playoff_importance = late_season = 0
            qb_injury_impact = key_player_out = 0
            home_third_down = away_third_down = 0
            home_top = away_top = 30
            
            # Target share (defaults)
            home_top_wr_share = home_target_concentration = 0
            away_top_wr_share = away_target_concentration = 0
            
            # CB vs WR matchups (defaults)
            home_wr_vs_away_cb = away_wr_vs_home_cb = cb_wr_matchup_advantage = 0
        
        # PFR features (from game_row if available)
        pfr_features = {}
        # game_row is already a Series if len > 0, or empty DataFrame if len == 0
        if len(game_row) > 0:
            # Extract all PFR features from the game row (team and player level)
            for col in game_row.index:
                if '_pfr_' in str(col) or '_qb_pfr_' in str(col) or '_rb_pfr_' in str(col) or '_wr_pfr_' in str(col) or '_te_pfr_' in str(col):
                    pfr_features[col] = game_row.get(col, 0)
        
        # Create feature row
        features = {
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            
            # Market features
            'market_spread': game.get('market_spread', 0),
            'market_implied_win_prob': game.get('market_implied_win_prob', 0.5),
            
            # Elo
            'elo_diff': elo_diff,
            
            # EPA differences (weighted recent form)
            'off_epa_diff': home_off_epa - away_off_epa,
            'def_epa_diff': home_def_epa - away_def_epa,
            'net_epa_diff': home_net_epa - away_net_epa,
            'success_rate_diff': home_success_rate - away_success_rate,
            
            # QB differences
            'qb_epa_diff': home_qb_epa - away_qb_epa,
            'qb_comp_pct_diff': home_qb_comp_pct - away_qb_comp_pct,
            
            # Head-to-head
            'h2h_games': h2h['h2h_games'],
            'h2h_win_rate': h2h['h2h_win_rate'],
            'h2h_avg_margin': h2h['avg_margin'],
            
            # Strength of schedule
            'sos_diff': home_sos - away_sos,
            
            # Home/away splits
            'home_advantage': home_split['home_advantage'],
            'away_advantage': -away_split['home_advantage'],  # Negative because they're away
            
            # Situational
            'is_home': 1,
            'is_divisional': game.get('is_divisional', 0),
            'is_primetime': game.get('is_primetime', 0),
            'is_thursday': game.get('is_thursday', 0),
            'is_monday': game.get('is_monday', 0),
            'rest_advantage': game.get('rest_advantage', 0),
            'is_short_week': game.get('is_short_week', 0),
            
            # Travel features
            'travel_distance': travel_distance,
            'timezone_change': timezone_change,
            'long_travel': long_travel,
            'eastward_travel': eastward_travel,
            
            # Weather features
            'is_dome': is_dome,
            'high_wind': high_wind,
            'cold_weather': cold_weather,
            'bad_weather': bad_weather,
            
            # Style matchup
            'home_pass_rate': home_pass_rate,
            'away_pass_rate': away_pass_rate,
            'style_mismatch': style_mismatch,
            
            # Defensive matchup
            'matchup_advantage': matchup_advantage,
            
            # Coaching matchup
            'coaching_win_rate': coaching_win_rate,
            
            # Turnover regression
            'home_turnover_regression': home_turnover_reg,
            'away_turnover_regression': away_turnover_reg,
            
            # Playoff implications
            'playoff_importance': playoff_importance,
            'late_season': late_season,
            
            # Injury features
            'qb_injury_impact': qb_injury_impact,
            'key_player_out': key_player_out,
            
            # Third down
            'third_down_pct_diff': home_third_down - away_third_down,
            
            # Time of possession
            'top_diff': home_top - away_top,
            
            # Target share
            'top_wr_target_share_diff': home_top_wr_share - away_top_wr_share,
            'target_concentration_diff': home_target_concentration - away_target_concentration,
            
            # CB vs WR matchups
            'home_wr_vs_away_cb_advantage': home_wr_vs_away_cb,
            'away_wr_vs_home_cb_advantage': away_wr_vs_home_cb,
            'cb_wr_matchup_advantage': cb_wr_matchup_advantage,
        }
        
        # Add PFR features
        features.update(pfr_features)
        
        # Add target if available
        if 'result' in game and not pd.isna(game['result']):
            features['home_win'] = 1 if game['result'] > 0 else 0
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    if len(features_df) == 0:
        print("No features created!")
        return None, None
    
    # Prepare X and y
    feature_cols = [
        # Market features
        'market_spread', 'market_implied_win_prob',
        
        # Elo
        'elo_diff',
        
        # EPA differences (weighted recent form)
        'off_epa_diff', 'def_epa_diff', 'net_epa_diff', 'success_rate_diff',
        
        # QB differences
        'qb_epa_diff', 'qb_comp_pct_diff',
        
        # Head-to-head
        'h2h_games', 'h2h_win_rate', 'h2h_avg_margin',
        
        # Strength of schedule
        'sos_diff',
        
        # Home/away splits
        'home_advantage', 'away_advantage',
        
        # Situational
        'is_home', 'is_divisional', 'is_primetime', 'is_thursday', 'is_monday',
        'rest_advantage', 'is_short_week',
        
        # Travel features
        'travel_distance', 'timezone_change', 'long_travel', 'eastward_travel',
        
        # Weather features
        'is_dome', 'high_wind', 'cold_weather', 'bad_weather',
        
        # Style matchup
        'home_pass_rate', 'away_pass_rate', 'style_mismatch',
        
        # Defensive matchup
        'matchup_advantage',
        
        # Coaching matchup
        'coaching_win_rate',
        
        # Turnover regression
        'home_turnover_regression', 'away_turnover_regression',
        
        # Playoff implications
        'playoff_importance', 'late_season',
        
        # Injury features
        'qb_injury_impact', 'key_player_out',
        
        # Third down
        'third_down_pct_diff',
        
        # Time of possession
        'top_diff',
        
        # Target share
        'top_wr_target_share_diff',
        'target_concentration_diff',
        
        # CB vs WR matchups
        'home_wr_vs_away_cb_advantage',
        'away_wr_vs_home_cb_advantage',
        'cb_wr_matchup_advantage',
        
        # PFR features (will be added dynamically if available)
    ]
    
    # Add PFR features if they exist in the DataFrame (team and player level)
    pfr_cols = [col for col in features_df.columns if '_pfr_' in col or '_qb_pfr_' in col or '_rb_pfr_' in col or '_wr_pfr_' in col or '_te_pfr_' in col]
    feature_cols.extend(pfr_cols)
    
    # Only use columns that exist
    feature_cols = [col for col in feature_cols if col in features_df.columns]
    
    X = features_df[feature_cols].copy()
    
    # Store metadata
    metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team']
    available_metadata = [col for col in metadata_cols if col in features_df.columns]
    X_metadata = features_df[available_metadata].copy() if available_metadata else None
    
    y = features_df['home_win'].copy() if 'home_win' in features_df.columns else None
    
    # Check if we have any valid targets
    has_valid_targets = False
    if y is not None:
        valid_target_mask = ~(pd.isna(y) | np.isinf(y))
        has_valid_targets = valid_target_mask.any()
    
    # Handle missing values
    if has_valid_targets:
        # Training mode (or mixed): Fill missing feature values, then filter by target
        # Fill missing features with 0 (neutral/default) before filtering by target
        X = X.fillna(0)
        y = y.loc[X.index]
        
        # Remove rows where target is NaN or invalid
        # UNLESS we want to keep games without targets (for prediction)
        valid_target_mask = ~(pd.isna(y) | np.isinf(y))
        
        if keep_games_without_targets:
            # Keep both games with and without targets
            # For games without targets, set y to NaN (will be filtered out later for training)
            print(f"  Keeping {valid_target_mask.sum()} games with targets and {(~valid_target_mask).sum()} games without targets")
            # Don't filter - keep all games
            # y will have NaN for games without targets, which is fine
        else:
            # Training mode: only keep games with valid targets
            X = X[valid_target_mask].copy()
            y = y[valid_target_mask].copy()
        
        # Ensure y is binary (0 or 1) for games with targets
        if keep_games_without_targets:
            # Only convert non-NaN targets to binary
            y_valid = y[valid_target_mask]
            if len(y_valid) > 0:
                y_valid = y_valid.astype(int)
                if not y_valid.isin([0, 1]).all():
                    print("  Warning: Converting non-binary targets to binary")
                    y_valid = (y_valid > 0.5).astype(int)
                y[valid_target_mask] = y_valid
        else:
            y = y.astype(int)
            if not y.isin([0, 1]).all():
                print("  Warning: Converting non-binary targets to binary")
                y = (y > 0.5).astype(int)
    else:
        # Prediction mode: No valid targets (all future games)
        # Fill missing values instead of dropping rows
        print("  Prediction mode: Filling missing feature values (games without results)")
        X = X.fillna(0)  # Fill missing features with 0 (neutral/default values)
        # Still remove rows where ALL features are missing (shouldn't happen, but safety check)
        X = X.dropna(how='all')
        y = None  # No targets for prediction
    
    # Update metadata to match X's index after filtering
    if X_metadata is not None:
        X_metadata = X_metadata.loc[X.index].copy()
        X._metadata = X_metadata
    
    print(f"\nPrepared {len(X)} games with {len(feature_cols)} enhanced features")
    if y is not None:
        print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


if __name__ == "__main__":
    current_season = get_current_season()
    min_season = max(2018, current_season - 6)
    
    X, y = prepare_enhanced_features(min_season=min_season, max_season=current_season)
    
    if X is not None and y is not None:
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

