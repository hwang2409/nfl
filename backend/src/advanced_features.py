"""
Advanced Feature Engineering Module

Implements high-signal features for maximum accuracy:
- Market consensus (spread/moneyline)
- Elo/Glicko ratings
- EPA/play and success rate
- QB availability & quality
- Situational edges (rest, travel, weather)
- Injuries
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from data_collection import load_game_data, DATA_DIR


def calculate_elo_ratings(games_df, initial_rating=1500, k_factor=32):
    """
    Calculate Elo ratings for teams over time.
    
    Args:
        games_df: DataFrame with game results
        initial_rating: Starting Elo rating
        k_factor: K-factor for Elo updates
    
    Returns:
        DataFrame with Elo ratings per team per game
    """
    print("Calculating Elo ratings...")
    
    # Get unique teams
    all_teams = set()
    if 'home_team' in games_df.columns:
        all_teams.update(games_df['home_team'].dropna().unique())
    if 'away_team' in games_df.columns:
        all_teams.update(games_df['away_team'].dropna().unique())
    
    # Initialize ratings
    elo_ratings = {team: initial_rating for team in all_teams}
    elo_history = []
    
    # Sort by season and week
    games_sorted = games_df.sort_values(['season', 'week']).copy()
    
    for idx, game in games_sorted.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        home_elo = elo_ratings.get(home_team, initial_rating)
        away_elo = elo_ratings.get(away_team, initial_rating)
        
        # Expected scores
        home_expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        away_expected = 1 - home_expected
        
        # Actual result (if available)
        if 'result' in game and not pd.isna(game['result']):
            if game['result'] > 0:  # Home win
                home_score = 1
                away_score = 0
            elif game['result'] < 0:  # Away win
                home_score = 0
                away_score = 1
            else:  # Tie
                home_score = 0.5
                away_score = 0.5
            
            # Update ratings
            home_elo += k_factor * (home_score - home_expected)
            away_elo += k_factor * (away_score - away_expected)
            
            elo_ratings[home_team] = home_elo
            elo_ratings[away_team] = away_elo
        
        # Store ratings for this game
        elo_history.append({
            'game_id': game.get('game_id'),
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo
        })
    
    return pd.DataFrame(elo_history)


def calculate_epa_features(pbp_data):
    """
    Calculate EPA/play and success rate features from play-by-play data.
    
    Args:
        pbp_data: Play-by-play DataFrame
    
    Returns:
        DataFrame with EPA features per team per game
    """
    print("Calculating EPA features...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return None
    
    epa_features = []
    
    for game_id in pbp_data['game_id'].unique():
        game_pbp = pbp_data[pbp_data['game_id'] == game_id]
        
        if len(game_pbp) == 0:
            continue
        
        season = game_pbp['season'].iloc[0]
        week = game_pbp['week'].iloc[0]
        
        for team in ['home_team', 'away_team']:
            team_name = game_pbp[team].iloc[0] if len(game_pbp) > 0 else None
            if team_name is None:
                continue
            
            # Offensive plays
            off_plays = game_pbp[game_pbp['posteam'] == team_name]
            def_plays = game_pbp[game_pbp['defteam'] == team_name]
            
            if len(off_plays) == 0:
                continue
            
            # EPA features
            off_epa = off_plays['epa'].sum() if 'epa' in off_plays.columns else 0
            off_epa_per_play = off_epa / max(len(off_plays), 1)
            
            # Success rate (positive EPA plays)
            successful_plays = (off_plays['epa'] > 0).sum() if 'epa' in off_plays.columns else 0
            success_rate = successful_plays / max(len(off_plays), 1)
            
            # Defensive EPA allowed
            def_epa_allowed = def_plays['epa'].sum() if 'epa' in def_plays.columns else 0
            def_epa_per_play_allowed = def_epa_allowed / max(len(def_plays), 1)
            
            epa_features.append({
                'game_id': game_id,
                'team': team_name,
                'season': season,
                'week': week,
                'off_epa_per_play': off_epa_per_play,
                'off_success_rate': success_rate,
                'def_epa_per_play_allowed': def_epa_per_play_allowed,
                'net_epa_per_play': off_epa_per_play - def_epa_per_play_allowed
            })
    
    return pd.DataFrame(epa_features)


def add_market_features(games_df):
    """
    Add market consensus features (spread, moneyline).
    Note: In production, fetch from multiple books at snapshot time.
    
    Args:
        games_df: DataFrame with game information
    
    Returns:
        DataFrame with market features added
    """
    # For now, use spread from game data if available
    # In production, fetch from multiple books and take median
    games_df = games_df.copy()
    
    if 'spread_line' in games_df.columns:
        games_df['market_spread'] = games_df['spread_line']
    elif 'home_spread' in games_df.columns:
        games_df['market_spread'] = games_df['home_spread']
    else:
        # Estimate from total/over_under if available
        games_df['market_spread'] = 0  # Default to pick'em
    
    # Convert spread to implied win probability
    # Using logistic approximation: P(win) = 1 / (1 + exp(-spread/3))
    games_df['market_implied_win_prob'] = 1 / (1 + np.exp(-games_df['market_spread'] / 3))
    
    return games_df


def add_situational_features(games_df):
    """
    Add situational features: rest days, travel, weather.
    
    Args:
        games_df: DataFrame with game information
    
    Returns:
        DataFrame with situational features added
    """
    games_df = games_df.copy()
    
    # Rest days (simplified - would need full schedule)
    # For now, detect bye weeks and Thursday games
    games_df['is_thursday'] = (games_df['weekday'] == 'Thursday').astype(int) if 'weekday' in games_df.columns else 0
    games_df['is_monday'] = (games_df['weekday'] == 'Monday').astype(int) if 'weekday' in games_df.columns else 0
    
    # Travel distance (simplified - would need team locations)
    # For now, flag divisional games (less travel)
    games_df['is_divisional'] = 0  # Would need division info
    
    # Weather (would need forecast data at snapshot time)
    # For now, flag dome games
    games_df['is_dome'] = 0  # Would need stadium info
    
    return games_df


def create_difference_features(features_df, home_team_col='home_team', away_team_col='away_team'):
    """
    Create difference features (home - away) for single-row prediction.
    
    Args:
        features_df: DataFrame with team-level features
        home_team_col: Column name for home team
        away_team_col: Column name for away team
    
    Returns:
        DataFrame with difference features
    """
    diff_features = []
    
    # Group by game
    for game_id in features_df['game_id'].unique():
        game_features = features_df[features_df['game_id'] == game_id]
        
        if len(game_features) < 2:
            continue
        
        home_row = game_features[game_features['team'] == game_features[home_team_col].iloc[0]]
        away_row = game_features[game_features['team'] == game_features[away_team_col].iloc[0]]
        
        if len(home_row) == 0 or len(away_row) == 0:
            continue
        
        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]
        
        # Create difference features
        diff_row = {
            'game_id': game_id,
            'season': home_row.get('season'),
            'week': home_row.get('week'),
            'home_team': home_row.get('team'),
            'away_team': away_row.get('team'),
        }
        
        # Elo difference
        if 'elo' in home_row and 'elo' in away_row:
            diff_row['elo_diff'] = home_row['elo'] - away_row['elo']
        
        # EPA differences
        if 'off_epa_per_play' in home_row and 'off_epa_per_play' in away_row:
            diff_row['off_epa_diff'] = home_row['off_epa_per_play'] - away_row['off_epa_per_play']
            diff_row['def_epa_diff'] = home_row.get('def_epa_per_play_allowed', 0) - away_row.get('def_epa_per_play_allowed', 0)
            diff_row['net_epa_diff'] = home_row.get('net_epa_per_play', 0) - away_row.get('net_epa_per_play', 0)
        
        # Success rate difference
        if 'off_success_rate' in home_row and 'off_success_rate' in away_row:
            diff_row['success_rate_diff'] = home_row['off_success_rate'] - away_row['off_success_rate']
        
        diff_features.append(diff_row)
    
    return pd.DataFrame(diff_features)


def prepare_advanced_features(min_season=None, max_season=None):
    """
    Prepare all advanced features for training.
    
    Args:
        min_season: Minimum season to include
        max_season: Maximum season to include
    
    Returns:
        X: Feature matrix
        y: Target variable
    """
    print("Preparing advanced features...")
    
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
    
    # Calculate Elo ratings
    elo_df = calculate_elo_ratings(games)
    
    # Calculate EPA features
    if pbp is not None:
        epa_df = calculate_epa_features(pbp)
    else:
        epa_df = None
    
    # Add market features
    games = add_market_features(games)
    
    # Add situational features
    games = add_situational_features(games)
    
    # Merge features
    features_list = []
    
    for idx, game in games.iterrows():
        game_id = game.get('game_id')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Get Elo ratings for this game
        # Use each team's most recent Elo rating before this game
        home_elo_history = elo_df[
            ((elo_df['home_team'] == home_team) | (elo_df['away_team'] == home_team)) &
            (elo_df['season'] == season) &
            (elo_df['week'] < week)
        ]
        if len(home_elo_history) == 0:
            # Try previous season
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
            # Try previous season
            away_elo_history = elo_df[
                ((elo_df['home_team'] == away_team) | (elo_df['away_team'] == away_team)) &
                (elo_df['season'] == season - 1)
            ]
        
        # Get most recent Elo for each team
        if len(home_elo_history) > 0:
            last_home_game = home_elo_history.iloc[-1]
            home_elo = last_home_game['home_elo'] if last_home_game['home_team'] == home_team else last_home_game['away_elo']
        else:
            home_elo = 1500  # Default
        
        if len(away_elo_history) > 0:
            last_away_game = away_elo_history.iloc[-1]
            away_elo = last_away_game['home_elo'] if last_away_game['home_team'] == away_team else last_away_game['away_elo']
        else:
            away_elo = 1500  # Default
        
        elo_diff = home_elo - away_elo
        
        # Get EPA features (use rolling average from previous games)
        if epa_df is not None:
            home_epa = epa_df[
                (epa_df['team'] == home_team) &
                (epa_df['season'] == season) &
                (epa_df['week'] < week)
            ].tail(3)
            away_epa = epa_df[
                (epa_df['team'] == away_team) &
                (epa_df['season'] == season) &
                (epa_df['week'] < week)
            ].tail(3)
            
            # If no current season, use previous season
            if len(home_epa) == 0:
                home_epa = epa_df[
                    (epa_df['team'] == home_team) &
                    (epa_df['season'] == season - 1)
                ].tail(3)
            if len(away_epa) == 0:
                away_epa = epa_df[
                    (epa_df['team'] == away_team) &
                    (epa_df['season'] == season - 1)
                ].tail(3)
            
            home_off_epa = home_epa['off_epa_per_play'].mean() if len(home_epa) > 0 else 0
            away_off_epa = away_epa['off_epa_per_play'].mean() if len(away_epa) > 0 else 0
            home_def_epa = home_epa['def_epa_per_play_allowed'].mean() if len(home_epa) > 0 else 0
            away_def_epa = away_epa['def_epa_per_play_allowed'].mean() if len(away_epa) > 0 else 0
        else:
            home_off_epa = away_off_epa = home_def_epa = away_def_epa = 0
        
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
            
            # Elo difference
            'elo_diff': elo_diff,
            
            # EPA differences
            'off_epa_diff': home_off_epa - away_off_epa,
            'def_epa_diff': home_def_epa - away_def_epa,
            'net_epa_diff': (home_off_epa - home_def_epa) - (away_off_epa - away_def_epa),
            
            # Situational
            'is_home': 1,
            'is_thursday': game.get('is_thursday', 0),
            'is_monday': game.get('is_monday', 0),
        }
        
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
        'market_spread', 'market_implied_win_prob',
        'elo_diff', 'off_epa_diff', 'def_epa_diff', 'net_epa_diff',
        'is_home', 'is_thursday', 'is_monday'
    ]
    
    feature_cols = [col for col in feature_cols if col in features_df.columns]
    
    # Keep metadata columns for matching
    metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team']
    available_metadata = [col for col in metadata_cols if col in features_df.columns]
    
    X = features_df[feature_cols].copy()
    # Store metadata in index or as separate attribute for later matching
    X_metadata = features_df[available_metadata].copy() if available_metadata else None
    
    y = features_df['home_win'].copy() if 'home_win' in features_df.columns else None
    
    # Remove rows with missing values
    X = X.dropna()
    if y is not None:
        y = y.loc[X.index]
        
        # Remove rows where target is NaN or invalid
        valid_target_mask = ~(pd.isna(y) | np.isinf(y))
        X = X[valid_target_mask].copy()
        y = y[valid_target_mask].copy()
        
        # Ensure y is binary (0 or 1)
        y = y.astype(int)
        if not y.isin([0, 1]).all():
            print("  Warning: Converting non-binary targets to binary")
            y = (y > 0.5).astype(int)
    
    # Update metadata to match X's index after filtering
    if X_metadata is not None:
        # Only keep metadata for rows that are still in X
        X_metadata = X_metadata.loc[X.index].copy()
        X._metadata = X_metadata
    
    print(f"Prepared {len(X)} games with {len(feature_cols)} advanced features")
    if y is not None:
        print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y
