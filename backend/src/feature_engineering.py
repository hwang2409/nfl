"""
Feature Engineering Module

Creates predictive features from raw NFL data for game prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).parent))
from data_collection import load_game_data, DATA_DIR, get_current_season

def calculate_rolling_stats(df, team_col='team', window=5):
    """
    Calculate rolling statistics for each team.
    
    Args:
        df: DataFrame with team stats
        team_col: Column name for team
        window: Number of games to look back
    
    Returns:
        DataFrame with rolling features
    """
    df = df.sort_values(['team', 'season', 'week'])
    
    # Calculate rolling averages for key metrics
    rolling_cols = [
        'off_points', 'off_total_yards', 'off_yards_per_play',
        'off_turnovers', 'off_third_down_pct',
        'def_points_allowed', 'def_total_yards_allowed', 'def_takeaways'
    ]
    
    for col in rolling_cols:
        if col in df.columns:
            df[f'{col}_rolling_{window}'] = (
                df.groupby(team_col)[col]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
    
    return df


def create_matchup_features(games_df, team_stats):
    """
    Create features for specific game matchups.
    
    Args:
        games_df: DataFrame with game information
        team_stats: DataFrame with team statistics
    
    Returns:
        DataFrame with matchup features
    """
    if len(games_df) > 1:
        print(f"Creating matchup features for {len(games_df)} games...")
    else:
        print("Creating matchup features...")
    
    matchup_features = []
    skipped_games = []
    
    for idx, game in games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        game_id = game.get('game_id')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Get recent stats for both teams (before this game)
        # First try current season, then fall back to previous season if needed
        home_stats = team_stats[
            (team_stats['team'] == home_team) &
            (team_stats['season'] == season) &
            (team_stats['week'] < week)
        ].tail(5)
        
        # If no current season stats (e.g., Week 1), use previous season
        if len(home_stats) == 0:
            home_stats = team_stats[
                (team_stats['team'] == home_team) &
                (team_stats['season'] == season - 1)
            ].tail(5)
        
        away_stats = team_stats[
            (team_stats['team'] == away_team) &
            (team_stats['season'] == season) &
            (team_stats['week'] < week)
        ].tail(5)
        
        # If no current season stats (e.g., Week 1), use previous season
        if len(away_stats) == 0:
            away_stats = team_stats[
                (team_stats['team'] == away_team) &
                (team_stats['season'] == season - 1)
            ].tail(5)
        
        # If still no stats, skip this game
        if len(home_stats) == 0 or len(away_stats) == 0:
            skipped_games.append(f"{away_team} @ {home_team} (Week {week})")
            continue
        
        # Calculate average stats
        home_off_ppg = home_stats['off_points'].mean()
        home_def_ppg_allowed = home_stats['def_points_allowed'].mean()
        home_off_ypg = home_stats['off_total_yards'].mean()
        home_def_ypg_allowed = home_stats['def_total_yards_allowed'].mean()
        
        away_off_ppg = away_stats['off_points'].mean()
        away_def_ppg_allowed = away_stats['def_points_allowed'].mean()
        away_off_ypg = away_stats['off_total_yards'].mean()
        away_def_ypg_allowed = away_stats['def_total_yards_allowed'].mean()
        
        # Create matchup features
        features = {
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            
            # Team offensive strength vs opponent defensive strength
            'home_off_vs_away_def': home_off_ppg - away_def_ppg_allowed,
            'away_off_vs_home_def': away_off_ppg - home_def_ppg_allowed,
            
            # Net advantage
            'home_net_advantage': (home_off_ppg - home_def_ppg_allowed) - \
                                 (away_off_ppg - away_def_ppg_allowed),
            
            # Yards differential
            'home_yards_diff': home_off_ypg - home_def_ypg_allowed,
            'away_yards_diff': away_off_ypg - away_def_ypg_allowed,
            
            # Strength of schedule (simplified - can be enhanced)
            'home_sos': home_stats['def_points_allowed'].mean(),  # Points allowed by opponents
            'away_sos': away_stats['def_points_allowed'].mean(),
            
            # Home field advantage (can be team-specific)
            'is_home': 1,
        }
        
        # Add actual result if available
        if 'result' in game:
            features['home_win'] = 1 if game['result'] > 0 else 0
            features['home_margin'] = game['result']
        
        matchup_features.append(features)
    
    if skipped_games:
        print(f"  Skipped {len(skipped_games)} games due to missing stats: {', '.join(skipped_games[:5])}")
        if len(skipped_games) > 5:
            print(f"  ... and {len(skipped_games) - 5} more")
    
    if len(matchup_features) == 0:
        print("  Warning: No matchup features created!")
        return pd.DataFrame()
    
    return pd.DataFrame(matchup_features)


def create_advanced_features(df):
    """
    Create advanced/derived features.
    
    Args:
        df: DataFrame with basic features
    
    Returns:
        DataFrame with additional features
    """
    # Point differential per game
    if 'home_net_advantage' in df.columns:
        df['home_net_advantage_abs'] = abs(df['home_net_advantage'])
    
    # Strength differential
    if 'home_yards_diff' in df.columns and 'away_yards_diff' in df.columns:
        df['yards_diff_differential'] = df['home_yards_diff'] - df['away_yards_diff']
    
    # Rest days (simplified - would need schedule data)
    # df['home_rest_days'] = ...
    # df['away_rest_days'] = ...
    
    return df


def prepare_training_data(min_season=None, max_season=None):
    """
    Main function to prepare all features for training.
    
    Args:
        min_season: Minimum season to include (default: None, uses all)
        max_season: Maximum season to include (default: None, uses all)
    
    Returns:
        X: Feature matrix
        y: Target variable (home team win)
    """
    print("Loading data...")
    pbp, games = load_game_data()
    
    if games is None:
        print("No game data found. Run data_collection.py first.")
        return None, None
    
    # Load team stats
    team_stats_path = DATA_DIR / "team_stats.parquet"
    if not team_stats_path.exists():
        print("Team stats not found. Run data_collection.py first.")
        return None, None
    
    team_stats = pd.read_parquet(team_stats_path)
    
    # Filter to recent seasons if specified
    if min_season is not None:
        games = games[games['season'] >= min_season]
        team_stats = team_stats[team_stats['season'] >= min_season]
        print(f"Filtered to seasons >= {min_season}")
    if max_season is not None:
        games = games[games['season'] <= max_season]
        team_stats = team_stats[team_stats['season'] <= max_season]
        print(f"Filtered to seasons <= {max_season}")
    
    # Calculate rolling stats
    print("Calculating rolling statistics...")
    team_stats = calculate_rolling_stats(team_stats)
    
    # Create matchup features
    matchup_df = create_matchup_features(games, team_stats)
    
    # Add advanced features
    matchup_df = create_advanced_features(matchup_df)
    
    # Prepare X and y
    feature_cols = [
        'home_off_vs_away_def', 'away_off_vs_home_def',
        'home_net_advantage', 'home_yards_diff', 'away_yards_diff',
        'home_sos', 'away_sos', 'is_home', 'yards_diff_differential'
    ]
    
    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in matchup_df.columns]
    
    X = matchup_df[feature_cols].copy()
    y = matchup_df['home_win'].copy() if 'home_win' in matchup_df.columns else None
    
    # Remove rows with missing values
    X = X.dropna()
    if y is not None:
        y = y.loc[X.index]
    
    print(f"Prepared {len(X)} games with {len(feature_cols)} features")
    
    # Save processed data
    X.to_parquet(DATA_DIR / "features.parquet", index=False)
    if y is not None:
        y.to_frame().to_parquet(DATA_DIR / "targets.parquet", index=False)
    
    return X, y


if __name__ == "__main__":
    # Use recent data by default (last 6 seasons)
    current_season = get_current_season()
    min_season = max(2018, current_season - 6)
    print(f"Preparing training data for seasons {min_season} to {current_season}")
    X, y = prepare_training_data(min_season=min_season, max_season=current_season)
    if X is not None:
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"\nFeatures:\n{X.columns.tolist()}")
        if y is not None:
            print(f"\nTarget distribution:\n{y.value_counts()}")

