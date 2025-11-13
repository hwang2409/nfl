"""
Enhanced Feature Engineering Module

Implements advanced features from IMPROVING_ACCURACY.md:
- Advanced EPA metrics (success rate, explosive plays, red zone)
- Home/away performance splits
- Divisional game indicators
- Rest days calculation
- Head-to-head history
- QB performance metrics
- Recent form weighting
- Strength of schedule
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from data_collection import load_game_data, DATA_DIR
from advanced_features import calculate_elo_ratings, add_market_features


# NFL divisions (for divisional game detection)
NFL_DIVISIONS = {
    'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
    'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
    'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
    'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
    'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
    'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA']
}


def get_team_division(team):
    """Get division for a team."""
    for div, teams in NFL_DIVISIONS.items():
        if team in teams:
            return div
    return None


def are_divisional_opponents(team1, team2):
    """Check if two teams are in the same division."""
    div1 = get_team_division(team1)
    div2 = get_team_division(team2)
    return div1 is not None and div1 == div2


def calculate_enhanced_epa_features(pbp_data):
    """
    Calculate enhanced EPA features including:
    - Success rate
    - Explosive play rate (15+ yard plays)
    - Red zone efficiency
    - Time of possession
    - Penalty impact
    
    Args:
        pbp_data: Play-by-play DataFrame
    
    Returns:
        DataFrame with enhanced EPA features per team per game
    """
    print("Calculating enhanced EPA features...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return None
    
    epa_features = []
    
    for game_id in pbp_data['game_id'].unique():
        game_pbp = pbp_data[pbp_data['game_id'] == game_id]
        
        if len(game_pbp) == 0:
            continue
        
        season = game_pbp['season'].iloc[0] if 'season' in game_pbp.columns else None
        week = game_pbp['week'].iloc[0] if 'week' in game_pbp.columns else None
        
        for team in ['home_team', 'away_team']:
            if team not in game_pbp.columns:
                continue
                
            team_name = game_pbp[team].iloc[0] if len(game_pbp) > 0 else None
            if team_name is None:
                continue
            
            # Offensive plays
            off_plays = game_pbp[game_pbp['posteam'] == team_name].copy()
            def_plays = game_pbp[game_pbp['defteam'] == team_name].copy()
            
            if len(off_plays) == 0:
                continue
            
            # Basic EPA features
            off_epa = off_plays['epa'].sum() if 'epa' in off_plays.columns else 0
            off_epa_per_play = off_epa / max(len(off_plays), 1)
            
            # Success rate (positive EPA plays)
            successful_plays = (off_plays['epa'] > 0).sum() if 'epa' in off_plays.columns else 0
            success_rate = successful_plays / max(len(off_plays), 1)
            
            # Explosive play rate (15+ yard gains)
            if 'yards_gained' in off_plays.columns:
                explosive_plays = (off_plays['yards_gained'] >= 15).sum()
                explosive_rate = explosive_plays / max(len(off_plays), 1)
            else:
                explosive_rate = 0
            
            # Red zone efficiency (inside opponent 20)
            if 'yardline_100' in off_plays.columns:
                redzone_plays = off_plays[off_plays['yardline_100'] <= 20]
                if len(redzone_plays) > 0:
                    redzone_tds = (redzone_plays['touchdown'] == 1).sum() if 'touchdown' in redzone_plays.columns else 0
                    redzone_efficiency = redzone_tds / max(len(redzone_plays), 1)
                else:
                    redzone_efficiency = 0
            else:
                redzone_efficiency = 0
            
            # Time of possession (if available)
            if 'time_of_possession' in game_pbp.columns:
                # This would need to be aggregated per team
                top_seconds = 0  # Would need to calculate from play data
            else:
                top_seconds = 0
            
            # Penalty impact
            if 'penalty' in off_plays.columns:
                penalty_yards = off_plays[off_plays['penalty'] == 1]['penalty_yards'].sum() if 'penalty_yards' in off_plays.columns else 0
            else:
                penalty_yards = 0
            
            # Defensive EPA allowed
            def_epa_allowed = def_plays['epa'].sum() if 'epa' in def_plays.columns else 0
            def_epa_per_play_allowed = def_epa_allowed / max(len(def_plays), 1)
            
            # Defensive success rate (stopping positive EPA)
            def_success_rate = ((def_plays['epa'] <= 0).sum() if 'epa' in def_plays.columns else 0) / max(len(def_plays), 1)
            
            epa_features.append({
                'game_id': game_id,
                'team': team_name,
                'season': season,
                'week': week,
                'off_epa_per_play': off_epa_per_play,
                'off_success_rate': success_rate,
                'off_explosive_rate': explosive_rate,
                'off_redzone_efficiency': redzone_efficiency,
                'def_epa_per_play_allowed': def_epa_per_play_allowed,
                'def_success_rate': def_success_rate,
                'net_epa_per_play': off_epa_per_play - def_epa_per_play_allowed,
                'penalty_yards': penalty_yards
            })
    
    return pd.DataFrame(epa_features)


def calculate_qb_features(pbp_data):
    """
    Calculate QB performance metrics:
    - QB EPA
    - Completion percentage
    - QBR (if available)
    - Drop to backup penalty
    
    Args:
        pbp_data: Play-by-play DataFrame
    
    Returns:
        DataFrame with QB features per team per game
    """
    print("Calculating QB features...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return None
    
    qb_features = []
    
    for game_id in pbp_data['game_id'].unique():
        game_pbp = pbp_data[pbp_data['game_id'] == game_id]
        
        if len(game_pbp) == 0:
            continue
        
        season = game_pbp['season'].iloc[0] if 'season' in game_pbp.columns else None
        week = game_pbp['week'].iloc[0] if 'week' in game_pbp.columns else None
        
        for team in ['home_team', 'away_team']:
            if team not in game_pbp.columns:
                continue
                
            team_name = game_pbp[team].iloc[0] if len(game_pbp) > 0 else None
            if team_name is None:
                continue
            
            # Passing plays
            pass_plays = game_pbp[
                (game_pbp['posteam'] == team_name) & 
                (game_pbp['play_type'] == 'pass')
            ].copy()
            
            if len(pass_plays) == 0:
                qb_features.append({
                    'game_id': game_id,
                    'team': team_name,
                    'season': season,
                    'week': week,
                    'qb_epa_per_play': 0,
                    'qb_completion_pct': 0,
                    'qb_dropback_count': 0
                })
                continue
            
            # QB EPA
            qb_epa = pass_plays['epa'].sum() if 'epa' in pass_plays.columns else 0
            qb_epa_per_play = qb_epa / max(len(pass_plays), 1)
            
            # Completion percentage
            if 'complete_pass' in pass_plays.columns:
                completions = pass_plays['complete_pass'].sum()
                qb_completion_pct = completions / max(len(pass_plays), 1)
            else:
                qb_completion_pct = 0
            
            qb_features.append({
                'game_id': game_id,
                'team': team_name,
                'season': season,
                'week': week,
                'qb_epa_per_play': qb_epa_per_play,
                'qb_completion_pct': qb_completion_pct,
                'qb_dropback_count': len(pass_plays)
            })
    
    return pd.DataFrame(qb_features)


def calculate_rest_days(games_df):
    """
    Calculate rest days between games for each team.
    
    Args:
        games_df: DataFrame with game information
    
    Returns:
        DataFrame with rest_days added
    """
    print("Calculating rest days...")
    
    games_df = games_df.copy()
    
    # Convert gameday to datetime if available
    if 'gameday' in games_df.columns:
        games_df['gameday'] = pd.to_datetime(games_df['gameday'], errors='coerce')
    
    games_df['home_rest_days'] = 7  # Default (normal week)
    games_df['away_rest_days'] = 7
    
    # Sort by season and week
    games_sorted = games_df.sort_values(['season', 'week']).copy()
    
    # Track last game date for each team
    last_game_date = {}
    
    for idx, game in games_sorted.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        gameday = game.get('gameday')
        
        if pd.isna(gameday) or pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Calculate rest days for home team
        if home_team in last_game_date:
            rest_days = (gameday - last_game_date[home_team]).days
            if rest_days > 0:  # Valid rest days
                games_df.loc[idx, 'home_rest_days'] = rest_days
        else:
            # First game of season or no previous game
            games_df.loc[idx, 'home_rest_days'] = 7  # Default
        
        # Calculate rest days for away team
        if away_team in last_game_date:
            rest_days = (gameday - last_game_date[away_team]).days
            if rest_days > 0:
                games_df.loc[idx, 'away_rest_days'] = rest_days
        else:
            games_df.loc[idx, 'away_rest_days'] = 7
        
        # Update last game date
        last_game_date[home_team] = gameday
        last_game_date[away_team] = gameday
    
    return games_df


def calculate_home_away_splits(games_df, epa_df=None):
    """
    Calculate home/away performance splits for each team.
    
    Args:
        games_df: DataFrame with game information
        epa_df: DataFrame with EPA features (optional)
    
    Returns:
        Dictionary with home/away splits per team
    """
    print("Calculating home/away splits...")
    
    splits = {}
    
    # Group by team and home/away
    for team in games_df['home_team'].dropna().unique():
        home_games = games_df[games_df['home_team'] == team]
        away_games = games_df[games_df['away_team'] == team]
        
        # Calculate win rates
        home_wins = (home_games['result'] > 0).sum() if 'result' in home_games.columns else 0
        home_total = len(home_games[home_games['result'].notna()])
        home_win_rate = home_wins / max(home_total, 1)
        
        away_wins = (away_games['result'] < 0).sum() if 'result' in away_games.columns else 0
        away_total = len(away_games[away_games['result'].notna()])
        away_win_rate = away_wins / max(away_total, 1)
        
        splits[team] = {
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'home_advantage': home_win_rate - away_win_rate
        }
    
    return splits


def calculate_head_to_head_history(games_df, home_team, away_team, season, week, n_games=5):
    """
    Calculate head-to-head history between two teams.
    
    Args:
        games_df: DataFrame with historical games
        home_team: Home team
        away_team: Away team
        season: Current season
        week: Current week
        n_games: Number of recent meetings to consider
    
    Returns:
        Dictionary with head-to-head stats
    """
    # Get historical matchups (before current game)
    matchups = games_df[
        ((games_df['home_team'] == home_team) & (games_df['away_team'] == away_team)) |
        ((games_df['home_team'] == away_team) & (games_df['away_team'] == home_team))
    ].copy()
    
    # Filter to games before current game
    matchups = matchups[
        (matchups['season'] < season) | 
        ((matchups['season'] == season) & (matchups['week'] < week))
    ]
    
    # Sort by most recent
    matchups = matchups.sort_values(['season', 'week'], ascending=False).head(n_games)
    
    if len(matchups) == 0:
        return {
            'h2h_games': 0,
            'home_team_wins': 0,
            'h2h_win_rate': 0.5,
            'avg_margin': 0
        }
    
    # Calculate wins for home team (in current context)
    home_wins = 0
    margins = []
    
    for idx, game in matchups.iterrows():
        if 'result' in game and not pd.isna(game['result']):
            if game['home_team'] == home_team:
                # Home team was home in this game
                if game['result'] > 0:
                    home_wins += 1
                    margins.append(game['result'])
                else:
                    margins.append(-game['result'])
            else:
                # Home team was away in this game
                if game['result'] < 0:
                    home_wins += 1
                    margins.append(-game['result'])
                else:
                    margins.append(game['result'])
    
    return {
        'h2h_games': len(matchups),
        'home_team_wins': home_wins,
        'h2h_win_rate': home_wins / max(len(matchups), 1),
        'avg_margin': np.mean(margins) if margins else 0
    }


def calculate_strength_of_schedule(games_df, elo_df, team, season, week):
    """
    Calculate strength of schedule for a team.
    
    Args:
        games_df: DataFrame with game information
        elo_df: DataFrame with Elo ratings
        team: Team name
        season: Current season
        week: Current week
    
    Returns:
        Average opponent Elo rating
    """
    # Get games played by this team before current week
    team_games = games_df[
        ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
        (games_df['season'] == season) &
        (games_df['week'] < week)
    ]
    
    if len(team_games) == 0:
        return 1500  # Default Elo
    
    opponent_elos = []
    
    for idx, game in team_games.iterrows():
        if game['home_team'] == team:
            opponent = game['away_team']
        else:
            opponent = game['home_team']
        
        # Get opponent's Elo before this game
        opponent_elo_history = elo_df[
            ((elo_df['home_team'] == opponent) | (elo_df['away_team'] == opponent)) &
            (elo_df['season'] == season) &
            (elo_df['week'] < game['week'])
        ]
        
        if len(opponent_elo_history) > 0:
            last_game = opponent_elo_history.iloc[-1]
            opponent_elo = last_game['home_elo'] if last_game['home_team'] == opponent else last_game['away_elo']
            opponent_elos.append(opponent_elo)
    
    return np.mean(opponent_elos) if opponent_elos else 1500


def calculate_recent_form_weighted(epa_df, team, season, week, n_games=5, decay=0.9):
    """
    Calculate weighted recent form (more recent games weighted higher).
    
    Args:
        epa_df: DataFrame with EPA features
        team: Team name
        season: Current season
        week: Current week
        n_games: Number of recent games to consider
        decay: Decay factor (0.9 means each game is 90% weight of next game)
    
    Returns:
        Dictionary with weighted averages
    """
    # Get recent games
    recent_games = epa_df[
        (epa_df['team'] == team) &
        (epa_df['season'] == season) &
        (epa_df['week'] < week)
    ].sort_values('week', ascending=False).head(n_games)
    
    # If no current season, use previous season
    if len(recent_games) == 0:
        recent_games = epa_df[
            (epa_df['team'] == team) &
            (epa_df['season'] == season - 1)
        ].sort_values('week', ascending=False).head(n_games)
    
    if len(recent_games) == 0:
        return {
            'weighted_off_epa': 0,
            'weighted_def_epa': 0,
            'weighted_net_epa': 0,
            'weighted_success_rate': 0
        }
    
    # Calculate weights (most recent = highest weight)
    weights = [decay ** i for i in range(len(recent_games))]
    weights = np.array(weights) / sum(weights)  # Normalize
    
    # Weighted averages
    weighted_off_epa = np.average(recent_games['off_epa_per_play'], weights=weights)
    weighted_def_epa = np.average(recent_games['def_epa_per_play_allowed'], weights=weights)
    weighted_net_epa = np.average(recent_games['net_epa_per_play'], weights=weights)
    weighted_success_rate = np.average(recent_games['off_success_rate'], weights=weights) if 'off_success_rate' in recent_games.columns else 0
    
    return {
        'weighted_off_epa': weighted_off_epa,
        'weighted_def_epa': weighted_def_epa,
        'weighted_net_epa': weighted_net_epa,
        'weighted_success_rate': weighted_success_rate
    }


def add_enhanced_situational_features(games_df):
    """
    Add enhanced situational features:
    - Divisional games
    - Primetime games
    - Rest days
    - Home/away splits
    
    Args:
        games_df: DataFrame with game information
    
    Returns:
        DataFrame with enhanced situational features
    """
    games_df = games_df.copy()
    
    # Divisional games
    games_df['is_divisional'] = games_df.apply(
        lambda row: are_divisional_opponents(row.get('home_team'), row.get('away_team')) if not pd.isna(row.get('home_team')) and not pd.isna(row.get('away_team')) else 0,
        axis=1
    ).astype(int)
    
    # Primetime games (Thursday, Sunday Night, Monday Night)
    if 'weekday' in games_df.columns:
        games_df['is_primetime'] = (
            (games_df['weekday'] == 'Thursday') |
            (games_df['weekday'] == 'Monday') |
            (games_df['gametime'].str.contains('20:', na=False) if 'gametime' in games_df.columns else False)
        ).astype(int)
    else:
        games_df['is_primetime'] = 0
    
    # Rest days (calculated separately)
    games_df = calculate_rest_days(games_df)
    
    # Rest advantage (difference in rest days)
    games_df['rest_advantage'] = games_df['home_rest_days'] - games_df['away_rest_days']
    
    # Short week indicator (Thursday game = short week)
    games_df['is_short_week'] = (games_df['is_thursday'] == 1).astype(int) if 'is_thursday' in games_df.columns else 0
    
    return games_df

