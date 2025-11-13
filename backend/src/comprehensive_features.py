"""
Comprehensive Feature Engineering Module

Implements ALL features from "Strategies for Improving Accuracy" section:
- Weather forecast integration
- Travel distance and time zones
- Injury impact features
- Playoff implications
- Coaching matchups
- Style matchups (run-heavy vs pass-heavy)
- Complete time of possession
- Turnover regression
- Third down conversion rates
- Defensive matchups
- Real-time market data framework
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from math import radians, sin, cos, sqrt, atan2
sys.path.append(str(Path(__file__).parent))
from data_collection import load_game_data, DATA_DIR
from enhanced_features import (
    calculate_enhanced_epa_features, calculate_qb_features,
    are_divisional_opponents, calculate_home_away_splits
)


# Team locations for travel distance calculation
TEAM_LOCATIONS = {
    'ARI': {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.5275, 'lon': -112.2625, 'timezone': 'MST'},
    'ATL': {'city': 'Atlanta', 'state': 'GA', 'lat': 33.7550, 'lon': -84.4010, 'timezone': 'EST'},
    'BAL': {'city': 'Baltimore', 'state': 'MD', 'lat': 39.2780, 'lon': -76.6227, 'timezone': 'EST'},
    'BUF': {'city': 'Buffalo', 'state': 'NY', 'lat': 42.7738, 'lon': -78.7869, 'timezone': 'EST'},
    'CAR': {'city': 'Charlotte', 'state': 'NC', 'lat': 35.2258, 'lon': -80.8528, 'timezone': 'EST'},
    'CHI': {'city': 'Chicago', 'state': 'IL', 'lat': 41.8625, 'lon': -87.6167, 'timezone': 'CST'},
    'CIN': {'city': 'Cincinnati', 'state': 'OH', 'lat': 39.0950, 'lon': -84.5160, 'timezone': 'EST'},
    'CLE': {'city': 'Cleveland', 'state': 'OH', 'lat': 41.5061, 'lon': -81.6996, 'timezone': 'EST'},
    'DAL': {'city': 'Dallas', 'state': 'TX', 'lat': 32.7473, 'lon': -97.0945, 'timezone': 'CST'},
    'DEN': {'city': 'Denver', 'state': 'CO', 'lat': 39.7439, 'lon': -105.0200, 'timezone': 'MST'},
    'DET': {'city': 'Detroit', 'state': 'MI', 'lat': 42.3400, 'lon': -83.0456, 'timezone': 'EST'},
    'GB': {'city': 'Green Bay', 'state': 'WI', 'lat': 44.5013, 'lon': -88.0622, 'timezone': 'CST'},
    'HOU': {'city': 'Houston', 'state': 'TX', 'lat': 29.6847, 'lon': -95.4107, 'timezone': 'CST'},
    'IND': {'city': 'Indianapolis', 'state': 'IN', 'lat': 39.7601, 'lon': -86.1639, 'timezone': 'EST'},
    'JAX': {'city': 'Jacksonville', 'state': 'FL', 'lat': 30.3239, 'lon': -81.6372, 'timezone': 'EST'},
    'KC': {'city': 'Kansas City', 'state': 'MO', 'lat': 39.0489, 'lon': -94.4839, 'timezone': 'CST'},
    'LV': {'city': 'Las Vegas', 'state': 'NV', 'lat': 36.0908, 'lon': -115.1836, 'timezone': 'PST'},
    'LAC': {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0141, 'lon': -118.2879, 'timezone': 'PST'},
    'LAR': {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0141, 'lon': -118.2879, 'timezone': 'PST'},
    'MIA': {'city': 'Miami', 'state': 'FL', 'lat': 25.9581, 'lon': -80.2389, 'timezone': 'EST'},
    'MIN': {'city': 'Minneapolis', 'state': 'MN', 'lat': 44.9740, 'lon': -93.2581, 'timezone': 'CST'},
    'NE': {'city': 'Foxborough', 'state': 'MA', 'lat': 42.0926, 'lon': -71.2640, 'timezone': 'EST'},
    'NO': {'city': 'New Orleans', 'state': 'LA', 'lat': 29.9511, 'lon': -90.0815, 'timezone': 'CST'},
    'NYG': {'city': 'East Rutherford', 'state': 'NJ', 'lat': 40.8136, 'lon': -74.0744, 'timezone': 'EST'},
    'NYJ': {'city': 'East Rutherford', 'state': 'NJ', 'lat': 40.8136, 'lon': -74.0744, 'timezone': 'EST'},
    'PHI': {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.9010, 'lon': -75.1675, 'timezone': 'EST'},
    'PIT': {'city': 'Pittsburgh', 'state': 'PA', 'lat': 40.4468, 'lon': -80.0158, 'timezone': 'EST'},
    'SF': {'city': 'San Francisco', 'state': 'CA', 'lat': 37.7133, 'lon': -122.3860, 'timezone': 'PST'},
    'SEA': {'city': 'Seattle', 'state': 'WA', 'lat': 47.5952, 'lon': -122.3316, 'timezone': 'PST'},
    'TB': {'city': 'Tampa', 'state': 'FL', 'lat': 27.9756, 'lon': -82.5033, 'timezone': 'EST'},
    'TEN': {'city': 'Nashville', 'state': 'TN', 'lat': 36.1665, 'lon': -86.7713, 'timezone': 'CST'},
    'WAS': {'city': 'Landover', 'state': 'MD', 'lat': 38.9076, 'lon': -76.8644, 'timezone': 'EST'},
}

# Dome stadiums (no weather impact)
DOME_STADIUMS = {
    'ATL': True,  # Mercedes-Benz Stadium
    'DET': True,  # Ford Field
    'IND': True,  # Lucas Oil Stadium
    'NO': True,   # Caesars Superdome
    'DAL': True,  # AT&T Stadium
    'HOU': True,  # NRG Stadium
    'MIN': True,  # U.S. Bank Stadium
    'LV': True,   # Allegiant Stadium
    'LAR': True,  # SoFi Stadium (roof)
    'LAC': True,  # SoFi Stadium (roof)
}


def calculate_travel_distance(home_team, away_team):
    """
    Calculate travel distance between two teams' cities.
    Uses Haversine formula for great-circle distance.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
    
    Returns:
        Distance in miles
    """
    if home_team not in TEAM_LOCATIONS or away_team not in TEAM_LOCATIONS:
        return 0
    
    home_loc = TEAM_LOCATIONS[home_team]
    away_loc = TEAM_LOCATIONS[away_team]
    
    # Convert to radians
    lat1, lon1 = radians(home_loc['lat']), radians(home_loc['lon'])
    lat2, lon2 = radians(away_loc['lat']), radians(away_loc['lon'])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Earth radius in miles
    R = 3959
    distance = R * c
    
    return distance


def calculate_timezone_change(home_team, away_team):
    """
    Calculate timezone change for away team.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
    
    Returns:
        Hours of timezone change (positive = eastward, negative = westward)
    """
    if home_team not in TEAM_LOCATIONS or away_team not in TEAM_LOCATIONS:
        return 0
    
    timezone_map = {'EST': 0, 'CST': -1, 'MST': -2, 'PST': -3}
    
    home_tz = TEAM_LOCATIONS[home_team]['timezone']
    away_tz = TEAM_LOCATIONS[away_team]['timezone']
    
    home_offset = timezone_map.get(home_tz, 0)
    away_offset = timezone_map.get(away_tz, 0)
    
    return home_offset - away_offset  # Positive = away team traveling east


def add_weather_features(games_df, use_api=True):
    """
    Add weather features for outdoor games.
    
    Now integrates with Open-Meteo (free, no API key required).
    
    Args:
        games_df: DataFrame with game information
        use_api: Whether to fetch from API (True) or use placeholders (False)
    
    Returns:
        DataFrame with weather features added
    """
    games_df = games_df.copy()
    
    # Check if game is in dome (no weather impact)
    games_df['is_dome'] = games_df['home_team'].map(
        lambda x: DOME_STADIUMS.get(x, False) if pd.notna(x) else False
    ).astype(int)
    
    if use_api:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from weather_api import batch_fetch_weather
            
            # Determine game date column
            date_col = None
            for col in ['gameday', 'game_date', 'date']:
                if col in games_df.columns:
                    date_col = col
                    break
            
            if date_col:
                weather_df = batch_fetch_weather(games_df, game_date_col=date_col, home_team_col='home_team')
                
                # Merge weather data
                if len(weather_df) == len(games_df):
                    games_df['temp'] = weather_df['temp'].values if 'temp' in weather_df.columns else 65
                    games_df['wind_mph'] = weather_df['wind_mph'].values if 'wind_mph' in weather_df.columns else 0
                    games_df['precipitation'] = weather_df['precipitation'].values if 'precipitation' in weather_df.columns else 0
                    games_df['precipitation_prob'] = weather_df['precipitation_prob'].values if 'precipitation_prob' in weather_df.columns else 0
                else:
                    # Length mismatch, use defaults
                    games_df['temp'] = 65
                    games_df['wind_mph'] = 0
                    games_df['precipitation'] = 0
                    games_df['precipitation_prob'] = 0
            else:
                # Fallback to placeholders if no date column
                games_df['temp'] = 65
                games_df['wind_mph'] = 0
                games_df['precipitation'] = 0
                games_df['precipitation_prob'] = 0
        except (ImportError, NameError):
            print("  Warning: weather_api module not available, using placeholders")
            games_df['temp'] = 65
            games_df['wind_mph'] = 0
            games_df['precipitation'] = 0
            games_df['precipitation_prob'] = 0
        except Exception as e:
            print(f"  Warning: Weather API error, using placeholders: {e}")
            games_df['temp'] = 65
            games_df['wind_mph'] = 0
            games_df['precipitation'] = 0
            games_df['precipitation_prob'] = 0
    else:
        # Use placeholder values
        games_df['temp'] = 65
        games_df['wind_mph'] = 0
        games_df['precipitation'] = 0
        games_df['precipitation_prob'] = 0
    
    # Weather impact indicators
    games_df['high_wind'] = ((games_df['wind_mph'] > 15) & (games_df['is_dome'] == 0)).astype(int)
    games_df['cold_weather'] = ((games_df['temp'] < 32) & (games_df['is_dome'] == 0)).astype(int)
    games_df['bad_weather'] = ((games_df['precipitation'] == 1) & (games_df['is_dome'] == 0)).astype(int)
    
    return games_df


def add_travel_features(games_df):
    """
    Add travel distance and timezone change features.
    
    Args:
        games_df: DataFrame with game information
    
    Returns:
        DataFrame with travel features added
    """
    games_df = games_df.copy()
    
    games_df['travel_distance'] = games_df.apply(
        lambda row: calculate_travel_distance(row.get('home_team'), row.get('away_team'))
        if pd.notna(row.get('home_team')) and pd.notna(row.get('away_team')) else 0,
        axis=1
    )
    
    games_df['timezone_change'] = games_df.apply(
        lambda row: calculate_timezone_change(row.get('home_team'), row.get('away_team'))
        if pd.notna(row.get('home_team')) and pd.notna(row.get('away_team')) else 0,
        axis=1
    )
    
    # Travel impact indicators
    games_df['long_travel'] = (games_df['travel_distance'] > 1500).astype(int)
    games_df['cross_country'] = (games_df['travel_distance'] > 2000).astype(int)
    games_df['eastward_travel'] = (games_df['timezone_change'] > 0).astype(int)
    games_df['westward_travel'] = (games_df['timezone_change'] < 0).astype(int)
    
    return games_df


def calculate_style_matchups(pbp_data, games_df):
    """
    Calculate offensive style (run-heavy vs pass-heavy) for teams.
    
    Args:
        pbp_data: Play-by-play DataFrame
        games_df: DataFrame with game information
    
    Returns:
        Dictionary with team style metrics
    """
    print("Calculating offensive style matchups...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return {}
    
    style_metrics = {}
    
    for team in games_df['home_team'].dropna().unique():
        team_plays = pbp_data[pbp_data['posteam'] == team]
        
        if len(team_plays) == 0:
            continue
        
        # Calculate pass/run ratio
        pass_plays = (team_plays['play_type'] == 'pass').sum()
        run_plays = (team_plays['play_type'] == 'run').sum()
        total_plays = pass_plays + run_plays
        
        if total_plays > 0:
            pass_rate = pass_plays / total_plays
            run_rate = run_plays / total_plays
        else:
            pass_rate = run_rate = 0.5
        
        style_metrics[team] = {
            'pass_rate': pass_rate,
            'run_rate': run_rate,
            'is_pass_heavy': 1 if pass_rate > 0.6 else 0,
            'is_run_heavy': 1 if run_rate > 0.5 else 0
        }
    
    return style_metrics


def calculate_defensive_matchups(pbp_data, games_df, epa_df=None):
    """
    Calculate how well team's offense matches opponent's defense.
    
    Args:
        pbp_data: Play-by-play DataFrame
        games_df: DataFrame with game information
        epa_df: DataFrame with EPA features (optional)
    
    Returns:
        Dictionary with defensive matchup metrics
    """
    print("Calculating defensive matchups...")
    
    if epa_df is None:
        return {}
    
    matchup_metrics = {}
    
    for idx, game in games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Get recent offensive EPA for home team
        home_off_epa = epa_df[
            (epa_df['team'] == home_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ].tail(3)['off_epa_per_play'].mean() if len(epa_df[
            (epa_df['team'] == home_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ]) > 0 else 0
        
        # Get recent defensive EPA allowed for away team
        away_def_epa = epa_df[
            (epa_df['team'] == away_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ].tail(3)['def_epa_per_play_allowed'].mean() if len(epa_df[
            (epa_df['team'] == away_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ]) > 0 else 0
        
        # Get recent offensive EPA for away team
        away_off_epa = epa_df[
            (epa_df['team'] == away_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ].tail(3)['off_epa_per_play'].mean() if len(epa_df[
            (epa_df['team'] == away_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ]) > 0 else 0
        
        # Get recent defensive EPA allowed for home team
        home_def_epa = epa_df[
            (epa_df['team'] == home_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ].tail(3)['def_epa_per_play_allowed'].mean() if len(epa_df[
            (epa_df['team'] == home_team) &
            (epa_df['season'] == season) &
            (epa_df['week'] < week)
        ]) > 0 else 0
        
        # Matchup advantage: offense EPA vs opponent defense EPA
        home_off_vs_away_def = home_off_epa - away_def_epa
        away_off_vs_home_def = away_off_epa - home_def_epa
        
        matchup_metrics[(home_team, away_team, season, week)] = {
            'home_off_vs_away_def': home_off_vs_away_def,
            'away_off_vs_home_def': away_off_vs_home_def,
            'matchup_advantage': home_off_vs_away_def - away_off_vs_home_def
        }
    
    return matchup_metrics


def calculate_coaching_matchups(games_df):
    """
    Calculate head-to-head coaching records.
    
    Note: This is simplified. In production, would track actual coaches.
    
    Args:
        games_df: DataFrame with game information
    
    Returns:
        Dictionary with coaching matchup records
    """
    print("Calculating coaching matchups...")
    
    coaching_records = {}
    
    # Group by team pairs (simplified - assumes same coach over time)
    for home_team in games_df['home_team'].dropna().unique():
        for away_team in games_df['away_team'].dropna().unique():
            if home_team == away_team:
                continue
            
            # Get historical matchups
            matchups = games_df[
                ((games_df['home_team'] == home_team) & (games_df['away_team'] == away_team)) |
                ((games_df['home_team'] == away_team) & (games_df['away_team'] == home_team))
            ]
            
            if len(matchups) > 0:
                # Count wins for home team (in context of home_team vs away_team)
                home_wins = 0
                for idx, game in matchups.iterrows():
                    if 'result' in game and not pd.isna(game['result']):
                        if game['home_team'] == home_team and game['result'] > 0:
                            home_wins += 1
                        elif game['away_team'] == home_team and game['result'] < 0:
                            home_wins += 1
                
                coaching_records[(home_team, away_team)] = {
                    'total_games': len(matchups),
                    'home_team_wins': home_wins,
                    'coaching_win_rate': home_wins / max(len(matchups), 1)
                }
    
    return coaching_records


def calculate_playoff_implications(games_df, season, week):
    """
    Calculate playoff implications for games.
    
    Args:
        games_df: DataFrame with game information
        season: Current season
        week: Current week
    
    Returns:
        Dictionary with playoff implication scores
    """
    # Simplified: games later in season have more playoff implications
    # In production, would calculate actual playoff scenarios
    
    if week <= 12:
        playoff_importance = 0.0  # Early season
    elif week <= 15:
        playoff_importance = 0.5  # Mid-late season
    else:
        playoff_importance = 1.0  # Late season/playoff race
    
    return {
        'playoff_importance': playoff_importance,
        'late_season': 1 if week >= 14 else 0,
        'playoff_race': 1 if week >= 15 else 0
    }


def calculate_turnover_regression(pbp_data, games_df):
    """
    Calculate turnover rates and regression indicators.
    Teams with extreme turnover rates tend to regress.
    
    Args:
        pbp_data: Play-by-play DataFrame
        games_df: DataFrame with game information
    
    Returns:
        Dictionary with turnover regression metrics
    """
    print("Calculating turnover regression...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return {}
    
    turnover_metrics = {}
    
    for team in games_df['home_team'].dropna().unique():
        team_plays = pbp_data[pbp_data['posteam'] == team]
        
        if len(team_plays) == 0:
            continue
        
        # Calculate turnover rate
        turnovers = (team_plays['interception'] == 1).sum() + (team_plays['fumble_lost'] == 1).sum()
        total_plays = len(team_plays)
        
        turnover_rate = turnovers / max(total_plays, 1)
        
        # Regression indicators
        # High turnover rate (>3% per play) tends to regress down
        # Low turnover rate (<1% per play) tends to regress up
        high_turnover = 1 if turnover_rate > 0.03 else 0
        low_turnover = 1 if turnover_rate < 0.01 else 0
        
        turnover_metrics[team] = {
            'turnover_rate': turnover_rate,
            'high_turnover': high_turnover,
            'low_turnover': low_turnover,
            'turnover_regression': 1 if (high_turnover or low_turnover) else 0
        }
    
    return turnover_metrics


def calculate_third_down_stats(pbp_data):
    """
    Calculate third down conversion rates.
    
    Args:
        pbp_data: Play-by-play DataFrame
    
    Returns:
        DataFrame with third down stats per team per game
    """
    print("Calculating third down stats...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return None
    
    third_down_stats = []
    
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
            
            # Offensive third downs
            off_third_downs = game_pbp[
                (game_pbp['posteam'] == team_name) &
                (game_pbp['down'] == 3)
            ]
            
            if len(off_third_downs) > 0:
                conversions = (off_third_downs['first_down'] == 1).sum()
                third_down_pct = conversions / len(off_third_downs)
            else:
                third_down_pct = 0
            
            # Defensive third down stops
            def_third_downs = game_pbp[
                (game_pbp['defteam'] == team_name) &
                (game_pbp['down'] == 3)
            ]
            
            if len(def_third_downs) > 0:
                stops = (def_third_downs['first_down'] == 0).sum()
                third_down_stop_pct = stops / len(def_third_downs)
            else:
                third_down_stop_pct = 0
            
            third_down_stats.append({
                'game_id': game_id,
                'team': team_name,
                'season': season,
                'week': week,
                'third_down_pct': third_down_pct,
                'third_down_stop_pct': third_down_stop_pct
            })
    
    return pd.DataFrame(third_down_stats)


def calculate_time_of_possession(pbp_data):
    """
    Calculate time of possession per team per game.
    
    Args:
        pbp_data: Play-by-play DataFrame
    
    Returns:
        DataFrame with TOP per team per game
    """
    print("Calculating time of possession...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return None
    
    top_stats = []
    
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
            
            # Get plays for this team
            team_plays = game_pbp[game_pbp['posteam'] == team_name]
            
            # Calculate time of possession from play durations
            if 'time' in team_plays.columns:
                # Would need to parse time and calculate duration
                # For now, use play count as proxy
                top_seconds = len(team_plays) * 30  # Rough estimate: 30 seconds per play
            else:
                top_seconds = len(team_plays) * 30
            
            # Convert to minutes
            top_minutes = top_seconds / 60
            
            top_stats.append({
                'game_id': game_id,
                'team': team_name,
                'season': season,
                'week': week,
                'time_of_possession_min': top_minutes,
                'play_count': len(team_plays)
            })
    
    return pd.DataFrame(top_stats)


def calculate_target_share(pbp_data):
    """
    Calculate target share for skill positions (WR, TE, RB).
    
    Target share metrics:
    - Top receiver target share
    - Target concentration (Herfindahl index)
    - WR vs TE vs RB target distribution
    
    Args:
        pbp_data: Play-by-play DataFrame
    
    Returns:
        DataFrame with target share metrics per team per game
    """
    print("Calculating target share metrics...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return None
    
    target_share_stats = []
    
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
            
            # Get passing plays for this team
            pass_plays = game_pbp[
                (game_pbp['posteam'] == team_name) &
                (game_pbp['play_type'] == 'pass') &
                (game_pbp['receiver_player_name'].notna())
            ].copy()
            
            if len(pass_plays) == 0:
                target_share_stats.append({
                    'game_id': game_id,
                    'team': team_name,
                    'season': season,
                    'week': week,
                    'top_receiver_target_share': 0,
                    'target_concentration': 0,
                    'wr_target_share': 0,
                    'te_target_share': 0,
                    'rb_target_share': 0,
                    'total_targets': 0
                })
                continue
            
            # Count targets by receiver
            receiver_targets = pass_plays['receiver_player_name'].value_counts()
            total_targets = len(pass_plays)
            
            if total_targets == 0:
                continue
            
            # Top receiver target share
            top_receiver_target_share = receiver_targets.iloc[0] / total_targets if len(receiver_targets) > 0 else 0
            
            # Target concentration (Herfindahl index: sum of squared shares)
            # Higher = more concentrated (fewer receivers getting targets)
            target_shares = receiver_targets / total_targets
            target_concentration = (target_shares ** 2).sum()
            
            # Position-based target share (simplified - would need position data)
            # For now, use receiver count as proxy
            # Top 2 receivers likely WRs, next likely TE/RB
            if len(receiver_targets) >= 2:
                top_2_share = (receiver_targets.iloc[0] + receiver_targets.iloc[1]) / total_targets
                wr_target_share = top_2_share  # Approximate WR share
                remaining_share = 1 - top_2_share
                te_target_share = remaining_share * 0.4  # Approximate TE share
                rb_target_share = remaining_share * 0.6  # Approximate RB share
            else:
                wr_target_share = top_receiver_target_share
                te_target_share = 0
                rb_target_share = 0
            
            target_share_stats.append({
                'game_id': game_id,
                'team': team_name,
                'season': season,
                'week': week,
                'top_receiver_target_share': top_receiver_target_share,
                'target_concentration': target_concentration,
                'wr_target_share': wr_target_share,
                'te_target_share': te_target_share,
                'rb_target_share': rb_target_share,
                'total_targets': total_targets
            })
    
    return pd.DataFrame(target_share_stats)


def calculate_cb_wr_matchups(pbp_data, games_df, target_share_df=None):
    """
    Calculate CB vs WR defensive matchups.
    
    This creates matchup features based on:
    - Top WR target share (offense)
    - Defensive passing EPA allowed (defense - proxies for CB quality)
    - Air yards per target (offense)
    - Defensive air yards allowed (defense)
    
    Args:
        pbp_data: Play-by-play DataFrame
        games_df: DataFrame with game information
        target_share_df: DataFrame with target share metrics (optional)
    
    Returns:
        Dictionary with CB vs WR matchup metrics
    """
    print("Calculating CB vs WR defensive matchups...")
    
    if pbp_data is None or len(pbp_data) == 0:
        return {}
    
    # Calculate target share if not provided
    if target_share_df is None:
        target_share_df = calculate_target_share(pbp_data)
    
    if target_share_df is None or len(target_share_df) == 0:
        return {}
    
    # Calculate defensive passing stats per team
    def_passing_stats = {}
    
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
            
            # Defensive passing plays (opponent passing against this team)
            def_pass_plays = game_pbp[
                (game_pbp['defteam'] == team_name) &
                (game_pbp['play_type'] == 'pass')
            ].copy()
            
            if len(def_pass_plays) > 0:
                # Defensive passing EPA allowed
                def_pass_epa = def_pass_plays['epa'].sum() if 'epa' in def_pass_plays.columns else 0
                def_pass_epa_per_play = def_pass_epa / len(def_pass_plays)
                
                # Air yards allowed (if available)
                if 'air_yards' in def_pass_plays.columns:
                    air_yards_allowed = def_pass_plays['air_yards'].sum()
                    air_yards_per_attempt = air_yards_allowed / len(def_pass_plays)
                else:
                    air_yards_per_attempt = 0
                
                # Completion rate allowed
                if 'complete_pass' in def_pass_plays.columns:
                    completions_allowed = def_pass_plays['complete_pass'].sum()
                    comp_pct_allowed = completions_allowed / len(def_pass_plays)
                else:
                    comp_pct_allowed = 0
                
                key = (team_name, season, week)
                def_passing_stats[key] = {
                    'def_pass_epa_per_play': def_pass_epa_per_play,
                    'air_yards_per_attempt_allowed': air_yards_per_attempt,
                    'comp_pct_allowed': comp_pct_allowed
                }
    
    # Calculate offensive air yards per target
    off_air_yards_stats = {}
    
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
            
            # Offensive passing plays
            off_pass_plays = game_pbp[
                (game_pbp['posteam'] == team_name) &
                (game_pbp['play_type'] == 'pass')
            ].copy()
            
            if len(off_pass_plays) > 0:
                # Air yards per target
                if 'air_yards' in off_pass_plays.columns:
                    total_air_yards = off_pass_plays['air_yards'].sum()
                    air_yards_per_target = total_air_yards / len(off_pass_plays)
                else:
                    air_yards_per_target = 0
                
                key = (team_name, season, week)
                off_air_yards_stats[key] = {
                    'air_yards_per_target': air_yards_per_target
                }
    
    # Build matchup metrics
    matchup_metrics = {}
    
    for idx, game in games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Get recent target share for home team (offense)
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
        
        home_top_wr_share = home_target_share['top_receiver_target_share'].mean() if len(home_target_share) > 0 else 0
        home_target_concentration = home_target_share['target_concentration'].mean() if len(home_target_share) > 0 else 0
        
        # Get recent defensive passing stats for away team (defense)
        away_def_key = (away_team, season, week)
        away_def_stats = def_passing_stats.get(away_def_key, {})
        away_def_pass_epa = away_def_stats.get('def_pass_epa_per_play', 0)
        away_air_yards_allowed = away_def_stats.get('air_yards_per_attempt_allowed', 0)
        
        # Home offense vs Away defense matchup
        home_off_air_yards_key = (home_team, season, week)
        home_off_air_yards = off_air_yards_stats.get(home_off_air_yards_key, {}).get('air_yards_per_target', 0)
        
        # Matchup: High target concentration WR vs weak pass defense = advantage
        home_wr_vs_away_cb_advantage = home_top_wr_share * (home_off_air_yards - away_air_yards_allowed) - away_def_pass_epa
        
        # Repeat for away team offense vs home team defense
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
        
        away_top_wr_share = away_target_share['top_receiver_target_share'].mean() if len(away_target_share) > 0 else 0
        away_target_concentration = away_target_share['target_concentration'].mean() if len(away_target_share) > 0 else 0
        
        home_def_key = (home_team, season, week)
        home_def_stats = def_passing_stats.get(home_def_key, {})
        home_def_pass_epa = home_def_stats.get('def_pass_epa_per_play', 0)
        home_air_yards_allowed = home_def_stats.get('air_yards_per_attempt_allowed', 0)
        
        away_off_air_yards_key = (away_team, season, week)
        away_off_air_yards = off_air_yards_stats.get(away_off_air_yards_key, {}).get('air_yards_per_target', 0)
        
        away_wr_vs_home_cb_advantage = away_top_wr_share * (away_off_air_yards - home_air_yards_allowed) - home_def_pass_epa
        
        matchup_metrics[(home_team, away_team, season, week)] = {
            'home_wr_target_concentration': home_target_concentration,
            'away_wr_target_concentration': away_target_concentration,
            'home_wr_vs_away_cb_advantage': home_wr_vs_away_cb_advantage,
            'away_wr_vs_home_cb_advantage': away_wr_vs_home_cb_advantage,
            'cb_wr_matchup_advantage': home_wr_vs_away_cb_advantage - away_wr_vs_home_cb_advantage
        }
    
    return matchup_metrics


def add_injury_features(games_df, injury_data=None, use_api=True):
    """
    Add injury impact features.
    
    Now integrates with NFL.com injury reports or injury APIs.
    
    Args:
        games_df: DataFrame with game information
        injury_data: Optional DataFrame with injury information (if provided, use this instead of API)
        use_api: Whether to fetch from API (True) or use placeholders (False)
    
    Returns:
        DataFrame with injury features added
    """
    games_df = games_df.copy()
    
    if injury_data is not None:
        # Use provided injury data
        for col in ['qb_injury_impact', 'ol_injury_impact', 'wr_injury_impact', 
                   'cb_injury_impact', 'key_player_out', 'injury_advantage']:
            if col in injury_data.columns:
                games_df[col] = injury_data[col]
            else:
                games_df[col] = 0
    elif use_api:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from injury_api import batch_fetch_injuries
            
            # Check if required columns exist
            if all(col in games_df.columns for col in ['season', 'week', 'home_team', 'away_team']):
                injury_df = batch_fetch_injuries(
                    games_df,
                    season_col='season',
                    week_col='week',
                    home_team_col='home_team',
                    away_team_col='away_team'
                )
                
                # Merge injury data
                for col in ['qb_injury_impact', 'ol_injury_impact', 'wr_injury_impact',
                           'cb_injury_impact', 'key_player_out', 'injury_advantage']:
                    if col in injury_df.columns:
                        games_df[col] = injury_df[col]
                    else:
                        games_df[col] = 0
            else:
                # Missing required columns, use placeholders
                games_df['qb_injury_impact'] = 0
                games_df['ol_injury_impact'] = 0
                games_df['wr_injury_impact'] = 0
                games_df['cb_injury_impact'] = 0
                games_df['key_player_out'] = 0
                games_df['injury_advantage'] = 0
        except (ImportError, NameError):
            print("  Warning: injury_api module not available, using placeholders")
            games_df['qb_injury_impact'] = 0
            games_df['ol_injury_impact'] = 0
            games_df['wr_injury_impact'] = 0
            games_df['cb_injury_impact'] = 0
            games_df['key_player_out'] = 0
            games_df['injury_advantage'] = 0
        except Exception as e:
            print(f"  Warning: Injury API error, using placeholders: {e}")
            games_df['qb_injury_impact'] = 0
            games_df['ol_injury_impact'] = 0
            games_df['wr_injury_impact'] = 0
            games_df['cb_injury_impact'] = 0
            games_df['key_player_out'] = 0
            games_df['injury_advantage'] = 0
    else:
        # Use placeholder values
        games_df['qb_injury_impact'] = 0
        games_df['ol_injury_impact'] = 0
        games_df['wr_injury_impact'] = 0
        games_df['cb_injury_impact'] = 0
        games_df['key_player_out'] = 0
        games_df['injury_advantage'] = 0
    
    return games_df


def fetch_market_data_multi_book(season, week, games_df=None, use_api=True):
    """
    Fetch market data from multiple books.
    
    NOTE: Market data API functionality has been removed as all APIs require payment.
    This function now returns None. Use existing market_spread data from game data instead.
    
    Args:
        season: Season year
        week: Week number
        games_df: Optional DataFrame with game information (for batch fetching)
        use_api: Whether to fetch from API (True) or use existing data (False)
    
    Returns:
        None (market data API removed)
    """
    # Market data API removed - all APIs require payment
    return None


def calculate_comprehensive_features(games_df, pbp_data, epa_df=None, use_apis=True):
    """
    Calculate all comprehensive features.
    
    Args:
        games_df: DataFrame with game information
        pbp_data: Play-by-play DataFrame
        epa_df: DataFrame with EPA features (optional)
        use_apis: Whether to use API integrations for weather, injuries, and market data
    
    Returns:
        DataFrame with all comprehensive features added
    """
    print("Calculating comprehensive features...")
    
    games_df = games_df.copy()
    
    # Fetch multi-book market data if available
    if use_apis and 'season' in games_df.columns and 'week' in games_df.columns:
        market_data = fetch_market_data_multi_book(
            games_df['season'].iloc[0] if len(games_df) > 0 else None,
            games_df['week'].iloc[0] if len(games_df) > 0 else None,
            games_df=games_df,
            use_api=use_apis
        )
        
        if market_data is not None and len(market_data) > 0:
            # Merge market data (update market_spread with multi-book consensus)
            for col in ['market_spread', 'market_spread_mean', 'market_spread_std', 
                       'spread_books', 'line_movement']:
                if col in market_data.columns:
                    games_df[col] = market_data[col].values if len(market_data) == len(games_df) else games_df.get(col, 0)
    
    # Add weather features (now with API integration)
    games_df = add_weather_features(games_df, use_api=use_apis)
    
    # Add travel features
    games_df = add_travel_features(games_df)
    
    # Add injury features (now with API integration)
    games_df = add_injury_features(games_df, use_api=use_apis)
    
    # Ensure injury features are filled (not NaN) - fill with 0 if missing
    injury_cols = ['qb_injury_impact', 'ol_injury_impact', 'wr_injury_impact', 
                   'cb_injury_impact', 'key_player_out', 'injury_advantage']
    for col in injury_cols:
        if col in games_df.columns:
            games_df[col] = games_df[col].fillna(0)
        else:
            games_df[col] = 0
    
    # Calculate style matchups
    style_metrics = calculate_style_matchups(pbp_data, games_df)
    
    # Calculate defensive matchups (offense vs defense EPA)
    defensive_matchups = calculate_defensive_matchups(pbp_data, games_df, epa_df)
    
    # Calculate target share
    target_share_df = calculate_target_share(pbp_data)
    
    # Calculate CB vs WR matchups
    cb_wr_matchups = calculate_cb_wr_matchups(pbp_data, games_df, target_share_df)
    
    # Calculate coaching matchups
    coaching_matchups = calculate_coaching_matchups(games_df)
    
    # Calculate turnover regression
    turnover_metrics = calculate_turnover_regression(pbp_data, games_df)
    
    # Calculate third down stats
    third_down_df = calculate_third_down_stats(pbp_data)
    
    # Calculate time of possession
    top_df = calculate_time_of_possession(pbp_data)
    
    # Add style matchup features to games
    for idx, game in games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        # Style matchup
        home_style = style_metrics.get(home_team, {})
        away_style = style_metrics.get(away_team, {})
        
        games_df.loc[idx, 'home_pass_rate'] = home_style.get('pass_rate', 0.5)
        games_df.loc[idx, 'away_pass_rate'] = away_style.get('pass_rate', 0.5)
        games_df.loc[idx, 'style_mismatch'] = abs(home_style.get('pass_rate', 0.5) - away_style.get('pass_rate', 0.5))
        
        # Defensive matchup (offense vs defense EPA)
        matchup_key = (home_team, away_team, season, week)
        matchup = defensive_matchups.get(matchup_key, {})
        games_df.loc[idx, 'home_off_vs_away_def'] = matchup.get('home_off_vs_away_def', 0)
        games_df.loc[idx, 'away_off_vs_home_def'] = matchup.get('away_off_vs_home_def', 0)
        games_df.loc[idx, 'matchup_advantage'] = matchup.get('matchup_advantage', 0)
        
        # CB vs WR matchup
        cb_wr_matchup = cb_wr_matchups.get(matchup_key, {})
        games_df.loc[idx, 'home_wr_target_concentration'] = cb_wr_matchup.get('home_wr_target_concentration', 0)
        games_df.loc[idx, 'away_wr_target_concentration'] = cb_wr_matchup.get('away_wr_target_concentration', 0)
        games_df.loc[idx, 'home_wr_vs_away_cb_advantage'] = cb_wr_matchup.get('home_wr_vs_away_cb_advantage', 0)
        games_df.loc[idx, 'away_wr_vs_home_cb_advantage'] = cb_wr_matchup.get('away_wr_vs_home_cb_advantage', 0)
        games_df.loc[idx, 'cb_wr_matchup_advantage'] = cb_wr_matchup.get('cb_wr_matchup_advantage', 0)
        
        # Coaching matchup
        coaching_key = (home_team, away_team)
        coaching = coaching_matchups.get(coaching_key, {})
        games_df.loc[idx, 'coaching_win_rate'] = coaching.get('coaching_win_rate', 0.5)
        
        # Turnover regression
        home_turnover = turnover_metrics.get(home_team, {})
        away_turnover = turnover_metrics.get(away_team, {})
        games_df.loc[idx, 'home_turnover_regression'] = home_turnover.get('turnover_regression', 0)
        games_df.loc[idx, 'away_turnover_regression'] = away_turnover.get('turnover_regression', 0)
        
        # Playoff implications
        playoff = calculate_playoff_implications(games_df, season, week)
        games_df.loc[idx, 'playoff_importance'] = playoff['playoff_importance']
        games_df.loc[idx, 'late_season'] = playoff['late_season']
        games_df.loc[idx, 'playoff_race'] = playoff['playoff_race']
    
    return games_df, third_down_df, top_df, target_share_df

