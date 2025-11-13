"""
Pro Football Reference Integration

Fetches and processes data from Pro Football Reference using the
pro-football-reference-web-scraper package.

Provides:
- Team game logs with detailed statistics
- Player game logs for key positions (QB, RB, WR, TE)
- Advanced statistics and metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from pro_football_reference_web_scraper import player_game_log as pfr_player
    from pro_football_reference_web_scraper import team_game_log as pfr_team
    PFR_AVAILABLE = True
except ImportError:
    PFR_AVAILABLE = False
    print("Warning: pro-football-reference-web-scraper not available. Install with: pip install pro-football-reference-web-scraper")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Team name mapping from nflverse abbreviations to PFR full names
TEAM_NAME_MAP = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LV': 'Las Vegas Raiders',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders'
}

# Reverse mapping
PFR_TO_ABBR = {v: k for k, v in TEAM_NAME_MAP.items()}


def get_team_pfr_name(team_abbr):
    """Convert team abbreviation to PFR full name."""
    return TEAM_NAME_MAP.get(team_abbr, None)


def get_team_abbr_from_pfr(pfr_name):
    """Convert PFR full name to team abbreviation."""
    return PFR_TO_ABBR.get(pfr_name, None)


def fetch_team_game_log(team_abbr, season, use_cache=True):
    """
    Fetch team game log from Pro Football Reference.
    
    Args:
        team_abbr: Team abbreviation (e.g., 'KC')
        season: Season year
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with team game log or None if failed
    """
    if not PFR_AVAILABLE:
        return None
    
    team_name = get_team_pfr_name(team_abbr)
    if not team_name:
        return None
    
    # Validate season - don't try to fetch future seasons or very old seasons
    from data_collection import get_current_season
    from datetime import datetime
    current_season = get_current_season()
    current_month = datetime.now().month
    
    # Be conservative: only fetch seasons that are definitely complete or in progress
    max_valid_season = current_season
    if current_month < 9:
        # Before September, current season hasn't started
        max_valid_season = current_season - 1
    
    if season > max_valid_season:
        # Future season - data won't be available yet
        return None
    if season < 2000:
        # Very old seasons may not be available
        return None
    
    # Check cache
    cache_file = DATA_DIR / "pfr" / f"team_{team_abbr}_{season}.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if use_cache and cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except:
            pass
    
    try:
        # Fetch from PFR
        game_log = pfr_team.get_team_game_log(team=team_name, season=season)
        
        if game_log is not None and len(game_log) > 0:
            # Validate the data - check for empty/invalid values
            # The PFR scraper sometimes returns DataFrames with empty strings
            # that cause issues when converted to numeric types
            if isinstance(game_log, pd.DataFrame):
                # Replace empty strings with NaN
                game_log = game_log.replace('', np.nan)
                # Drop rows that are completely empty
                game_log = game_log.dropna(how='all')
            
            if len(game_log) > 0:
                # Save to cache
                game_log.to_parquet(cache_file, index=False)
                return game_log
    except ValueError as e:
        # Handle specific ValueError for int conversion (empty string to int)
        if "invalid literal for int()" in str(e):
            # Silently skip - this happens when PFR data isn't ready yet
            return None
        else:
            # Silently skip other ValueError - likely data parsing issue
            return None
    except (IndexError, KeyError) as e:
        # Handle "list index out of range" and similar errors
        # This happens when PFR page structure is different (season hasn't started)
        # or data is incomplete
        return None
    except Exception as e:
        # Check if it's a data parsing error
        error_str = str(e).lower()
        if ("invalid literal" in error_str or "empty" in error_str or 
            "cannot convert" in error_str or "index out of range" in error_str or
            "list index" in error_str):
            # Data format issue - likely season hasn't started or data incomplete
            return None
        # Only print warning for unexpected errors
        if "warning" not in error_str.lower():
            print(f"  Warning: Failed to fetch PFR team data for {team_abbr} {season}: {e}")
        return None
    
    return None


def fetch_player_game_log(player_name, position, season, use_cache=True):
    """
    Fetch player game log from Pro Football Reference.
    
    Args:
        player_name: Player full name as it appears on PFR
        position: 'QB', 'RB', 'WR', or 'TE'
        season: Season year
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with player game log or None if failed
    """
    if not PFR_AVAILABLE:
        return None
    
    if position not in ['QB', 'RB', 'WR', 'TE']:
        return None
    
    # Validate season - don't try to fetch future seasons
    from data_collection import get_current_season
    from datetime import datetime
    current_season = get_current_season()
    current_month = datetime.now().month
    
    # Be conservative: only fetch seasons that are definitely complete or in progress
    max_valid_season = current_season
    if current_month < 9:
        # Before September, current season hasn't started
        max_valid_season = current_season - 1
    
    if season > max_valid_season:
        # Future season - data won't be available yet
        return None
    if season < 2000:
        # Very old seasons may not be available
        return None
    
    # Check cache
    cache_file = DATA_DIR / "pfr" / f"player_{player_name.replace(' ', '_')}_{position}_{season}.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if use_cache and cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except:
            pass
    
    try:
        # Fetch from PFR
        game_log = pfr_player.get_player_game_log(player=player_name, position=position, season=season)
        
        if game_log is not None and len(game_log) > 0:
            # Validate the data - check for empty/invalid values
            if isinstance(game_log, pd.DataFrame):
                # Replace empty strings with NaN
                game_log = game_log.replace('', np.nan)
                # Drop rows that are completely empty
                game_log = game_log.dropna(how='all')
            
            if len(game_log) > 0:
                # Save to cache
                game_log.to_parquet(cache_file, index=False)
                return game_log
    except ValueError as e:
        # Handle specific ValueError for int conversion (empty string to int)
        if "invalid literal for int()" in str(e):
            # Silently skip - this happens when PFR data isn't ready yet
            return None
        # Silently fail for player data (many players won't exist or API may have issues)
        return None
    except Exception as e:
        # Check if it's a data parsing error
        error_str = str(e).lower()
        if "invalid literal" in error_str or "empty" in error_str or "cannot convert" in error_str:
            # Data format issue - likely season hasn't started or data incomplete
            return None
        # Silently fail for player data (many players won't exist)
        return None
    
    return None


def batch_fetch_team_logs(teams, seasons, use_cache=True):
    """
    Batch fetch team game logs for multiple teams and seasons.
    
    Args:
        teams: List of team abbreviations
        seasons: List of season years
        use_cache: Whether to use cached data
    
    Returns:
        Dictionary mapping (team, season) -> DataFrame
    """
    if not PFR_AVAILABLE:
        return {}
    
    # Filter out future seasons and seasons that likely don't have data yet
    from data_collection import get_current_season
    from datetime import datetime
    current_season = get_current_season()
    current_month = datetime.now().month
    
    # NFL season starts in September, so if we're in the current year but before September,
    # the season hasn't started yet. Also, if we're in the current season but it's very early,
    # PFR data might not be available yet (scraper can fail on incomplete seasons).
    # Be conservative: only fetch seasons that are definitely complete or well into the season
    # For the current season, only include it if we're past mid-September (week 2-3)
    current_day = datetime.now().day
    max_valid_season = current_season
    if current_month < 9:
        # Before September, current season hasn't started
        max_valid_season = current_season - 1
    elif current_month == 9 and current_day < 20:
        # Very early in September (before week 2-3), data might not be ready
        # Be conservative and skip current season to avoid scraper errors
        max_valid_season = current_season - 1
    
    valid_seasons = [s for s in seasons if s <= max_valid_season and s >= 2000]
    
    if len(valid_seasons) == 0:
        return {}
    
    print(f"Fetching PFR team data for {len(teams)} teams across {len(valid_seasons)} seasons...")
    if len(valid_seasons) < len(seasons):
        skipped = len(seasons) - len(valid_seasons)
        print(f"  Skipping {skipped} future/invalid seasons (max valid: {max_valid_season})")
    
    team_logs = {}
    to_fetch = []
    
    # First pass: Check cache
    for team in teams:
        for season in valid_seasons:
            cache_file = DATA_DIR / "pfr" / f"team_{team}_{season}.parquet"
            if use_cache and cache_file.exists():
                try:
                    team_logs[(team, season)] = pd.read_parquet(cache_file)
                except:
                    to_fetch.append((team, season))
            else:
                to_fetch.append((team, season))
    
    # Second pass: Fetch uncached
    if to_fetch:
        print(f"  Fetching {len(to_fetch)} team logs from PFR...")
        for team, season in tqdm(to_fetch, desc="  PFR Team Logs"):
            log = fetch_team_game_log(team, season, use_cache=False)
            if log is not None:
                team_logs[(team, season)] = log
            # Rate limiting - be nice to PFR
            time.sleep(0.5)
    
    print(f"  Fetched {len(team_logs)} team logs")
    return team_logs


def calculate_pfr_team_features(team_logs, team_abbr, season, week):
    """
    Calculate team features from PFR game logs.
    
    Args:
        team_logs: Dictionary of (team, season) -> DataFrame
        team_abbr: Team abbreviation
        season: Season year
        week: Week number (features calculated before this week)
    
    Returns:
        Dictionary of team features
    """
    if not team_logs:
        return {}
    
    # Get team log for this season
    team_log = team_logs.get((team_abbr, season))
    
    if team_log is None or len(team_log) == 0:
        # Try previous season
        team_log = team_logs.get((team_abbr, season - 1))
    
    if team_log is None or len(team_log) == 0:
        return {}
    
    # Filter to games before this week
    # PFR game logs use 'week' column
    if 'week' in team_log.columns:
        try:
            team_log['week'] = pd.to_numeric(team_log['week'], errors='coerce')
            recent_games = team_log[team_log['week'] < week].tail(5)
        except:
            recent_games = team_log.tail(5)
    else:
        # No week column, use last 5 games
        recent_games = team_log.tail(5)
    
    if len(recent_games) == 0:
        return {}
    
    features = {}
    
    # PFR columns (using snake_case from the package)
    # Offensive stats
    if 'pass_yds' in recent_games.columns:
        features['pfr_pass_yds_avg'] = pd.to_numeric(recent_games['pass_yds'], errors='coerce').mean()
    
    if 'rush_yds' in recent_games.columns:
        features['pfr_rush_yds_avg'] = pd.to_numeric(recent_games['rush_yds'], errors='coerce').mean()
    
    if 'tot_yds' in recent_games.columns:
        features['pfr_total_yds_avg'] = pd.to_numeric(recent_games['tot_yds'], errors='coerce').mean()
    
    # Points
    if 'points_for' in recent_games.columns:
        features['pfr_points_avg'] = pd.to_numeric(recent_games['points_for'], errors='coerce').mean()
    
    # Opponent stats (defensive)
    if 'points_allowed' in recent_games.columns:
        features['pfr_opp_points_avg'] = pd.to_numeric(recent_games['points_allowed'], errors='coerce').mean()
    
    if 'opp_tot_yds' in recent_games.columns:
        features['pfr_opp_total_yds_avg'] = pd.to_numeric(recent_games['opp_tot_yds'], errors='coerce').mean()
    
    if 'opp_pass_yds' in recent_games.columns:
        features['pfr_opp_pass_yds_avg'] = pd.to_numeric(recent_games['opp_pass_yds'], errors='coerce').mean()
    
    if 'opp_rush_yds' in recent_games.columns:
        features['pfr_opp_rush_yds_avg'] = pd.to_numeric(recent_games['opp_rush_yds'], errors='coerce').mean()
    
    # Fill NaN values with 0
    features = {k: (v if not pd.isna(v) else 0) for k, v in features.items()}
    
    return features


def calculate_pfr_features_for_games(games_df, min_season=None, max_season=None):
    """
    Calculate PFR features for all games in the DataFrame.
    
    Args:
        games_df: DataFrame with game information
        min_season: Minimum season to fetch
        max_season: Maximum season to fetch
    
    Returns:
        DataFrame with PFR features added
    """
    if not PFR_AVAILABLE:
        print("  PFR not available, skipping PFR features")
        return games_df
    
    if games_df is None or len(games_df) == 0:
        return games_df
    
    print("Calculating PFR features...")
    
    # Get unique teams and seasons
    if min_season is None:
        min_season = games_df['season'].min() if 'season' in games_df.columns else 2020
    if max_season is None:
        max_season = games_df['season'].max() if 'season' in games_df.columns else 2024
    
    teams = set()
    if 'home_team' in games_df.columns:
        teams.update(games_df['home_team'].dropna().unique())
    if 'away_team' in games_df.columns:
        teams.update(games_df['away_team'].dropna().unique())
    
    teams = [t for t in teams if t in TEAM_NAME_MAP]
    seasons = list(range(min_season, max_season + 1))
    
    # Fetch team logs
    team_logs = batch_fetch_team_logs(teams, seasons, use_cache=True)
    
    # Calculate features for each game
    pfr_features_list = []
    
    for idx, game in games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team) or pd.isna(season) or pd.isna(week):
            pfr_features_list.append({})
            continue
        
        # Get features for both teams
        home_features = calculate_pfr_team_features(team_logs, home_team, season, week)
        away_features = calculate_pfr_team_features(team_logs, away_team, season, week)
        
        # Create difference features
        game_features = {}
        
        # Home team features (prefixed with home_)
        for key, value in home_features.items():
            game_features[f'home_{key}'] = value
        
        # Away team features (prefixed with away_)
        for key, value in away_features.items():
            game_features[f'away_{key}'] = value
        
        # Difference features
        for key in home_features.keys():
            home_key = f'home_{key}'
            away_key = f'away_{key}'
            if home_key in game_features and away_key in game_features:
                game_features[f'{key}_diff'] = game_features[home_key] - game_features[away_key]
        
        pfr_features_list.append(game_features)
    
    # Add features to games DataFrame
    pfr_features_df = pd.DataFrame(pfr_features_list, index=games_df.index)
    
    # Merge with games DataFrame
    for col in pfr_features_df.columns:
        games_df[col] = pfr_features_df[col]
    
    # Fill missing values
    pfr_cols = [col for col in pfr_features_df.columns]
    for col in pfr_cols:
        if col in games_df.columns:
            games_df[col] = games_df[col].fillna(0)
    
    print(f"  Added {len(pfr_cols)} PFR features")
    
    return games_df

