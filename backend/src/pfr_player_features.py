"""
Pro Football Reference Player-Level Features

Identifies and tracks key players (QB, RB, WR, TE) for each team
and calculates player-level statistics from PFR data.
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

from pfr_integration import (
    DATA_DIR, TEAM_NAME_MAP, get_team_pfr_name,
    fetch_team_game_log, batch_fetch_team_logs
)

# Player database - maps (team, season, position) -> player name as it appears on PFR
# This can be expanded over time or built from historical data
PLAYER_DATABASE = {
    # Format: (team_abbr, season, position): 'Player Name'
    # Examples for 2024 season (can be expanded)
    ('KC', 2024, 'QB'): 'Patrick Mahomes',
    ('BUF', 2024, 'QB'): 'Josh Allen',
    ('BAL', 2024, 'QB'): 'Lamar Jackson',
    ('CIN', 2024, 'QB'): 'Joe Burrow',
    ('MIA', 2024, 'QB'): 'Tua Tagovailoa',
    ('DAL', 2024, 'QB'): 'Dak Prescott',
    ('PHI', 2024, 'QB'): 'Jalen Hurts',
    ('SF', 2024, 'QB'): 'Brock Purdy',
    ('DET', 2024, 'QB'): 'Jared Goff',
    ('GB', 2024, 'QB'): 'Jordan Love',
    ('HOU', 2024, 'QB'): 'C.J. Stroud',
    ('LAR', 2024, 'QB'): 'Matthew Stafford',
    ('TB', 2024, 'QB'): 'Baker Mayfield',
    ('PIT', 2024, 'QB'): 'Russell Wilson',
    ('ATL', 2024, 'QB'): 'Kirk Cousins',
    ('MIN', 2024, 'QB'): 'Sam Darnold',
    ('NO', 2024, 'QB'): 'Derek Carr',
    ('IND', 2024, 'QB'): 'Anthony Richardson',
    ('JAX', 2024, 'QB'): 'Trevor Lawrence',
    ('TEN', 2024, 'QB'): 'Will Levis',
    ('DEN', 2024, 'QB'): 'Bo Nix',
    ('LV', 2024, 'QB'): 'Gardner Minshew',
    ('LAC', 2024, 'QB'): 'Justin Herbert',
    ('NYJ', 2024, 'QB'): 'Aaron Rodgers',
    ('NE', 2024, 'QB'): 'Drake Maye',
    ('NYG', 2024, 'QB'): 'Daniel Jones',
    ('WAS', 2024, 'QB'): 'Jayden Daniels',
    ('CHI', 2024, 'QB'): 'Caleb Williams',
    ('CAR', 2024, 'QB'): 'Bryce Young',
    ('ARI', 2024, 'QB'): 'Kyler Murray',
    ('SEA', 2024, 'QB'): 'Geno Smith',
    ('CLE', 2024, 'QB'): 'Deshaun Watson',
    
    # Top RBs for 2024
    ('SF', 2024, 'RB'): 'Christian McCaffrey',
    ('BUF', 2024, 'RB'): 'James Cook',
    ('DET', 2024, 'RB'): 'Jahmyr Gibbs',
    ('BAL', 2024, 'RB'): 'Derrick Henry',
    ('MIA', 2024, 'RB'): 'De\'Von Achane',
    ('KC', 2024, 'RB'): 'Isiah Pacheco',
    ('DAL', 2024, 'RB'): 'Tony Pollard',
    ('PHI', 2024, 'RB'): 'Saquon Barkley',
    ('GB', 2024, 'RB'): 'Josh Jacobs',
    ('HOU', 2024, 'RB'): 'Joe Mixon',
    ('LAR', 2024, 'RB'): 'Kyren Williams',
    ('TB', 2024, 'RB'): 'Rachaad White',
    ('PIT', 2024, 'RB'): 'Najee Harris',
    ('ATL', 2024, 'RB'): 'Bijan Robinson',
    ('MIN', 2024, 'RB'): 'Aaron Jones',
    ('NO', 2024, 'RB'): 'Alvin Kamara',
    ('IND', 2024, 'RB'): 'Jonathan Taylor',
    ('JAX', 2024, 'RB'): 'Travis Etienne',
    ('TEN', 2024, 'RB'): 'Tony Pollard',
    ('DEN', 2024, 'RB'): 'Javonte Williams',
    ('LV', 2024, 'RB'): 'Zamir White',
    ('LAC', 2024, 'RB'): 'Gus Edwards',
    ('NYJ', 2024, 'RB'): 'Breece Hall',
    ('NE', 2024, 'RB'): 'Rhamondre Stevenson',
    ('NYG', 2024, 'RB'): 'Devin Singletary',
    ('WAS', 2024, 'RB'): 'Brian Robinson',
    ('CHI', 2024, 'RB'): 'D\'Andre Swift',
    ('CAR', 2024, 'RB'): 'Chuba Hubbard',
    ('ARI', 2024, 'RB'): 'James Conner',
    ('SEA', 2024, 'RB'): 'Kenneth Walker',
    ('CLE', 2024, 'RB'): 'Nick Chubb',
    
    # Top WRs for 2024
    ('MIA', 2024, 'WR'): 'Tyreek Hill',
    ('MIN', 2024, 'WR'): 'Justin Jefferson',
    ('SF', 2024, 'WR'): 'Deebo Samuel',
    ('CIN', 2024, 'WR'): 'Ja\'Marr Chase',
    ('DAL', 2024, 'WR'): 'CeeDee Lamb',
    ('PHI', 2024, 'WR'): 'A.J. Brown',
    ('DET', 2024, 'WR'): 'Amon-Ra St. Brown',
    ('HOU', 2024, 'WR'): 'Nico Collins',
    ('LAR', 2024, 'WR'): 'Cooper Kupp',
    ('TB', 2024, 'WR'): 'Mike Evans',
    ('PIT', 2024, 'WR'): 'George Pickens',
    ('ATL', 2024, 'WR'): 'Drake London',
    ('NO', 2024, 'WR'): 'Chris Olave',
    ('IND', 2024, 'WR'): 'Michael Pittman',
    ('JAX', 2024, 'WR'): 'Calvin Ridley',
    ('KC', 2024, 'WR'): 'Rashee Rice',
    ('BUF', 2024, 'WR'): 'Stefon Diggs',
    ('BAL', 2024, 'WR'): 'Zay Flowers',
    ('GB', 2024, 'WR'): 'Jayden Reed',
    ('TEN', 2024, 'WR'): 'DeAndre Hopkins',
    ('DEN', 2024, 'WR'): 'Courtland Sutton',
    ('LV', 2024, 'WR'): 'Davante Adams',
    ('LAC', 2024, 'WR'): 'Keenan Allen',
    ('NYJ', 2024, 'WR'): 'Garrett Wilson',
    ('NE', 2024, 'WR'): 'Kendrick Bourne',
    ('NYG', 2024, 'WR'): 'Malik Nabers',
    ('WAS', 2024, 'WR'): 'Terry McLaurin',
    ('CHI', 2024, 'WR'): 'D.J. Moore',
    ('CAR', 2024, 'WR'): 'Diontae Johnson',
    ('ARI', 2024, 'WR'): 'Marvin Harrison',
    ('SEA', 2024, 'WR'): 'DK Metcalf',
    ('CLE', 2024, 'WR'): 'Amari Cooper',
}


def normalize_player_name(name):
    """
    Normalize player name from nfl_data_py format to PFR format.
    
    nfl_data_py uses abbreviated format like "P.Mahomes"
    PFR uses full name like "Patrick Mahomes"
    
    Args:
        name: Player name in nfl_data_py format
    
    Returns:
        Player name in PFR format (or original if no match found)
    """
    if not name or pd.isna(name):
        return name
    
    # If already in full format (has space), return as-is
    if ' ' in str(name):
        return name
    
    # Try to match with manual database first (most reliable)
    for key, pfr_name in PLAYER_DATABASE.items():
        # Check if abbreviated name matches
        abbrev = name.replace('.', '')
        pfr_abbrev = ''.join([w[0] for w in pfr_name.split() if w])
        if abbrev.lower() == pfr_abbrev.lower():
            return pfr_name
    
    # If no match, try to expand common patterns
    # For now, return as-is and let PFR API handle it
    # The PFR scraper might be able to handle abbreviated names
    return name


def identify_key_players_from_pbp(pbp_data, teams, seasons):
    """
    Automatically identify key players from play-by-play data.
    
    Args:
        pbp_data: Play-by-play DataFrame
        teams: List of team abbreviations
        seasons: List of season years
    
    Returns:
        Dictionary mapping (team, season, position) -> player_name (in PFR format)
    """
    if pbp_data is None or len(pbp_data) == 0:
        return {}
    
    print("  Identifying key players from play-by-play data...")
    player_map = {}
    
    for season in seasons:
        season_pbp = pbp_data[pbp_data['season'] == season].copy()
        if len(season_pbp) == 0:
            continue
        
        for team in teams:
            team_pbp = season_pbp[season_pbp['posteam'] == team].copy()
            if len(team_pbp) == 0:
                continue
            
            # Identify QB (most pass attempts)
            if 'passer_player_name' in team_pbp.columns:
                qb_counts = team_pbp['passer_player_name'].value_counts()
                if len(qb_counts) > 0:
                    top_qb = qb_counts.index[0]
                    if pd.notna(top_qb) and top_qb:
                        normalized_name = normalize_player_name(top_qb)
                        player_map[(team, season, 'QB')] = normalized_name
            
            # Identify RB (most rush attempts)
            if 'rusher_player_name' in team_pbp.columns:
                rb_counts = team_pbp['rusher_player_name'].value_counts()
                if len(rb_counts) > 0:
                    top_rb = rb_counts.index[0]
                    if pd.notna(top_rb) and top_rb:
                        normalized_name = normalize_player_name(top_rb)
                        player_map[(team, season, 'RB')] = normalized_name
            
            # Identify WR (most targets/receptions)
            if 'receiver_player_name' in team_pbp.columns:
                # Filter to actual receptions/targets (not just any play)
                receiving_plays = team_pbp[
                    (team_pbp['play_type'] == 'pass') & 
                    (team_pbp['receiver_player_name'].notna())
                ]
                if len(receiving_plays) > 0:
                    wr_counts = receiving_plays['receiver_player_name'].value_counts()
                    if len(wr_counts) > 0:
                        top_wr = wr_counts.index[0]
                        if pd.notna(top_wr) and top_wr:
                            normalized_name = normalize_player_name(top_wr)
                            player_map[(team, season, 'WR')] = normalized_name
    
    return player_map


def build_player_database(pbp_data=None, teams=None, seasons=None, use_cache=True):
    """
    Build player database automatically from play-by-play data.
    
    Args:
        pbp_data: Play-by-play DataFrame (if None, will load from data_collection)
        teams: List of team abbreviations (if None, will use all teams)
        seasons: List of season years (if None, will use available seasons)
        use_cache: Whether to use cached database
    
    Returns:
        Dictionary mapping (team, season, position) -> player_name
    """
    # Check cache
    cache_file = DATA_DIR / "pfr" / "player_database.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if use_cache and cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            # Convert DataFrame back to dictionary
            player_db = {}
            for _, row in cached_df.iterrows():
                key = (row['team'], row['season'], row['position'])
                player_db[key] = row['player_name']
            print(f"  Loaded {len(player_db)} players from cache")
            return player_db
        except:
            pass
    
    # Load data if not provided
    if pbp_data is None:
        try:
            from data_collection import load_game_data
            pbp_data, _ = load_game_data()
        except:
            pbp_data = None
    
    if pbp_data is None or len(pbp_data) == 0:
        print("  No play-by-play data available, using manual database")
        return PLAYER_DATABASE.copy()
    
    # Get teams and seasons if not provided
    if teams is None:
        teams = list(TEAM_NAME_MAP.keys())
    
    if seasons is None:
        if 'season' in pbp_data.columns:
            seasons = sorted(pbp_data['season'].unique().tolist())
        else:
            seasons = [2024]  # Default
    
    # Start with manual database as fallback
    player_db = PLAYER_DATABASE.copy()
    
    # Identify players from PBP data
    auto_players = identify_key_players_from_pbp(pbp_data, teams, seasons)
    
    # Merge auto-identified players (they override manual entries)
    player_db.update(auto_players)
    
    # Save to cache
    try:
        db_list = [{'team': k[0], 'season': k[1], 'position': k[2], 'player_name': v} 
                   for k, v in player_db.items()]
        db_df = pd.DataFrame(db_list)
        db_df.to_parquet(cache_file, index=False)
        print(f"  Cached {len(player_db)} players to database")
    except Exception as e:
        print(f"  Warning: Could not cache player database: {e}")
    
    return player_db


def get_player_name(team_abbr, season, position, pbp_data=None, player_db=None):
    """
    Get player name for a team/season/position.
    
    Uses automatic identification from play-by-play data, with fallback to manual database.
    
    Args:
        team_abbr: Team abbreviation
        season: Season year
        position: 'QB', 'RB', 'WR', or 'TE'
        pbp_data: Optional play-by-play data (for auto-identification)
        player_db: Optional pre-built player database
    
    Returns:
        Player name as it appears on PFR, or None if not found
    """
    # Use provided database or build one
    if player_db is None:
        player_db = build_player_database(pbp_data=pbp_data, use_cache=True)
    
    # Try exact match first
    key = (team_abbr, season, position)
    if key in player_db:
        return player_db[key]
    
    # Try previous season (players often stay on same team)
    key_prev = (team_abbr, season - 1, position)
    if key_prev in player_db:
        return player_db[key_prev]
    
    # Try next season (for early season predictions)
    key_next = (team_abbr, season + 1, position)
    if key_next in player_db:
        return player_db[key_next]
    
    return None


def fetch_player_game_log_cached(player_name, position, season, use_cache=True):
    """
    Fetch player game log with caching.
    
    Args:
        player_name: Player name as it appears on PFR
        position: 'QB', 'RB', 'WR', or 'TE'
        season: Season year
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with player game log or None if failed
    """
    if not PFR_AVAILABLE or not player_name:
        return None
    
    # Check cache
    safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
    cache_file = DATA_DIR / "pfr" / "players" / f"{safe_name}_{position}_{season}.parquet"
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
            # Save to cache
            game_log.to_parquet(cache_file, index=False)
            return game_log
    except Exception as e:
        # Silently fail - many players won't exist or API may have issues
        return None
    
    return None


def calculate_qb_features_from_pfr(player_log, week):
    """
    Calculate QB features from PFR player game log.
    
    Args:
        player_log: DataFrame with QB game log
        week: Current week (features calculated before this week)
    
    Returns:
        Dictionary of QB features
    """
    if player_log is None or len(player_log) == 0:
        return {}
    
    # Filter to games before this week
    if 'week' in player_log.columns:
        try:
            player_log['week'] = pd.to_numeric(player_log['week'], errors='coerce')
            recent_games = player_log[player_log['week'] < week].tail(5)
        except:
            recent_games = player_log.tail(5)
    else:
        recent_games = player_log.tail(5)
    
    if len(recent_games) == 0:
        return {}
    
    features = {}
    
    # QB-specific columns from PFR
    if 'pass_yds' in recent_games.columns:
        features['qb_pfr_pass_yds_avg'] = pd.to_numeric(recent_games['pass_yds'], errors='coerce').mean()
    
    if 'pass_td' in recent_games.columns:
        features['qb_pfr_pass_td_avg'] = pd.to_numeric(recent_games['pass_td'], errors='coerce').mean()
    
    if 'pass_int' in recent_games.columns:
        features['qb_pfr_int_avg'] = pd.to_numeric(recent_games['pass_int'], errors='coerce').mean()
    
    if 'pass_att' in recent_games.columns:
        att = pd.to_numeric(recent_games['pass_att'], errors='coerce')
        comp = pd.to_numeric(recent_games.get('pass_cmp', att), errors='coerce')
        features['qb_pfr_comp_pct'] = (comp / att.replace(0, 1)).mean() if att.sum() > 0 else 0
    
    if 'pass_yds' in recent_games.columns and 'pass_att' in recent_games.columns:
        yds = pd.to_numeric(recent_games['pass_yds'], errors='coerce')
        att = pd.to_numeric(recent_games['pass_att'], errors='coerce')
        features['qb_pfr_yds_per_att'] = (yds / att.replace(0, 1)).mean() if att.sum() > 0 else 0
    
    # Fill NaN values
    features = {k: (v if not pd.isna(v) else 0) for k, v in features.items()}
    
    return features


def calculate_rb_features_from_pfr(player_log, week):
    """
    Calculate RB features from PFR player game log.
    
    Args:
        player_log: DataFrame with RB game log
        week: Current week (features calculated before this week)
    
    Returns:
        Dictionary of RB features
    """
    if player_log is None or len(player_log) == 0:
        return {}
    
    # Filter to games before this week
    if 'week' in player_log.columns:
        try:
            player_log['week'] = pd.to_numeric(player_log['week'], errors='coerce')
            recent_games = player_log[player_log['week'] < week].tail(5)
        except:
            recent_games = player_log.tail(5)
    else:
        recent_games = player_log.tail(5)
    
    if len(recent_games) == 0:
        return {}
    
    features = {}
    
    # RB-specific columns
    if 'rush_yds' in recent_games.columns:
        features['rb_pfr_rush_yds_avg'] = pd.to_numeric(recent_games['rush_yds'], errors='coerce').mean()
    
    if 'rush_td' in recent_games.columns:
        features['rb_pfr_rush_td_avg'] = pd.to_numeric(recent_games['rush_td'], errors='coerce').mean()
    
    if 'rush_att' in recent_games.columns:
        att = pd.to_numeric(recent_games['rush_att'], errors='coerce')
        yds = pd.to_numeric(recent_games.get('rush_yds', 0), errors='coerce')
        features['rb_pfr_yds_per_carry'] = (yds / att.replace(0, 1)).mean() if att.sum() > 0 else 0
    
    if 'rec_yds' in recent_games.columns:
        features['rb_pfr_rec_yds_avg'] = pd.to_numeric(recent_games['rec_yds'], errors='coerce').mean()
    
    if 'rec_td' in recent_games.columns:
        features['rb_pfr_rec_td_avg'] = pd.to_numeric(recent_games['rec_td'], errors='coerce').mean()
    
    # Fill NaN values
    features = {k: (v if not pd.isna(v) else 0) for k, v in features.items()}
    
    return features


def calculate_wr_features_from_pfr(player_log, week):
    """
    Calculate WR features from PFR player game log.
    
    Args:
        player_log: DataFrame with WR game log
        week: Current week (features calculated before this week)
    
    Returns:
        Dictionary of WR features
    """
    if player_log is None or len(player_log) == 0:
        return {}
    
    # Filter to games before this week
    if 'week' in player_log.columns:
        try:
            player_log['week'] = pd.to_numeric(player_log['week'], errors='coerce')
            recent_games = player_log[player_log['week'] < week].tail(5)
        except:
            recent_games = player_log.tail(5)
    else:
        recent_games = player_log.tail(5)
    
    if len(recent_games) == 0:
        return {}
    
    features = {}
    
    # WR-specific columns
    if 'rec_yds' in recent_games.columns:
        features['wr_pfr_rec_yds_avg'] = pd.to_numeric(recent_games['rec_yds'], errors='coerce').mean()
    
    if 'rec_td' in recent_games.columns:
        features['wr_pfr_rec_td_avg'] = pd.to_numeric(recent_games['rec_td'], errors='coerce').mean()
    
    if 'rec' in recent_games.columns:
        features['wr_pfr_receptions_avg'] = pd.to_numeric(recent_games['rec'], errors='coerce').mean()
    
    if 'targets' in recent_games.columns:
        targets = pd.to_numeric(recent_games['targets'], errors='coerce')
        rec = pd.to_numeric(recent_games.get('rec', 0), errors='coerce')
        features['wr_pfr_catch_rate'] = (rec / targets.replace(0, 1)).mean() if targets.sum() > 0 else 0
    
    if 'rec_yds' in recent_games.columns and 'rec' in recent_games.columns:
        yds = pd.to_numeric(recent_games['rec_yds'], errors='coerce')
        rec = pd.to_numeric(recent_games['rec'], errors='coerce')
        features['wr_pfr_yds_per_rec'] = (yds / rec.replace(0, 1)).mean() if rec.sum() > 0 else 0
    
    # Fill NaN values
    features = {k: (v if not pd.isna(v) else 0) for k, v in features.items()}
    
    return features


def calculate_te_features_from_pfr(player_log, week):
    """
    Calculate TE features from PFR player game log.
    Similar to WR but for tight ends.
    
    Args:
        player_log: DataFrame with TE game log
        week: Current week (features calculated before this week)
    
    Returns:
        Dictionary of TE features
    """
    # TEs use similar stats to WRs
    return calculate_wr_features_from_pfr(player_log, week)


def calculate_player_features_for_games(games_df, pbp_data=None, min_season=None, max_season=None):
    """
    Calculate player-level features for all games.
    
    Args:
        games_df: DataFrame with game information
        pbp_data: Optional play-by-play data (for auto-identifying players)
        min_season: Minimum season to fetch
        max_season: Maximum season to fetch
    
    Returns:
        DataFrame with player features added
    """
    if not PFR_AVAILABLE:
        print("  PFR not available, skipping player features")
        return games_df
    
    if games_df is None or len(games_df) == 0:
        return games_df
    
    print("Calculating PFR player-level features...")
    
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
    
    # Build player database automatically
    player_db = build_player_database(pbp_data=pbp_data, teams=teams, seasons=seasons, use_cache=True)
    
    # Fetch player logs for key positions
    player_logs = {}  # (team, season, position) -> DataFrame
    
    positions = ['QB', 'RB', 'WR', 'TE']
    to_fetch = []
    
    for team in teams:
        for season in seasons:
            for position in positions:
                player_name = get_player_name(team, season, position, pbp_data=pbp_data, player_db=player_db)
                if player_name:
                    cache_key = (team, season, position, player_name)
                    safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
                    cache_file = DATA_DIR / "pfr" / "players" / f"{safe_name}_{position}_{season}.parquet"
                    
                    if cache_file.exists():
                        try:
                            player_logs[(team, season, position)] = pd.read_parquet(cache_file)
                        except:
                            to_fetch.append((team, season, position, player_name))
                    else:
                        to_fetch.append((team, season, position, player_name))
    
    # Fetch uncached player logs
    if to_fetch:
        print(f"  Fetching {len(to_fetch)} player logs from PFR...")
        for team, season, position, player_name in tqdm(to_fetch, desc="  PFR Player Logs"):
            log = fetch_player_game_log_cached(player_name, position, season, use_cache=False)
            if log is not None:
                player_logs[(team, season, position)] = log
            # Rate limiting
            time.sleep(0.5)
    
    # Calculate features for each game
    player_features_list = []
    
    for idx, game in games_df.iterrows():
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        season = game.get('season')
        week = game.get('week')
        
        if pd.isna(home_team) or pd.isna(away_team) or pd.isna(season) or pd.isna(week):
            player_features_list.append({})
            continue
        
        game_features = {}
        
        # Calculate features for each position
        for position in positions:
            # Home team
            home_key = (home_team, season, position)
            home_log = player_logs.get(home_key)
            if home_log is None and season > min_season:
                # Try previous season
                home_log = player_logs.get((home_team, season - 1, position))
            
            # Away team
            away_key = (away_team, season, position)
            away_log = player_logs.get(away_key)
            if away_log is None and season > min_season:
                # Try previous season
                away_log = player_logs.get((away_team, season - 1, position))
            
            # Calculate position-specific features
            if position == 'QB':
                home_features = calculate_qb_features_from_pfr(home_log, week)
                away_features = calculate_qb_features_from_pfr(away_log, week)
            elif position == 'RB':
                home_features = calculate_rb_features_from_pfr(home_log, week)
                away_features = calculate_rb_features_from_pfr(away_log, week)
            elif position == 'WR':
                home_features = calculate_wr_features_from_pfr(home_log, week)
                away_features = calculate_wr_features_from_pfr(away_log, week)
            elif position == 'TE':
                home_features = calculate_te_features_from_pfr(home_log, week)
                away_features = calculate_te_features_from_pfr(away_log, week)
            else:
                home_features = {}
                away_features = {}
            
            # Add home/away features
            for key, value in home_features.items():
                game_features[f'home_{key}'] = value
            
            for key, value in away_features.items():
                game_features[f'away_{key}'] = value
            
            # Add difference features
            for key in home_features.keys():
                home_key = f'home_{key}'
                away_key = f'away_{key}'
                if home_key in game_features and away_key in game_features:
                    game_features[f'{key}_diff'] = game_features[home_key] - game_features[away_key]
        
        player_features_list.append(game_features)
    
    # Add features to games DataFrame
    player_features_df = pd.DataFrame(player_features_list, index=games_df.index)
    
    # Merge with games DataFrame
    for col in player_features_df.columns:
        games_df[col] = player_features_df[col]
    
    # Fill missing values
    player_cols = [col for col in player_features_df.columns]
    for col in player_cols:
        if col in games_df.columns:
            games_df[col] = games_df[col].fillna(0)
    
    print(f"  Added {len(player_cols)} player-level features")
    
    return games_df

