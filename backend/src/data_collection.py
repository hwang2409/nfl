"""
NFL Data Collection Module

Collects historical NFL game data using nfl_data_py and other sources.
"""

import pandas as pd
import nfl_data_py as nfl
from pathlib import Path
import pickle
from tqdm import tqdm
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_current_season():
    """Get the current NFL season year."""
    now = datetime.now()
    # NFL season starts in September, so if we're before September, use previous year
    if now.month < 9:
        return now.year - 1
    return now.year


def collect_game_data(start_year=None, end_year=None):
    """
    Collect play-by-play and game data for specified years.
    
    Args:
        start_year: First season to collect (default: 2018 for recent data)
        end_year: Last season to collect (default: current season)
    """
    # Default to recent data: last 6-7 seasons
    if end_year is None:
        end_year = get_current_season()
    if start_year is None:
        # Use last 6 seasons for recent data focus
        start_year = max(2018, end_year - 6)
    
    print(f"Collecting NFL data from {start_year} to {end_year}...")
    print(f"Focusing on recent data for better prediction accuracy")
    
    all_pbp = []
    all_games = []
    
    for year in tqdm(range(start_year, end_year + 1), desc="Collecting seasons"):
        try:
            # Get play-by-play data
            # nfl_data_py API: import_pbp_data(years) or import_pbp(years)
            try:
                if hasattr(nfl, 'import_pbp_data'):
                    pbp = nfl.import_pbp_data([year])
                elif hasattr(nfl, 'import_pbp'):
                    pbp = nfl.import_pbp([year])
                else:
                    # Try as a module function
                    pbp = nfl.import_pbp_data([year])
            except AttributeError:
                # If the above doesn't work, try alternative import
                pbp = None
                print(f"  Warning: Could not find pbp import function for {year}")
            
            if pbp is not None and len(pbp) > 0:
                all_pbp.append(pbp)
            
            # Get game data (schedules)
            # nfl_data_py API: import_schedules(years)
            try:
                if hasattr(nfl, 'import_schedules'):
                    games = nfl.import_schedules([year])
                elif hasattr(nfl, 'import_schedule'):
                    games = nfl.import_schedule([year])
                else:
                    games = nfl.import_schedules([year])
            except AttributeError:
                games = None
                print(f"  Warning: Could not find schedule import function for {year}")
            
            if games is not None and len(games) > 0:
                all_games.append(games)
            
            pbp_len = len(pbp) if pbp is not None else 0
            games_len = len(games) if games is not None else 0
            print(f"✓ Collected {year}: {pbp_len} plays, {games_len} games")
        except Exception as e:
            print(f"✗ Error collecting {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all years
    if all_pbp:
        combined_pbp = pd.concat(all_pbp, ignore_index=True)
        combined_pbp.to_parquet(DATA_DIR / "pbp_data.parquet", index=False)
        print(f"Saved play-by-play data: {len(combined_pbp)} plays")
    
    if all_games:
        combined_games = pd.concat(all_games, ignore_index=True)
        combined_games.to_parquet(DATA_DIR / "game_data.parquet", index=False)
        print(f"Saved game data: {len(combined_games)} games")
    
    return combined_pbp if all_pbp else None, combined_games if all_games else None


def load_game_data():
    """Load previously collected game data."""
    pbp_path = DATA_DIR / "pbp_data.parquet"
    games_path = DATA_DIR / "game_data.parquet"
    
    pbp = pd.read_parquet(pbp_path) if pbp_path.exists() else None
    games = pd.read_parquet(games_path) if games_path.exists() else None
    
    return pbp, games


def get_team_stats(pbp_data, games_data):
    """
    Calculate team-level statistics from play-by-play data.
    
    Args:
        pbp_data: Play-by-play DataFrame
        games_data: Games DataFrame
    
    Returns:
        DataFrame with team statistics per game
    """
    if pbp_data is None or len(pbp_data) == 0:
        print("No play-by-play data available")
        return None
    
    print("Calculating team statistics...")
    
    # Group by game and team
    team_stats = []
    
    for game_id in tqdm(pbp_data['game_id'].unique(), desc="Processing games"):
        game_pbp = pbp_data[pbp_data['game_id'] == game_id]
        
        for team in ['home_team', 'away_team']:
            team_name = game_pbp[team].iloc[0] if len(game_pbp) > 0 else None
            if team_name is None:
                continue
            
            # Filter plays for this team
            team_plays = game_pbp[
                (game_pbp['posteam'] == team_name) | 
                (game_pbp['defteam'] == team_name)
            ]
            
            if len(team_plays) == 0:
                continue
            
            # Offensive stats
            off_plays = team_plays[team_plays['posteam'] == team_name]
            def_plays = team_plays[team_plays['defteam'] == team_name]
            
            stats = {
                'game_id': game_id,
                'team': team_name,
                'is_home': team == 'home_team',
                'season': game_pbp['season'].iloc[0],
                'week': game_pbp['week'].iloc[0],
                
                # Offensive stats
                'off_points': off_plays['touchdown'].sum() * 6 + 
                             (off_plays['field_goal_result'] == 'made').sum() * 3,
                'off_total_yards': off_plays['yards_gained'].sum(),
                'off_pass_yards': off_plays[off_plays['play_type'] == 'pass']['yards_gained'].sum(),
                'off_rush_yards': off_plays[off_plays['play_type'] == 'run']['yards_gained'].sum(),
                'off_turnovers': (off_plays['interception'] == 1).sum() + 
                               (off_plays['fumble_lost'] == 1).sum(),
                'off_third_down_conversions': len(off_plays[
                    (off_plays['down'] == 3) & 
                    (off_plays['first_down'] == 1)
                ]),
                'off_third_down_attempts': len(off_plays[off_plays['down'] == 3]),
                
                # Defensive stats
                'def_points_allowed': def_plays['touchdown'].sum() * 6 + 
                                    (def_plays['field_goal_result'] == 'made').sum() * 3,
                'def_total_yards_allowed': def_plays['yards_gained'].sum(),
                'def_takeaways': (def_plays['interception'] == 1).sum() + 
                               (def_plays['fumble_lost'] == 1).sum(),
            }
            
            # Calculate rates
            stats['off_third_down_pct'] = (stats['off_third_down_conversions'] / 
                                          max(stats['off_third_down_attempts'], 1))
            stats['off_yards_per_play'] = stats['off_total_yards'] / max(len(off_plays), 1)
            
            team_stats.append(stats)
    
    team_stats_df = pd.DataFrame(team_stats)
    team_stats_df.to_parquet(DATA_DIR / "team_stats.parquet", index=False)
    print(f"Saved team statistics: {len(team_stats_df)} team-games")
    
    return team_stats_df


if __name__ == "__main__":
    # Collect most recent data (default: last 6-7 seasons)
    current_season = get_current_season()
    pbp, games = collect_game_data(start_year=max(2018, current_season - 6), end_year=current_season)
    
    # Calculate team stats
    if pbp is not None:
        team_stats = get_team_stats(pbp, games)

