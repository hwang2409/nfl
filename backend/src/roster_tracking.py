"""
Roster Change Tracking Module

Tracks NFL roster changes over time by comparing snapshots of team rosters.
Detects player additions, releases, trades, and position changes.

Uses multiple data sources:
1. nfl_data_py roster data (if available)
2. Play-by-play data to identify active players
3. Pro Football Reference for player information
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    NFL_DATA_AVAILABLE = False
    print("Warning: nfl_data_py not available. Install with: pip install nfl_data_py")

from pfr_integration import TEAM_NAME_MAP, DATA_DIR
from data_collection import load_game_data, get_current_season

# Key positions to track
KEY_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S', 'K']

# Roster change types
CHANGE_TYPES = {
    'ADDED': 'Player added to roster',
    'REMOVED': 'Player removed from roster',
    'TRADED': 'Player traded to different team',
    'POSITION_CHANGE': 'Player position changed',
    'ACTIVE': 'Player became active',
    'INACTIVE': 'Player became inactive',
}


def fetch_rosters_nfl_data_py(season: int, week: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Fetch roster data from nfl_data_py.
    
    Args:
        season: Season year
        week: Week number (optional, for weekly rosters)
    
    Returns:
        DataFrame with roster data or None if failed
    """
    if not NFL_DATA_AVAILABLE:
        return None
    
    try:
        # Try to import rosters
        if hasattr(nfl, 'import_rosters'):
            rosters = nfl.import_rosters([season])
        elif hasattr(nfl, 'import_roster'):
            rosters = nfl.import_roster([season])
        else:
            # Try alternative: use depth charts or player data
            try:
                # nfl_data_py might have depth charts
                if hasattr(nfl, 'import_depth_charts'):
                    rosters = nfl.import_depth_charts([season])
                else:
                    return None
            except:
                return None
        
        if rosters is None or len(rosters) == 0:
            return None
        
        # Standardize column names
        # Expected columns: player_name, team, position, season, week (optional)
        if 'team' not in rosters.columns and 'team_abbr' in rosters.columns:
            rosters['team'] = rosters['team_abbr']
        
        if week is not None and 'week' in rosters.columns:
            rosters = rosters[rosters['week'] == week]
        
        return rosters
    
    except Exception as e:
        print(f"  Warning: Could not fetch rosters from nfl_data_py: {e}")
        return None


def build_roster_from_pbp(pbp_data: pd.DataFrame, teams: List[str], 
                          season: int, week: Optional[int] = None) -> Dict[Tuple[str, str], str]:
    """
    Build roster snapshot from play-by-play data.
    
    Identifies active players by their participation in games.
    
    Args:
        pbp_data: Play-by-play DataFrame
        teams: List of team abbreviations
        season: Season year
        week: Week number (optional, for weekly rosters)
    
    Returns:
        Dictionary mapping (team, position) -> player_name
    """
    if pbp_data is None or len(pbp_data) == 0:
        return {}
    
    # Filter by season and week
    season_pbp = pbp_data[pbp_data['season'] == season].copy()
    if week is not None and 'week' in season_pbp.columns:
        season_pbp = season_pbp[season_pbp['week'] == week]
    
    roster = {}
    
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
                    roster[(team, 'QB')] = str(top_qb).strip()
        
        # Identify RB (most rush attempts)
        if 'rusher_player_name' in team_pbp.columns:
            rb_counts = team_pbp['rusher_player_name'].value_counts()
            if len(rb_counts) > 0:
                top_rb = rb_counts.index[0]
                if pd.notna(top_rb) and top_rb:
                    roster[(team, 'RB')] = str(top_rb).strip()
        
        # Identify WR (most receptions/targets)
        if 'receiver_player_name' in team_pbp.columns:
            wr_counts = team_pbp[
                (team_pbp['receiver_player_name'].notna()) &
                (team_pbp['play_type'] == 'pass')
            ]['receiver_player_name'].value_counts()
            if len(wr_counts) > 0:
                top_wr = wr_counts.index[0]
                if pd.notna(top_wr) and top_wr:
                    roster[(team, 'WR')] = str(top_wr).strip()
        
        # Identify TE (similar to WR but typically fewer targets)
        # For now, use second most targeted receiver as TE approximation
        if 'receiver_player_name' in team_pbp.columns:
            te_counts = team_pbp[
                (team_pbp['receiver_player_name'].notna()) &
                (team_pbp['play_type'] == 'pass')
            ]['receiver_player_name'].value_counts()
            if len(te_counts) > 1:
                # Second most targeted might be TE
                top_te = te_counts.index[1]
                if pd.notna(top_te) and top_te:
                    roster[(team, 'TE')] = str(top_te).strip()
    
    return roster


def get_current_roster_snapshot(season: Optional[int] = None, 
                                week: Optional[int] = None,
                                use_cache: bool = True) -> Dict[Tuple[str, str], str]:
    """
    Get current roster snapshot from available data sources.
    
    Args:
        season: Season year (default: current season)
        week: Week number (optional)
        use_cache: Whether to use cached snapshot
    
    Returns:
        Dictionary mapping (team, position) -> player_name
    """
    if season is None:
        season = get_current_season()
    
    # Check cache
    cache_file = DATA_DIR / "rosters" / f"snapshot_{season}_{week or 'current'}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                roster_dict = json.load(f)
            # Convert string keys back to tuples
            roster = {tuple(k.split('|')): v for k, v in roster_dict.items()}
            return roster
        except:
            pass
    
    roster = {}
    
    # Try nfl_data_py first
    rosters_df = fetch_rosters_nfl_data_py(season, week)
    if rosters_df is not None and len(rosters_df) > 0:
        # Convert to roster dictionary
        for _, row in rosters_df.iterrows():
            team = row.get('team') or row.get('team_abbr')
            position = row.get('position')
            player_name = row.get('player_name') or row.get('name')
            
            if team and position and player_name:
                # Normalize position (take first 2 chars for QB, RB, etc.)
                pos = str(position).upper()[:2]
                if pos in KEY_POSITIONS or any(pos.startswith(p) for p in KEY_POSITIONS):
                    roster[(team, pos)] = str(player_name).strip()
    
    # Fallback to play-by-play data
    if len(roster) == 0:
        pbp_data, _ = load_game_data()
        if pbp_data is not None:
            teams = list(TEAM_NAME_MAP.keys())
            roster = build_roster_from_pbp(pbp_data, teams, season, week)
    
    # Cache the snapshot
    if len(roster) > 0:
        try:
            # Convert tuple keys to strings for JSON
            roster_dict = {f"{k[0]}|{k[1]}": v for k, v in roster.items()}
            with open(cache_file, 'w') as f:
                json.dump(roster_dict, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not cache roster snapshot: {e}")
    
    return roster


def compare_rosters(old_roster: Dict[Tuple[str, str], str],
                   new_roster: Dict[Tuple[str, str], str]) -> List[Dict]:
    """
    Compare two roster snapshots and identify changes.
    
    Args:
        old_roster: Previous roster snapshot
        new_roster: Current roster snapshot
    
    Returns:
        List of change dictionaries with details
    """
    changes = []
    
    # Find players added
    for key, player in new_roster.items():
        if key not in old_roster:
            changes.append({
                'type': 'ADDED',
                'team': key[0],
                'position': key[1],
                'player': player,
                'previous_team': None,
                'previous_position': None,
                'timestamp': datetime.now().isoformat(),
            })
        elif old_roster[key] != player:
            # Player changed at this position
            changes.append({
                'type': 'POSITION_CHANGE' if old_roster[key] == player else 'REMOVED',
                'team': key[0],
                'position': key[1],
                'player': player,
                'previous_player': old_roster[key],
                'previous_team': key[0],
                'previous_position': key[1],
                'timestamp': datetime.now().isoformat(),
            })
    
    # Find players removed
    for key, player in old_roster.items():
        if key not in new_roster:
            changes.append({
                'type': 'REMOVED',
                'team': key[0],
                'position': key[1],
                'player': player,
                'previous_team': key[0],
                'previous_position': key[1],
                'timestamp': datetime.now().isoformat(),
            })
        else:
            # Check if player moved to different team/position
            if old_roster[key] != new_roster[key]:
                # Check if player exists elsewhere
                found_elsewhere = False
                for new_key, new_player in new_roster.items():
                    if new_player == old_roster[key] and new_key != key:
                        changes.append({
                            'type': 'TRADED' if new_key[0] != key[0] else 'POSITION_CHANGE',
                            'team': new_key[0],
                            'position': new_key[1],
                            'player': new_player,
                            'previous_team': key[0],
                            'previous_position': key[1],
                            'timestamp': datetime.now().isoformat(),
                        })
                        found_elsewhere = True
                        break
    
    return changes


def track_roster_changes(season: Optional[int] = None, 
                        week: Optional[int] = None,
                        save_changes: bool = True) -> pd.DataFrame:
    """
    Track roster changes by comparing current snapshot to previous snapshots.
    
    Args:
        season: Season year (default: current)
        week: Week number (optional)
        save_changes: Whether to save changes to file
    
    Returns:
        DataFrame with roster changes
    """
    if season is None:
        season = get_current_season()
    
    print(f"Tracking roster changes for {season} Week {week or 'current'}...")
    
    # Get current roster
    current_roster = get_current_roster_snapshot(season, week, use_cache=False)
    print(f"  Current roster: {len(current_roster)} players")
    
    # Load previous snapshots
    roster_history_file = DATA_DIR / "rosters" / "roster_history.json"
    roster_history_file.parent.mkdir(parents=True, exist_ok=True)
    
    previous_snapshots = {}
    if roster_history_file.exists():
        try:
            with open(roster_history_file, 'r') as f:
                previous_snapshots = json.load(f)
                # Convert string keys back to tuples
                previous_snapshots = {
                    k: {tuple(k2.split('|')): v2 for k2, v2 in v.items()}
                    for k, v in previous_snapshots.items()
                }
        except:
            pass
    
    # Find most recent snapshot for this season/week
    snapshot_key = f"{season}_{week or 'current'}"
    previous_roster = previous_snapshots.get(snapshot_key, {})
    
    # If no previous snapshot, try to find the most recent one
    if len(previous_roster) == 0:
        # Look for any previous snapshot
        for key in sorted(previous_snapshots.keys(), reverse=True):
            if key.startswith(f"{season}_"):
                previous_roster = previous_snapshots[key]
                break
    
    # Compare rosters
    if len(previous_roster) > 0:
        changes = compare_rosters(previous_roster, current_roster)
        print(f"  Found {len(changes)} roster changes")
    else:
        print("  No previous snapshot found - this is the first snapshot")
        changes = []
    
    # Save current snapshot to history
    current_roster_dict = {f"{k[0]}|{k[1]}": v for k, v in current_roster.items()}
    previous_snapshots[snapshot_key] = current_roster_dict
    
    # Keep only recent snapshots (last 20)
    if len(previous_snapshots) > 20:
        sorted_keys = sorted(previous_snapshots.keys(), reverse=True)
        for key in sorted_keys[20:]:
            del previous_snapshots[key]
    
    # Save history
    try:
        with open(roster_history_file, 'w') as f:
            json.dump(previous_snapshots, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not save roster history: {e}")
    
    # Save changes if any
    if save_changes and len(changes) > 0:
        changes_df = pd.DataFrame(changes)
        changes_file = DATA_DIR / "rosters" / f"changes_{season}_{week or 'current'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        changes_df.to_csv(changes_file, index=False)
        print(f"  Changes saved to {changes_file.name}")
    
    if len(changes) > 0:
        return pd.DataFrame(changes)
    else:
        return pd.DataFrame()


def get_roster_changes_summary(season: Optional[int] = None,
                              weeks_back: int = 4) -> pd.DataFrame:
    """
    Get summary of roster changes over recent weeks.
    
    Args:
        season: Season year (default: current)
        weeks_back: Number of weeks to look back
    
    Returns:
        DataFrame with summary of changes
    """
    if season is None:
        season = get_current_season()
    
    # Load all change files
    changes_dir = DATA_DIR / "rosters"
    changes_files = list(changes_dir.glob(f"changes_{season}_*.csv"))
    
    if len(changes_files) == 0:
        return pd.DataFrame()
    
    # Load and combine all changes
    all_changes = []
    for file in sorted(changes_files)[-weeks_back:]:
        try:
            df = pd.read_csv(file)
            df['source_file'] = file.name
            all_changes.append(df)
        except:
            continue
    
    if len(all_changes) == 0:
        return pd.DataFrame()
    
    combined = pd.concat(all_changes, ignore_index=True)
    
    # Group by type and team
    summary = combined.groupby(['type', 'team']).size().reset_index(name='count')
    summary = summary.sort_values(['type', 'count'], ascending=[True, False])
    
    return summary


def update_player_database_from_changes(season: Optional[int] = None):
    """
    Update the player database (pfr_player_features.PLAYER_DATABASE) based on detected roster changes.
    
    This function can be called to automatically update the player database when roster changes are detected.
    
    Args:
        season: Season year (default: current)
    """
    if season is None:
        season = get_current_season()
    
    print(f"Updating player database from roster changes for {season}...")
    
    # Get current roster snapshot
    current_roster = get_current_roster_snapshot(season, use_cache=True)
    
    # Load current player database
    from pfr_player_features import build_player_database
    player_db = build_player_database(use_cache=True)
    
    # Update database with current roster
    updated_count = 0
    for (team, position), player_name in current_roster.items():
        key = (team, season, position)
        if key not in player_db or player_db[key] != player_name:
            player_db[key] = player_name
            updated_count += 1
    
    print(f"  Updated {updated_count} player entries")
    
    # Save updated database
    cache_file = DATA_DIR / "pfr" / "player_database.parquet"
    try:
        db_list = [{'team': k[0], 'season': k[1], 'position': k[2], 'player_name': v} 
                   for k, v in player_db.items()]
        db_df = pd.DataFrame(db_list)
        db_df.to_parquet(cache_file, index=False)
        print(f"  Saved updated database to {cache_file}")
    except Exception as e:
        print(f"  Warning: Could not save updated database: {e}")
    
    return player_db


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Track NFL roster changes")
    parser.add_argument("--season", type=int, help="Season year (default: current)")
    parser.add_argument("--week", type=int, help="Week number (optional)")
    parser.add_argument("--summary", action="store_true", help="Show summary of recent changes")
    parser.add_argument("--update-db", action="store_true", help="Update player database from changes")
    
    args = parser.parse_args()
    
    if args.summary:
        summary = get_roster_changes_summary(season=args.season)
        if len(summary) > 0:
            print("\nRoster Changes Summary:")
            print(summary.to_string(index=False))
        else:
            print("No roster changes found in recent weeks.")
    elif args.update_db:
        update_player_database_from_changes(season=args.season)
    else:
        changes = track_roster_changes(season=args.season, week=args.week)
        if len(changes) > 0:
            print("\nRoster Changes Detected:")
            print(changes[['type', 'team', 'position', 'player', 'previous_player']].to_string(index=False))
        else:
            print("\nNo roster changes detected.")


