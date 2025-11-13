"""
Injury Report Integration Module

Fetches injury reports from NFL.com or other sources.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time
from bs4 import BeautifulSoup
from tqdm import tqdm

load_dotenv()

# NFL team abbreviations mapping
NFL_TEAMS = {
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
    'WAS': 'Washington Commanders',
}

# Key positions that significantly impact performance
KEY_POSITIONS = {
    'QB': 3.0,  # Highest impact
    'RB': 1.5,
    'WR': 1.2,
    'TE': 1.0,
    'OL': 1.5,  # Offensive line (aggregate)
    'DL': 1.2,  # Defensive line (aggregate)
    'LB': 1.0,
    'CB': 1.2,
    'S': 1.0,
    'K': 0.5,
    'P': 0.3,
}

# Status mapping
STATUS_MAP = {
    'Out': 2,
    'Doubtful': 1.5,
    'Questionable': 1,
    'Probable': 0.5,
    'Healthy': 0,
}


def fetch_injury_report_nfl_com(team_abbr, season, week):
    """
    Fetch injury report from NFL.com (scraping).
    
    Uses the main injuries page (https://www.nfl.com/injuries/) and finds
    the team's section by matching team abbreviation.
    
    Args:
        team_abbr: Team abbreviation (e.g., 'KC')
        season: Season year (not used, but kept for API compatibility)
        week: Week number (not used, but kept for API compatibility)
    
    Returns:
        List of injury dictionaries or None if failed
    """
    try:
        if team_abbr not in NFL_TEAMS:
            return None
        
        # NFL.com uses a centralized injuries page
        url = "https://www.nfl.com/injuries/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all injury report units (one per team)
        injury_units = soup.find_all('section', class_=lambda x: x and 'injury-report__unit' in str(x))
        
        # Find the unit for our team
        team_unit = None
        for unit in injury_units:
            # Look for team abbreviation in the unit
            team_abbr_elem = unit.find('span', class_=lambda x: x and 'team-abbreviation' in str(x))
            if team_abbr_elem:
                unit_abbr = team_abbr_elem.get_text(strip=True)
                if unit_abbr == team_abbr:
                    team_unit = unit
                    break
        
        if not team_unit:
            return None
        
        # Find the table in this unit
        table = team_unit.find('table')
        if not table:
            return None
        
        # Parse injury rows (skip header row)
        rows = table.find_all('tr')[1:]
        injuries = []
        
        for row in rows:
            try:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 5:
                    player = cells[0].get_text(strip=True)
                    position = cells[1].get_text(strip=True)
                    injury_desc = cells[2].get_text(strip=True)
                    practice_status = cells[3].get_text(strip=True)
                    game_status = cells[4].get_text(strip=True)
                    
                    # Use game status if available, otherwise use practice status
                    status = game_status if game_status else practice_status
                    
                    # Map common statuses
                    status_upper = status.upper()
                    if 'OUT' in status_upper or 'DID NOT PARTICIPATE' in status_upper:
                        mapped_status = 'Out'
                    elif 'DOUBTFUL' in status_upper:
                        mapped_status = 'Doubtful'
                    elif 'QUESTIONABLE' in status_upper:
                        mapped_status = 'Questionable'
                    elif 'PROBABLE' in status_upper:
                        mapped_status = 'Probable'
                    elif 'FULL PARTICIPATION' in status_upper or 'FULL' in status_upper:
                        mapped_status = 'Healthy'
                    else:
                        mapped_status = status
                    
                    injuries.append({
                        'player': player,
                        'position': position,
                        'status': mapped_status,
                        'practice_status': practice_status,
                        'game_status': game_status,
                        'injury': injury_desc,
                        'team': team_abbr
                    })
            except Exception as e:
                continue
        
        return injuries if injuries else None
        
    except Exception as e:
        print(f"  Warning: NFL.com scraping error for {team_abbr}: {e}")
        return None


def fetch_injury_report_api(team_abbr, season, week, api_key=None):
    """
    Fetch injury report from a third-party API (if available).
    
    Args:
        team_abbr: Team abbreviation
        season: Season year
        week: Week number
        api_key: API key (or from env INJURY_API_KEY)
    
    Returns:
        List of injury dictionaries or None if failed
    """
    if api_key is None:
        api_key = os.getenv('INJURY_API_KEY')
    
    if not api_key:
        return None
    
    # Placeholder for API integration
    # Example: SportsDataIO, RapidAPI, etc.
    try:
        # url = f"https://api.example.com/injuries?team={team_abbr}&season={season}&week={week}"
        # headers = {'Authorization': f'Bearer {api_key}'}
        # response = requests.get(url, headers=headers, timeout=10)
        # if response.status_code == 200:
        #     return response.json()
        return None
    except Exception as e:
        print(f"  Warning: Injury API error: {e}")
        return None


def calculate_injury_impact(injuries, team_abbr):
    """
    Calculate injury impact scores for a team.
    
    Args:
        injuries: List of injury dictionaries
        team_abbr: Team abbreviation
    
    Returns:
        Dictionary with injury impact scores
    """
    if not injuries:
        return {
            'qb_injury_impact': 0,
            'ol_injury_impact': 0,
            'wr_injury_impact': 0,
            'cb_injury_impact': 0,
            'key_player_out': 0,
            'total_injury_impact': 0
        }
    
    qb_impact = 0
    ol_impact = 0
    wr_impact = 0
    cb_impact = 0
    key_player_out = 0
    total_impact = 0
    
    # Assume starting roster sizes (simplified)
    OL_STARTERS = 5
    WR_STARTERS = 3
    CB_STARTERS = 2
    
    ol_active = OL_STARTERS
    wr_active = WR_STARTERS
    cb_active = CB_STARTERS
    
    for injury in injuries:
        position = injury.get('position', '').upper()
        status = injury.get('status', '').strip()
        
        status_score = STATUS_MAP.get(status, 0)
        position_weight = KEY_POSITIONS.get(position, 0.5)
        
        impact = status_score * position_weight
        total_impact += impact
        
        # Position-specific impacts
        if position == 'QB':
            qb_impact = max(qb_impact, status_score)
            if status_score >= 2:  # Out
                key_player_out = 1
        elif position in ['OT', 'OG', 'C']:  # Offensive line
            ol_active -= status_score / 2
        elif position == 'WR':
            wr_active -= status_score / 2
        elif position == 'CB':
            cb_active -= status_score / 2
        
        # Check for star players (simplified - would need player database)
        if position in ['QB', 'RB', 'WR'] and status_score >= 2:
            key_player_out = 1
    
    # Calculate percentage impacts
    ol_injury_impact = max(0, 1 - (ol_active / OL_STARTERS))
    wr_injury_impact = max(0, 1 - (wr_active / WR_STARTERS))
    cb_injury_impact = max(0, 1 - (cb_active / CB_STARTERS))
    
    return {
        'qb_injury_impact': qb_impact,
        'ol_injury_impact': ol_injury_impact,
        'wr_injury_impact': wr_injury_impact,
        'cb_injury_impact': cb_injury_impact,
        'key_player_out': key_player_out,
        'total_injury_impact': min(total_impact / 10, 1.0)  # Normalize to 0-1
    }


def get_injury_data_for_game(home_team, away_team, season, week, use_cache=True):
    """
    Get injury data for both teams in a game.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: Season year
        week: Week number
        use_cache: Whether to use cached injury data
    
    Returns:
        Dictionary with combined injury impacts
    """
    # Check cache
    cache_file = Path(__file__).parent.parent / "data" / "injury_cache.parquet"
    if use_cache and cache_file.exists():
        try:
            cache_df = pd.read_parquet(cache_file)
            cache_key = f"{home_team}_{away_team}_{season}_{week}"
            cached = cache_df[cache_df['key'] == cache_key]
            if len(cached) > 0:
                return cached.iloc[0].to_dict()
        except:
            pass
    
    # Fetch injuries for both teams
    home_injuries = fetch_injury_report_nfl_com(home_team, season, week)
    away_injuries = fetch_injury_report_nfl_com(away_team, season, week)
    
    # Calculate impacts
    home_impact = calculate_injury_impact(home_injuries or [], home_team)
    away_impact = calculate_injury_impact(away_injuries or [], away_team)
    
    # Combine impacts
    result = {
        'key': f"{home_team}_{away_team}_{season}_{week}",
        'home_team': home_team,
        'away_team': away_team,
        'season': season,
        'week': week,
        'home_qb_injury_impact': home_impact['qb_injury_impact'],
        'away_qb_injury_impact': away_impact['qb_injury_impact'],
        'home_ol_injury_impact': home_impact['ol_injury_impact'],
        'away_ol_injury_impact': away_impact['ol_injury_impact'],
        'home_wr_injury_impact': home_impact['wr_injury_impact'],
        'away_wr_injury_impact': away_impact['wr_injury_impact'],
        'home_cb_injury_impact': home_impact['cb_injury_impact'],
        'away_cb_injury_impact': away_impact['cb_injury_impact'],
        'home_key_player_out': home_impact['key_player_out'],
        'away_key_player_out': away_impact['key_player_out'],
        'home_total_injury_impact': home_impact['total_injury_impact'],
        'away_total_injury_impact': away_impact['total_injury_impact'],
        'injury_advantage': away_impact['total_injury_impact'] - home_impact['total_injury_impact']
    }
    
    # Cache the result
    if use_cache:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if cache_file.exists():
                cache_df = pd.read_parquet(cache_file)
                cache_df = cache_df[cache_df['key'] != result['key']]
                cache_df = pd.concat([cache_df, pd.DataFrame([result])], ignore_index=True)
            else:
                cache_df = pd.DataFrame([result])
            
            cache_df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"  Warning: Could not cache injury data: {e}")
    
    return result


def batch_fetch_injuries(games_df, season_col='season', week_col='week',
                         home_team_col='home_team', away_team_col='away_team'):
    """
    Batch fetch injury data for multiple games.
    
    Efficiently checks cache first, then only fetches from API for uncached games.
    
    Args:
        games_df: DataFrame with game information
        season_col: Column name for season
        week_col: Column name for week
        home_team_col: Column name for home team
        away_team_col: Column name for away team
    
    Returns:
        DataFrame with injury features added
    """
    print("Fetching injury data...")
    
    # Load cache once at the start
    cache_file = Path(__file__).parent.parent / "data" / "injury_cache.parquet"
    cache_df = None
    if cache_file.exists():
        try:
            cache_df = pd.read_parquet(cache_file)
        except:
            cache_df = None
    
    injury_features = []
    games_to_fetch = []  # Games that need API fetching (list of (position, home_team, away_team, season, week))
    full_injury_data_cache = {}  # Store full injury data for caching
    
    # First pass: Check cache for all games
    print("  Checking cache...")
    for pos, (idx, row) in enumerate(games_df.iterrows()):
        home_team = row.get(home_team_col)
        away_team = row.get(away_team_col)
        season = row.get(season_col)
        week = row.get(week_col)
        
        if pd.isna(home_team) or pd.isna(away_team) or pd.isna(season) or pd.isna(week):
            injury_features.append({
                'qb_injury_impact': 0,
                'ol_injury_impact': 0,
                'wr_injury_impact': 0,
                'cb_injury_impact': 0,
                'key_player_out': 0,
                'injury_advantage': 0
            })
            continue
        
        cache_key = f"{home_team}_{away_team}_{season}_{week}"
        
        # Check cache
        cached_injury = None
        if cache_df is not None:
            cached = cache_df[cache_df['key'] == cache_key]
            if len(cached) > 0:
                cached_injury = cached.iloc[0].to_dict()
        
        if cached_injury:
            # Extract features from cached data (home team perspective)
            injury_features.append({
                'qb_injury_impact': cached_injury.get('home_qb_injury_impact', 0) - cached_injury.get('away_qb_injury_impact', 0),
                'ol_injury_impact': cached_injury.get('home_ol_injury_impact', 0),
                'wr_injury_impact': cached_injury.get('home_wr_injury_impact', 0),
                'cb_injury_impact': cached_injury.get('home_cb_injury_impact', 0),
                'key_player_out': cached_injury.get('home_key_player_out', 0),
                'injury_advantage': cached_injury.get('injury_advantage', 0)
            })
        else:
            # Need to fetch from API
            injury_features.append(None)  # Placeholder
            games_to_fetch.append((pos, home_team, away_team, season, week))
    
    # Second pass: Fetch from API for uncached games
    if games_to_fetch:
        print(f"  Fetching {len(games_to_fetch)} games from NFL.com...")
        for pos, home_team, away_team, season, week in tqdm(games_to_fetch, desc="  Injuries API"):
            injury_data = get_injury_data_for_game(home_team, away_team, season, week, use_cache=False)  # Don't check cache again
            
            # Store full data for caching
            cache_key = f"{home_team}_{away_team}_{season}_{week}"
            if injury_data:
                full_injury_data_cache[cache_key] = injury_data
                # Extract features for the game (home team perspective)
                injury_features[pos] = {
                    'qb_injury_impact': injury_data.get('home_qb_injury_impact', 0) - injury_data.get('away_qb_injury_impact', 0),
                    'ol_injury_impact': injury_data.get('home_ol_injury_impact', 0),
                    'wr_injury_impact': injury_data.get('home_wr_injury_impact', 0),
                    'cb_injury_impact': injury_data.get('home_cb_injury_impact', 0),
                    'key_player_out': injury_data.get('home_key_player_out', 0),
                    'injury_advantage': injury_data.get('injury_advantage', 0)
                }
            else:
                # Default values if API fails
                injury_features[pos] = {
                    'qb_injury_impact': 0,
                    'ol_injury_impact': 0,
                    'wr_injury_impact': 0,
                    'cb_injury_impact': 0,
                    'key_player_out': 0,
                    'injury_advantage': 0
                }
            
            # Rate limiting - be nice to NFL.com
            time.sleep(1)
        
        # Save newly fetched injury data to cache
        if full_injury_data_cache:
            print("  Saving injury data to cache...")
            new_cache_entries = list(full_injury_data_cache.values())
            
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                if cache_df is not None:
                    # Remove any existing entries for these keys
                    new_keys = {e['key'] for e in new_cache_entries}
                    cache_df = cache_df[~cache_df['key'].isin(new_keys)]
                    # Append new entries
                    cache_df = pd.concat([cache_df, pd.DataFrame(new_cache_entries)], ignore_index=True)
                else:
                    cache_df = pd.DataFrame(new_cache_entries)
                
                cache_df.to_parquet(cache_file, index=False)
                print(f"  Saved {len(new_cache_entries)} entries to cache")
            except Exception as e:
                print(f"  Warning: Failed to save cache: {e}")
    else:
        print("  All games found in cache!")
    
    injury_df = pd.DataFrame(injury_features)
    
    return injury_df

