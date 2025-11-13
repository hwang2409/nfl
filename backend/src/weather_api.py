"""
Weather API Integration Module

Fetches weather forecasts for NFL games using Open-Meteo (free, no API key required).
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import time
from tqdm import tqdm
import openmeteo_requests

load_dotenv()

# Initialize Open-Meteo API client
_openmeteo = openmeteo_requests.Client()

# Track Open-Meteo API calls for status reporting
_openmeteo_success_count = 0
_openmeteo_total_attempts = 0
_openmeteo_debug_info = []  # Store debug info for failures

# Team locations (from comprehensive_features.py)
TEAM_LOCATIONS = {
    'ARI': {'city': 'Phoenix', 'state': 'AZ', 'lat': 33.5275, 'lon': -112.2625},
    'ATL': {'city': 'Atlanta', 'state': 'GA', 'lat': 33.7550, 'lon': -84.4010},
    'BAL': {'city': 'Baltimore', 'state': 'MD', 'lat': 39.2780, 'lon': -76.6227},
    'BUF': {'city': 'Buffalo', 'state': 'NY', 'lat': 42.7738, 'lon': -78.7869},
    'CAR': {'city': 'Charlotte', 'state': 'NC', 'lat': 35.2258, 'lon': -80.8528},
    'CHI': {'city': 'Chicago', 'state': 'IL', 'lat': 41.8625, 'lon': -87.6167},
    'CIN': {'city': 'Cincinnati', 'state': 'OH', 'lat': 39.0950, 'lon': -84.5160},
    'CLE': {'city': 'Cleveland', 'state': 'OH', 'lat': 41.5061, 'lon': -81.6996},
    'DAL': {'city': 'Dallas', 'state': 'TX', 'lat': 32.7473, 'lon': -97.0945},
    'DEN': {'city': 'Denver', 'state': 'CO', 'lat': 39.7439, 'lon': -105.0200},
    'DET': {'city': 'Detroit', 'state': 'MI', 'lat': 42.3400, 'lon': -83.0456},
    'GB': {'city': 'Green Bay', 'state': 'WI', 'lat': 44.5013, 'lon': -88.0622},
    'HOU': {'city': 'Houston', 'state': 'TX', 'lat': 29.6847, 'lon': -95.4107},
    'IND': {'city': 'Indianapolis', 'state': 'IN', 'lat': 39.7601, 'lon': -86.1639},
    'JAX': {'city': 'Jacksonville', 'state': 'FL', 'lat': 30.3239, 'lon': -81.6372},
    'KC': {'city': 'Kansas City', 'state': 'MO', 'lat': 39.0489, 'lon': -94.4839},
    'LV': {'city': 'Las Vegas', 'state': 'NV', 'lat': 36.0908, 'lon': -115.1836},
    'LAC': {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0141, 'lon': -118.2879},
    'LAR': {'city': 'Los Angeles', 'state': 'CA', 'lat': 34.0141, 'lon': -118.2879},
    'MIA': {'city': 'Miami', 'state': 'FL', 'lat': 25.9581, 'lon': -80.2389},
    'MIN': {'city': 'Minneapolis', 'state': 'MN', 'lat': 44.9740, 'lon': -93.2581},
    'NE': {'city': 'Foxborough', 'state': 'MA', 'lat': 42.0926, 'lon': -71.2640},
    'NO': {'city': 'New Orleans', 'state': 'LA', 'lat': 29.9511, 'lon': -90.0815},
    'NYG': {'city': 'East Rutherford', 'state': 'NJ', 'lat': 40.8136, 'lon': -74.0744},
    'NYJ': {'city': 'East Rutherford', 'state': 'NJ', 'lat': 40.8136, 'lon': -74.0744},
    'PHI': {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.9010, 'lon': -75.1675},
    'PIT': {'city': 'Pittsburgh', 'state': 'PA', 'lat': 40.4468, 'lon': -80.0158},
    'SF': {'city': 'San Francisco', 'state': 'CA', 'lat': 37.7133, 'lon': -122.3860},
    'SEA': {'city': 'Seattle', 'state': 'WA', 'lat': 47.5952, 'lon': -122.3316},
    'TB': {'city': 'Tampa', 'state': 'FL', 'lat': 27.9756, 'lon': -82.5033},
    'TEN': {'city': 'Nashville', 'state': 'TN', 'lat': 36.1665, 'lon': -86.7713},
    'WAS': {'city': 'Landover', 'state': 'MD', 'lat': 38.9076, 'lon': -76.8644},
}

DOME_STADIUMS = {
    'ATL', 'DET', 'IND', 'NO', 'DAL', 'HOU', 'MIN', 'LV', 'LAR', 'LAC'
}


def fetch_weather_openmeteo(lat, lon, game_date):
    """
    Fetch weather data from Open-Meteo API (free, no API key required).
    
    Uses Archive API (/v1/archive) for historical dates and Forecast API (/v1/forecast) 
    for future dates (up to 10 days).
    
    Args:
        lat: Latitude
        lon: Longitude
        game_date: Game date (datetime)
    
    Returns:
        Dictionary with weather data or None if failed
    """
    global _openmeteo_total_attempts, _openmeteo_success_count, _openmeteo_debug_info
    
    # Parse game date
    today = datetime.now().date()
    if hasattr(game_date, 'date'):
        game_date_obj = game_date.date()
    elif isinstance(game_date, str):
        try:
            game_date_obj = pd.to_datetime(game_date).date()
        except:
            game_date_obj = None
    else:
        game_date_obj = game_date
    
    if not game_date_obj:
        return None
    
    game_date_str = game_date_obj.isoformat()
    
    # Check if date is more than 10 days in the FUTURE (forecast API limit)
    # Historical dates can be fetched (no limit)
    days_ahead = (game_date_obj - today).days
    if days_ahead > 10:
        debug_entry = {
            'game_date': game_date_str,
            'lat': lat,
            'lon': lon,
            'status_code': None,
            'error': f"Date is {days_ahead} days in the future (max 10 days for forecast API)"
        }
        _openmeteo_debug_info.append(debug_entry)
        return None
    
    try:
        _openmeteo_total_attempts += 1
        
        # Debug: Track request
        debug_entry = {
            'game_date': game_date_str,
            'lat': lat,
            'lon': lon,
            'status_code': None,
            'error': None
        }
        
        # Determine which API to use based on date
        days_ahead = (game_date_obj - today).days
        
        if days_ahead < 0:
            # Historical date: use Archive API
            url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            # Future date: use Forecast API
            url = "https://api.open-meteo.com/v1/forecast"
        
        # Use start_date/end_date (don't use forecast_days)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": game_date_obj.isoformat(),
            "end_date": game_date_obj.isoformat(),
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "precipitation_probability_max",
                "windspeed_10m_max",
                "windgusts_10m_max",
                "weathercode"
            ],
            "timezone": "America/New_York",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch"
        }
        
        # Ensure forecast_days is NOT in params (it conflicts with start_date/end_date)
        if "forecast_days" in params:
            del params["forecast_days"]
        
        try:
            # Use the openmeteo-requests package with the appropriate endpoint
            responses = _openmeteo.weather_api(url, params=params)
        except Exception as e:
            debug_entry['error'] = f"API request error: {type(e).__name__}: {str(e)}"
            _openmeteo_debug_info.append(debug_entry)
            return None
        
        if not responses or len(responses) == 0:
            debug_entry['error'] = "No response from API"
            _openmeteo_debug_info.append(debug_entry)
            return None
        
        response = responses[0]
        
        # Extract daily data using the package's API
        daily = response.Daily()
        dates_raw = daily.Time()
        
        # Convert dates to list (handle both single values and arrays)
        if hasattr(dates_raw, '__iter__') and not isinstance(dates_raw, (str, bytes)):
            try:
                dates = list(dates_raw)
            except TypeError:
                dates = [dates_raw]
        else:
            dates = [dates_raw] if dates_raw is not None else []
        
        # Get weather variables
        temps_max = daily.Variables(0).ValuesAsNumpy()  # temperature_2m_max
        temps_min = daily.Variables(1).ValuesAsNumpy()  # temperature_2m_min
        temps_mean = daily.Variables(2).ValuesAsNumpy()  # temperature_2m_mean
        precip = daily.Variables(3).ValuesAsNumpy()  # precipitation_sum
        precip_prob = daily.Variables(4).ValuesAsNumpy()  # precipitation_probability_max
        wind_speed = daily.Variables(5).ValuesAsNumpy()  # windspeed_10m_max
        wind_gust = daily.Variables(6).ValuesAsNumpy()  # windgusts_10m_max
        weathercode = daily.Variables(7).ValuesAsNumpy()  # weathercode
        
        # Convert dates to strings for comparison
        date_strings = []
        for d in dates:
            try:
                if isinstance(d, (int, float)):
                    date_strings.append(datetime.fromtimestamp(d).strftime('%Y-%m-%d'))
                else:
                    date_strings.append(datetime.fromtimestamp(d).strftime('%Y-%m-%d'))
            except Exception:
                date_strings.append(str(d)[:10])  # Take first 10 chars (YYYY-MM-DD)
        
        # Check if date exists in response
        if game_date_str not in date_strings:
            debug_entry['error'] = f"Date {game_date_str} not found in response"
            debug_entry['available_dates'] = date_strings[:5] if len(date_strings) > 5 else date_strings
            debug_entry['total_dates'] = len(date_strings)
            _openmeteo_debug_info.append(debug_entry)
            return None
        
        # Found the date - extract data
        for i, date_str in enumerate(date_strings):
            if date_str == game_date_str:
                # Map weathercode to conditions (WMO Weather interpretation codes)
                code = int(weathercode[i]) if i < len(weathercode) else 0
                conditions = _map_weathercode_to_conditions(code)
                
                _openmeteo_success_count += 1
                return {
                    'temp': float(temps_mean[i]) if i < len(temps_mean) else float((temps_max[i] + temps_min[i]) / 2) if i < len(temps_max) and i < len(temps_min) else 65.0,
                    'temp_min': float(temps_min[i]) if i < len(temps_min) else 0.0,
                    'temp_max': float(temps_max[i]) if i < len(temps_max) else 0.0,
                    'wind_mph': float(wind_speed[i]) if i < len(wind_speed) else 0.0,
                    'wind_gust_mph': float(wind_gust[i]) if i < len(wind_gust) else 0.0,
                    'precipitation': float(precip[i]) if i < len(precip) else 0.0,
                    'precipitation_prob': float(precip_prob[i] / 100.0) if i < len(precip_prob) else 0.0,  # Convert percentage to 0-1
                    'humidity': 0.0,  # Open-Meteo free tier doesn't include humidity
                    'conditions': conditions,
                    'description': conditions
                }
        
        # Should not reach here, but just in case
        debug_entry['error'] = "Date found but loop didn't return"
        _openmeteo_debug_info.append(debug_entry)
        return None
    except Exception as e:
        # Capture exception in debug info
        debug_entry = {
            'game_date': game_date.strftime('%Y-%m-%d') if hasattr(game_date, 'strftime') else str(game_date),
            'lat': lat,
            'lon': lon,
            'status_code': None,
            'error': f"Exception: {type(e).__name__}: {str(e)}"
        }
        _openmeteo_debug_info.append(debug_entry)
        return None


def _map_weathercode_to_conditions(code):
    """
    Map WMO Weather interpretation codes to condition strings.
    
    Args:
        code: Weather code (0-99)
    
    Returns:
        Condition string
    """
    # WMO Weather interpretation codes (WW)
    if code == 0:
        return 'Clear'
    elif code in [1, 2, 3]:
        return 'Partly Cloudy'
    elif code in [45, 48]:
        return 'Foggy'
    elif code in [51, 53, 55, 56, 57]:
        return 'Drizzle'
    elif code in [61, 63, 65, 66, 67]:
        return 'Rain'
    elif code in [71, 73, 75, 77]:
        return 'Snow'
    elif code in [80, 81, 82]:
        return 'Rain Showers'
    elif code in [85, 86]:
        return 'Snow Showers'
    elif code in [95, 96, 99]:
        return 'Thunderstorm'
    else:
        return 'Unknown'


def get_weather_for_game(home_team, game_date, use_cache=True):
    """
    Get weather forecast for a game.
    
    Args:
        home_team: Home team abbreviation
        game_date: Game date (datetime or string)
        use_cache: Whether to use cached weather data
    
    Returns:
        Dictionary with weather features
    """
    # Check if dome
    if home_team in DOME_STADIUMS:
        return {
            'temp': 72,  # Indoor temperature
            'wind_mph': 0,
            'precipitation': 0,
            'precipitation_prob': 0,
            'is_dome': True
        }
    
    # Get team location
    if home_team not in TEAM_LOCATIONS:
        return None
    
    location = TEAM_LOCATIONS[home_team]
    lat, lon = location['lat'], location['lon']
    
    # Parse game date
    if isinstance(game_date, str):
        try:
            game_date = pd.to_datetime(game_date)
        except:
            return None
    
    # Check cache
    cache_file = Path(__file__).parent.parent / "data" / "weather_cache.parquet"
    if use_cache and cache_file.exists():
        try:
            cache_df = pd.read_parquet(cache_file)
            game_date_str = game_date.date().isoformat() if hasattr(game_date, 'date') else str(game_date)
            cache_key = f"{home_team}_{game_date_str}"
            cached = cache_df[cache_df['key'] == cache_key]
            if len(cached) > 0:
                result = cached.iloc[0].to_dict()
                # Remove cache-specific keys before returning
                result.pop('key', None)
                result.pop('home_team', None)
                result.pop('game_date', None)
                return result
        except:
            pass
    
    # Fetch from Open-Meteo (free, no API key required)
    weather_data = fetch_weather_openmeteo(lat, lon, game_date)
    
    if weather_data is None:
        # Return default values if API fails
        return {
            'temp': 65,  # Default temperature
            'wind_mph': 0,
            'precipitation': 0,
            'precipitation_prob': 0,
            'is_dome': False
        }
    
    weather_data['is_dome'] = False
    
    # Cache the result
    if use_cache:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            game_date_str = game_date.date().isoformat() if hasattr(game_date, 'date') else str(game_date)
            cache_data = {
                'key': f"{home_team}_{game_date_str}",
                'home_team': home_team,
                'game_date': game_date_str,  # Store as string to avoid parquet type inference issues
                **weather_data
            }
            
            if cache_file.exists():
                cache_df = pd.read_parquet(cache_file)
                cache_df = cache_df[cache_df['key'] != cache_data['key']]
                cache_df = pd.concat([cache_df, pd.DataFrame([cache_data])], ignore_index=True)
            else:
                cache_df = pd.DataFrame([cache_data])
            
            # Ensure all columns have proper types for parquet
            cache_df = cache_df.astype({
                'key': 'string',
                'home_team': 'string',
                'game_date': 'string',
                'temp': 'float64',
                'wind_mph': 'float64',
                'precipitation': 'float64',
                'precipitation_prob': 'float64',
                'is_dome': 'bool'
            }, errors='ignore')  # Ignore columns that don't exist
            
            cache_df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"  Warning: Could not cache weather data: {e}")
    
    return weather_data


def batch_fetch_weather(games_df, game_date_col='gameday', home_team_col='home_team'):
    """
    Batch fetch weather for multiple games.
    
    Efficiently checks cache first, then only fetches from API for uncached games.
    
    Args:
        games_df: DataFrame with game information
        game_date_col: Column name for game date
        home_team_col: Column name for home team
    
    Returns:
        DataFrame with weather features added
    """
    global _openmeteo_success_count, _openmeteo_total_attempts, _openmeteo_debug_info
    
    # Reset counters at start of batch
    _openmeteo_success_count = 0
    _openmeteo_total_attempts = 0
    _openmeteo_debug_info = []
    
    print("Fetching weather data...")
    
    # Load cache once at the start
    cache_file = Path(__file__).parent.parent / "data" / "weather_cache.parquet"
    cache_df = None
    if cache_file.exists():
        try:
            cache_df = pd.read_parquet(cache_file)
        except:
            cache_df = None
    
    weather_features = []
    games_to_fetch = []  # Games that need API fetching (list of (position, home_team, game_date))
    
    # First pass: Check cache for all games
    print("  Checking cache...")
    for pos, (idx, row) in enumerate(games_df.iterrows()):
        home_team = row.get(home_team_col)
        game_date = row.get(game_date_col)
        
        if pd.isna(home_team) or pd.isna(game_date):
            weather_features.append({
                'temp': 65,
                'wind_mph': 0,
                'precipitation': 0,
                'precipitation_prob': 0,
                'is_dome': False
            })
            continue
        
        # Check if dome (no API call needed)
        if home_team in DOME_STADIUMS:
            weather_features.append({
                'temp': 72,
                'wind_mph': 0,
                'precipitation': 0,
                'precipitation_prob': 0,
                'is_dome': True
            })
            continue
        
        # Parse game date for cache lookup
        if isinstance(game_date, str):
            try:
                game_date_parsed = pd.to_datetime(game_date)
            except:
                game_date_parsed = game_date
        else:
            game_date_parsed = game_date
        
        game_date_str = game_date_parsed.date().isoformat() if hasattr(game_date_parsed, 'date') else str(game_date_parsed)
        cache_key = f"{home_team}_{game_date_str}"
        
        # Check cache
        cached_weather = None
        if cache_df is not None:
            cached = cache_df[cache_df['key'] == cache_key]
            if len(cached) > 0:
                cached_weather = cached.iloc[0].to_dict()
                # Remove cache-specific keys
                cached_weather.pop('key', None)
                cached_weather.pop('home_team', None)
                cached_weather.pop('game_date', None)
        
        if cached_weather:
            weather_features.append(cached_weather)
        else:
            # Need to fetch from API
            weather_features.append(None)  # Placeholder
            games_to_fetch.append((pos, home_team, game_date_parsed))
    
    # Second pass: Fetch from API for uncached games
    if games_to_fetch:
        print(f"  Fetching {len(games_to_fetch)} games from Open-Meteo API...")
        for pos, home_team, game_date in tqdm(games_to_fetch, desc="  Weather API"):
            weather = get_weather_for_game(home_team, game_date, use_cache=False)  # Don't check cache again
            
            if weather:
                weather_features[pos] = weather
            else:
                # Default values if API fails
                weather_features[pos] = {
                    'temp': 65,
                    'wind_mph': 0,
                    'precipitation': 0,
                    'precipitation_prob': 0,
                    'is_dome': False
                }
            
            # Rate limiting - be nice to APIs
            time.sleep(0.5)
        
        # Save newly fetched weather data to cache
        print("  Saving weather data to cache...")
        new_cache_entries = []
        for pos, home_team, game_date in games_to_fetch:
            weather = weather_features[pos]
            if weather:
                game_date_str = game_date.date().isoformat() if hasattr(game_date, 'date') else str(game_date)
                cache_entry = {
                    'key': f"{home_team}_{game_date_str}",
                    'home_team': home_team,
                    'game_date': game_date_str,
                    **weather
                }
                new_cache_entries.append(cache_entry)
        
        if new_cache_entries:
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
                
                # Ensure all columns have proper types for parquet
                cache_df = cache_df.astype({
                    'key': 'string',
                    'home_team': 'string',
                    'game_date': 'string',
                    'temp': 'float64',
                    'wind_mph': 'float64',
                    'precipitation': 'float64',
                    'precipitation_prob': 'float64',
                    'is_dome': 'bool'
                }, errors='ignore')
                
                cache_df.to_parquet(cache_file, index=False)
                print(f"  Saved {len(new_cache_entries)} entries to cache")
            except Exception as e:
                print(f"  Warning: Failed to save cache: {e}")
    else:
        print("  All games found in cache!")
    
    weather_df = pd.DataFrame(weather_features)
    
    # Print status summary
    cached_count = len(weather_features) - len(games_to_fetch) if games_to_fetch else len(weather_features)
    print(f"  Weather fetch complete: {cached_count} from cache, {_openmeteo_success_count}/{_openmeteo_total_attempts} successful from Open-Meteo API")
    
    # Print debug info if there were failures
    if _openmeteo_debug_info:
        # Filter out dates beyond 10 days (forecast limit)
        future_limit_errors = [e for e in _openmeteo_debug_info if 'days in the future' in e.get('error', '')]
        api_failures = [e for e in _openmeteo_debug_info if 'days in the future' not in e.get('error', '')]
        
        if future_limit_errors:
            print(f"\n  Note: {len(future_limit_errors)} dates skipped (more than 10 days in the future - forecast API limit)")
        
        if api_failures:
            print(f"\n  Debug: {len(api_failures)} API call failures detected")
            
            # Group errors by type
            error_counts = {}
            for entry in api_failures:
                error_type = entry.get('error', 'Unknown')
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            print("  Failure breakdown:")
            for error_type, count in error_counts.items():
                print(f"    - {error_type}: {count} times")
        
            # Show sample failures (first 3 unique errors)
            print("\n  Sample failure details (first 3 unique errors):")
            seen_errors = set()
            shown = 0
            for entry in api_failures:
                error_type = entry.get('error', 'Unknown')
                if error_type not in seen_errors and shown < 3:
                    seen_errors.add(error_type)
                    shown += 1
                    print(f"    Error: {error_type}")
                    print(f"      Game date: {entry.get('game_date', 'N/A')}")
                    print(f"      Location: ({entry.get('lat', 'N/A')}, {entry.get('lon', 'N/A')})")
                    if 'available_dates' in entry:
                        print(f"      Available dates: {entry.get('available_dates', [])}")
                    if 'response_text' in entry:
                        print(f"      Response: {entry.get('response_text', '')[:100]}...")
                    print()
        
        # Clear debug info for next run
        _openmeteo_debug_info = []
    
    return weather_df

