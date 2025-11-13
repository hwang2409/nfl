#!/usr/bin/env python3
"""
Python script to download NFL team logos
Run with: python3 scripts/download-logos.py
"""

import os
import urllib.request
from pathlib import Path

teams = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
]

# Create directory if it doesn't exist
script_dir = Path(__file__).parent
teams_dir = script_dir.parent / 'public' / 'images' / 'teams'
teams_dir.mkdir(parents=True, exist_ok=True)

def download_logo(team, url, filepath):
    """Download a logo from URL to filepath"""
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        return False

def get_logo_urls(team):
    """Get possible URLs for a team logo"""
    team_lower = team.lower()
    return [
        f'https://a.espncdn.com/i/teamlogos/nfl/500/{team_lower}.png',
        f'https://a.espncdn.com/i/teamlogos/nfl/500-dark/{team_lower}.png',
    ]

print('Downloading NFL team logos...\n')

success_count = 0
failed_teams = []

for team in teams:
    filepath = teams_dir / f'{team}.png'
    urls = get_logo_urls(team)
    
    downloaded = False
    for url in urls:
        if download_logo(team, url, filepath):
            print(f'✓ Downloaded {team}')
            success_count += 1
            downloaded = True
            break
    
    if not downloaded:
        print(f'✗ Failed to download {team}')
        failed_teams.append(team)

print(f'\nDone!')
print(f'Successfully downloaded: {success_count} logos')
print(f'Failed: {len(failed_teams)} logos')

if failed_teams:
    print(f'\nFailed teams: {", ".join(failed_teams)}')
    print('\nYou can manually download these logos from:')
    print('- https://www.espn.com/nfl/teams')
    print('- https://www.nfl.com/teams')
    print('- https://logos-world.net/nfl-logos/')
    print(f'\nSave them as {{TEAM_ABBREV}}.png in {teams_dir}/')

