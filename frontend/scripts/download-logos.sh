#!/bin/bash

# Script to download NFL team logos
# This script downloads logos from a public CDN and saves them to public/images/teams/

# Create directory if it doesn't exist
mkdir -p public/images/teams

# Base URL for NFL logos (using a reliable CDN)
# You can replace this with your preferred source
BASE_URL="https://a.espncdn.com/i/teamlogos/nfl/500"

# Array of all NFL team abbreviations
teams=(
  "ARI" "ATL" "BAL" "BUF" "CAR" "CHI" "CIN" "CLE"
  "DAL" "DEN" "DET" "GB" "HOU" "IND" "JAX" "KC"
  "LV" "LAC" "LAR" "MIA" "MIN" "NE" "NO" "NYG"
  "NYJ" "PHI" "PIT" "SF" "SEA" "TB" "TEN" "WAS"
)

echo "Downloading NFL team logos..."

# Download each team logo
for team in "${teams[@]}"; do
  # ESPN uses lowercase team abbreviations in their URLs
  team_lower=$(echo "$team" | tr '[:upper:]' '[:lower:]')
  
  # Try different URL patterns
  url="${BASE_URL}/${team_lower}.png"
  
  echo "Downloading $team..."
  
  # Download with curl, save to public/images/teams/
  curl -L -o "public/images/teams/${team}.png" "$url" 2>/dev/null
  
  if [ $? -eq 0 ]; then
    echo "✓ Downloaded $team"
  else
    echo "✗ Failed to download $team from $url"
    echo "  You may need to download this logo manually from:"
    echo "  https://www.espn.com/nfl/teams or another source"
  fi
done

echo ""
echo "Done! Logos saved to public/images/teams/"
echo ""
echo "If some logos failed to download, you can:"
echo "1. Visit https://www.espn.com/nfl/teams"
echo "2. Right-click on each team logo and 'Save Image As...'"
echo "3. Save as {TEAM_ABBREV}.png in public/images/teams/"
echo ""
echo "Alternative sources:"
echo "- https://www.nfl.com/teams (official NFL site)"
echo "- https://logos-world.net/nfl-logos/"

