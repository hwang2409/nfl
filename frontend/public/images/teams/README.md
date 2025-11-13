# NFL Team Logos

This directory contains NFL team logos used in the application.

## File Naming Convention

Each logo should be named using the team abbreviation (e.g., `KC.png`, `BUF.png`, `SF.png`).

## Downloading Logos

### Option 1: Using the Script (Recommended)

Run the download script:

```bash
# Using Node.js
node scripts/download-logos.js

# Or using bash
chmod +x scripts/download-logos.sh
./scripts/download-logos.sh
```

### Option 2: Manual Download

1. Visit one of these sources:
   - [ESPN NFL Teams](https://www.espn.com/nfl/teams)
   - [NFL.com Teams](https://www.nfl.com/teams)
   - [Logos World - NFL Logos](https://logos-world.net/nfl-logos/)

2. For each team:
   - Right-click on the team logo
   - Select "Save Image As..."
   - Save as `{TEAM_ABBREV}.png` (e.g., `KC.png`, `BUF.png`)

3. Place all logos in this directory: `public/images/teams/`

## Supported Teams

All 32 NFL teams are supported:

**AFC East:** BUF, MIA, NE, NYJ  
**AFC North:** BAL, CIN, CLE, PIT  
**AFC South:** HOU, IND, JAX, TEN  
**AFC West:** DEN, KC, LV, LAC  

**NFC East:** DAL, NYG, PHI, WAS  
**NFC North:** CHI, DET, GB, MIN  
**NFC South:** ATL, CAR, NO, TB  
**NFC West:** ARI, LAR, SF, SEA  

## Image Requirements

- **Format:** PNG (recommended) or SVG
- **Size:** 256x256 pixels or larger (will be scaled down)
- **Background:** Transparent preferred
- **Aspect Ratio:** Square (1:1)

## Fallback Behavior

If a logo file is missing, the application will:
1. Try to load the logo from the path
2. If it fails, display the team abbreviation as a fallback

## Notes

- Logos are served statically from the `public` directory
- Next.js Image component is used for optimized loading
- Logos are cached by the browser for better performance

