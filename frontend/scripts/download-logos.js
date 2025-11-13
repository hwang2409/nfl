/**
 * Node.js script to download NFL team logos
 * Run with: node scripts/download-logos.js
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const teams = [
  'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
  'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
  'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
  'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
];

// Create directory if it doesn't exist
const teamsDir = path.join(__dirname, '..', 'public', 'images', 'teams');
if (!fs.existsSync(teamsDir)) {
  fs.mkdirSync(teamsDir, { recursive: true });
}

// Function to download a file
function downloadFile(url, filepath) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    
    protocol.get(url, (response) => {
      if (response.statusCode === 200) {
        const fileStream = fs.createWriteStream(filepath);
        response.pipe(fileStream);
        fileStream.on('finish', () => {
          fileStream.close();
          resolve();
        });
      } else if (response.statusCode === 301 || response.statusCode === 302) {
        // Handle redirects
        downloadFile(response.headers.location, filepath)
          .then(resolve)
          .catch(reject);
      } else {
        reject(new Error(`Failed to download: ${response.statusCode}`));
      }
    }).on('error', reject);
  });
}

// Alternative URLs to try for each team
const getLogoUrls = (team) => {
  const teamLower = team.toLowerCase();
  return [
    `https://a.espncdn.com/i/teamlogos/nfl/500/${teamLower}.png`,
    `https://a.espncdn.com/i/teamlogos/nfl/500-dark/${teamLower}.png`,
    `https://static.www.nfl.com/image/private/t_headshot_desktop/league/${teamLower}`,
  ];
};

async function downloadLogos() {
  console.log('Downloading NFL team logos...\n');
  
  const results = { success: [], failed: [] };
  
  for (const team of teams) {
    const filepath = path.join(teamsDir, `${team}.png`);
    const urls = getLogoUrls(team);
    
    let downloaded = false;
    
    for (const url of urls) {
      try {
        await downloadFile(url, filepath);
        console.log(`✓ Downloaded ${team}`);
        results.success.push(team);
        downloaded = true;
        break;
      } catch (error) {
        // Try next URL
        continue;
      }
    }
    
    if (!downloaded) {
      console.log(`✗ Failed to download ${team}`);
      results.failed.push(team);
    }
  }
  
  console.log(`\nDone!`);
  console.log(`Successfully downloaded: ${results.success.length} logos`);
  console.log(`Failed: ${results.failed.length} logos`);
  
  if (results.failed.length > 0) {
    console.log(`\nFailed teams: ${results.failed.join(', ')}`);
    console.log('\nYou can manually download these logos from:');
    console.log('- https://www.espn.com/nfl/teams');
    console.log('- https://www.nfl.com/teams');
    console.log('- https://logos-world.net/nfl-logos/');
    console.log('\nSave them as {TEAM_ABBREV}.png in public/images/teams/');
  }
}

downloadLogos().catch(console.error);

