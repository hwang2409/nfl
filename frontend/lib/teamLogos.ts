/**
 * NFL Team Logo Mapping
 * 
 * This file maps NFL team abbreviations to their logo image paths.
 * Logos are stored in /public/images/teams/ directory.
 * 
 * To add logos:
 * 1. Download team logos (PNG format, recommended size: 256x256 or larger)
 * 2. Save them in public/images/teams/ with the filename: {teamAbbrev}.png
 * 3. The mapping below will automatically reference them
 */

export const NFL_TEAMS = {
  // AFC East
  'BUF': { name: 'Buffalo Bills', logo: '/images/teams/BUF.png' },
  'MIA': { name: 'Miami Dolphins', logo: '/images/teams/MIA.png' },
  'NE': { name: 'New England Patriots', logo: '/images/teams/NE.png' },
  'NYJ': { name: 'New York Jets', logo: '/images/teams/NYJ.png' },
  
  // AFC North
  'BAL': { name: 'Baltimore Ravens', logo: '/images/teams/BAL.png' },
  'CIN': { name: 'Cincinnati Bengals', logo: '/images/teams/CIN.png' },
  'CLE': { name: 'Cleveland Browns', logo: '/images/teams/CLE.png' },
  'PIT': { name: 'Pittsburgh Steelers', logo: '/images/teams/PIT.png' },
  
  // AFC South
  'HOU': { name: 'Houston Texans', logo: '/images/teams/HOU.png' },
  'IND': { name: 'Indianapolis Colts', logo: '/images/teams/IND.png' },
  'JAX': { name: 'Jacksonville Jaguars', logo: '/images/teams/JAX.png' },
  'TEN': { name: 'Tennessee Titans', logo: '/images/teams/TEN.png' },
  
  // AFC West
  'DEN': { name: 'Denver Broncos', logo: '/images/teams/DEN.png' },
  'KC': { name: 'Kansas City Chiefs', logo: '/images/teams/KC.png' },
  'LV': { name: 'Las Vegas Raiders', logo: '/images/teams/LV.png' },
  'LAC': { name: 'Los Angeles Chargers', logo: '/images/teams/LAC.png' },
  
  // NFC East
  'DAL': { name: 'Dallas Cowboys', logo: '/images/teams/DAL.png' },
  'NYG': { name: 'New York Giants', logo: '/images/teams/NYG.png' },
  'PHI': { name: 'Philadelphia Eagles', logo: '/images/teams/PHI.png' },
  'WAS': { name: 'Washington Commanders', logo: '/images/teams/WAS.png' },
  
  // NFC North
  'CHI': { name: 'Chicago Bears', logo: '/images/teams/CHI.png' },
  'DET': { name: 'Detroit Lions', logo: '/images/teams/DET.png' },
  'GB': { name: 'Green Bay Packers', logo: '/images/teams/GB.png' },
  'MIN': { name: 'Minnesota Vikings', logo: '/images/teams/MIN.png' },
  
  // NFC South
  'ATL': { name: 'Atlanta Falcons', logo: '/images/teams/ATL.png' },
  'CAR': { name: 'Carolina Panthers', logo: '/images/teams/CAR.png' },
  'NO': { name: 'New Orleans Saints', logo: '/images/teams/NO.png' },
  'TB': { name: 'Tampa Bay Buccaneers', logo: '/images/teams/TB.png' },
  
  // NFC West
  'ARI': { name: 'Arizona Cardinals', logo: '/images/teams/ARI.png' },
  'LAR': { name: 'Los Angeles Rams', logo: '/images/teams/LAR.png' },
  'SF': { name: 'San Francisco 49ers', logo: '/images/teams/SF.png' },
  'SEA': { name: 'Seattle Seahawks', logo: '/images/teams/SEA.png' },
} as const;

export type TeamAbbreviation = keyof typeof NFL_TEAMS;

/**
 * Get team logo path for a given team abbreviation
 */
export function getTeamLogo(teamAbbrev: string): string {
  const team = NFL_TEAMS[teamAbbrev as TeamAbbreviation];
  return team?.logo || '/images/teams/default.png';
}

/**
 * Get team full name for a given team abbreviation
 */
export function getTeamName(teamAbbrev: string): string {
  const team = NFL_TEAMS[teamAbbrev as TeamAbbreviation];
  return team?.name || teamAbbrev;
}

/**
 * Get team short name (without city) for a given team abbreviation
 * e.g., "Kansas City Chiefs" -> "Chiefs"
 */
export function getTeamShortName(teamAbbrev: string): string {
  const fullName = getTeamName(teamAbbrev);
  // Remove city names - team name is usually the last word(s)
  const parts = fullName.split(' ');
  // Handle special cases
  if (fullName.includes('New York')) {
    return parts.slice(2).join(' '); // "New York Jets" -> "Jets"
  }
  if (fullName.includes('Tampa Bay')) {
    return parts.slice(2).join(' '); // "Tampa Bay Buccaneers" -> "Buccaneers"
  }
  if (fullName.includes('Green Bay')) {
    return parts.slice(2).join(' '); // "Green Bay Packers" -> "Packers"
  }
  if (fullName.includes('Las Vegas')) {
    return parts.slice(2).join(' '); // "Las Vegas Raiders" -> "Raiders"
  }
  if (fullName.includes('Los Angeles')) {
    return parts.slice(2).join(' '); // "Los Angeles Rams" -> "Rams"
  }
  if (fullName.includes('San Francisco')) {
    return parts.slice(2).join(' '); // "San Francisco 49ers" -> "49ers"
  }
  if (fullName.includes('New England')) {
    return parts.slice(2).join(' '); // "New England Patriots" -> "Patriots"
  }
  if (fullName.includes('New Orleans')) {
    return parts.slice(2).join(' '); // "New Orleans Saints" -> "Saints"
  }
  // For most teams, return everything after the first word
  return parts.slice(1).join(' ');
}

/**
 * Check if a team abbreviation is valid
 */
export function isValidTeam(teamAbbrev: string): boolean {
  return teamAbbrev in NFL_TEAMS;
}

/**
 * Get all team abbreviations
 */
export function getAllTeamAbbreviations(): TeamAbbreviation[] {
  return Object.keys(NFL_TEAMS) as TeamAbbreviation[];
}

