import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Prediction {
  season: number;
  week: number;
  game_date?: string;
  gametime?: string;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  away_win_probability: number;
  predicted_winner: string;
  confidence: number;
}

export interface QBStats {
  name: string;
  team: string;
  qbr: number;
  passingYards: number;
  touchdowns: number;
  interceptions: number;
}

export interface TeamStats {
  epa: number;
  dvoa: number;
  elo: number;
}

export interface Injury {
  player: string;
  position: string;
  status: string;
  impact: string;
}

export interface BettingSpread {
  source: string;
  spread: number;
  overUnder: number;
}

export interface TimelineDataPoint {
  timestamp: string;
  home_win_probability: number;
  away_win_probability: number;
  confidence: number;
}

export interface Weather {
  temperature: number;
  condition: string;
  windSpeed: number;
  humidity?: number;
  precipitation?: number;
  isDome: boolean;
}

export interface Stadium {
  name: string;
  city: string;
  state: string;
  capacity?: number;
  surface?: string;
  roofType?: string;
}

export interface GameDetails {
  prediction: Prediction;
  homeQB?: QBStats;
  awayQB?: QBStats;
  homeTeamStats?: TeamStats;
  awayTeamStats?: TeamStats;
  injuries?: Injury[];
  bettingSpreads?: BettingSpread[];
  timeline?: TimelineDataPoint[];
  weather?: Weather;
  stadium?: Stadium;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export const predictionsApi = {
  // Get upcoming predictions
  async getUpcoming(season?: number, week?: number, minConfidence?: number): Promise<Prediction[]> {
    const params: Record<string, string | number> = {};
    if (season) params.season = season;
    if (week) params.week = week;
    if (minConfidence) params.min_confidence = minConfidence;

    const response = await api.get('/api/v1/predictions/upcoming', { params });
    return response.data;
  },

  // Get specific game prediction
  async getGamePrediction(
    homeTeam: string,
    awayTeam: string,
    season?: number,
    week?: number
  ): Promise<Prediction> {
    const params: Record<string, string | number> = {};
    if (season) params.season = season;
    if (week) params.week = week;

    const response = await api.get(
      `/api/v1/predictions/game/${homeTeam}/${awayTeam}`,
      { params }
    );
    return response.data;
  },

  // Get team's upcoming games
  async getTeamPredictions(team: string, season?: number): Promise<Prediction[]> {
    const params: Record<string, string | number> = {};
    if (season) params.season = season;

    const response = await api.get(
      `/api/v1/predictions/team/${team}/upcoming`,
      { params }
    );
    return response.data;
  },

  // Health check
  async healthCheck(): Promise<{ status: string }> {
    const response = await api.get('/api/v1/health');
    return response.data;
  },

  // Model status
  async getModelStatus(): Promise<{
    model_loaded: boolean;
    model_version?: string;
    cache_enabled: boolean;
  }> {
    const response = await api.get('/api/v1/status');
    return response.data;
  },

  // Get detailed game information (includes QB stats, team stats, injuries, betting, timeline)
  async getGameDetails(
    homeTeam: string,
    awayTeam: string,
    season?: number,
    week?: number
  ): Promise<GameDetails | null> {
    const params: Record<string, string | number> = {};
    if (season) params.season = season;
    if (week) params.week = week;

    try {
      const response = await api.get(
        `/api/v1/predictions/game/${homeTeam}/${awayTeam}/details`,
        { params }
      );
      return response.data;
    } catch (error) {
      // Endpoint doesn't exist yet, return null to use placeholders
      return null;
    }
  },

  // Get prediction timeline for a specific game
  async getPredictionTimeline(
    homeTeam: string,
    awayTeam: string,
    season?: number,
    week?: number
  ): Promise<TimelineDataPoint[]> {
    const params: Record<string, string | number> = {};
    if (season) params.season = season;
    if (week) params.week = week;

    try {
      const response = await api.get(
        `/api/v1/predictions/timeline/${homeTeam}/${awayTeam}`,
        { params }
      );
      return response.data;
    } catch (error) {
      // Endpoint doesn't exist yet, return empty array to use placeholders
      return [];
    }
  },
};

export default api;

