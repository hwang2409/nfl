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
};

export default api;

