'use client';

import { useState, useEffect } from 'react';
import { predictionsApi, Prediction } from '@/lib/api';
import PredictionCard from '@/components/PredictionCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import ErrorState from '@/components/ErrorState';
import EmptyState from '@/components/EmptyState';
import PredictionAccuracyCard from '@/components/PredictionAccuracyCard';
import { Filter, Search } from 'lucide-react';

// Placeholder predictions for when API is unavailable
const placeholderPredictions: Prediction[] = [
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-15',
    gametime: '4:25 PM EST',
    home_team: 'KC',
    away_team: 'BUF',
    home_win_probability: 0.58,
    away_win_probability: 0.42,
    predicted_winner: 'KC',
    confidence: 0.82,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-15',
    gametime: '8:20 PM EST',
    home_team: 'SF',
    away_team: 'DAL',
    home_win_probability: 0.65,
    away_win_probability: 0.35,
    predicted_winner: 'SF',
    confidence: 0.78,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-15',
    gametime: '1:00 PM EST',
    home_team: 'PHI',
    away_team: 'GB',
    home_win_probability: 0.52,
    away_win_probability: 0.48,
    predicted_winner: 'PHI',
    confidence: 0.71,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-15',
    gametime: '1:00 PM EST',
    home_team: 'BAL',
    away_team: 'MIA',
    home_win_probability: 0.61,
    away_win_probability: 0.39,
    predicted_winner: 'BAL',
    confidence: 0.85,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-16',
    gametime: '1:00 PM EST',
    home_team: 'DET',
    away_team: 'TB',
    home_win_probability: 0.55,
    away_win_probability: 0.45,
    predicted_winner: 'DET',
    confidence: 0.73,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-16',
    gametime: '4:25 PM EST',
    home_team: 'CIN',
    away_team: 'CLE',
    home_win_probability: 0.48,
    away_win_probability: 0.52,
    predicted_winner: 'CLE',
    confidence: 0.69,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-16',
    gametime: '4:05 PM EST',
    home_team: 'LAC',
    away_team: 'DEN',
    home_win_probability: 0.57,
    away_win_probability: 0.43,
    predicted_winner: 'LAC',
    confidence: 0.76,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-16',
    gametime: '4:25 PM EST',
    home_team: 'SEA',
    away_team: 'ARI',
    home_win_probability: 0.63,
    away_win_probability: 0.37,
    predicted_winner: 'SEA',
    confidence: 0.81,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-17',
    gametime: '8:15 PM EST',
    home_team: 'NYJ',
    away_team: 'NE',
    home_win_probability: 0.45,
    away_win_probability: 0.55,
    predicted_winner: 'NE',
    confidence: 0.68,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-17',
    gametime: '1:00 PM EST',
    home_team: 'JAX',
    away_team: 'HOU',
    home_win_probability: 0.54,
    away_win_probability: 0.46,
    predicted_winner: 'JAX',
    confidence: 0.74,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-17',
    gametime: '1:00 PM EST',
    home_team: 'PIT',
    away_team: 'IND',
    home_win_probability: 0.50,
    away_win_probability: 0.50,
    predicted_winner: 'PIT',
    confidence: 0.70,
  },
  {
    season: 2024,
    week: 15,
    game_date: '2024-12-17',
    gametime: '1:00 PM EST',
    home_team: 'NO',
    away_team: 'ATL',
    home_win_probability: 0.56,
    away_win_probability: 0.44,
    predicted_winner: 'NO',
    confidence: 0.72,
  },
];

export default function Home() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [season, setSeason] = useState<number>(new Date().getFullYear());
  const [week, setWeek] = useState<number | undefined>(undefined);
  const [minConfidence, setMinConfidence] = useState<number>(0);
  const [apiStatus, setApiStatus] = useState<{ connected: boolean; modelLoaded: boolean }>({
    connected: false,
    modelLoaded: false,
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  const loadPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      // Check API health first
      try {
        const health = await predictionsApi.healthCheck();
        setApiStatus((prev) => ({ ...prev, connected: true }));

        const status = await predictionsApi.getModelStatus();
        setApiStatus((prev) => ({ ...prev, modelLoaded: status.model_loaded }));
      } catch (err: any) {
        console.warn('API health check failed:', err);
        setApiStatus((prev) => ({ ...prev, connected: false }));
        // Don't return early - still try to show error message
      }

      const data = await predictionsApi.getUpcoming(
        season,
        week,
        minConfidence > 0 ? minConfidence : undefined
      );
      // If API returns empty array or no data, use placeholders
      if (!data || data.length === 0) {
        setError('No predictions available from API. Showing sample data.');
        setPredictions(placeholderPredictions);
      } else {
        setPredictions(data);
      }
    } catch (err: any) {
      // Handle different error types
      const errorMessage = err.response?.data?.detail 
        || err.response?.data?.message 
        || err.message 
        || 'Failed to load predictions';
      setError(errorMessage);
      console.error('Error loading predictions:', err);
      // Use placeholder data when API fails
      setPredictions(placeholderPredictions);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPredictions();
  }, [season, week, minConfidence]);

  const currentYear = new Date().getFullYear();
  const seasons = Array.from({ length: 5 }, (_, i) => currentYear - i);
  const weeks = Array.from({ length: 18 }, (_, i) => i + 1);

  // Filter predictions by search query, week, and confidence
  const filteredPredictions = predictions.filter((prediction) => {
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      if (
        !prediction.home_team.toLowerCase().includes(query) &&
        !prediction.away_team.toLowerCase().includes(query)
      ) {
        return false;
      }
    }
    
    // Week filter
    if (week !== undefined && prediction.week !== week) {
      return false;
    }
    
    // Confidence filter
    if (minConfidence > 0 && prediction.confidence < minConfidence) {
      return false;
    }
    
    return true;
  });

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* API Status and Filters Bar */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row gap-4">
            {/* API Status */}
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  apiStatus.connected ? 'bg-green-500' : 'bg-red-500'
                }`}
              ></div>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                {apiStatus.connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500" />
              <input
                type="text"
                placeholder="Search teams..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-nfl-primary dark:focus:ring-nfl-accent focus:border-transparent text-sm"
              />
            </div>

            {/* Filters Toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors whitespace-nowrap"
            >
              <Filter className="w-4 h-4" />
              <span>Filters</span>
            </button>
          </div>

          {/* Filters */}
          {showFilters && (
            <div className="flex flex-wrap items-center gap-4 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">Season:</label>
                <select
                  value={season}
                  onChange={(e) => setSeason(Number(e.target.value))}
                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-nfl-primary dark:focus:ring-nfl-accent focus:border-transparent text-sm"
                >
                  {seasons.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">Week:</label>
                <select
                  value={week || 'all'}
                  onChange={(e) =>
                    setWeek(e.target.value === 'all' ? undefined : Number(e.target.value))
                  }
                  className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-nfl-primary dark:focus:ring-nfl-accent focus:border-transparent text-sm"
                >
                  <option value="all">All Weeks</option>
                  {weeks.map((w) => (
                    <option key={w} value={w}>
                      Week {w}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2 min-w-[200px]">
                <label className="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
                  Min Confidence: {(minConfidence * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(Number(e.target.value))}
                  className="flex-1"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Model Accuracy Card */}
        {!loading && (
          <div className="mb-6">
            <PredictionAccuracyCard />
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="mb-6">
            <ErrorState
              message={error}
              onRetry={loadPredictions}
              showPlaceholderInfo={predictions.length > 0}
            />
          </div>
        )}

        {/* Results Count */}
        {!loading && filteredPredictions.length > 0 && (
          <div className="mb-6">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {filteredPredictions.length} prediction{filteredPredictions.length !== 1 ? 's' : ''} found
              {error && <span className="text-gray-400 dark:text-gray-500 ml-2">(placeholder data)</span>}
            </p>
          </div>
        )}

        {/* Predictions */}
        {loading ? (
          <LoadingSpinner />
        ) : filteredPredictions.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredPredictions.map((prediction, index) => (
              <PredictionCard key={index} prediction={prediction} />
            ))}
          </div>
        )}
      </main>

      {/* Minimal Footer */}
      <footer className="mt-16 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            NFL Game Predictions • Powered by Machine Learning
          </p>
        </div>
      </footer>
    </div>
  );
}
