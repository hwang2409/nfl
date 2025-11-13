'use client';

import { useState, useEffect } from 'react';
import { predictionsApi, Prediction } from '@/lib/api';
import PredictionCard from '@/components/PredictionCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import ErrorMessage from '@/components/ErrorMessage';

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

  const loadPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      // Check API health first
      try {
        await predictionsApi.healthCheck();
        setApiStatus((prev) => ({ ...prev, connected: true }));

        const status = await predictionsApi.getModelStatus();
        setApiStatus((prev) => ({ ...prev, modelLoaded: status.model_loaded }));
      } catch (err) {
        console.warn('API health check failed:', err);
      }

      const data = await predictionsApi.getUpcoming(
        season,
        week,
        minConfidence > 0 ? minConfidence : undefined
      );
      setPredictions(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load predictions');
      console.error('Error loading predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPredictions();
  }, [season, week, minConfidence]);

  const currentYear = new Date().getFullYear();
  const seasons = Array.from({ length: 5 }, (_, i) => currentYear - i);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-nfl-primary text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">NFL Game Predictions</h1>
              <p className="text-nfl-accent mt-1">AI-powered predictions with confidence scores</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-sm opacity-90">API Status</div>
                <div className="flex items-center gap-2 mt-1">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      apiStatus.connected ? 'bg-green-400' : 'bg-red-400'
                    }`}
                  ></div>
                  <span className="text-sm">
                    {apiStatus.connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Season</label>
              <select
                value={season}
                onChange={(e) => setSeason(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-nfl-primary"
              >
                {seasons.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Week (Optional)</label>
              <input
                type="number"
                min="1"
                max="18"
                value={week || ''}
                onChange={(e) =>
                  setWeek(e.target.value ? Number(e.target.value) : undefined)
                }
                placeholder="All weeks"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-nfl-primary"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Min Confidence: {(minConfidence * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6">
            <ErrorMessage message={error} onRetry={loadPredictions} />
          </div>
        )}

        {/* Predictions */}
        {loading ? (
          <LoadingSpinner />
        ) : predictions.length === 0 ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <p className="text-gray-600 text-lg">
              {error
                ? 'Unable to load predictions. Make sure the API server is running.'
                : 'No predictions found for the selected filters.'}
            </p>
            {!error && (
              <p className="text-gray-500 text-sm mt-2">
                Try adjusting the season, week, or confidence filter.
              </p>
            )}
          </div>
        ) : (
          <>
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-gray-800">
                {predictions.length} Prediction{predictions.length !== 1 ? 's' : ''} Found
              </h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {predictions.map((prediction, index) => (
                <PredictionCard key={index} prediction={prediction} />
              ))}
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white mt-12 py-6">
        <div className="container mx-auto px-4 text-center text-sm">
          <p>NFL Game Predictions • Powered by Machine Learning</p>
          <p className="text-gray-400 mt-2">
            API: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </p>
        </div>
      </footer>
    </div>
  );
}

