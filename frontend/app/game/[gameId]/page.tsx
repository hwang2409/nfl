'use client';

import { useParams, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { predictionsApi, Prediction } from '@/lib/api';
import { ArrowLeft, Activity, TrendingUp, AlertCircle, DollarSign } from 'lucide-react';
import { getTeamName, getTeamLogo, getTeamShortName } from '@/lib/teamLogos';
import Image from 'next/image';
import LoadingSpinner from '@/components/LoadingSpinner';

// Placeholder data generator - replace with real API data
const generatePlaceholderData = (prediction: Prediction) => {
  const homeQB = {
    name: `${getTeamShortName(prediction.home_team)} QB`,
    team: prediction.home_team,
    qbr: 75 + Math.random() * 20,
    passingYards: 250 + Math.random() * 100,
    touchdowns: 1 + Math.floor(Math.random() * 3),
    interceptions: Math.floor(Math.random() * 2),
  };

  const awayQB = {
    name: `${getTeamShortName(prediction.away_team)} QB`,
    team: prediction.away_team,
    qbr: 75 + Math.random() * 20,
    passingYards: 250 + Math.random() * 100,
    touchdowns: 1 + Math.floor(Math.random() * 3),
    interceptions: Math.floor(Math.random() * 2),
  };

  const homeTeamStats = {
    epa: 0.1 + Math.random() * 0.2,
    dvoa: -5 + Math.random() * 20,
    elo: 1500 + Math.random() * 200,
  };

  const awayTeamStats = {
    epa: 0.1 + Math.random() * 0.2,
    dvoa: -5 + Math.random() * 20,
    elo: 1500 + Math.random() * 200,
  };

  const injuries = [
    { player: 'Sample Player', position: 'WR', status: 'Questionable', impact: 'Medium' },
    { player: 'Another Player', position: 'CB', status: 'Probable', impact: 'Low' },
  ];

  const bettingSpreads = [
    { source: 'DraftKings', spread: -3.5, overUnder: 48.5 },
    { source: 'FanDuel', spread: -3.0, overUnder: 49.0 },
    { source: 'Caesars', spread: -4.0, overUnder: 48.0 },
  ];

  return { homeQB, awayQB, homeTeamStats, awayTeamStats, injuries, bettingSpreads };
};

export default function GameDetailPage() {
  const params = useParams();
  const router = useRouter();
  const gameId = params?.gameId as string;
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Parse gameId (format: "homeTeam-awayTeam-week-season")
    const parts = gameId?.split('-');
    if (!parts || parts.length < 2) {
      setError('Invalid game ID');
      setLoading(false);
      return;
    }

    const loadGame = async () => {
      try {
        // Try to get the game from API
        // For now, we'll use placeholder data
        // In the future, you can call: predictionsApi.getGamePrediction(homeTeam, awayTeam, season, week)
        setError('Game data not available from API. Using placeholder data.');
      } catch (err: any) {
        setError(err.message || 'Failed to load game data');
      } finally {
        setLoading(false);
      }
    };

    loadGame();
  }, [gameId]);

  // For now, create a placeholder prediction from the gameId
  // In production, this would come from the API
  const getPredictionFromId = (): Prediction | null => {
    const parts = gameId?.split('-');
    if (!parts || parts.length < 2) return null;

    const homeTeam = parts[0];
    const awayTeam = parts[1];
    const week = parts[2] ? parseInt(parts[2]) : 15;
    const season = parts[3] ? parseInt(parts[3]) : new Date().getFullYear();

    return {
      season,
      week,
      game_date: new Date().toISOString().split('T')[0],
      gametime: '1:00 PM EST',
      home_team: homeTeam,
      away_team: awayTeam,
      home_win_probability: 0.55,
      away_win_probability: 0.45,
      predicted_winner: homeTeam,
      confidence: 0.75,
    };
  };

  const currentPrediction = prediction || getPredictionFromId();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <LoadingSpinner />
        </div>
      </div>
    );
  }

  if (!currentPrediction) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <button
            onClick={() => router.back()}
            className="mb-4 flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
            <p className="text-gray-600 dark:text-gray-400">Game not found</p>
          </div>
        </div>
      </div>
    );
  }

  const {
    home_team,
    away_team,
    home_win_probability,
    away_win_probability,
    game_date,
    gametime,
    week,
    confidence,
  } = currentPrediction;

  const homeProb = Math.round(home_win_probability * 100);
  const awayProb = Math.round(away_win_probability * 100);
  const confPercent = Math.round(confidence * 100);

  const homeName = getTeamShortName(home_team);
  const awayName = getTeamShortName(away_team);
  const homeFullName = getTeamName(home_team);
  const awayFullName = getTeamName(away_team);
  const homeLogo = getTeamLogo(home_team);
  const awayLogo = getTeamLogo(away_team);

  const {
    homeQB,
    awayQB,
    homeTeamStats,
    awayTeamStats,
    injuries,
    bettingSpreads,
  } = generatePlaceholderData(currentPrediction);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '';
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Back Button */}
        <button
          onClick={() => router.back()}
          className="mb-6 flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Predictions
        </button>

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
            <p className="text-yellow-600 dark:text-yellow-500 text-sm">{error}</p>
          </div>
        )}

        {/* Game Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <div className="text-center mb-4">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Week {week} • {formatDate(game_date)}
            </h1>
            {gametime && (
              <p className="text-gray-600 dark:text-gray-400">{gametime}</p>
            )}
          </div>

          <div className="flex items-center justify-center gap-8">
            <div className="flex flex-col items-center">
              <div className="w-20 h-20 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center overflow-hidden mb-2">
                <Image
                  src={awayLogo}
                  alt={awayFullName}
                  width={80}
                  height={80}
                  className="object-contain"
                />
              </div>
              <div className="font-semibold text-lg text-gray-900 dark:text-white">{awayName}</div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{awayProb}%</div>
            </div>

            <div className="text-2xl font-bold text-gray-400 dark:text-gray-500">@</div>

            <div className="flex flex-col items-center">
              <div className="w-20 h-20 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center overflow-hidden mb-2">
                <Image
                  src={homeLogo}
                  alt={homeFullName}
                  width={80}
                  height={80}
                  className="object-contain"
                />
              </div>
              <div className="font-semibold text-lg text-gray-900 dark:text-white">{homeName}</div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{homeProb}%</div>
            </div>
          </div>

          <div className="text-center mt-4 text-sm text-gray-600 dark:text-gray-400">
            Confidence: <span className="font-semibold">{confPercent}%</span>
          </div>
        </div>

        {/* QB Performance */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Quarterback Performance
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <div className="font-semibold text-gray-900 dark:text-white mb-3">{awayQB.name}</div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">QBR:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {awayQB.qbr.toFixed(1)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Yards:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {awayQB.passingYards.toFixed(0)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">TDs:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {awayQB.touchdowns}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">INTs:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {awayQB.interceptions}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <div className="font-semibold text-gray-900 dark:text-white mb-3">{homeQB.name}</div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">QBR:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {homeQB.qbr.toFixed(1)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">Yards:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {homeQB.passingYards.toFixed(0)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">TDs:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {homeQB.touchdowns}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">INTs:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {homeQB.interceptions}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Team Stats */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Team Statistics
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <div className="font-semibold text-gray-900 dark:text-white mb-3">{awayName}</div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">EPA/Play:</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {awayTeamStats.epa > 0 ? '+' : ''}{awayTeamStats.epa.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">DVOA:</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {awayTeamStats.dvoa > 0 ? '+' : ''}{awayTeamStats.dvoa.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Elo Rating:</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {awayTeamStats.elo.toFixed(0)}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <div className="font-semibold text-gray-900 dark:text-white mb-3">{homeName}</div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">EPA/Play:</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {homeTeamStats.epa > 0 ? '+' : ''}{homeTeamStats.epa.toFixed(3)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">DVOA:</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {homeTeamStats.dvoa > 0 ? '+' : ''}{homeTeamStats.dvoa.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Elo Rating:</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {homeTeamStats.elo.toFixed(0)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Injury Report */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            Injury Report
          </h2>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            {injuries.length > 0 ? (
              <div className="space-y-2">
                {injuries.map((injury, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700 last:border-0"
                  >
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">
                        {injury.player} ({injury.position})
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {injury.status} • Impact: {injury.impact}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-600 dark:text-gray-400 text-sm">No significant injuries reported</p>
            )}
          </div>
        </div>

        {/* Betting Spread Comparison */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Betting Spread Comparison
          </h2>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="space-y-3">
              {bettingSpreads.map((spread, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between py-2 border-b border-gray-200 dark:border-gray-700 last:border-0"
                >
                  <div className="font-medium text-gray-900 dark:text-white">{spread.source}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    <span className="mr-4">
                      Spread: {spread.spread > 0 ? '+' : ''}{spread.spread} ({homeName})
                    </span>
                    <span>O/U: {spread.overUnder}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

