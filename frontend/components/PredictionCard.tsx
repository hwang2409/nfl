'use client';

import { Prediction } from '@/lib/api';

interface PredictionCardProps {
  prediction: Prediction;
}

const teamColors: Record<string, { primary: string; secondary: string }> = {
  KC: { primary: 'bg-red-600', secondary: 'bg-yellow-400' },
  BUF: { primary: 'bg-blue-600', secondary: 'bg-red-600' },
  SF: { primary: 'bg-red-600', secondary: 'bg-yellow-400' },
  DAL: { primary: 'bg-blue-600', secondary: 'bg-silver-400' },
  PHI: { primary: 'bg-green-600', secondary: 'bg-silver-400' },
  GB: { primary: 'bg-green-600', secondary: 'bg-yellow-400' },
  BAL: { primary: 'bg-purple-600', secondary: 'bg-yellow-400' },
  CIN: { primary: 'bg-orange-600', secondary: 'bg-black' },
  MIA: { primary: 'bg-teal-600', secondary: 'bg-orange-400' },
  LAC: { primary: 'bg-blue-600', secondary: 'bg-yellow-400' },
  // Add more teams as needed
};

const getTeamColor = (team: string) => {
  return teamColors[team] || { primary: 'bg-gray-600', secondary: 'bg-gray-400' };
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.7) return 'bg-green-500';
  if (confidence >= 0.4) return 'bg-yellow-500';
  return 'bg-gray-500';
};

export default function PredictionCard({ prediction }: PredictionCardProps) {
  const {
    home_team,
    away_team,
    home_win_probability,
    away_win_probability,
    predicted_winner,
    confidence,
    game_date,
    gametime,
    week,
  } = prediction;

  const homeColor = getTeamColor(home_team);
  const awayColor = getTeamColor(away_team);
  const confidenceColor = getConfidenceColor(confidence);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '';
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="text-sm text-gray-600">
          {game_date && (
            <div>
              <span className="font-semibold">Week {week}</span> • {formatDate(game_date)}
              {gametime && <span className="ml-2">{gametime}</span>}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Confidence</span>
          <div className="flex items-center gap-1">
            <div className={`w-2 h-2 rounded-full ${confidenceColor}`}></div>
            <span className="text-sm font-semibold">{(confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Teams */}
      <div className="space-y-3">
        {/* Away Team */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1">
            <div className={`w-12 h-12 rounded-full ${awayColor.primary} flex items-center justify-center text-white font-bold text-lg`}>
              {away_team}
            </div>
            <div className="flex-1">
              <div className="font-semibold text-lg">{away_team}</div>
              <div className="text-sm text-gray-600">
                {(away_win_probability * 100).toFixed(1)}% win probability
              </div>
            </div>
          </div>
          {predicted_winner === away_team && (
            <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-semibold">
              Winner
            </div>
          )}
        </div>

        {/* VS */}
        <div className="text-center text-gray-400 font-semibold">@</div>

        {/* Home Team */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1">
            <div className={`w-12 h-12 rounded-full ${homeColor.primary} flex items-center justify-center text-white font-bold text-lg`}>
              {home_team}
            </div>
            <div className="flex-1">
              <div className="font-semibold text-lg">{home_team}</div>
              <div className="text-sm text-gray-600">
                {(home_win_probability * 100).toFixed(1)}% win probability
              </div>
            </div>
          </div>
          {predicted_winner === home_team && (
            <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-semibold">
              Winner
            </div>
          )}
        </div>
      </div>

      {/* Probability Bars */}
      <div className="mt-4 pt-4 border-t">
        <div className="flex gap-2">
          <div className="flex-1">
            <div className="text-xs text-gray-600 mb-1">{away_team}</div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${away_win_probability * 100}%` }}
              ></div>
            </div>
          </div>
          <div className="flex-1">
            <div className="text-xs text-gray-600 mb-1">{home_team}</div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-red-600 h-2 rounded-full transition-all"
                style={{ width: `${home_win_probability * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

