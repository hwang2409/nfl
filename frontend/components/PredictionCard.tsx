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

  // Ensure probabilities sum to 100% (handle rounding)
  const homeProb = Math.max(0, Math.min(1, home_win_probability));
  const awayProb = Math.max(0, Math.min(1, away_win_probability));
  const totalProb = homeProb + awayProb;
  
  // Normalize if they don't sum to 1.0 (due to rounding)
  const normalizedHomeProb = totalProb > 0 ? homeProb / totalProb : 0.5;
  const normalizedAwayProb = totalProb > 0 ? awayProb / totalProb : 0.5;
  
  // Format as percentages
  const homeProbPercent = (normalizedHomeProb * 100).toFixed(1);
  const awayProbPercent = (normalizedAwayProb * 100).toFixed(1);
  
  // Confidence is already 0-1, convert to percentage
  const confidencePercent = (confidence * 100).toFixed(0);

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
            <span className="text-sm font-semibold">{confidencePercent}%</span>
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
                {awayProbPercent}% win probability
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
                {homeProbPercent}% win probability
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
        <div className="space-y-2">
          <div>
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>{away_team}</span>
              <span className="font-semibold">{awayProbPercent}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all"
                style={{ width: `${awayProbPercent}%` }}
              ></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>{home_team}</span>
              <span className="font-semibold">{homeProbPercent}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-red-600 h-3 rounded-full transition-all"
                style={{ width: `${homeProbPercent}%` }}
              ></div>
            </div>
          </div>
        </div>
        {/* Confidence indicator */}
        <div className="mt-3 pt-3 border-t">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-600">Model Confidence:</span>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${confidenceColor}`}></div>
              <span className="font-semibold text-gray-800">
                {confidencePercent}% 
                {confidence >= 0.7 && ' (High)'}
                {confidence >= 0.4 && confidence < 0.7 && ' (Medium)'}
                {confidence < 0.4 && ' (Low)'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

