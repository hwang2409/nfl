'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Prediction } from '@/lib/api';
import { Clock, TrendingUp } from 'lucide-react';
import { getTeamLogo, getTeamShortName } from '@/lib/teamLogos';

interface PredictionCardProps {
  prediction: Prediction;
}

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.7) return 'text-green-600';
  if (confidence >= 0.4) return 'text-yellow-600';
  return 'text-gray-500';
};

interface TeamLogoProps {
  teamAbbrev: string;
  teamName: string;
  logoPath: string;
}

function TeamLogo({ teamAbbrev, teamName, logoPath }: TeamLogoProps) {
  const [imageError, setImageError] = useState(false);

  if (imageError) {
    // Fallback: show team abbreviation in a colored circle
    return (
      <div className="w-12 h-12 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
        <span className="text-gray-600 dark:text-gray-300 font-semibold text-sm">{teamAbbrev}</span>
      </div>
    );
  }

  return (
    <div className="w-12 h-12 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center overflow-hidden flex-shrink-0">
      <Image
        src={logoPath}
        alt={`${teamName} logo`}
        width={48}
        height={48}
        className="object-contain"
        onError={() => setImageError(true)}
      />
    </div>
  );
}

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
    season,
  } = prediction;

  const confidenceColorClass = getConfidenceColor(confidence);

  // Ensure probabilities sum to 100% (handle rounding)
  const homeProb = Math.max(0, Math.min(1, home_win_probability));
  const awayProb = Math.max(0, Math.min(1, away_win_probability));
  const totalProb = homeProb + awayProb;
  
  // Normalize if they don't sum to 1.0 (due to rounding)
  const normalizedHomeProb = totalProb > 0 ? homeProb / totalProb : 0.5;
  const normalizedAwayProb = totalProb > 0 ? awayProb / totalProb : 0.5;
  
  // Format as percentages
  const homeProbPercent = Math.round(normalizedHomeProb * 100);
  const awayProbPercent = Math.round(normalizedAwayProb * 100);
  
  // Confidence is already 0-1, convert to percentage
  const confidencePercent = Math.round(confidence * 100);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return '';
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  const homeLogo = getTeamLogo(home_team);
  const awayLogo = getTeamLogo(away_team);
  const homeName = getTeamShortName(home_team);
  const awayName = getTeamShortName(away_team);

  // Create game ID for routing: homeTeam-awayTeam-week-season
  const gameId = `${home_team}-${away_team}-${week}-${season}`;

  const cardContent = (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md transition-all duration-200 overflow-hidden cursor-pointer">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <Clock className="w-4 h-4" />
            <span>Week {week}</span>
            {game_date && (
              <>
                <span>•</span>
                <span>{formatDate(game_date)}</span>
              </>
            )}
            {gametime && (
              <>
                <span>•</span>
                <span>{gametime}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            <TrendingUp className={`w-3.5 h-3.5 ${confidenceColorClass}`} />
            <span className={`text-xs font-medium ${confidenceColorClass}`}>
              {confidencePercent}%
            </span>
          </div>
        </div>
      </div>

      {/* Game Content */}
      <div className="p-4">
        {/* Away Team */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <TeamLogo
              teamAbbrev={away_team}
              teamName={awayName}
              logoPath={awayLogo}
            />
            <div className="min-w-0 flex-1">
              <div className="font-semibold text-gray-900 dark:text-white truncate">{awayName}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Away</div>
            </div>
          </div>
          <div className="text-right flex-shrink-0">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">{awayProbPercent}%</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Win Probability</div>
          </div>
        </div>

        {/* Divider */}
        <div className="flex items-center gap-2 my-3">
          <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700"></div>
          <span className="text-xs text-gray-400 dark:text-gray-500 font-medium">VS</span>
          <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700"></div>
        </div>

        {/* Home Team */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TeamLogo
              teamAbbrev={home_team}
              teamName={homeName}
              logoPath={homeLogo}
            />
            <div className="min-w-0 flex-1">
              <div className="font-semibold text-gray-900 dark:text-white truncate">{homeName}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Home</div>
            </div>
          </div>
          <div className="text-right flex-shrink-0">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">{homeProbPercent}%</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Win Probability</div>
          </div>
        </div>

        {/* Probability Bar */}
        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
          <div className="flex gap-2 mb-2">
            <div 
              className="h-2 bg-blue-500 rounded-full transition-all"
              style={{ width: `${awayProbPercent}%` }}
            />
            <div 
              className="h-2 bg-red-500 rounded-full transition-all"
              style={{ width: `${homeProbPercent}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-xs">
            <div className="flex items-center gap-2">
              <span className="text-gray-500 dark:text-gray-400 truncate">{away_team}</span>
              {predicted_winner === away_team && (
                <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full font-medium whitespace-nowrap">
                  Winner
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {predicted_winner === home_team && (
                <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full font-medium whitespace-nowrap">
                  Winner
                </span>
              )}
              <span className="text-gray-500 dark:text-gray-400 truncate">{home_team}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <Link href={`/game/${gameId}`} className="block">
      {cardContent}
    </Link>
  );
}
