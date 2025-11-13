'use client';

import { X, TrendingUp, Users, Activity, DollarSign, AlertCircle } from 'lucide-react';
import { Prediction } from '@/lib/api';
import { getTeamName, getTeamLogo } from '@/lib/teamLogos';
import Image from 'next/image';

interface GameDetailsModalProps {
  prediction: Prediction;
  isOpen: boolean;
  onClose: () => void;
}

interface QBStats {
  name: string;
  team: string;
  qbr: number;
  passingYards: number;
  touchdowns: number;
  interceptions: number;
}

interface TeamStats {
  epa: number;
  dvoa: number;
  elo: number;
}

interface Injury {
  player: string;
  position: string;
  status: string;
  impact: string;
}

interface BettingSpread {
  source: string;
  spread: number;
  overUnder: number;
}

// Placeholder data generator - replace with real API data
const generatePlaceholderData = (prediction: Prediction) => {
  const homeQB: QBStats = {
    name: `${prediction.home_team} QB`,
    team: prediction.home_team,
    qbr: 75 + Math.random() * 20,
    passingYards: 250 + Math.random() * 100,
    touchdowns: 1 + Math.floor(Math.random() * 3),
    interceptions: Math.floor(Math.random() * 2),
  };

  const awayQB: QBStats = {
    name: `${prediction.away_team} QB`,
    team: prediction.away_team,
    qbr: 75 + Math.random() * 20,
    passingYards: 250 + Math.random() * 100,
    touchdowns: 1 + Math.floor(Math.random() * 3),
    interceptions: Math.floor(Math.random() * 2),
  };

  const homeTeamStats: TeamStats = {
    epa: 0.1 + Math.random() * 0.2,
    dvoa: -5 + Math.random() * 20,
    elo: 1500 + Math.random() * 200,
  };

  const awayTeamStats: TeamStats = {
    epa: 0.1 + Math.random() * 0.2,
    dvoa: -5 + Math.random() * 20,
    elo: 1500 + Math.random() * 200,
  };

  const injuries: Injury[] = [
    {
      player: 'Sample Player',
      position: 'WR',
      status: 'Questionable',
      impact: 'Medium',
    },
    {
      player: 'Another Player',
      position: 'CB',
      status: 'Probable',
      impact: 'Low',
    },
  ];

  const bettingSpreads: BettingSpread[] = [
    { source: 'DraftKings', spread: -3.5, overUnder: 48.5 },
    { source: 'FanDuel', spread: -3.0, overUnder: 49.0 },
    { source: 'Caesars', spread: -4.0, overUnder: 48.0 },
  ];

  return {
    homeQB,
    awayQB,
    homeTeamStats,
    awayTeamStats,
    injuries,
    bettingSpreads,
  };
};

export default function GameDetailsModal({ prediction, isOpen, onClose }: GameDetailsModalProps) {
  if (!isOpen) return null;

  const {
    home_team,
    away_team,
    home_win_probability,
    away_win_probability,
    game_date,
    gametime,
    week,
    confidence,
  } = prediction;

  const homeProb = Math.round(home_win_probability * 100);
  const awayProb = Math.round(away_win_probability * 100);
  const confPercent = Math.round(confidence * 100);

  const homeName = getTeamName(home_team);
  const awayName = getTeamName(away_team);
  const homeLogo = getTeamLogo(home_team);
  const awayLogo = getTeamLogo(away_team);

  const {
    homeQB,
    awayQB,
    homeTeamStats,
    awayTeamStats,
    injuries,
    bettingSpreads,
  } = generatePlaceholderData(prediction);

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
    <div
      className="fixed inset-0 z-50 overflow-y-auto"
      onClick={onClose}
    >
      <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div
          className="fixed inset-0 transition-opacity bg-gray-500 bg-opacity-75 dark:bg-gray-900 dark:bg-opacity-75"
          aria-hidden="true"
        ></div>

        {/* Modal panel */}
        <div
          className="inline-block align-bottom bg-white dark:bg-gray-800 rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gray-50 dark:bg-gray-900 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Game Details - Week {week}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {formatDate(game_date)} {gametime && `• ${gametime}`}
                </p>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="px-6 py-4 max-h-[calc(100vh-200px)] overflow-y-auto">
            {/* Game Matchup */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4 flex-1">
                  <div className="w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center overflow-hidden">
                    <Image
                      src={awayLogo}
                      alt={awayName}
                      width={64}
                      height={64}
                      className="object-contain"
                    />
                  </div>
                  <div>
                    <div className="font-semibold text-lg text-gray-900 dark:text-white">{awayName}</div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{awayProb}%</div>
                  </div>
                </div>
                <div className="text-gray-400 dark:text-gray-500 font-medium">@</div>
                <div className="flex items-center gap-4 flex-1 justify-end">
                  <div className="text-right">
                    <div className="font-semibold text-lg text-gray-900 dark:text-white">{homeName}</div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{homeProb}%</div>
                  </div>
                  <div className="w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center overflow-hidden">
                    <Image
                      src={homeLogo}
                      alt={homeName}
                      width={64}
                      height={64}
                      className="object-contain"
                    />
                  </div>
                </div>
              </div>
              <div className="text-center text-sm text-gray-600 dark:text-gray-400">
                Confidence: <span className="font-semibold">{confPercent}%</span>
              </div>
            </div>

            {/* QB Performance */}
            <div className="mb-6">
              <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Quarterback Performance
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Away QB */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <div className="font-semibold text-gray-900 dark:text-white mb-2">{awayQB.name}</div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
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

                {/* Home QB */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <div className="font-semibold text-gray-900 dark:text-white mb-2">{homeQB.name}</div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
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

            {/* Team Stats (EPA, DVOA, Elo) */}
            <div className="mb-6">
              <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Team Statistics
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Away Team Stats */}
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

                {/* Home Team Stats */}
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

            {/* Injury List */}
            <div className="mb-6">
              <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                Injury Report
              </h4>
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
            <div>
              <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <DollarSign className="w-5 h-5" />
                Betting Spread Comparison
              </h4>
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

          {/* Footer */}
          <div className="bg-gray-50 dark:bg-gray-900 px-6 py-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={onClose}
              className="w-full bg-nfl-primary hover:bg-nfl-primary/90 text-white font-medium py-2 px-4 rounded-lg transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

