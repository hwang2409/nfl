'use client';

import { useParams, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { predictionsApi, Prediction, QBStats, TeamStats, Injury, BettingSpread, TimelineDataPoint, Weather, Stadium } from '@/lib/api';
import { ArrowLeft, Activity, TrendingUp, AlertCircle, DollarSign } from 'lucide-react';
import { getTeamName, getTeamLogo, getTeamShortName } from '@/lib/teamLogos';
import Image from 'next/image';
import LoadingSpinner from '@/components/LoadingSpinner';
import ProbabilityTimeline from '@/components/ProbabilityTimeline';
import WeatherStadium from '@/components/WeatherStadium';

// Placeholder data generator - replace with real API data
const generatePlaceholderData = (prediction: Prediction) => {
  const homeQB: QBStats = {
    name: `${getTeamShortName(prediction.home_team)} QB`,
    team: prediction.home_team,
    qbr: 75 + Math.random() * 20,
    passingYards: 250 + Math.random() * 100,
    touchdowns: 1 + Math.floor(Math.random() * 3),
    interceptions: Math.floor(Math.random() * 2),
  };

  const awayQB: QBStats = {
    name: `${getTeamShortName(prediction.away_team)} QB`,
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
    { player: 'Sample Player', position: 'WR', status: 'Questionable', impact: 'Medium' },
    { player: 'Another Player', position: 'CB', status: 'Probable', impact: 'Low' },
  ];

  const bettingSpreads: BettingSpread[] = [
    { source: 'DraftKings', spread: -3.5, overUnder: 48.5 },
    { source: 'FanDuel', spread: -3.0, overUnder: 49.0 },
    { source: 'Caesars', spread: -4.0, overUnder: 48.0 },
  ];

  // Stadium mapping
  const stadiumMap: Record<string, Stadium> = {
    'BUF': { name: 'Highmark Stadium', city: 'Orchard Park', state: 'NY', capacity: 71608, surface: 'FieldTurf', roofType: 'Open' },
    'MIA': { name: 'Hard Rock Stadium', city: 'Miami Gardens', state: 'FL', capacity: 65326, surface: 'FieldTurf', roofType: 'Open' },
    'NE': { name: 'Gillette Stadium', city: 'Foxborough', state: 'MA', capacity: 65878, surface: 'FieldTurf', roofType: 'Open' },
    'NYJ': { name: 'MetLife Stadium', city: 'East Rutherford', state: 'NJ', capacity: 82500, surface: 'FieldTurf', roofType: 'Open' },
    'NYG': { name: 'MetLife Stadium', city: 'East Rutherford', state: 'NJ', capacity: 82500, surface: 'FieldTurf', roofType: 'Open' },
    'BAL': { name: 'M&T Bank Stadium', city: 'Baltimore', state: 'MD', capacity: 71008, surface: 'FieldTurf', roofType: 'Open' },
    'CIN': { name: 'Paycor Stadium', city: 'Cincinnati', state: 'OH', capacity: 65515, surface: 'FieldTurf', roofType: 'Open' },
    'CLE': { name: 'Cleveland Browns Stadium', city: 'Cleveland', state: 'OH', capacity: 67595, surface: 'FieldTurf', roofType: 'Open' },
    'PIT': { name: 'Acrisure Stadium', city: 'Pittsburgh', state: 'PA', capacity: 68400, surface: 'Grass', roofType: 'Open' },
    'HOU': { name: 'NRG Stadium', city: 'Houston', state: 'TX', capacity: 72220, surface: 'FieldTurf', roofType: 'Retractable' },
    'IND': { name: 'Lucas Oil Stadium', city: 'Indianapolis', state: 'IN', capacity: 67000, surface: 'FieldTurf', roofType: 'Retractable' },
    'JAX': { name: 'EverBank Stadium', city: 'Jacksonville', state: 'FL', capacity: 67814, surface: 'Grass', roofType: 'Open' },
    'TEN': { name: 'Nissan Stadium', city: 'Nashville', state: 'TN', capacity: 69143, surface: 'FieldTurf', roofType: 'Open' },
    'DEN': { name: 'Empower Field at Mile High', city: 'Denver', state: 'CO', capacity: 76125, surface: 'Grass', roofType: 'Open' },
    'KC': { name: 'GEHA Field at Arrowhead Stadium', city: 'Kansas City', state: 'MO', capacity: 76416, surface: 'Grass', roofType: 'Open' },
    'LV': { name: 'Allegiant Stadium', city: 'Las Vegas', state: 'NV', capacity: 65000, surface: 'FieldTurf', roofType: 'Dome' },
    'LAC': { name: 'SoFi Stadium', city: 'Inglewood', state: 'CA', capacity: 70240, surface: 'FieldTurf', roofType: 'Fixed' },
    'LAR': { name: 'SoFi Stadium', city: 'Inglewood', state: 'CA', capacity: 70240, surface: 'FieldTurf', roofType: 'Fixed' },
    'DAL': { name: 'AT&T Stadium', city: 'Arlington', state: 'TX', capacity: 80000, surface: 'FieldTurf', roofType: 'Retractable' },
    'PHI': { name: 'Lincoln Financial Field', city: 'Philadelphia', state: 'PA', capacity: 69596, surface: 'Grass', roofType: 'Open' },
    'WAS': { name: 'FedExField', city: 'Landover', state: 'MD', capacity: 82000, surface: 'Grass', roofType: 'Open' },
    'CHI': { name: 'Soldier Field', city: 'Chicago', state: 'IL', capacity: 61500, surface: 'Grass', roofType: 'Open' },
    'DET': { name: 'Ford Field', city: 'Detroit', state: 'MI', capacity: 65000, surface: 'FieldTurf', roofType: 'Dome' },
    'GB': { name: 'Lambeau Field', city: 'Green Bay', state: 'WI', capacity: 81441, surface: 'Grass', roofType: 'Open' },
    'MIN': { name: 'U.S. Bank Stadium', city: 'Minneapolis', state: 'MN', capacity: 66655, surface: 'FieldTurf', roofType: 'Fixed' },
    'ATL': { name: 'Mercedes-Benz Stadium', city: 'Atlanta', state: 'GA', capacity: 71000, surface: 'FieldTurf', roofType: 'Retractable' },
    'CAR': { name: 'Bank of America Stadium', city: 'Charlotte', state: 'NC', capacity: 75523, surface: 'Grass', roofType: 'Open' },
    'NO': { name: 'Caesars Superdome', city: 'New Orleans', state: 'LA', capacity: 73208, surface: 'FieldTurf', roofType: 'Fixed' },
    'TB': { name: 'Raymond James Stadium', city: 'Tampa', state: 'FL', capacity: 65890, surface: 'Grass', roofType: 'Open' },
    'ARI': { name: 'State Farm Stadium', city: 'Glendale', state: 'AZ', capacity: 63400, surface: 'FieldTurf', roofType: 'Retractable' },
    'SF': { name: 'Levi\'s Stadium', city: 'Santa Clara', state: 'CA', capacity: 68500, surface: 'FieldTurf', roofType: 'Open' },
    'SEA': { name: 'Lumen Field', city: 'Seattle', state: 'WA', capacity: 68000, surface: 'FieldTurf', roofType: 'Open' },
  };

  const domeStadiums = ['ATL', 'DET', 'IND', 'NO', 'DAL', 'HOU', 'MIN', 'LV', 'LAR', 'LAC'];
  const isDome = domeStadiums.includes(prediction.home_team);

  const weather: Weather = {
    temperature: isDome ? 72 : 45 + Math.floor(Math.random() * 40),
    condition: isDome ? 'Indoor' : ['Sunny', 'Partly Cloudy', 'Cloudy', 'Clear'][Math.floor(Math.random() * 4)],
    windSpeed: isDome ? 0 : Math.floor(Math.random() * 20),
    humidity: isDome ? undefined : 40 + Math.floor(Math.random() * 40),
    precipitation: isDome ? 0 : Math.random() < 0.3 ? Math.floor(Math.random() * 30) : 0,
    isDome,
  };

  const stadium = stadiumMap[prediction.home_team] || {
    name: `${getTeamName(prediction.home_team)} Stadium`,
    city: 'Unknown',
    state: 'Unknown',
  };

  return { homeQB, awayQB, homeTeamStats, awayTeamStats, injuries, bettingSpreads, weather, stadium };
};

export default function GameDetailPage() {
  const params = useParams();
  const router = useRouter();
  const gameId = params?.gameId as string;
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gameDetails, setGameDetails] = useState<{
    homeQB?: QBStats;
    awayQB?: QBStats;
    homeTeamStats?: TeamStats;
    awayTeamStats?: TeamStats;
    injuries?: Injury[];
    bettingSpreads?: BettingSpread[];
    timeline?: TimelineDataPoint[];
    weather?: Weather;
    stadium?: Stadium;
  } | null>(null);

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
        const homeTeam = parts[0];
        const awayTeam = parts[1];
        const week = parts[2] ? parseInt(parts[2]) : undefined;
        const season = parts[3] ? parseInt(parts[3]) : undefined;

        // Try to get detailed game data from API
        const details = await predictionsApi.getGameDetails(homeTeam, awayTeam, season, week);
        
        if (details) {
          // Use real API data
          setPrediction(details.prediction);
          setGameDetails({
            homeQB: details.homeQB,
            awayQB: details.awayQB,
            homeTeamStats: details.homeTeamStats,
            awayTeamStats: details.awayTeamStats,
            injuries: details.injuries,
            bettingSpreads: details.bettingSpreads,
            timeline: details.timeline,
            weather: details.weather,
            stadium: details.stadium,
          });
          setError(null);
        } else {
          // Try to get basic prediction
          try {
            const gameData = await predictionsApi.getGamePrediction(homeTeam, awayTeam, season, week);
            setPrediction(gameData);
            setError(null);
          } catch (apiErr) {
            // API failed, use placeholder
            setError('Game data not available from API. Using placeholder data.');
          }

          // Try to get timeline separately
          try {
            const timeline = await predictionsApi.getPredictionTimeline(homeTeam, awayTeam, season, week);
            if (timeline && timeline.length > 0) {
              setGameDetails((prev) => ({ ...prev, timeline }));
            }
          } catch (timelineErr) {
            // Timeline not available, will use placeholder
          }
        }
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

  // Use API data if available, otherwise use placeholders
  const placeholderData = generatePlaceholderData(currentPrediction);
  const homeQB = gameDetails?.homeQB || placeholderData.homeQB;
  const awayQB = gameDetails?.awayQB || placeholderData.awayQB;
  const homeTeamStats = gameDetails?.homeTeamStats || placeholderData.homeTeamStats;
  const awayTeamStats = gameDetails?.awayTeamStats || placeholderData.awayTeamStats;
  const injuries = gameDetails?.injuries || placeholderData.injuries;
  const bettingSpreads = gameDetails?.bettingSpreads || placeholderData.bettingSpreads;
  const timelineData = gameDetails?.timeline;
  const weather = gameDetails?.weather || placeholderData.weather;
  const stadium = gameDetails?.stadium || placeholderData.stadium;

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

        {/* Probability Timeline Graph and Weather/Stadium */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6 items-stretch">
          <div className="lg:col-span-2">
            <ProbabilityTimeline
              data={timelineData}
              homeTeam={homeName}
              awayTeam={awayName}
              currentHomeProb={homeProb / 100}
              currentAwayProb={awayProb / 100}
            />
          </div>
          <div className="lg:col-span-1">
            <WeatherStadium
              weather={weather}
              stadium={stadium}
            />
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

