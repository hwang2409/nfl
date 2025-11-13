'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TimelineDataPoint } from '@/lib/api';

interface ProbabilityTimelineProps {
  data?: TimelineDataPoint[];
  homeTeam: string;
  awayTeam: string;
  currentHomeProb: number;
  currentAwayProb: number;
}

type TimeRange = '1d' | '1w' | '1m' | 'all';

// Generate placeholder timeline data
const generatePlaceholderTimeline = (
  homeTeam: string,
  awayTeam: string,
  currentHomeProb: number,
  currentAwayProb: number,
  daysBack: number = 30
): TimelineDataPoint[] => {
  const data: TimelineDataPoint[] = [];
  const now = new Date();
  
  // Generate data points - more frequent for recent data
  // For last 24 hours: every hour
  // For days 2-7: every 6 hours
  // For days 8-30: every 12 hours
  
  // Last 24 hours (hourly)
  for (let i = 24; i >= 0; i--) {
    const date = new Date(now);
    date.setHours(date.getHours() - i);
    date.setMinutes(0, 0, 0);
    
    const variation = (Math.random() - 0.5) * 0.06;
    const timeFactor = i / 24;
    
    const homeProb = Math.max(0.1, Math.min(0.9, currentHomeProb + variation * timeFactor));
    const awayProb = 1 - homeProb;
    const confidence = 0.7 + (1 - timeFactor) * 0.25;
    
    data.push({
      timestamp: date.toISOString(),
      home_win_probability: Math.round(homeProb * 100) / 100,
      away_win_probability: Math.round(awayProb * 100) / 100,
      confidence: Math.round(confidence * 100) / 100,
    });
  }
  
  // Days 2-7 (every 6 hours)
  for (let day = 2; day <= 7; day++) {
    for (let hour = 0; hour < 24; hour += 6) {
      const date = new Date(now);
      date.setDate(date.getDate() - day);
      date.setHours(hour, 0, 0, 0);
      
      const variation = (Math.random() - 0.5) * 0.08;
      const timeFactor = day / 7;
      
      const homeProb = Math.max(0.1, Math.min(0.9, currentHomeProb + variation * timeFactor));
      const awayProb = 1 - homeProb;
      const confidence = 0.65 + (1 - timeFactor) * 0.3;
      
      data.push({
        timestamp: date.toISOString(),
        home_win_probability: Math.round(homeProb * 100) / 100,
        away_win_probability: Math.round(awayProb * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
      });
    }
  }
  
  // Days 8-30 (every 12 hours)
  for (let day = 8; day <= daysBack; day++) {
    for (let hour = 0; hour < 24; hour += 12) {
      const date = new Date(now);
      date.setDate(date.getDate() - day);
      date.setHours(hour, 0, 0, 0);
      
      const variation = (Math.random() - 0.5) * 0.1;
      const timeFactor = day / daysBack;
      
      const homeProb = Math.max(0.1, Math.min(0.9, currentHomeProb + variation * timeFactor));
      const awayProb = 1 - homeProb;
      const confidence = 0.6 + (1 - timeFactor) * 0.35;
      
      data.push({
        timestamp: date.toISOString(),
        home_win_probability: Math.round(homeProb * 100) / 100,
        away_win_probability: Math.round(awayProb * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
      });
    }
  }
  
  // Sort by timestamp
  return data.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
};

export default function ProbabilityTimeline({
  data,
  homeTeam,
  awayTeam,
  currentHomeProb,
  currentAwayProb,
}: ProbabilityTimelineProps) {
  const [mounted, setMounted] = useState(false);
  const [timeRange, setTimeRange] = useState<TimeRange>('1w');

  useEffect(() => {
    setMounted(true);
  }, []);

  // Use provided data or generate placeholder
  const allTimelineData = data && data.length > 0 
    ? data 
    : generatePlaceholderTimeline(
        homeTeam,
        awayTeam,
        currentHomeProb,
        currentAwayProb,
        30 // Generate 30 days of data
      );

  // Filter data based on selected time range
  const filterDataByRange = (data: TimelineDataPoint[], range: TimeRange): TimelineDataPoint[] => {
    const now = new Date();
    const cutoff = new Date();
    
    switch (range) {
      case '1d':
        cutoff.setDate(now.getDate() - 1);
        break;
      case '1w':
        cutoff.setDate(now.getDate() - 7);
        break;
      case '1m':
        cutoff.setDate(now.getDate() - 30);
        break;
      case 'all':
        return data; // Return all data
    }
    
    return data.filter((point) => new Date(point.timestamp) >= cutoff);
  };

  const timelineData = filterDataByRange(allTimelineData, timeRange);

  // Format X axis label based on time range
  const formatXAxisLabel = (timestamp: string, range: TimeRange): string => {
    const date = new Date(timestamp);
    
    switch (range) {
      case '1d':
        return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
      case '1w':
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      case '1m':
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      case 'all':
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      default:
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  // Format data for chart
  const chartData = timelineData.map((point) => ({
    time: formatXAxisLabel(point.timestamp, timeRange),
    timestamp: point.timestamp,
    [homeTeam]: Math.round(point.home_win_probability * 100),
    [awayTeam]: Math.round(point.away_win_probability * 100),
    confidence: Math.round(point.confidence * 100),
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3">
          <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p
              key={index}
              className="text-sm"
              style={{ color: entry.color }}
            >
              {entry.name}: <span className="font-semibold">{entry.value}%</span>
            </p>
          ))}
          {payload[0]?.payload?.confidence && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Confidence: {payload[0].payload.confidence}%
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  // Calculate optimal tick interval based on data length
  const getTickInterval = (dataLength: number, range: TimeRange): number => {
    switch (range) {
      case '1d':
        return Math.max(1, Math.floor(dataLength / 8)); // ~8 ticks for 1 day
      case '1w':
        return Math.max(1, Math.floor(dataLength / 7)); // ~7 ticks for 1 week
      case '1m':
        return Math.max(1, Math.floor(dataLength / 10)); // ~10 ticks for 1 month
      case 'all':
        return Math.max(1, Math.floor(dataLength / 12)); // ~12 ticks for all
      default:
        return 1;
    }
  };

  // Prevent hydration mismatch
  if (!mounted) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="h-80 w-full flex items-center justify-center">
          <p className="text-gray-500 dark:text-gray-400">Loading chart...</p>
        </div>
      </div>
    );
  }

  const tickInterval = getTickInterval(chartData.length, timeRange);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 flex flex-col h-full">
      {/* Time Range Selector */}
      <div className="flex items-center justify-end mb-4 flex-shrink-0">
        <div className="flex items-center gap-2">
          {(['1d', '1w', '1m', 'all'] as TimeRange[]).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                timeRange === range
                  ? 'bg-nfl-primary dark:bg-nfl-accent text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {range === 'all' ? 'All' : range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      
      <div className="h-80 w-full flex-shrink-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" className="dark:stroke-gray-700" />
            <XAxis
              dataKey="time"
              stroke="#6b7280"
              className="dark:stroke-gray-400"
              tick={{ fill: '#6b7280', fontSize: 11 }}
              tickFormatter={(value) => value}
              interval={tickInterval}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#6b7280"
              className="dark:stroke-gray-400"
              tick={{ fill: '#6b7280', fontSize: 11 }}
              tickFormatter={(value) => `${value}%`}
              ticks={[0, 20, 40, 60, 80, 100]}
            />
            <Tooltip 
              content={<CustomTooltip />}
              cursor={{ stroke: '#6b7280', strokeWidth: 1, strokeDasharray: '3 3' }}
              allowEscapeViewBox={{ x: true, y: true }}
            />
            <Legend
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="line"
            />
            <Line
              type="monotone"
              dataKey={homeTeam}
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              activeDot={false}
              name={`${homeTeam}`}
            />
            <Line
              type="monotone"
              dataKey={awayTeam}
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={false}
              name={`${awayTeam}`}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

