'use client';

import { Cloud, MapPin, Thermometer, Wind, Droplet, Home } from 'lucide-react';
import { Weather, Stadium } from '@/lib/api';

interface WeatherStadiumProps {
  weather?: Weather;
  stadium?: Stadium;
}

export default function WeatherStadium({ weather, stadium }: WeatherStadiumProps) {
  const getWeatherIcon = (condition: string, isDome: boolean) => {
    if (isDome) {
      return <Home className="w-5 h-5 text-gray-600 dark:text-gray-400" />;
    }
    const lowerCondition = condition.toLowerCase();
    if (lowerCondition.includes('rain') || lowerCondition.includes('storm')) {
      return <Droplet className="w-5 h-5 text-blue-600 dark:text-blue-400" />;
    }
    if (lowerCondition.includes('cloud')) {
      return <Cloud className="w-5 h-5 text-gray-600 dark:text-gray-400" />;
    }
    return <Cloud className="w-5 h-5 text-yellow-500 dark:text-yellow-400" />;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 h-full flex flex-col">
      {/* Stadium Section */}
      {stadium && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-2">
            <MapPin className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Location</h3>
          </div>
          {/* Big Location */}
          <div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {stadium.name}
            </div>
            <div className="text-xl font-semibold text-gray-700 dark:text-gray-300 mt-1">
              {stadium.city}, {stadium.state}
            </div>
            {/* Compact details */}
            {(stadium.surface || stadium.roofType) && (
              <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 dark:text-gray-500">
                {stadium.surface && (
                  <span>Field: {stadium.surface}</span>
                )}
                {stadium.roofType && (
                  <span>• Roof: {stadium.roofType}</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Weather Section */}
      {weather && (
        <div className="flex-1 flex flex-col">
          <div className="flex items-center gap-2 mb-4">
            {getWeatherIcon(weather.condition, weather.isDome)}
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Weather</h3>
          </div>
          {/* Big Temperature */}
          <div className="mb-4">
            <div className="text-4xl font-bold text-gray-900 dark:text-white">
              {weather.temperature}°F
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {weather.isDome ? 'Indoor' : weather.condition}
            </div>
          </div>
          {/* Smaller details */}
          {!weather.isDome && (
            <div className="space-y-1.5 border-t border-gray-200 dark:border-gray-700 pt-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 dark:text-gray-500 flex items-center gap-1">
                  <Wind className="w-3 h-3" />
                  Wind
                </span>
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  {weather.windSpeed} mph
                </span>
              </div>
              {weather.humidity !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-500">Humidity</span>
                  <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    {weather.humidity}%
                  </span>
                </div>
              )}
              {weather.precipitation !== undefined && weather.precipitation > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-500 flex items-center gap-1">
                    <Droplet className="w-3 h-3" />
                    Precipitation
                  </span>
                  <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    {weather.precipitation}%
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

