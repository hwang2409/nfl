'use client';

import { TrendingUp, Target, BarChart3, CheckCircle } from 'lucide-react';

interface ModelStats {
  accuracy?: number;
  rocAuc?: number;
  logLoss?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
}

interface PredictionAccuracyCardProps {
  stats?: ModelStats;
  className?: string;
}

const defaultStats: ModelStats = {
  accuracy: 0.89,
  rocAuc: 0.95,
  logLoss: 0.27,
  precision: 0.87,
  recall: 0.85,
  f1Score: 0.86,
};

export default function PredictionAccuracyCard({ stats = defaultStats, className = '' }: PredictionAccuracyCardProps) {
  const displayStats = { ...defaultStats, ...stats };

  const formatPercentage = (value: number | undefined) => {
    if (value === undefined) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDecimal = (value: number | undefined) => {
    if (value === undefined) return 'N/A';
    return value.toFixed(3);
  };

  const getAccuracyColor = (accuracy: number | undefined) => {
    if (!accuracy) return 'text-gray-500';
    if (accuracy >= 0.85) return 'text-green-600 dark:text-green-400';
    if (accuracy >= 0.75) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Model Performance</h3>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {/* Accuracy */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">Accuracy</span>
          </div>
          <div className={`text-2xl font-bold ${getAccuracyColor(displayStats.accuracy)}`}>
            {formatPercentage(displayStats.accuracy)}
          </div>
        </div>

        {/* ROC-AUC */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">ROC-AUC</span>
          </div>
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {formatDecimal(displayStats.rocAuc)}
          </div>
        </div>

        {/* Log Loss */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">Log Loss</span>
          </div>
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {formatDecimal(displayStats.logLoss)}
          </div>
        </div>

        {/* Precision */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">Precision</span>
          </div>
          <div className="text-xl font-bold text-green-600 dark:text-green-400">
            {formatPercentage(displayStats.precision)}
          </div>
        </div>

        {/* Recall */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">Recall</span>
          </div>
          <div className="text-xl font-bold text-orange-600 dark:text-orange-400">
            {formatPercentage(displayStats.recall)}
          </div>
        </div>

        {/* F1 Score */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">F1 Score</span>
          </div>
          <div className="text-xl font-bold text-indigo-600 dark:text-indigo-400">
            {formatPercentage(displayStats.f1Score)}
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
          Metrics based on cross-validation performance
        </p>
      </div>
    </div>
  );
}

