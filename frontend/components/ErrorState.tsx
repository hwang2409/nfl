'use client';

import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
  showPlaceholderInfo?: boolean;
}

export default function ErrorState({ message, onRetry, showPlaceholderInfo = false }: ErrorStateProps) {
  return (
    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <AlertTriangle className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
        </div>
        <div className="flex-1">
          <h3 className="text-yellow-800 dark:text-yellow-400 font-semibold text-base mb-1">
            {showPlaceholderInfo ? 'Using Placeholder Data' : 'Error Loading Predictions'}
          </h3>
          <p className="text-yellow-700 dark:text-yellow-500 text-sm mb-4">
            {message}
            {showPlaceholderInfo && ' Displaying sample predictions for demonstration.'}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="inline-flex items-center gap-2 bg-yellow-600 dark:bg-yellow-700 text-white px-4 py-2 rounded-lg hover:bg-yellow-700 dark:hover:bg-yellow-600 transition-colors text-sm font-medium"
            >
              <RefreshCw className="w-4 h-4" />
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

