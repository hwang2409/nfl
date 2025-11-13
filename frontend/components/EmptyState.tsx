'use client';

import { Search, Filter } from 'lucide-react';

interface EmptyStateProps {
  message?: string;
  suggestion?: string;
}

export default function EmptyState({ 
  message = 'No games found for the selected filters.',
  suggestion = 'Try adjusting the season, week, or confidence filter.'
}: EmptyStateProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-12 text-center">
      <div className="flex flex-col items-center">
        <div className="w-16 h-16 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center mb-4">
          <Search className="w-8 h-8 text-gray-400 dark:text-gray-500" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          {message}
        </h3>
        <p className="text-gray-500 dark:text-gray-400 text-sm max-w-md">
          {suggestion}
        </p>
        <div className="mt-6 flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
          <Filter className="w-4 h-4" />
          <span>Use the filters above to refine your search</span>
        </div>
      </div>
    </div>
  );
}

