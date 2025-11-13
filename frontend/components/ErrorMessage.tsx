interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-red-800 font-semibold text-sm">Error</h3>
          <p className="text-red-600 text-sm mt-1">{message}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="ml-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
