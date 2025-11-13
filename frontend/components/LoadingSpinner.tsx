export default function LoadingSpinner() {
  return (
    <div className="flex flex-col justify-center items-center py-12">
      <div className="animate-spin rounded-full h-12 w-12 border-2 border-gray-200 dark:border-gray-700 border-t-gray-900 dark:border-t-white"></div>
      <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">Loading predictions...</p>
    </div>
  );
}
