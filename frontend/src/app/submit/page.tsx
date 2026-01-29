'use client';

/**
 * RETINA - Image Submission Page
 * ==============================
 *
 * Allows users to submit image IDs for anomaly detection.
 *
 * In the current implementation, images are referenced by ID only
 * (no actual file upload). This page provides:
 * - Image ID input
 * - Model selection
 * - Priority setting
 * - Submission feedback
 *
 * Note: This is a client component to handle form state and submission.
 */

import { useState } from 'react';
import { submitImage, SubmitImageResponse, ModelType } from '@/lib/api';

export default function SubmitPage() {
  // Form state
  const [imageId, setImageId] = useState('');
  const [modelType, setModelType] = useState<ModelType>('patchcore');
  const [priority, setPriority] = useState(5);

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<SubmitImageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Recent submissions for quick reference
  const [recentSubmissions, setRecentSubmissions] = useState<SubmitImageResponse[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!imageId.trim()) {
      setError('Please enter an image ID');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const response = await submitImage({
        image_id: imageId.trim(),
        model_type: modelType,
        priority,
        source: 'web_ui',
      });

      setResult(response);
      setRecentSubmissions((prev) => [response, ...prev].slice(0, 5));

      // Clear form on success
      setImageId('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Submit Image
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Submit an image ID for anomaly detection analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Submission Form */}
        <div className="lg:col-span-2 card">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
            New Submission
          </h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Image ID Input */}
            <div>
              <label
                htmlFor="imageId"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Image ID
              </label>
              <input
                type="text"
                id="imageId"
                value={imageId}
                onChange={(e) => setImageId(e.target.value)}
                placeholder="e.g., img_2024_001_front_panel"
                className="input"
                disabled={isSubmitting}
              />
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Enter a unique identifier for the image to analyze
              </p>
            </div>

            {/* Model Selection */}
            <div>
              <label
                htmlFor="modelType"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Detection Model
              </label>
              <select
                id="modelType"
                value={modelType}
                onChange={(e) => setModelType(e.target.value as ModelType)}
                className="input"
                disabled={isSubmitting}
              >
                <option value="patchcore">PatchCore (Memory-bank)</option>
                <option value="padim">PaDiM (Gaussian)</option>
                <option value="winclip">WinCLIP (Zero-shot)</option>
              </select>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                {modelType === 'patchcore' &&
                  'Feature embedding with k-NN distance scoring'}
                {modelType === 'padim' &&
                  'Multivariate Gaussian modeling of patch distributions'}
                {modelType === 'winclip' &&
                  'Zero-shot detection using CLIP text-image similarity'}
              </p>
            </div>

            {/* Priority Slider */}
            <div>
              <label
                htmlFor="priority"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Priority: {priority}
              </label>
              <input
                type="range"
                id="priority"
                min="0"
                max="10"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isSubmitting}
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>Low (0)</span>
                <span>Normal (5)</span>
                <span>High (10)</span>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-red-700 dark:text-red-300">⚠️ {error}</p>
              </div>
            )}

            {/* Success Message */}
            {result && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                <p className="text-green-700 dark:text-green-300 font-medium">
                  ✅ {result.message}
                </p>
                <p className="text-sm text-green-600 dark:text-green-400 mt-1">
                  Job ID: <code className="font-mono">{result.job_id}</code>
                </p>
                {result.queue_position && (
                  <p className="text-sm text-green-600 dark:text-green-400">
                    Queue Position: #{result.queue_position}
                  </p>
                )}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting || !imageId.trim()}
              className="btn-primary w-full"
            >
              {isSubmitting ? (
                <>
                  <span className="animate-spin mr-2">⏳</span>
                  Submitting...
                </>
              ) : (
                <>
                  <span className="mr-2">🔍</span>
                  Submit for Analysis
                </>
              )}
            </button>
          </form>
        </div>

        {/* Recent Submissions Sidebar */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Recent Submissions
          </h2>

          {recentSubmissions.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              No submissions yet. Submit an image to see it here.
            </p>
          ) : (
            <ul className="space-y-3">
              {recentSubmissions.map((submission, idx) => (
                <li
                  key={submission.job_id}
                  className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                >
                  <div className="flex items-center justify-between">
                    <span className="badge badge-info">
                      {submission.status}
                    </span>
                    <span className="text-xs text-gray-500">#{idx + 1}</span>
                  </div>
                  <p className="text-sm font-mono text-gray-700 dark:text-gray-300 mt-2 truncate">
                    {submission.job_id.slice(0, 8)}...
                  </p>
                  <a
                    href={`/results?id=${submission.job_id}`}
                    className="text-sm text-blue-600 dark:text-blue-400 hover:underline mt-1 inline-block"
                  >
                    View Result →
                  </a>
                </li>
              ))}
            </ul>
          )}

          {/* Batch Submission Hint */}
          <div className="mt-6 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>Tip:</strong> For batch submissions, use the API directly:
            </p>
            <code className="text-xs font-mono text-blue-700 dark:text-blue-300 mt-1 block">
              POST /api/images/submit
            </code>
          </div>
        </div>
      </div>

      {/* Information Panel */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="font-medium text-gray-900 dark:text-white mb-3">
          ℹ️ About Image Submission
        </h3>
        <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
          <p>
            In the current implementation, images are referenced by ID only.
            The system automatically selects the appropriate pipeline stage:
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong>Stage 1 (Unsupervised)</strong>: Used when fewer than 100
              labels exist
            </li>
            <li>
              <strong>Stage 2 (Supervised)</strong>: Activated after collecting
              enough labeled samples
            </li>
          </ul>
          <p className="mt-2">
            Results can be viewed on the{' '}
            <a href="/results" className="text-blue-600 dark:text-blue-400 hover:underline">
              Results page
            </a>{' '}
            once processing completes.
          </p>
        </div>
      </div>
    </div>
  );
}
