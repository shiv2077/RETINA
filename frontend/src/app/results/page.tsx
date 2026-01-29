'use client';

/**
 * RETINA - Results Page
 * =====================
 *
 * Displays inference results and allows querying by job/image ID.
 *
 * Features:
 * - Search by job ID or image ID
 * - Anomaly score visualization
 * - Stage-specific output display
 * - Active learning metadata
 */

import { useState } from 'react';
import { getResult, ResultResponse, InferenceResult } from '@/lib/api';

export default function ResultsPage() {
  const [searchId, setSearchId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ResultResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!searchId.trim()) {
      setError('Please enter a job ID or image ID');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await getResult(searchId.trim());
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch result');
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Inference Results
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          View anomaly detection results by job ID or image ID
        </p>
      </div>

      {/* Search Form */}
      <div className="card">
        <form onSubmit={handleSearch} className="flex gap-4">
          <input
            type="text"
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            placeholder="Enter job ID or image ID"
            className="input flex-1"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !searchId.trim()}
            className="btn-primary"
          >
            {isLoading ? '🔄 Loading...' : '🔍 Search'}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-red-700 dark:text-red-300">⚠️ {error}</p>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="space-y-6">
          {result.found && result.result ? (
            <ResultCard result={result.result} />
          ) : (
            <div className="card text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">
                No result found for this ID
              </p>
              <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">
                {result.message}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Initial State */}
      {!result && !error && (
        <div className="card text-center py-12">
          <p className="text-gray-500 dark:text-gray-400">
            Enter a job ID or image ID to view results
          </p>
          <p className="text-sm text-gray-400 dark:text-gray-500 mt-2">
            Job IDs are UUIDs returned when submitting images
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Result card component showing full inference details.
 */
function ResultCard({ result }: { result: InferenceResult }) {
  const isCompleted = result.status === 'completed';
  const isPending = ['pending', 'queued', 'processing'].includes(result.status);
  const isFailed = result.status === 'failed';

  // Determine anomaly class for styling
  const getAnomalyClass = (score: number) => {
    if (score < 0.3) return 'normal';
    if (score < 0.7) return 'uncertain';
    return 'anomaly';
  };

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Job ID</p>
            <p className="font-mono text-lg text-gray-900 dark:text-white">
              {result.job_id}
            </p>
          </div>
          <StatusBadge status={result.status} />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-500 dark:text-gray-400">Image ID</p>
            <p className="font-medium text-gray-900 dark:text-white">
              {result.image_id}
            </p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Stage</p>
            <p className="font-medium text-gray-900 dark:text-white">
              Stage {result.stage}{' '}
              {result.stage === 1 ? '(Unsupervised)' : '(Supervised)'}
            </p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Model</p>
            <p className="font-medium text-gray-900 dark:text-white uppercase">
              {result.model_used || 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-gray-500 dark:text-gray-400">Processing Time</p>
            <p className="font-medium text-gray-900 dark:text-white">
              {result.processing_time_ms
                ? `${result.processing_time_ms}ms`
                : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Pending State */}
      {isPending && (
        <div className="card text-center py-8">
          <div className="text-4xl mb-4 animate-pulse">⏳</div>
          <p className="text-gray-600 dark:text-gray-400">
            Job is {result.status}...
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            Refresh the page to check for updates
          </p>
        </div>
      )}

      {/* Failed State */}
      {isFailed && result.error && (
        <div className="card bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <h3 className="text-lg font-semibold text-red-700 dark:text-red-300 mb-2">
            ❌ Inference Failed
          </h3>
          <p className="text-red-600 dark:text-red-400">
            <strong>Error Code:</strong> {result.error.code}
          </p>
          <p className="text-red-600 dark:text-red-400 mt-1">
            <strong>Message:</strong> {result.error.message}
          </p>
        </div>
      )}

      {/* Completed State - Anomaly Score */}
      {isCompleted && result.anomaly_score !== undefined && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Anomaly Detection Result
          </h3>

          {/* Score Visualization */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Anomaly Score
              </span>
              <span
                className={`text-2xl font-bold ${
                  result.is_anomaly
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-green-600 dark:text-green-400'
                }`}
              >
                {(result.anomaly_score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="anomaly-score-bar">
              <div
                className={`anomaly-score-fill ${getAnomalyClass(
                  result.anomaly_score
                )}`}
                style={{ width: `${result.anomaly_score * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>Normal</span>
              <span>Uncertain</span>
              <span>Anomaly</span>
            </div>
          </div>

          {/* Classification Result */}
          <div className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div
              className={`text-4xl ${
                result.is_anomaly ? 'text-red-500' : 'text-green-500'
              }`}
            >
              {result.is_anomaly ? '⚠️' : '✅'}
            </div>
            <div>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {result.is_anomaly ? 'Anomaly Detected' : 'Normal Sample'}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Confidence: {((result.confidence || 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stage 1 Output */}
      {isCompleted && result.stage1_output && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Stage 1 Details (Unsupervised)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricBox
              label="Feature Distance"
              value={result.stage1_output.feature_distance?.toFixed(3) || 'N/A'}
            />
            <MetricBox
              label="CLIP Similarity"
              value={result.stage1_output.clip_similarity?.toFixed(3) || 'N/A'}
            />
            <MetricBox
              label="Heatmap"
              value={result.stage1_output.heatmap_available ? 'Available' : 'Not generated'}
            />
          </div>
        </div>
      )}

      {/* Stage 2 Output */}
      {isCompleted && result.stage2_output && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Stage 2 Details (Supervised)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricBox
              label="Defect Category"
              value={result.stage2_output.defect_category || 'N/A'}
            />
            <MetricBox
              label="Embedding Distance"
              value={result.stage2_output.embedding_distance?.toFixed(3) || 'N/A'}
            />
          </div>

          {/* Category Probabilities */}
          {result.stage2_output.category_probabilities && (
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Category Probabilities
              </p>
              <div className="space-y-2">
                {Object.entries(result.stage2_output.category_probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([category, probability]) => (
                    <div key={category} className="flex items-center gap-2">
                      <span className="w-24 text-sm text-gray-600 dark:text-gray-400 capitalize">
                        {category}
                      </span>
                      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${probability * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400 w-12">
                        {(probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Active Learning Info */}
      {isCompleted && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Active Learning Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricBox
              label="Uncertainty Score"
              value={(result.active_learning.uncertainty_score * 100).toFixed(1) + '%'}
            />
            <MetricBox
              label="In Labeling Pool"
              value={result.active_learning.in_labeling_pool ? 'Yes' : 'No'}
            />
            <MetricBox
              label="Labeled"
              value={result.active_learning.labeled ? 'Yes' : 'No'}
            />
          </div>

          {result.active_learning.uncertainty_score > 0.3 &&
            !result.active_learning.labeled && (
              <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  ⚡ This sample has high uncertainty and may be valuable for labeling.{' '}
                  <a
                    href="/label"
                    className="underline hover:no-underline font-medium"
                  >
                    Go to labeling interface →
                  </a>
                </p>
              </div>
            )}
        </div>
      )}
    </div>
  );
}

/**
 * Status badge component.
 */
function StatusBadge({ status }: { status: string }) {
  const statusConfig: Record<
    string,
    { class: string; icon: string }
  > = {
    pending: { class: 'badge-info', icon: '⏳' },
    queued: { class: 'badge-info', icon: '📋' },
    processing: { class: 'badge-warning', icon: '⚙️' },
    completed: { class: 'badge-success', icon: '✅' },
    failed: { class: 'badge-danger', icon: '❌' },
  };

  const config = statusConfig[status] || { class: 'badge-info', icon: '❓' };

  return (
    <span className={`badge ${config.class}`}>
      {config.icon} {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

/**
 * Metric display box.
 */
function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
      <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
      <p className="text-lg font-semibold text-gray-900 dark:text-white">
        {value}
      </p>
    </div>
  );
}
