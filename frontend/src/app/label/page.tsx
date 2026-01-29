'use client';

/**
 * RETINA - Active Learning Labeling Page
 * ======================================
 *
 * Interface for domain experts to label samples for active learning.
 *
 * This page:
 * - Shows samples from the labeling pool (ordered by uncertainty)
 * - Allows binary labeling (normal/anomaly)
 * - Supports defect categorization
 * - Tracks progress towards Stage 2
 *
 * The labeling interface is designed for efficiency:
 * - Single-click classification
 * - Keyboard shortcuts
 * - Batch labeling support (planned)
 */

import { useState, useEffect } from 'react';
import {
  getLabelingPool,
  submitLabel,
  LabelingPoolResponse,
  UnlabeledSample,
  SubmitLabelResponse,
} from '@/lib/api';

// Predefined defect categories (matches Python worker)
const DEFECT_CATEGORIES = [
  'scratch',
  'dent',
  'contamination',
  'discoloration',
  'crack',
  'other',
];

export default function LabelPage() {
  // Pool data
  const [pool, setPool] = useState<LabelingPoolResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Current sample being labeled
  const [currentSample, setCurrentSample] = useState<UnlabeledSample | null>(null);
  const [labelResult, setLabelResult] = useState<SubmitLabelResponse | null>(null);

  // Labeling form state
  const [defectCategory, setDefectCategory] = useState<string>('');
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Labeler identity (in real app, this would come from auth)
  const [labelerId] = useState('expert_001');

  // Fetch labeling pool on mount
  useEffect(() => {
    fetchPool();
  }, []);

  const fetchPool = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await getLabelingPool();
      setPool(response);

      // Set first sample as current
      if (response.samples.length > 0) {
        setCurrentSample(response.samples[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load labeling pool');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLabel = async (isAnomaly: boolean) => {
    if (!currentSample) return;

    setIsSubmitting(true);
    setLabelResult(null);

    try {
      const response = await submitLabel({
        image_id: currentSample.image_id,
        is_anomaly: isAnomaly,
        defect_category: isAnomaly && defectCategory ? defectCategory : undefined,
        confidence: 'high',
        labeled_by: labelerId,
        notes: notes || undefined,
      });

      setLabelResult(response);

      // Move to next sample
      if (pool) {
        const currentIdx = pool.samples.findIndex(
          (s) => s.image_id === currentSample.image_id
        );
        const nextSample = pool.samples[currentIdx + 1];

        if (nextSample) {
          setCurrentSample(nextSample);
        } else {
          setCurrentSample(null);
        }

        // Update pool counts
        setPool({
          ...pool,
          labels_collected: response.total_labels,
          samples: pool.samples.filter((s) => s.image_id !== currentSample.image_id),
        });
      }

      // Reset form
      setDefectCategory('');
      setNotes('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit label');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Progress percentage
  const progressPercent = pool
    ? Math.min((pool.labels_collected / pool.stage2_threshold) * 100, 100)
    : 0;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Active Learning Labeling
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Review uncertain samples and provide labels to improve the model
        </p>
      </div>

      {/* Progress Card */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Stage 2 Progress
          </h2>
          <span
            className={`badge ${
              progressPercent >= 100 ? 'badge-success' : 'badge-info'
            }`}
          >
            {progressPercent >= 100 ? 'Stage 2 Active!' : `${progressPercent.toFixed(0)}%`}
          </span>
        </div>

        <div className="progress-bar mb-2">
          <div
            className="progress-bar-fill"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
          <span>{pool?.labels_collected ?? 0} labels collected</span>
          <span>{pool?.stage2_threshold ?? 100} needed for Stage 2</span>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-red-700 dark:text-red-300">⚠️ {error}</p>
          <button
            onClick={fetchPool}
            className="mt-2 text-sm text-red-600 dark:text-red-400 underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Success Message */}
      {labelResult && (
        <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <p className="text-green-700 dark:text-green-300">
            ✅ {labelResult.message}
          </p>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="card text-center py-12">
          <div className="text-4xl mb-4 animate-spin">🔄</div>
          <p className="text-gray-600 dark:text-gray-400">Loading samples...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && (!pool || pool.samples.length === 0) && (
        <div className="card text-center py-12">
          <div className="text-4xl mb-4">🎉</div>
          <p className="text-gray-600 dark:text-gray-400">
            No samples in the labeling pool
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            Submit more images for analysis to populate the pool
          </p>
          <a href="/submit" className="btn-primary mt-4 inline-block">
            Submit Images →
          </a>
        </div>
      )}

      {/* Labeling Interface */}
      {!isLoading && currentSample && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Current Sample */}
          <div className="lg:col-span-2 card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Current Sample
              </h2>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {pool?.samples.length} remaining
              </span>
            </div>

            {/* Sample Info */}
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg mb-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Image ID
                  </p>
                  <p className="font-mono text-gray-900 dark:text-white">
                    {currentSample.image_id}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Uncertainty Score
                  </p>
                  <p className="font-semibold text-amber-600 dark:text-amber-400">
                    {(currentSample.uncertainty_score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Model's Prediction */}
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                  Model's Prediction (Anomaly Score)
                </p>
                <div className="anomaly-score-bar">
                  <div
                    className={`anomaly-score-fill ${
                      currentSample.anomaly_score < 0.3
                        ? 'normal'
                        : currentSample.anomaly_score < 0.7
                        ? 'uncertain'
                        : 'anomaly'
                    }`}
                    style={{ width: `${currentSample.anomaly_score * 100}%` }}
                  />
                </div>
                <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {(currentSample.anomaly_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Image Placeholder */}
            <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center mb-6">
              <div className="text-center">
                <div className="text-6xl mb-2">🖼️</div>
                <p className="text-gray-500 dark:text-gray-400">
                  Image Preview
                </p>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                  (Actual images will be displayed when storage is implemented)
                </p>
              </div>
            </div>

            {/* Labeling Actions */}
            <div className="space-y-4">
              {/* Quick Classification Buttons */}
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => handleLabel(false)}
                  disabled={isSubmitting}
                  className="btn-success h-16 text-lg"
                >
                  <span className="mr-2">✅</span>
                  Normal
                  <span className="ml-2 text-xs opacity-75">(N)</span>
                </button>
                <button
                  onClick={() => handleLabel(true)}
                  disabled={isSubmitting}
                  className="btn-danger h-16 text-lg"
                >
                  <span className="mr-2">⚠️</span>
                  Anomaly
                  <span className="ml-2 text-xs opacity-75">(A)</span>
                </button>
              </div>

              {/* Defect Category (for anomalies) */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Defect Category (optional, for anomalies)
                </label>
                <div className="flex flex-wrap gap-2">
                  {DEFECT_CATEGORIES.map((category) => (
                    <button
                      key={category}
                      onClick={() =>
                        setDefectCategory(
                          defectCategory === category ? '' : category
                        )
                      }
                      className={`px-3 py-1 rounded-full text-sm ${
                        defectCategory === category
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {category}
                    </button>
                  ))}
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Notes (optional)
                </label>
                <input
                  type="text"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add any observations..."
                  className="input"
                />
              </div>
            </div>
          </div>

          {/* Queue Sidebar */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Labeling Queue
            </h2>

            <div className="space-y-2 max-h-[500px] overflow-y-auto">
              {pool?.samples.slice(0, 10).map((sample, idx) => (
                <button
                  key={sample.image_id}
                  onClick={() => setCurrentSample(sample)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    currentSample?.image_id === sample.image_id
                      ? 'bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500'
                      : 'bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-mono text-gray-700 dark:text-gray-300 truncate">
                      {sample.image_id.slice(0, 20)}...
                    </span>
                    <span className="text-xs text-gray-500">
                      #{idx + 1}
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Uncertainty: {(sample.uncertainty_score * 100).toFixed(0)}%
                  </div>
                </button>
              ))}
            </div>

            {pool && pool.samples.length > 10 && (
              <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-3">
                +{pool.samples.length - 10} more samples
              </p>
            )}
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <h3 className="font-medium text-gray-900 dark:text-white mb-2">
          ⌨️ Keyboard Shortcuts (Coming Soon)
        </h3>
        <div className="text-sm text-gray-600 dark:text-gray-400 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">N</kbd> Normal
          </div>
          <div>
            <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">A</kbd> Anomaly
          </div>
          <div>
            <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">←</kbd> Previous
          </div>
          <div>
            <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded">→</kbd> Next
          </div>
        </div>
      </div>
    </div>
  );
}
