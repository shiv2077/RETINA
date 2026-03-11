'use client';

/**
 * RETINA Dashboard - Home Page
 * ============================
 *
 * Main dashboard showing:
 * - System health status
 * - Current pipeline stage
 * - Processing statistics
 * - Active learning progress
 *
 * Client component that fetches data on mount.
 */

import { useState, useEffect } from 'react';
import { getHealth, HealthResponse } from '@/lib/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface StatusData {
  pipeline?: Record<string, any>;
  labeling?: Record<string, any>;
  models?: Record<string, any>;
}

export default function DashboardPage() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [statusData, setStatusData] = useState<StatusData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthResponse, statusResponse] = await Promise.all([
          getHealth().catch(() => null),
          fetch(`${API_URL}/status`).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);

        setIsHealthy(healthResponse?.status === 'healthy');
        setStatusData(statusResponse);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to connect to backend');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const labelingStats = statusData?.labeling || {};
  const modelInfo = statusData?.models || {};

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">Connecting to backend...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          System Dashboard
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Monitor the multi-stage anomaly detection pipeline
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 dark:bg-red-900/20 dark:border-red-800">
          <div className="flex items-center">
            <span className="text-red-600 dark:text-red-400 font-medium">
              ⚠️ Connection Error:
            </span>
            <span className="ml-2 text-red-700 dark:text-red-300">{error}</span>
          </div>
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">
            Make sure the backend is running: <code>python -m src.backend.app</code>
          </p>
        </div>
      )}

      {/* Status Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* System Health */}
        <StatusCard
          title="System Status"
          value={isHealthy ? 'Healthy' : 'Offline'}
          icon={isHealthy ? '✅' : '❌'}
          color={isHealthy ? 'green' : 'red'}
        />

        {/* Pipeline Stage */}
        <StatusCard
          title="Pipeline"
          value={modelInfo.bgad ? 'BGAD Active' : 'PatchCore Only'}
          icon={modelInfo.bgad ? '🎯' : '🔍'}
          subtitle={
            modelInfo.bgad
              ? 'Cascade inference enabled'
              : 'Unsupervised detection'
          }
          color="blue"
        />

        {/* Labels Completed */}
        <StatusCard
          title="Labeling Progress"
          value={`${labelingStats.completed ?? 0}`}
          icon="📊"
          subtitle={`${labelingStats.pending ?? 0} pending`}
          color="purple"
        />

        {/* Queue Size */}
        <StatusCard
          title="Queue Size"
          value={`${labelingStats.total ?? 0}`}
          icon="🏷️"
          subtitle="Images in labeling queue"
          color="amber"
        />
      </div>

      {/* Active Learning Progress */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Active Learning Progress
        </h2>

        <div className="space-y-4">
          {/* Progress Bar */}
          <div>
            <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
              <span>Labeling Completion</span>
              <span>
                {labelingStats.completed ?? 0} / {labelingStats.total ?? 0} labeled
              </span>
            </div>
            <div className="w-full h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-600 rounded-full transition-all"
                style={{ width: `${labelingStats.progress_percent ?? 0}%` }}
              />
            </div>
          </div>

          {/* Model Status */}
          <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                BGAD (Supervised Learning)
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {modelInfo.bgad
                  ? 'Boundary-guided anomaly detection active'
                  : 'Collect labeled data to enable BGAD training'}
              </p>
            </div>
            <div
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                modelInfo.bgad
                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                  : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300'
              }`}
            >
              {modelInfo.bgad ? 'Trained' : 'Pending'}
            </div>
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Active Models
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* PatchCore */}
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Stage 1 Model
              </span>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
                Unsupervised
              </span>
            </div>
            <p className="text-lg font-semibold text-gray-900 dark:text-white uppercase">
              PatchCore
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Memory-bank based anomaly detection
            </p>
            {Array.isArray(modelInfo.patchcore) && modelInfo.patchcore.length > 0 && (
              <p className="text-xs text-gray-500 mt-2">
                Categories: {modelInfo.patchcore.join(', ')}
              </p>
            )}
          </div>

          {/* BGAD */}
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Stage 2 Model
              </span>
              <span
                className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                  modelInfo.bgad
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                    : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300'
                }`}
              >
                {modelInfo.bgad ? 'Active' : 'Pending'}
              </span>
            </div>
            <p className="text-lg font-semibold text-gray-900 dark:text-white uppercase">
              BGAD
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Boundary-guided push-pull contrastive learning
            </p>
          </div>
        </div>
      </div>

      {/* Info Note */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <h3 className="font-medium text-blue-900 dark:text-blue-200 mb-2">
          Pipeline Overview
        </h3>
        <p className="text-sm text-blue-800 dark:text-blue-300">
          Four-stage cascade: BGAD screens images at the edge (Stage 1), uncertain cases route to
          GPT-4V for zero-shot analysis (Stage 2), flagged images enter the annotation queue
          (Stage 3), and nightly retrain fine-tunes BGAD with new labels (Stage 4).
        </p>
      </div>
    </div>
  );
}

/**
 * Status card component for displaying key metrics.
 */
function StatusCard({
  title,
  value,
  icon,
  subtitle,
  color = 'gray',
}: {
  title: string;
  value: string;
  icon: string;
  subtitle?: string;
  color?: 'green' | 'red' | 'blue' | 'purple' | 'amber' | 'gray';
}) {
  const colorClasses = {
    green: 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800',
    red: 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800',
    blue: 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800',
    purple: 'bg-purple-50 border-purple-200 dark:bg-purple-900/20 dark:border-purple-800',
    amber: 'bg-amber-50 border-amber-200 dark:bg-amber-900/20 dark:border-amber-800',
    gray: 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700',
  };

  return (
    <div className={`rounded-lg border p-6 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
          {title}
        </span>
        <span className="text-2xl">{icon}</span>
      </div>
      <p className="mt-2 text-2xl font-bold text-gray-900 dark:text-white">
        {value}
      </p>
      {subtitle && (
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          {subtitle}
        </p>
      )}
    </div>
  );
}
