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
 * This is a server component that fetches data on each request.
 */

import { getSystemStatus, getHealth, SystemStatusResponse } from '@/lib/api';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function DashboardPage() {
  let systemStatus: SystemStatusResponse | null = null;
  let isHealthy = false;
  let error: string | null = null;

  try {
    // Fetch system status from backend
    const [healthResponse, statusResponse] = await Promise.all([
      getHealth().catch(() => null),
      getSystemStatus().catch(() => null),
    ]);

    isHealthy = healthResponse?.status === 'healthy';
    systemStatus = statusResponse;
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to connect to backend';
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
            Make sure the backend is running: <code>docker compose up</code>
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

        {/* Current Stage */}
        <StatusCard
          title="Pipeline Stage"
          value={`Stage ${systemStatus?.current_stage ?? '-'}`}
          icon={systemStatus?.current_stage === 2 ? '🎯' : '🔍'}
          subtitle={
            systemStatus?.current_stage === 2
              ? 'Supervised Classification'
              : 'Unsupervised Detection'
          }
          color="blue"
        />

        {/* Jobs Processed */}
        <StatusCard
          title="Jobs Completed"
          value={systemStatus?.stats.jobs_completed?.toLocaleString() ?? '-'}
          icon="📊"
          subtitle={`${systemStatus?.stats.queue_length ?? 0} in queue`}
          color="purple"
        />

        {/* Labels Collected */}
        <StatusCard
          title="Labels Collected"
          value={systemStatus?.stats.labels_collected?.toLocaleString() ?? '-'}
          icon="🏷️"
          subtitle={`of ${systemStatus?.labels_for_stage2 ?? 100} for Stage 2`}
          color="amber"
        />
      </div>

      {/* Active Learning Progress */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Active Learning Progress
        </h2>

        <div className="space-y-4">
          {/* Progress Bar */}
          <div>
            <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
              <span>Progress to Stage 2</span>
              <span>
                {systemStatus?.stats.labels_collected ?? 0} /{' '}
                {systemStatus?.labels_for_stage2 ?? 100} labels
              </span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-bar-fill"
                style={{ width: `${systemStatus?.stage2_progress ?? 0}%` }}
              />
            </div>
          </div>

          {/* Stage 2 Status */}
          <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Stage 2 (Supervised Learning)
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {systemStatus?.stage2_available
                  ? 'Push-pull contrastive model is active'
                  : 'Collect more labels to enable supervised classification'}
              </p>
            </div>
            <div
              className={`badge ${
                systemStatus?.stage2_available ? 'badge-success' : 'badge-warning'
              }`}
            >
              {systemStatus?.stage2_available ? 'Available' : 'Locked'}
            </div>
          </div>

          {/* Labeling Pool */}
          <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div>
              <p className="font-medium text-gray-900 dark:text-white">
                Labeling Pool
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Uncertain samples awaiting expert review
              </p>
            </div>
            <div className="text-2xl font-bold text-kuleuven-blue">
              {systemStatus?.stats.labeling_pool_size ?? 0}
            </div>
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Active Models
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Stage 1 Model */}
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Stage 1 Model
              </span>
              <span className="badge badge-info">Unsupervised</span>
            </div>
            <p className="text-lg font-semibold text-gray-900 dark:text-white uppercase">
              {systemStatus?.active_models.stage1_model ?? 'PatchCore'}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Memory-bank based anomaly detection
            </p>
          </div>

          {/* Stage 2 Model */}
          <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Stage 2 Model
              </span>
              <span
                className={`badge ${
                  systemStatus?.stage2_available ? 'badge-success' : 'badge-warning'
                }`}
              >
                {systemStatus?.stage2_available ? 'Active' : 'Pending'}
              </span>
            </div>
            <p className="text-lg font-semibold text-gray-900 dark:text-white uppercase">
              {systemStatus?.active_models.stage2_model ?? 'Push-Pull'}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Contrastive learning classifier
            </p>
          </div>
        </div>
      </div>

      {/* Research Note */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <h3 className="font-medium text-blue-900 dark:text-blue-200 mb-2">
          📚 Research Note
        </h3>
        <p className="text-sm text-blue-800 dark:text-blue-300">
          This system implements a two-stage pipeline for industrial visual inspection.
          Stage 1 uses unsupervised methods (PatchCore, WinCLIP) for initial anomaly detection
          with high recall. As labels are collected via active learning, Stage 2 enables
          supervised classification with push-pull contrastive learning for higher precision.
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
