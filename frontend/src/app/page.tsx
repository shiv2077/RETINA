'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  GitBranch,
  AlertTriangle,
  Tag,
  Cpu,
  Zap,
} from 'lucide-react';
import { getHealth, getSystemStatus, type SystemStatusResponse } from '@/lib/api';
import GlassCard from '@/components/GlassCard';
import Card from '@/components/Card';
import StatusCard from '@/components/StatusCard';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import SectionHeader from '@/components/SectionHeader';
import Badge from '@/components/Badge';
import ErrorBanner from '@/components/ErrorBanner';
import LoadingSkeleton from '@/components/LoadingSkeleton';

interface AlertEntry {
  image_id: string;
  anomaly_score: number;
  is_anomaly: boolean;
  timestamp: string;
  model_used?: string;
}

export default function DashboardPage() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [isHealthy, setIsHealthy] = useState(false);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [health, sys] = await Promise.all([
        getHealth().catch(() => null),
        getSystemStatus().catch(() => null),
      ]);
      setIsHealthy(health?.status === 'healthy');
      if (sys) setStatus(sys);
      setLastUpdated(new Date());
      setError(null);
    } catch {
      setError('Unable to reach backend. Is the server running on port 3001?');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10_000);
    return () => clearInterval(id);
  }, [fetchData]);

  const stage = status?.current_stage ?? 1;
  const stage2Available = status?.stage2_available ?? false;
  const labelsCollected = status?.stats?.labels_collected ?? 0;
  const labelsForStage2 = status?.labels_for_stage2 ?? 100;
  const stage2Progress = status?.stage2_progress ?? 0;
  const queueLength = status?.stats?.queue_length ?? 0;
  const jobsCompleted = status?.stats?.jobs_completed ?? 0;
  const stage1Model = status?.active_models?.stage1_model ?? 'PatchCore';

  const progressPct = labelsForStage2 > 0
    ? Math.min((labelsCollected / labelsForStage2) * 100, 100)
    : 0;

  return (
    <div>
      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">System Overview</h1>
          <p className="text-sm text-text-tertiary mt-1">
            Real-time anomaly detection pipeline
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge color={stage2Available ? 'kul' : 'pass'} dot>
            Stage {stage} Active
          </Badge>
          {lastUpdated && (
            <span className="text-text-tertiary text-xs">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-6">
          <ErrorBanner message={error} onRetry={fetchData} />
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatusCard
          title="Pipeline Status"
          value={isHealthy ? (stage2Available ? 'Stage 2' : 'Stage 1') : 'Offline'}
          subtitle={isHealthy ? (stage2Available ? 'Supervised active' : 'Unsupervised') : 'Backend unreachable'}
          icon={GitBranch}
          color="kul"
          loading={loading}
        />
        <StatusCard
          title="Images Processed"
          value={jobsCompleted}
          subtitle="Total inference jobs"
          icon={Activity}
          loading={loading}
        />
        <StatusCard
          title="Queue Length"
          value={queueLength}
          subtitle="Pending inference jobs"
          icon={AlertTriangle}
          color={queueLength > 10 ? 'warn' : 'default'}
          loading={loading}
        />
        <StatusCard
          title="Labels Collected"
          value={`${labelsCollected} / ${labelsForStage2}`}
          subtitle={`${Math.round(stage2Progress)}% to Stage 2`}
          icon={Tag}
          color="pass"
          loading={loading}
        />
      </div>

      {/* Live feed + Active learning */}
      <div className="grid grid-cols-3 gap-6 mb-8">

        {/* Live alerts feed */}
        <GlassCard padding="md" className="col-span-2">
          <SectionHeader
            title="Live Anomaly Feed"
            action={
              <span className="flex items-center gap-1.5 text-xs text-state-pass">
                <span className="w-1.5 h-1.5 rounded-full bg-state-pass animate-pulse" />
                Live
              </span>
            }
          />
          {loading ? (
            <LoadingSkeleton lines={4} heights={['h-10', 'h-10', 'h-10', 'h-10']} />
          ) : alerts.length === 0 ? (
            <div className="flex flex-col items-center py-12 text-center">
              <Activity className="w-8 h-8 text-surface-border mb-3" />
              <p className="text-text-tertiary text-sm">No anomalies detected</p>
              <p className="text-text-disabled text-xs mt-1">
                Waiting for inference results...
              </p>
            </div>
          ) : (
            <div className="overflow-y-auto max-h-80 -mx-2">
              {alerts.map((alert, i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 py-2.5 px-2 border-b border-surface-border last:border-0"
                >
                  <div className="w-10 h-10 rounded-lg bg-surface-overlay flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-text-primary truncate font-mono">
                      {alert.image_id}
                    </p>
                    <p className="text-xs text-text-tertiary">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                  <AnomalyScoreBar
                    score={alert.anomaly_score}
                    size="sm"
                    showValue
                    className="w-24"
                  />
                  <Badge color={alert.is_anomaly ? 'alert' : 'pass'}>
                    {alert.is_anomaly ? 'Anomaly' : 'Normal'}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

        {/* Active learning progress */}
        <GlassCard padding="md" className="col-span-1">
          <SectionHeader title="Active Learning" />

          <div className="text-center mb-4">
            <p className="font-mono text-3xl font-semibold text-text-primary tabular-nums">
              {loading ? '—' : labelsCollected}
            </p>
            <p className="text-xs text-text-tertiary mt-1">labels collected</p>
          </div>

          <div className="space-y-1 mb-1">
            <div className="w-full h-1 bg-surface-overlay rounded-full overflow-hidden">
              <div
                className="h-full bg-kul-accent rounded-full transition-all duration-500"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
          <p className="text-xs text-text-tertiary text-center mb-6">
            Stage 2 activates at {labelsForStage2} labels
          </p>

          <div className="border-t border-surface-border pt-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-primary">{stage1Model}</span>
              <Badge color="pass">Active</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className={['text-sm', stage2Available ? 'text-text-primary' : 'text-text-tertiary'].join(' ')}>
                BGAD
              </span>
              <Badge color={stage2Available ? 'kul' : 'default'}>
                {stage2Available ? 'Active' : 'Pending'}
              </Badge>
            </div>
          </div>

          <p className="text-xs text-text-disabled text-center mt-4">
            Expert labels unlock supervised detection
          </p>
        </GlassCard>
      </div>

      {/* Model cards */}
      <div className="grid grid-cols-2 gap-4">
        {/* Stage 1 */}
        <Card padding="md" className="border-l-2 border-l-state-pass">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
              Stage 1 — Unsupervised
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-state-pass animate-pulse" />
              <span className="text-state-pass text-xs">Active</span>
            </span>
          </div>
          <div className="flex items-end justify-between mt-3 mb-1">
            <div>
              <p className="text-lg font-semibold text-text-primary">PatchCore</p>
              <p className="text-xs text-text-tertiary mt-0.5">Memory bank · k-NN</p>
            </div>
            <Cpu className="w-6 h-6 text-state-pass/40" />
          </div>
          <div className="mt-4">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-text-tertiary">AUC</span>
              <span className="text-xs font-mono text-text-primary">0.895</span>
            </div>
            <AnomalyScoreBar score={0.895} size="xs" />
          </div>
        </Card>

        {/* Stage 2 */}
        <Card
          padding="md"
          className={[
            'border-l-2',
            stage2Available ? 'border-l-kul-accent' : 'border-l-surface-border opacity-60',
          ].join(' ')}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
              Stage 2 — Supervised
            </span>
            {stage2Available ? (
              <span className="inline-flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-kul-accent animate-pulse" />
                <span className="text-kul-accent text-xs">Active</span>
              </span>
            ) : (
              <Badge color="default">Pending</Badge>
            )}
          </div>
          <div className="flex items-end justify-between mt-3 mb-1">
            <div>
              <p className="text-lg font-semibold text-text-primary">BGAD</p>
              <p className="text-xs text-text-tertiary mt-0.5">
                Boundary-guided · Push-pull
              </p>
            </div>
            <Zap className="w-6 h-6 text-kul-accent/40" />
          </div>
          <div className="mt-4">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-text-tertiary">AUC</span>
              <span className="text-xs font-mono text-text-primary">0.930</span>
            </div>
            <AnomalyScoreBar score={0.930} size="xs" />
          </div>
        </Card>
      </div>
    </div>
  );
}
