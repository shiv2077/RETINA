'use client';

import { useState, useEffect, useCallback } from 'react';
import { Activity, GitBranch, Tag, Cpu, Zap } from 'lucide-react';
import { getHealth, getLabelPoolV2, type PoolItem } from '@/lib/api';
import GlassCard from '@/components/GlassCard';
import Card from '@/components/Card';
import StatusCard from '@/components/StatusCard';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import SectionHeader from '@/components/SectionHeader';
import Badge from '@/components/Badge';
import ErrorBanner from '@/components/ErrorBanner';
import LoadingSkeleton from '@/components/LoadingSkeleton';

export default function DashboardPage() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [pool, setPool] = useState<PoolItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [health, poolRes] = await Promise.all([
        getHealth().catch(() => null),
        getLabelPoolV2(10).catch(() => null),
      ]);
      setIsHealthy(health?.status === 'ok');
      setPool(poolRes?.pool ?? []);
      setLastUpdated(new Date());
      setError(null);
    } catch {
      setError('Unable to reach backend. Is the API running on port 3001?');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10_000);
    return () => clearInterval(id);
  }, [fetchData]);

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
          <Badge color={isHealthy ? 'pass' : 'alert'} dot>
            {isHealthy ? 'Backend Online' : 'Backend Offline'}
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

      {/* Stat cards — sourced from GET /health and GET /api/labels/pool */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
        <StatusCard
          title="Pipeline Status"
          value={isHealthy ? 'Online' : 'Offline'}
          subtitle={isHealthy ? 'API and Redis reachable' : 'API unreachable'}
          icon={GitBranch}
          color={isHealthy ? 'kul' : 'default'}
          loading={loading}
        />
        <StatusCard
          title="Awaiting Review"
          value={pool.length}
          subtitle="Images in the active learning pool"
          icon={Tag}
          color="pass"
          loading={loading}
        />
      </div>

      {/* Active learning pool — GET /api/labels/pool */}
      <GlassCard padding="md" className="mb-8">
        <SectionHeader
          title="Awaiting Expert Review"
          subtitle="Highest-uncertainty images queued for labelling"
        />
        {loading ? (
          <LoadingSkeleton lines={4} heights={['h-10', 'h-10', 'h-10', 'h-10']} />
        ) : pool.length === 0 ? (
          <div className="flex flex-col items-center py-12 text-center">
            <Activity className="w-8 h-8 text-surface-border mb-3" />
            <p className="text-text-tertiary text-sm">Review queue is empty</p>
            <p className="text-text-disabled text-xs mt-1">
              Submit images to populate the active learning pool
            </p>
          </div>
        ) : (
          <div className="overflow-y-auto max-h-80 -mx-2">
            {pool.map(item => (
              <div
                key={item.image_id}
                className="flex items-center gap-3 py-2.5 px-2 border-b border-surface-border last:border-0"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-text-primary truncate font-mono">
                    {item.image_id}
                  </p>
                  <p className="text-xs text-text-tertiary">
                    {item.product_class ?? 'unidentified product'}
                  </p>
                </div>
                <AnomalyScoreBar
                  score={item.anomaly_score ?? item.score}
                  size="sm"
                  showValue
                  className="w-24"
                />
              </div>
            ))}
          </div>
        )}
      </GlassCard>

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
        <Card padding="md" className="border-l-2 border-l-kul-accent">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
              Stage 2 — Supervised
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-kul-accent animate-pulse" />
              <span className="text-kul-accent text-xs">Active</span>
            </span>
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
