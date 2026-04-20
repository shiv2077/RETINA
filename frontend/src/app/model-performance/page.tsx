'use client';

import { useEffect, useState } from 'react';
import { Activity, Zap, GitBranch } from 'lucide-react';
import { getSystemStatus, type SystemStatusResponse } from '@/lib/api';
import Card from '@/components/Card';
import GlassCard from '@/components/GlassCard';
import StatusCard from '@/components/StatusCard';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import SectionHeader from '@/components/SectionHeader';
import Badge from '@/components/Badge';
import ErrorBanner from '@/components/ErrorBanner';
import LoadingSkeleton from '@/components/LoadingSkeleton';

interface BenchmarkRow {
  model: string;
  type: string;
  dataset: string;
  auc: string;
  aucValue: number;
  notes: string;
  stage: 1 | 2;
  active: boolean;
}

// Preserved exactly from original — accurate research data
const BENCHMARKS: BenchmarkRow[] = [
  {
    model: 'PatchCore',
    type: 'Unsupervised',
    dataset: 'MVTec AD (avg)',
    auc: '0.895',
    aucValue: 0.895,
    notes: 'Memory-bank k-NN, one-class learner, no defect labels needed',
    stage: 1,
    active: true,
  },
  {
    model: 'PaDiM',
    type: 'Unsupervised',
    dataset: 'MVTec AD (avg)',
    auc: '0.884',
    aucValue: 0.884,
    notes: 'Multivariate Gaussian per patch, fastest inference of the three',
    stage: 1,
    active: false,
  },
  {
    model: 'GPT-4o Vision',
    type: 'Zero-shot VLM',
    dataset: 'MVTec AD (est.)',
    auc: '0.856+',
    aucValue: 0.856,
    notes: 'Zero-shot, no training, produces human-readable defect descriptions',
    stage: 1,
    active: true,
  },
  {
    model: 'AdaCLIP',
    type: 'Zero-shot VLM',
    dataset: 'Decospan (wood)',
    auc: '~0.72',
    aucValue: 0.72,
    notes: 'CLIP with learnable prompts, ran on proprietary Decospan dataset',
    stage: 1,
    active: false,
  },
  {
    model: 'BGAD',
    type: 'Supervised',
    dataset: 'MVTec AD (avg)',
    auc: '0.930',
    aucValue: 0.930,
    notes: 'Best-in-class with labeled anomaly data, requires 50-200 labels',
    stage: 2,
    active: false,
  },
  {
    model: 'Custom Push-Pull',
    type: 'Supervised',
    dataset: 'Decospan (wood)',
    auc: '0.860',
    aucValue: 0.860,
    notes: 'Trained with 100 normal + 200 anomaly samples, EfficientNet-B0 backbone',
    stage: 2,
    active: false,
  },
];

interface DefectCategory {
  dutch: string;
  english: string;
  description: string;
  trainCount: number;
  testCount: number;
}

// Preserved exactly from original — accurate dataset labels
const DECOSPAN_CATEGORIES: DefectCategory[] = [
  { dutch: 'deuk',        english: 'Dent',           description: 'Physical indentation from impact or pressure',  trainCount: 25, testCount: 92  },
  { dutch: 'krassen',     english: 'Scratches',       description: 'Surface abrasion marks, linear or irregular',    trainCount: 25, testCount: 92  },
  { dutch: 'vlekken',     english: 'Stains',          description: 'Discolouration from contamination or moisture',  trainCount: 25, testCount: 91  },
  { dutch: 'open voeg',   english: 'Open Joint',      description: 'Gap between wood veneer segments',              trainCount: 25, testCount: 91  },
  { dutch: 'open fout',   english: 'Open Defect',     description: 'General open surface defect or void',           trainCount: 25, testCount: 91  },
  { dutch: 'open knop',   english: 'Open Knot',       description: 'Natural wood knot that has opened or split',    trainCount: 25, testCount: 91  },
  { dutch: 'snijfout',    english: 'Cutting Error',   description: 'Incorrect cut angle or dimension from sawing',  trainCount: 25, testCount: 40  },
  { dutch: 'barst',       english: 'Crack',           description: 'Structural crack through the material',         trainCount: 0,  testCount: 53  },
  { dutch: 'scheef',      english: 'Skewed',          description: 'Misaligned or crooked orientation',             trainCount: 0,  testCount: 8   },
  { dutch: 'stuk fineer', english: 'Broken Veneer',   description: 'Damaged or detached wood veneer layer',         trainCount: 0,  testCount: 62  },
  { dutch: 'zaag kort',   english: 'Cut Too Short',   description: 'Board cut shorter than specification',          trainCount: 0,  testCount: 29  },
  { dutch: 'zaag lang',   english: 'Cut Too Long',    description: 'Board cut longer than specification',           trainCount: 0,  testCount: 57  },
  { dutch: 'not_labeled', english: 'Unlabelled',      description: 'Anomalous samples awaiting expert labelling',   trainCount: 25, testCount: 795 },
];

export default function ModelPerformancePage() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [statusError, setStatusError] = useState(false);

  useEffect(() => {
    getSystemStatus()
      .then(s => { setStatus(s); setLoading(false); })
      .catch(() => { setStatusError(true); setLoading(false); });
  }, []);

  const activeStage = status?.current_stage ?? 1;

  return (
    <div>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-text-primary">Model Performance</h1>
        <p className="text-sm text-text-tertiary mt-1">
          Benchmark results across datasets, active model status, and defect category reference
        </p>
      </div>

      {/* Live status cards */}
      {statusError ? (
        <div className="mb-8">
          <ErrorBanner message="Could not reach backend — model status unavailable." />
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <StatusCard
            title="Active Pipeline Stage"
            value={loading ? '—' : `Stage ${activeStage}`}
            subtitle={status?.stage2_available ? 'Supervised active' : 'Unsupervised / VLM'}
            icon={GitBranch}
            color="kul"
            loading={loading}
          />
          <StatusCard
            title="Stage 1 Model"
            value={status?.active_models?.stage1_model ?? '—'}
            subtitle="Current zero-shot detector"
            icon={Activity}
            color="pass"
            loading={loading}
          />
          <StatusCard
            title="Stage 2 Progress"
            value={status ? `${Math.round(status.stage2_progress)}%` : '—'}
            subtitle={status ? `${status.stats.labels_collected} / ${status.labels_for_stage2} labels` : 'Loading…'}
            icon={Zap}
            color={status?.stage2_available ? 'pass' : 'default'}
            loading={loading}
          />
        </div>
      )}

      {/* Benchmark table */}
      <section className="mb-8">
        <SectionHeader title="Benchmark Results" />
        <Card padding="none" className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-surface-border bg-surface-overlay">
                  {['Model', 'Type', 'Dataset', 'AUC', 'Stage', 'Notes'].map(h => (
                    <th
                      key={h}
                      className="px-4 py-3 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {BENCHMARKS.map(row => (
                  <tr
                    key={`${row.model}-${row.dataset}`}
                    className={[
                      'border-b border-surface-border transition-colors',
                      row.active ? 'bg-kul-blue/8' : 'hover:bg-surface-overlay/50',
                    ].join(' ')}
                  >
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-text-primary">{row.model}</span>
                        {row.active && (
                          <span className="inline-flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-state-pass animate-pulse" />
                            <span className="text-state-pass text-xs">Active</span>
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-text-secondary whitespace-nowrap">{row.type}</td>
                    <td className="px-4 py-3 text-xs text-text-secondary whitespace-nowrap">{row.dataset}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-semibold text-text-primary">{row.auc}</span>
                        <AnomalyScoreBar score={row.aucValue} size="xs" className="w-16" />
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Badge color={row.stage === 1 ? 'kul' : 'purple'}>
                        Stage {row.stage}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-xs text-text-tertiary max-w-xs">{row.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      {/* Decospan categories */}
      <section className="mb-8">
        <SectionHeader
          title="Decospan Wood Defect Categories"
          subtitle="Proprietary dataset from KU Leuven Vakantiejob internship · 300 train · 10,285 test · Dutch names are canonical identifiers"
        />
        <Card padding="none" className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-surface-border bg-surface-overlay">
                  {['Dutch (canonical)', 'English', 'Description', 'Train', 'Test'].map(h => (
                    <th
                      key={h}
                      className="px-4 py-3 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {DECOSPAN_CATEGORIES.map(cat => (
                  <tr
                    key={cat.dutch}
                    className="border-b border-surface-border hover:bg-surface-overlay/50 transition-colors"
                  >
                    <td className="px-4 py-2.5 font-mono text-sm text-text-primary whitespace-nowrap">
                      {cat.dutch}
                    </td>
                    <td className="px-4 py-2.5 text-sm font-medium text-text-secondary whitespace-nowrap">
                      {cat.english}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-text-tertiary">{cat.description}</td>
                    <td className="px-4 py-2.5 text-center">
                      {cat.trainCount > 0 ? (
                        <span className="inline-flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-kul-accent" />
                          <span className="text-xs font-mono text-text-primary">{cat.trainCount}</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-surface-border" />
                          <span className="text-xs text-text-disabled">—</span>
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-xs font-mono text-center text-text-secondary">
                      {cat.testCount}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      {/* Architecture note */}
      <GlassCard padding="md" className="border-kul-accent/20 bg-kul-blue/5">
        <h2 className="text-base font-semibold text-text-primary mb-4">
          RETINA Two-Stage Architecture
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <p className="font-medium text-text-primary mb-2">Stage 1 — Unsupervised / Zero-Shot</p>
            <ul className="space-y-1.5 text-text-tertiary text-xs">
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-state-pass mt-1.5 flex-shrink-0" />
                PatchCore: memory-bank k-NN, trains on normal images only
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-state-pass mt-1.5 flex-shrink-0" />
                GPT-4o Vision: zero-shot, no training, text explanations
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-state-pass mt-1.5 flex-shrink-0" />
                Active from day one — no labelled defect data required
              </li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-text-primary mb-2">Stage 2 — Supervised (Active Learning)</p>
            <ul className="space-y-1.5 text-text-tertiary text-xs">
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-kul-accent mt-1.5 flex-shrink-0" />
                Activates after {status?.labels_for_stage2 ?? '—'} expert labels collected
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-kul-accent mt-1.5 flex-shrink-0" />
                Custom Push-Pull contrastive model (EfficientNet-B0 backbone)
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-kul-accent mt-1.5 flex-shrink-0" />
                Current progress:{' '}
                {status
                  ? `${Math.round(status.stage2_progress)}% (${status.stats.labels_collected}/${status.labels_for_stage2} labels)`
                  : '—'}
              </li>
            </ul>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
