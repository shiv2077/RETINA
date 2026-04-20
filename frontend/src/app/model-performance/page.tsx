'use client';

import { Brain, TrendingUp } from 'lucide-react';
import Card from '@/components/Card';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import Badge from '@/components/Badge';
import SectionHeader from '@/components/SectionHeader';

const STAGE1_BENCHMARKS = [
  { method: 'PatchCore',  dataset: 'MVTec AD', image_auroc: 0.895, type: 'Memory Bank',     notes: 'Primary Stage 1 model' },
  { method: 'PaDiM',      dataset: 'MVTec AD', image_auroc: 0.884, type: 'Gaussian',         notes: 'Fastest inference' },
  { method: 'WinCLIP',    dataset: 'MVTec AD', image_auroc: 0.856, type: 'Zero-shot',        notes: 'VLM baseline' },
  { method: 'GPT-4o',     dataset: 'MVTec AD', image_auroc: null,  type: 'Zero-shot VLM',   notes: 'Cold-start fallback' },
];

const STAGE2_BENCHMARKS = [
  { method: 'BGAD',       dataset: 'MVTec AD', image_auroc: 0.930, type: 'Supervised',       notes: 'Requires masks' },
  { method: 'Push-Pull',  dataset: 'Decospan', image_auroc: 0.860, type: 'Supervised',       notes: 'No masks, 100-200 samples' },
];

const DECOSPAN_CATEGORIES = [
  { dutch: 'krassen',   english: 'scratches',      category: 'surface' },
  { dutch: 'deuk',      english: 'dent',           category: 'structural' },
  { dutch: 'vlekken',   english: 'stains',         category: 'surface' },
  { dutch: 'open voeg', english: 'open joint',     category: 'structural' },
  { dutch: 'open fout', english: 'open defect',    category: 'structural' },
  { dutch: 'open knop', english: 'open knot',      category: 'structural' },
  { dutch: 'snijfout',  english: 'cutting error',  category: 'process' },
  { dutch: 'barst',     english: 'crack',          category: 'structural' },
  { dutch: 'scheef',    english: 'skewed',         category: 'process' },
  { dutch: 'stuk fineer', english: 'broken veneer', category: 'structural' },
];

const CATEGORY_COLOR: Record<string, string> = {
  surface:    'bg-state-warnSubtle text-state-warn border-state-warn/20',
  structural: 'bg-state-alertSubtle text-state-alert border-state-alert/20',
  process:    'bg-kul-blue/10 text-kul-accent border-kul-accent/20',
};

function BenchmarkTable({ rows }: { rows: typeof STAGE1_BENCHMARKS }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full">
        <thead>
          <tr className="border-b border-surface-border">
            {['Method', 'Dataset', 'Image AUROC', 'Type', 'Notes'].map(h => (
              <th key={h} className="pb-3 px-4 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.method} className="border-b border-surface-border hover:bg-surface-overlay/50">
              <td className="py-3 px-4 text-sm font-medium text-text-primary">{row.method}</td>
              <td className="py-3 px-4 text-xs font-mono text-text-secondary">{row.dataset}</td>
              <td className="py-3 px-4">
                {row.image_auroc !== null ? (
                  <div className="flex items-center gap-2">
                    <span className={[
                      'text-xs font-mono font-semibold',
                      row.image_auroc >= 0.9 ? 'text-state-pass' :
                      row.image_auroc >= 0.7 ? 'text-state-warn' : 'text-state-alert',
                    ].join(' ')}>
                      {(row.image_auroc * 100).toFixed(1)}%
                    </span>
                    <AnomalyScoreBar score={row.image_auroc} size="xs" className="w-16" />
                  </div>
                ) : (
                  <span className="text-xs text-text-disabled">TBD</span>
                )}
              </td>
              <td className="py-3 px-4"><Badge color="default">{row.type}</Badge></td>
              <td className="py-3 px-4 text-xs text-text-tertiary">{row.notes}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPerformancePage() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-text-primary">Model Performance</h1>
        <p className="text-sm text-text-tertiary mt-1">
          Reference benchmarks from published results — any new model must exceed baseline by AUC &gt; 0.02
        </p>
      </div>

      {/* Stage 1 */}
      <Card padding="md" className="mb-6">
        <SectionHeader
          title="Stage 1 — Unsupervised Detection"
          subtitle="Trained on normal images only. MVTec AD dataset."
        />
        <BenchmarkTable rows={STAGE1_BENCHMARKS} />
      </Card>

      {/* Stage 2 */}
      <Card padding="md" className="mb-6">
        <SectionHeader
          title="Stage 2 — Supervised Classification"
          subtitle="Requires expert-labeled defect samples from active learning."
        />
        <BenchmarkTable rows={STAGE2_BENCHMARKS} />
        <div className="mt-4 p-3 bg-surface-overlay rounded-lg">
          <p className="text-xs text-text-tertiary">
            <span className="text-kul-accent font-medium">Stage 2 activates</span> automatically
            when the active learning pool reaches the configured label threshold (default: 200 labels).
            Before activation, all Stage 1 anomaly flags route to the Expert Review queue.
          </p>
        </div>
      </Card>

      {/* Decospan taxonomy */}
      <Card padding="md">
        <SectionHeader
          title="Decospan Defect Taxonomy"
          subtitle="Dutch names are canonical. English names are display-only and must not be used as code identifiers."
        />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mt-2">
          {DECOSPAN_CATEGORIES.map(({ dutch, english, category }) => (
            <div
              key={dutch}
              className={['rounded-lg border px-4 py-3', CATEGORY_COLOR[category]].join(' ')}
            >
              <p className="text-sm font-mono font-semibold">{dutch}</p>
              <p className="text-xs opacity-70 mt-0.5">{english}</p>
              <Badge color="default" className="mt-2 text-[10px]">{category}</Badge>
            </div>
          ))}
        </div>

        <div className="mt-6 flex items-start gap-3 p-3 bg-surface-overlay rounded-lg">
          <TrendingUp className="w-4 h-4 text-kul-accent flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs text-text-tertiary leading-relaxed">
              <span className="text-text-secondary font-medium">Dataset location:</span>{' '}
              KU Leuven HPC — <span className="font-mono">/scratch/leuven/369/vsc36963/Vakantiejob/Decospan/Dataset</span>
              {' '}(not in repo — must be transferred separately for training).
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
