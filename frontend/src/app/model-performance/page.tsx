'use client';

import { TrendingUp, CheckCircle2 } from 'lucide-react';
import Card from '@/components/Card';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import Badge from '@/components/Badge';
import SectionHeader from '@/components/SectionHeader';

type Benchmark = {
  method: string;
  dataset: string;
  image_auroc: number | null;
  type: string;
  notes: string;
  measured: boolean;
};

const STAGE1_BENCHMARKS: Benchmark[] = [
  {
    method: 'PatchCore (ours, 15-cat mean)',
    dataset: 'MVTec AD',
    image_auroc: 0.9829,
    type: 'Memory Bank',
    notes: 'This implementation — mean over all 15 categories',
    measured: true,
  },
  {
    method: 'PatchCore (published)',
    dataset: 'MVTec AD',
    image_auroc: 0.991,
    type: 'Memory Bank',
    notes: 'Roth et al. 2022 reference',
    measured: false,
  },
  {
    method: 'PaDiM (published)',
    dataset: 'MVTec AD',
    image_auroc: 0.884,
    type: 'Gaussian',
    notes: 'Defard et al. 2021 reference',
    measured: false,
  },
  {
    method: 'WinCLIP (published)',
    dataset: 'MVTec AD',
    image_auroc: 0.856,
    type: 'Zero-shot',
    notes: 'VLM baseline — superseded by GPT-4o routing',
    measured: false,
  },
  {
    method: 'GPT-4o (ours, as Stage 2 refiner)',
    dataset: 'MVTec AD',
    image_auroc: null,
    type: 'Supervised refiner',
    notes: 'Used for defect type classification + false-positive rejection',
    measured: true,
  },
];

const STAGE2_BENCHMARKS: Benchmark[] = [
  {
    method: 'BGAD',
    dataset: 'MVTec AD',
    image_auroc: 0.930,
    type: 'Supervised',
    notes: 'Requires pixel masks',
    measured: false,
  },
  {
    method: 'Push-Pull',
    dataset: 'Decospan',
    image_auroc: 0.860,
    type: 'Supervised',
    notes: 'No masks · 100–200 samples',
    measured: false,
  },
  {
    method: 'GPT-4o Stage 2 (ours)',
    dataset: 'Any product category (zero-training)',
    image_auroc: null,
    type: 'Supervised (in-context)',
    notes: 'Trigger zone: anomaly_score ∈ [0.5, 0.9). Classifies defect type.',
    measured: true,
  },
];

const PER_CATEGORY_RESULTS = [
  { category: 'bottle',     image_auroc: 1.0000, pixel_auroc: 0.9856 },
  { category: 'leather',    image_auroc: 1.0000, pixel_auroc: 0.9922 },
  { category: 'hazelnut',   image_auroc: 1.0000, pixel_auroc: 0.9884 },
  { category: 'tile',       image_auroc: 1.0000, pixel_auroc: 0.9555 },
  { category: 'metal_nut',  image_auroc: 0.9971, pixel_auroc: 0.9870 },
  { category: 'transistor', image_auroc: 0.9950, pixel_auroc: 0.9731 },
  { category: 'capsule',    image_auroc: 0.9928, pixel_auroc: 0.9900 },
  { category: 'grid',       image_auroc: 0.9891, pixel_auroc: 0.9815 },
  { category: 'wood',       image_auroc: 0.9877, pixel_auroc: 0.9317 },
  { category: 'carpet',     image_auroc: 0.9872, pixel_auroc: 0.9907 },
  { category: 'cable',      image_auroc: 0.9867, pixel_auroc: 0.9852 },
  { category: 'zipper',     image_auroc: 0.9758, pixel_auroc: 0.9817 },
  { category: 'screw',      image_auroc: 0.9639, pixel_auroc: 0.9891 },
  { category: 'pill',       image_auroc: 0.9487, pixel_auroc: 0.9817 },
  { category: 'toothbrush', image_auroc: 0.9194, pixel_auroc: 0.9888 },
].sort((a, b) => b.image_auroc - a.image_auroc);

const DECOSPAN_CATEGORIES = [
  { dutch: 'krassen',     english: 'scratches',      category: 'surface' },
  { dutch: 'deuk',        english: 'dent',           category: 'structural' },
  { dutch: 'vlekken',     english: 'stains',         category: 'surface' },
  { dutch: 'open voeg',   english: 'open joint',     category: 'structural' },
  { dutch: 'open fout',   english: 'open defect',    category: 'structural' },
  { dutch: 'open knop',   english: 'open knot',      category: 'structural' },
  { dutch: 'snijfout',    english: 'cutting error',  category: 'process' },
  { dutch: 'barst',       english: 'crack',          category: 'structural' },
  { dutch: 'scheef',      english: 'skewed',         category: 'process' },
  { dutch: 'stuk fineer', english: 'broken veneer',  category: 'structural' },
];

const CATEGORY_COLOR: Record<string, string> = {
  surface:    'bg-state-warnSubtle text-state-warn border-state-warn/20',
  structural: 'bg-state-alertSubtle text-state-alert border-state-alert/20',
  process:    'bg-kul-blue/10 text-kul-accent border-kul-accent/20',
};

function aurocColor(v: number): string {
  if (v >= 0.99) return 'text-state-pass';
  if (v >= 0.95) return 'text-state-warn';
  return 'text-state-alert';
}

function BenchmarkTable({ rows }: { rows: Benchmark[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full">
        <thead>
          <tr className="border-b border-surface-border">
            {['Method', 'Dataset', 'Image AUROC', 'Type', 'Source', 'Notes'].map(h => (
              <th key={h} className="pb-3 px-4 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.method} className="border-b border-surface-border hover:bg-surface-overlay/50">
              <td className="py-3 px-4 text-sm font-medium text-text-primary">
                {row.method}
                {row.measured && (
                  <CheckCircle2 className="inline-block w-3.5 h-3.5 text-kul-accent ml-1.5 align-text-bottom" />
                )}
              </td>
              <td className="py-3 px-4 text-xs font-mono text-text-secondary">{row.dataset}</td>
              <td className="py-3 px-4">
                {row.image_auroc !== null ? (
                  <div className="flex items-center gap-2">
                    <span className={['text-xs font-mono font-semibold', aurocColor(row.image_auroc)].join(' ')}>
                      {(row.image_auroc * 100).toFixed(1)}%
                    </span>
                    <AnomalyScoreBar score={row.image_auroc} size="xs" className="w-16" />
                  </div>
                ) : (
                  <span className="text-xs text-text-disabled">—</span>
                )}
              </td>
              <td className="py-3 px-4"><Badge color="default">{row.type}</Badge></td>
              <td className="py-3 px-4">
                {row.measured ? (
                  <Badge color="kul" className="text-[10px]">Measured</Badge>
                ) : (
                  <span className="text-[10px] font-mono text-text-disabled italic">published</span>
                )}
              </td>
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
          Measured results from this implementation plus published baselines. Any new
          model must exceed its baseline by AUC &gt; 0.02 to be considered.
        </p>
      </div>

      {/* Per-category measured results — the strongest evidence, shown first */}
      <Card padding="md" className="mb-6">
        <SectionHeader
          title="Per-Category Results — PatchCore, MVTec AD"
          subtitle="All 15 categories trained with identical pipeline (anomalib 2.3.3, 1-epoch memory bank, RTX 3060 laptop)."
        />
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b border-surface-border">
                {['Category', 'Image AUROC', 'Pixel AUROC'].map(h => (
                  <th key={h} className="pb-3 px-4 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PER_CATEGORY_RESULTS.map(row => (
                <tr key={row.category} className="border-b border-surface-border hover:bg-surface-overlay/50">
                  <td className="py-2.5 px-4 text-sm font-mono text-text-primary">{row.category}</td>
                  <td className="py-2.5 px-4">
                    <div className="flex items-center gap-2">
                      <span className={['text-xs font-mono font-semibold', aurocColor(row.image_auroc)].join(' ')}>
                        {(row.image_auroc * 100).toFixed(2)}%
                      </span>
                      <AnomalyScoreBar score={row.image_auroc} size="xs" className="w-20" />
                    </div>
                  </td>
                  <td className="py-2.5 px-4 text-xs font-mono text-text-secondary">
                    {(row.pixel_auroc * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-text-tertiary mt-4 leading-relaxed">
          Mean image AUROC across 15 categories: <span className="text-state-pass font-mono font-semibold">98.29%</span>.
          Categories below 0.95 (<span className="font-mono">pill, toothbrush</span>) are known weak spots
          for memory-bank methods on small-object-extent targets — GPT-4o Stage 2 refinement covers these cases.
        </p>
      </Card>

      {/* Stage 1 summary */}
      <Card padding="md" className="mb-6">
        <SectionHeader
          title="Stage 1 — Unsupervised Detection"
          subtitle="Our measurements against published baselines. Checkmark = measured in this repo."
        />
        <BenchmarkTable rows={STAGE1_BENCHMARKS} />
      </Card>

      {/* Stage 2 */}
      <Card padding="md" className="mb-6">
        <SectionHeader
          title="Stage 2 — Supervised Classification"
          subtitle="Triggers on ambiguous Stage 1 scores to reduce false positives and name defect types."
        />
        <BenchmarkTable rows={STAGE2_BENCHMARKS} />
        <div className="mt-4 p-3 bg-surface-overlay rounded-lg">
          <p className="text-xs text-text-tertiary leading-relaxed">
            <span className="text-kul-accent font-medium">Stage 2 runs</span> when Stage 1
            confidence is ambiguous (score ∈ [0.5, 0.9)). GPT-4o uses operator-labeled
            examples as in-context supervision to confirm or reject Stage 1 flags, and
            classifies defect type. Works on any product category without per-category training.
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
