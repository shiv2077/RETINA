'use client';

import { useState, useEffect } from 'react';
import { BarChart2, Target, Cpu, Zap, Brain } from 'lucide-react';
import { getSystemStatus, type SystemStatusResponse } from '@/lib/api';
import Card from '@/components/Card';
import GlassCard from '@/components/GlassCard';
import StatusCard from '@/components/StatusCard';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import SectionHeader from '@/components/SectionHeader';
import ErrorBanner from '@/components/ErrorBanner';
import Badge from '@/components/Badge';

interface EvaluationResult {
  category: string;
  image_auroc: number;
  pixel_auroc: number;
  f1_score: number;
  precision: number;
  recall: number;
  threshold: number;
  confusion_matrix: { tp: number; tn: number; fp: number; fn: number };
}

const MVTEC_BENCHMARKS = [
  { method: 'RETINA (Ours)',  image_auroc: null,  pixel_auroc: null,  type: 'Hybrid',          highlight: true  },
  { method: 'PatchCore',      image_auroc: 0.991, pixel_auroc: 0.981, type: 'Memory Bank',      highlight: false },
  { method: 'EfficientAD',    image_auroc: 0.991, pixel_auroc: 0.968, type: 'Student-Teacher',  highlight: false },
  { method: 'PaDiM',          image_auroc: 0.953, pixel_auroc: 0.975, type: 'Gaussian',         highlight: false },
  { method: 'DRAEM',          image_auroc: 0.980, pixel_auroc: 0.973, type: 'Reconstruction',   highlight: false },
];

export default function ResultsPage() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [evaluations, setEvaluations] = useState<Record<string, EvaluationResult>>({});
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const sys = await getSystemStatus().catch(() => null);
        if (sys) setStatus(sys);
      } catch {
        setError('Backend unreachable — model status unavailable.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const aggregateMetrics = Object.values(evaluations);
  const meanAUROC = aggregateMetrics.length > 0
    ? aggregateMetrics.reduce((s, e) => s + e.image_auroc, 0) / aggregateMetrics.length
    : null;
  const meanF1 = aggregateMetrics.length > 0
    ? aggregateMetrics.reduce((s, e) => s + e.f1_score, 0) / aggregateMetrics.length
    : null;

  const runEvaluation = async (category: string) => {
    setEvaluating(true);
    // Endpoint not yet implemented — graceful no-op
    await new Promise(r => setTimeout(r, 500));
    setEvaluating(false);
  };

  return (
    <div>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-text-primary">Evaluation Dashboard</h1>
        <p className="text-sm text-text-tertiary mt-1">
          Model benchmark results and per-category performance
        </p>
      </div>

      {error && (
        <div className="mb-6">
          <ErrorBanner message={error} onRetry={() => setError(null)} />
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatusCard
          title="Mean AUROC"
          value={meanAUROC !== null ? `${(meanAUROC * 100).toFixed(1)}%` : '—'}
          subtitle={`Across ${aggregateMetrics.length} categories`}
          icon={BarChart2}
          color="kul"
          loading={loading}
        />
        <StatusCard
          title="Mean F1 Score"
          value={meanF1 !== null ? `${(meanF1 * 100).toFixed(1)}%` : '—'}
          subtitle="Across evaluated categories"
          icon={Target}
          color="pass"
          loading={loading}
        />
        <StatusCard
          title="Stage 1 Model"
          value={status?.active_models?.stage1_model ?? '—'}
          subtitle="Current unsupervised detector"
          icon={Cpu}
          loading={loading}
        />
        <StatusCard
          title="Stage 2 Model"
          value={status?.stage2_available ? 'BGAD' : 'Pending'}
          subtitle={status?.stage2_available ? 'Supervised active' : 'Collecting labels'}
          icon={Zap}
          color={status?.stage2_available ? 'kul' : 'default'}
          loading={loading}
        />
      </div>

      {/* Per-category evaluation */}
      <Card padding="md" className="mb-6">
        <SectionHeader
          title="Per-Category Evaluation"
          action={
            <select
              value={selectedCategory}
              onChange={e => setSelectedCategory(e.target.value)}
              className="bg-surface-overlay border border-surface-border rounded-lg px-3 py-1.5 text-xs text-text-primary focus:border-kul-accent transition-colors"
            >
              <option value="all">All Categories</option>
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          }
        />

        {categories.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-48 text-center">
            <Brain className="w-10 h-10 text-surface-border mb-3" />
            <p className="text-sm text-text-tertiary">No evaluation data</p>
            <p className="text-xs text-text-disabled mt-1">
              Run inference on images to generate evaluation metrics
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-surface-border">
                  {['Category', 'Image AUROC', 'Pixel AUROC', 'F1', 'Precision', 'Recall', ''].map(h => (
                    <th
                      key={h}
                      className="pb-3 px-4 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {categories
                  .filter(c => selectedCategory === 'all' || c === selectedCategory)
                  .map(category => {
                    const ev = evaluations[category];
                    return (
                      <tr key={category} className="border-b border-surface-border hover:bg-surface-overlay/50">
                        <td className="py-3 px-4 font-medium text-text-primary capitalize text-sm">
                          {category}
                        </td>
                        {ev ? (
                          <>
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                <span className={[
                                  'text-xs font-mono font-semibold',
                                  ev.image_auroc >= 0.9 ? 'text-state-pass' :
                                  ev.image_auroc >= 0.7 ? 'text-state-warn' : 'text-state-alert',
                                ].join(' ')}>
                                  {(ev.image_auroc * 100).toFixed(1)}%
                                </span>
                                <AnomalyScoreBar score={ev.image_auroc} size="xs" className="w-16" />
                              </div>
                            </td>
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-mono text-text-primary">
                                  {(ev.pixel_auroc * 100).toFixed(1)}%
                                </span>
                                <AnomalyScoreBar score={ev.pixel_auroc} size="xs" className="w-16" />
                              </div>
                            </td>
                            <td className="py-3 px-4 text-xs font-mono text-text-primary">
                              {(ev.f1_score * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-xs font-mono text-text-primary">
                              {(ev.precision * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-xs font-mono text-text-primary">
                              {(ev.recall * 100).toFixed(1)}%
                            </td>
                          </>
                        ) : (
                          <td colSpan={5} className="py-3 px-4 text-xs text-text-disabled">
                            Not evaluated
                          </td>
                        )}
                        <td className="py-3 px-4">
                          <button
                            onClick={() => runEvaluation(category)}
                            disabled={evaluating}
                            className="px-3 py-1 bg-kul-blue/10 hover:bg-kul-blue/20 border border-kul-accent/20 text-kul-accent rounded text-xs transition-colors disabled:opacity-40"
                          >
                            {evaluating ? '...' : 'Evaluate'}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Confusion matrix */}
      {selectedCategory !== 'all' && evaluations[selectedCategory]?.confusion_matrix && (
        <div className="grid grid-cols-2 gap-6 mb-6">
          <Card padding="md">
            <SectionHeader title={`Confusion Matrix — ${selectedCategory}`} />
            <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto text-sm">
              <div />
              <div className="text-center text-xs text-text-tertiary">Pred Normal</div>
              <div className="text-center text-xs text-text-tertiary">Pred Anomaly</div>
              <div className="text-right text-xs text-text-tertiary flex items-center justify-end">Actual Normal</div>
              <div className="p-4 bg-state-passSubtle border border-state-pass/20 rounded text-center font-semibold text-state-pass">
                {evaluations[selectedCategory].confusion_matrix.tn}
              </div>
              <div className="p-4 bg-state-alertSubtle border border-state-alert/20 rounded text-center font-semibold text-state-alert">
                {evaluations[selectedCategory].confusion_matrix.fp}
              </div>
              <div className="text-right text-xs text-text-tertiary flex items-center justify-end">Actual Anomaly</div>
              <div className="p-4 bg-state-alertSubtle border border-state-alert/20 rounded text-center font-semibold text-state-alert">
                {evaluations[selectedCategory].confusion_matrix.fn}
              </div>
              <div className="p-4 bg-state-passSubtle border border-state-pass/20 rounded text-center font-semibold text-state-pass">
                {evaluations[selectedCategory].confusion_matrix.tp}
              </div>
            </div>
          </Card>

          <Card padding="md">
            <SectionHeader title="Metric Breakdown" />
            <div className="space-y-4">
              {(['image_auroc', 'pixel_auroc', 'f1_score', 'precision', 'recall'] as const).map(key => {
                const val = evaluations[selectedCategory][key];
                return (
                  <div key={key}>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-text-tertiary capitalize">{key.replace('_', ' ')}</span>
                      <span className="font-mono font-semibold text-text-primary">
                        {(val * 100).toFixed(1)}%
                      </span>
                    </div>
                    <AnomalyScoreBar score={val} size="sm" />
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      )}

      {/* MVTec benchmark */}
      <Card padding="md">
        <SectionHeader
          title="MVTec AD Benchmark"
          subtitle="State-of-the-art comparison on the MVTec Anomaly Detection dataset"
        />
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b border-surface-border">
                {['Method', 'Image AUROC', 'Pixel AUROC', 'Type'].map(h => (
                  <th key={h} className="pb-3 px-4 text-left text-xs font-medium text-text-tertiary uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MVTEC_BENCHMARKS.map(row => (
                <tr
                  key={row.method}
                  className={[
                    'border-b border-surface-border',
                    row.highlight ? 'bg-kul-blue/8' : 'hover:bg-surface-overlay/50',
                  ].join(' ')}
                >
                  <td className="py-3 px-4">
                    <span className={['text-sm font-medium', row.highlight ? 'text-kul-accent' : 'text-text-primary'].join(' ')}>
                      {row.method}
                    </span>
                    {row.highlight && meanAUROC === null && (
                      <Badge color="kul" className="ml-2 text-[10px]">Live</Badge>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    {row.image_auroc !== null ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-text-primary">
                          {(row.image_auroc * 100).toFixed(1)}%
                        </span>
                        <AnomalyScoreBar score={row.image_auroc} size="xs" className="w-16" />
                      </div>
                    ) : (
                      <span className="text-xs text-text-disabled">
                        {meanAUROC !== null ? `${(meanAUROC * 100).toFixed(1)}%` : 'No eval data'}
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-4 text-xs font-mono text-text-primary">
                    {row.pixel_auroc !== null ? `${(row.pixel_auroc * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-3 px-4">
                    <Badge color="default">{row.type}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
