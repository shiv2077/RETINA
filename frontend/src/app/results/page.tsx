'use client';

/**
 * RETINA - Evaluation & Results Dashboard
 * =======================================
 *
 * Comprehensive evaluation dashboard showing:
 * - Model performance metrics (AUROC, F1, etc.)
 * - Per-category breakdown
 * - Confusion matrix visualization
 * - ROC curves
 * - Recent inference results
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface EvaluationResult {
  category: string;
  image_auroc: number;
  pixel_auroc: number;
  f1_score: number;
  precision: number;
  recall: number;
  threshold: number;
  confusion_matrix: {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
  };
}

interface ModelStatus {
  patchcore_trained: boolean;
  bgad_trained: boolean;
  category: string | null;
}

interface RecentResult {
  image_id: string;
  anomaly_score: number;
  is_anomaly: boolean;
  timestamp: string;
}

export default function ResultsPage() {
  const [evaluations, setEvaluations] = useState<Record<string, EvaluationResult>>({});
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [categories, setCategories] = useState<string[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [recentResults, setRecentResults] = useState<RecentResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusRes, categoriesRes, evalsRes] = await Promise.all([
          fetch(`${API_URL}/status`).catch(() => null),
          fetch(`${API_URL}/categories`).catch(() => null),
          fetch(`${API_URL}/evaluations`).catch(() => null),
        ]);

        if (statusRes?.ok) {
          const data = await statusRes.json();
          setModelStatus(data.pipeline);
        }

        if (categoriesRes?.ok) {
          const data = await categoriesRes.json();
          setCategories(data.categories || []);
        }

        if (evalsRes?.ok) {
          const data = await evalsRes.json();
          setEvaluations(data.evaluations || {});
        }
      } catch (e) {
        console.error('Failed to fetch data:', e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Run evaluation
  const runEvaluation = async (category: string) => {
    setEvaluating(true);
    try {
      const res = await fetch(`${API_URL}/pipeline/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category }),
      });

      if (res.ok) {
        const data = await res.json();
        setEvaluations(prev => ({
          ...prev,
          [category]: data.result,
        }));
      }
    } catch (e) {
      console.error('Evaluation failed:', e);
    } finally {
      setEvaluating(false);
    }
  };

  // Calculate aggregate metrics
  const aggregateMetrics = Object.values(evaluations);
  const meanAUROC = aggregateMetrics.length > 0
    ? aggregateMetrics.reduce((sum, e) => sum + e.image_auroc, 0) / aggregateMetrics.length
    : 0;
  const meanF1 = aggregateMetrics.length > 0
    ? aggregateMetrics.reduce((sum, e) => sum + e.f1_score, 0) / aggregateMetrics.length
    : 0;

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold">RETINA</h1>
                <p className="text-xs text-slate-400">Evaluation Dashboard</p>
              </div>
            </Link>
          </div>

          <nav className="flex items-center space-x-6">
            <Link href="/" className="text-slate-400 hover:text-white transition">Dashboard</Link>
            <Link href="/label" className="text-slate-400 hover:text-white transition">Label</Link>
            <Link href="/results" className="text-blue-400 font-medium">Results</Link>
            <Link href="/demo" className="text-slate-400 hover:text-white transition">Demo</Link>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Aggregate Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="text-slate-400 text-sm">Mean AUROC</div>
              <span className="text-2xl">📊</span>
            </div>
            <div className="text-3xl font-bold mt-2">
              {(meanAUROC * 100).toFixed(2)}%
            </div>
            <div className="text-sm text-slate-500 mt-1">
              Across {aggregateMetrics.length} categories
            </div>
          </div>

          <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="text-slate-400 text-sm">Mean F1 Score</div>
              <span className="text-2xl">🎯</span>
            </div>
            <div className="text-3xl font-bold mt-2">
              {(meanF1 * 100).toFixed(2)}%
            </div>
          </div>

          <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="text-slate-400 text-sm">PatchCore</div>
              <span className={`w-3 h-3 rounded-full ${modelStatus?.patchcore_trained ? 'bg-green-500' : 'bg-red-500'}`}></span>
            </div>
            <div className="text-xl font-bold mt-2">
              {modelStatus?.patchcore_trained ? 'Trained' : 'Not Trained'}
            </div>
            <div className="text-sm text-slate-500 mt-1">
              {modelStatus?.category || 'No category'}
            </div>
          </div>

          <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="text-slate-400 text-sm">BGAD</div>
              <span className={`w-3 h-3 rounded-full ${modelStatus?.bgad_trained ? 'bg-green-500' : 'bg-red-500'}`}></span>
            </div>
            <div className="text-xl font-bold mt-2">
              {modelStatus?.bgad_trained ? 'Trained' : 'Not Trained'}
            </div>
            <div className="text-sm text-slate-500 mt-1">
              Supervised refinement
            </div>
          </div>
        </div>

        {/* Per-Category Results */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Per-Category Evaluation</h2>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
            >
              <option value="all">All Categories</option>
              {categories.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-slate-400">Category</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-400">Image AUROC</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-400">Pixel AUROC</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-400">F1 Score</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-400">Precision</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-400">Recall</th>
                  <th className="text-center py-3 px-4 text-sm font-semibold text-slate-400">Actions</th>
                </tr>
              </thead>
              <tbody>
                {categories
                  .filter(cat => selectedCategory === 'all' || cat === selectedCategory)
                  .map(category => {
                    const eval_ = evaluations[category];
                    return (
                      <tr key={category} className="border-b border-slate-700/50 hover:bg-slate-800/50">
                        <td className="py-3 px-4 font-medium capitalize">{category}</td>
                        <td className="py-3 px-4 text-center">
                          {eval_ ? (
                            <span className={`font-semibold ${eval_.image_auroc >= 0.9 ? 'text-green-400' : eval_.image_auroc >= 0.7 ? 'text-yellow-400' : 'text-red-400'}`}>
                              {(eval_.image_auroc * 100).toFixed(2)}%
                            </span>
                          ) : (
                            <span className="text-slate-500">-</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-center">
                          {eval_ ? (
                            <span className={`font-semibold ${eval_.pixel_auroc >= 0.9 ? 'text-green-400' : eval_.pixel_auroc >= 0.7 ? 'text-yellow-400' : 'text-red-400'}`}>
                              {(eval_.pixel_auroc * 100).toFixed(2)}%
                            </span>
                          ) : (
                            <span className="text-slate-500">-</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-center">
                          {eval_ ? `${(eval_.f1_score * 100).toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-3 px-4 text-center">
                          {eval_ ? `${(eval_.precision * 100).toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-3 px-4 text-center">
                          {eval_ ? `${(eval_.recall * 100).toFixed(2)}%` : '-'}
                        </td>
                        <td className="py-3 px-4 text-center">
                          <button
                            onClick={() => runEvaluation(category)}
                            disabled={evaluating}
                            className="px-3 py-1 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 rounded text-sm"
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
        </div>

        {/* Confusion Matrix (if available for selected category) */}
        {selectedCategory !== 'all' && evaluations[selectedCategory]?.confusion_matrix && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
              <h3 className="text-lg font-semibold mb-4">Confusion Matrix - {selectedCategory}</h3>
              <div className="grid grid-cols-3 gap-2 max-w-xs mx-auto">
                <div></div>
                <div className="text-center text-sm text-slate-400">Pred Normal</div>
                <div className="text-center text-sm text-slate-400">Pred Anomaly</div>
                
                <div className="text-right text-sm text-slate-400 flex items-center justify-end">Actual Normal</div>
                <div className="p-4 bg-green-500/20 border border-green-500/50 rounded text-center font-semibold text-green-400">
                  {evaluations[selectedCategory].confusion_matrix.tn}
                </div>
                <div className="p-4 bg-red-500/20 border border-red-500/50 rounded text-center font-semibold text-red-400">
                  {evaluations[selectedCategory].confusion_matrix.fp}
                </div>
                
                <div className="text-right text-sm text-slate-400 flex items-center justify-end">Actual Anomaly</div>
                <div className="p-4 bg-red-500/20 border border-red-500/50 rounded text-center font-semibold text-red-400">
                  {evaluations[selectedCategory].confusion_matrix.fn}
                </div>
                <div className="p-4 bg-green-500/20 border border-green-500/50 rounded text-center font-semibold text-green-400">
                  {evaluations[selectedCategory].confusion_matrix.tp}
                </div>
              </div>
              <div className="flex justify-center space-x-4 mt-4 text-sm text-slate-400">
                <span>TN: True Negative</span>
                <span>FP: False Positive</span>
                <span>FN: False Negative</span>
                <span>TP: True Positive</span>
              </div>
            </div>

            <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
              <h3 className="text-lg font-semibold mb-4">Performance Breakdown</h3>
              <div className="space-y-4">
                {[
                  { label: 'Image AUROC', value: evaluations[selectedCategory].image_auroc },
                  { label: 'Pixel AUROC', value: evaluations[selectedCategory].pixel_auroc },
                  { label: 'F1 Score', value: evaluations[selectedCategory].f1_score },
                  { label: 'Precision', value: evaluations[selectedCategory].precision },
                  { label: 'Recall', value: evaluations[selectedCategory].recall },
                ].map(({ label, value }) => (
                  <div key={label}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-400">{label}</span>
                      <span className="font-semibold">{(value * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          value >= 0.9 ? 'bg-green-500' : value >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* MVTec Benchmark Comparison */}
        <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
          <h3 className="text-lg font-semibold mb-4">MVTec AD Benchmark Reference</h3>
          <p className="text-slate-400 text-sm mb-4">
            Comparison with state-of-the-art methods on MVTec Anomaly Detection dataset
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-2 px-4">Method</th>
                  <th className="text-center py-2 px-4">Image AUROC</th>
                  <th className="text-center py-2 px-4">Pixel AUROC</th>
                  <th className="text-center py-2 px-4">Type</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                <tr className="border-b border-slate-700/50 bg-blue-500/10">
                  <td className="py-2 px-4 font-semibold text-blue-400">RETINA (Ours)</td>
                  <td className="py-2 px-4 text-center">{(meanAUROC * 100).toFixed(1)}%</td>
                  <td className="py-2 px-4 text-center">-</td>
                  <td className="py-2 px-4 text-center">Hybrid</td>
                </tr>
                <tr className="border-b border-slate-700/50">
                  <td className="py-2 px-4">PatchCore</td>
                  <td className="py-2 px-4 text-center">99.1%</td>
                  <td className="py-2 px-4 text-center">98.1%</td>
                  <td className="py-2 px-4 text-center">Memory Bank</td>
                </tr>
                <tr className="border-b border-slate-700/50">
                  <td className="py-2 px-4">EfficientAD</td>
                  <td className="py-2 px-4 text-center">99.1%</td>
                  <td className="py-2 px-4 text-center">96.8%</td>
                  <td className="py-2 px-4 text-center">Student-Teacher</td>
                </tr>
                <tr className="border-b border-slate-700/50">
                  <td className="py-2 px-4">PaDiM</td>
                  <td className="py-2 px-4 text-center">95.3%</td>
                  <td className="py-2 px-4 text-center">97.5%</td>
                  <td className="py-2 px-4 text-center">Gaussian</td>
                </tr>
                <tr className="border-b border-slate-700/50">
                  <td className="py-2 px-4">DRAEM</td>
                  <td className="py-2 px-4 text-center">98.0%</td>
                  <td className="py-2 px-4 text-center">97.3%</td>
                  <td className="py-2 px-4 text-center">Reconstruction</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
