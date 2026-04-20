'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import {
  Upload,
  X,
  Loader2,
  Search,
  BarChart2,
  Sparkles,
  Tag,
} from 'lucide-react';
import { predictCascade, submitImage, type CascadeResponse } from '@/lib/api';
import Card from '@/components/Card';
import GlassCard from '@/components/GlassCard';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import HeatmapOverlay from '@/components/HeatmapOverlay';
import Badge from '@/components/Badge';
import SectionHeader from '@/components/SectionHeader';
import ErrorBanner from '@/components/ErrorBanner';

type InferenceMode = 'cascade' | 'standard';

const MVTec_CATEGORIES = [
  'bottle','cable','capsule','carpet','grid',
  'hazelnut','leather','metal_nut','pill','screw',
  'tile','toothbrush','transistor','wood','zipper',
];

interface RecentEntry {
  filename: string;
  anomaly_score: number;
  is_anomaly: boolean;
  model_used: string;
}

export default function SubmitPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [mode, setMode] = useState<InferenceMode>('cascade');
  const [category, setCategory] = useState('bottle');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<CascadeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentEntry[]>([]);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (PNG, JPG, etc.)');
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
    setResult(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      if (mode === 'cascade') {
        const data = await predictCascade(selectedFile, {
          normal_threshold: 0.2,
          anomaly_threshold: 0.8,
          use_vlm_fallback: true,
        });
        setResult(data);
        setRecent(prev => [{
          filename: selectedFile.name,
          anomaly_score: data.anomaly_score,
          is_anomaly: data.is_anomaly,
          model_used: data.model_used,
        }, ...prev].slice(0, 5));
      } else {
        await submitImage({ image_id: selectedFile.name });
        setError('Standard mode: image queued for processing. Check Results for output.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed. Is the backend running?');
    } finally {
      setIsSubmitting(false);
    }
  };

  const heatmapB64: string | null = null;

  return (
    <div>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-text-primary">Submit for Analysis</h1>
        <p className="text-sm text-text-tertiary mt-1">
          Upload images for anomaly detection
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-5 gap-8">

          {/* Left: form */}
          <div className="col-span-3 space-y-4">

            {/* Mode selector */}
            <Card padding="md">
              <p className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3">
                Detection Mode
              </p>
              <div className="grid grid-cols-2 gap-3">
                {([
                  { id: 'cascade' as const, label: 'Cascade', desc: 'Stage 1 → Stage 2 automatically', recommended: true },
                  { id: 'standard' as const, label: 'Standard', desc: 'Submit to queue only', recommended: false },
                ] as const).map(({ id, label, desc, recommended }) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setMode(id)}
                    className={[
                      'p-3 rounded-xl border-2 text-left transition-all duration-150',
                      mode === id
                        ? 'border-kul-accent bg-kul-blue/10'
                        : 'border-surface-border hover:border-surface-borderhover',
                    ].join(' ')}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium text-text-primary">{label}</span>
                      {recommended && (
                        <Badge color="kul" className="text-[10px] px-1.5 py-0">
                          Recommended
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-text-tertiary">{desc}</p>
                  </button>
                ))}
              </div>

              {mode === 'standard' && (
                <div className="mt-3">
                  <label className="block text-xs text-text-tertiary mb-1.5">
                    Product Category
                  </label>
                  <select
                    value={category}
                    onChange={e => setCategory(e.target.value)}
                    className="w-full bg-surface-overlay border border-surface-border rounded-lg px-3 py-2 text-sm text-text-primary focus:border-kul-accent transition-colors"
                  >
                    {MVTec_CATEGORIES.map(c => (
                      <option key={c} value={c} className="bg-surface-raised">{c}</option>
                    ))}
                  </select>
                </div>
              )}
            </Card>

            {/* Drop zone */}
            <GlassCard padding="none">
              {!selectedFile ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
                  onDragLeave={() => setIsDragOver(false)}
                  onClick={() => fileInputRef.current?.click()}
                  className={[
                    'flex flex-col items-center justify-center p-12 cursor-pointer',
                    'border-2 border-dashed rounded-2xl transition-all duration-200',
                    isDragOver
                      ? 'border-kul-accent bg-kul-blue/5'
                      : 'border-surface-border hover:border-kul-accent/40 hover:bg-kul-blue/5',
                  ].join(' ')}
                >
                  <Upload className="w-10 h-10 text-text-disabled mb-3" />
                  <p className="text-sm text-text-secondary">Drop images here</p>
                  <p className="text-xs text-text-disabled mt-1">or click to browse</p>
                  <p className="text-xs text-text-disabled mt-3">PNG · JPG · JPEG</p>
                </div>
              ) : (
                <div className="relative">
                  {result?.bgad_score !== undefined ? (
                    <HeatmapOverlay
                      imageSrc={previewUrl!}
                      heatmapBase64={heatmapB64}
                      className="max-h-72"
                    />
                  ) : (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={previewUrl!}
                      alt="Preview"
                      className="w-full max-h-72 object-contain rounded-t-2xl"
                    />
                  )}
                  <button
                    type="button"
                    onClick={clearSelection}
                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-surface-overlay border border-surface-border flex items-center justify-center hover:bg-surface-border transition-colors"
                  >
                    <X className="w-3.5 h-3.5 text-text-tertiary" />
                  </button>
                  <div className="px-4 py-3 border-t border-surface-border">
                    <p className="text-xs text-text-secondary truncate">{selectedFile.name}</p>
                    <p className="text-xs text-text-disabled">
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
              )}
            </GlassCard>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFileSelect(f); }}
            />

            {error && <ErrorBanner message={error} onRetry={() => setError(null)} />}

            {/* Submit button */}
            <button
              type="submit"
              disabled={isSubmitting || !selectedFile}
              className="w-full h-11 bg-kul-blue hover:bg-kul-light text-white font-medium text-sm rounded-xl transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  {mode === 'cascade' ? 'Run Cascade Analysis' : 'Queue for Processing'}
                </>
              )}
            </button>
          </div>

          {/* Right: results */}
          <div className="col-span-2 space-y-4">
            {!result ? (
              <Card padding="md" className="flex flex-col items-center justify-center min-h-64">
                <BarChart2 className="w-10 h-10 text-surface-border mb-4" />
                <p className="text-sm text-text-tertiary">Results appear here</p>
                <p className="text-xs text-text-disabled mt-1">Submit an image to begin</p>
              </Card>
            ) : (
              <>
                {/* Verdict card */}
                <Card
                  padding="md"
                  alert={result.is_anomaly}
                  className={result.is_anomaly ? '' : 'border-state-pass/20'}
                >
                  <div className="flex items-center justify-between mb-4">
                    <p className={[
                      'text-xl font-semibold',
                      result.is_anomaly ? 'text-state-alert' : 'text-state-pass',
                    ].join(' ')}>
                      {result.is_anomaly ? 'Anomaly Detected' : 'Normal'}
                    </p>
                    <Badge color={result.is_anomaly ? 'alert' : 'pass'}>
                      {(result.confidence * 100).toFixed(0)}% conf.
                    </Badge>
                  </div>

                  <p className="text-xs text-text-tertiary mb-2">Anomaly Score</p>
                  <AnomalyScoreBar score={result.anomaly_score} size="lg" showValue />

                  <div className="grid grid-cols-2 gap-3 mt-4">
                    {([
                      ['Model', result.model_used],
                      ['Route', result.routing_case?.replace(/_/g, ' ') ?? '—'],
                      ['Time', result.processing_time_ms ? `${result.processing_time_ms}ms` : '—'],
                      ['Score', result.anomaly_score.toFixed(4)],
                    ] as [string, string][]).map(([label, val]) => (
                      <div key={label}>
                        <p className="text-xs text-text-tertiary">{label}</p>
                        <p className="text-xs font-mono text-text-primary">{val}</p>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* GPT-4V analysis */}
                {result.vlm_result && (
                  <Card padding="md" className="border-l-2 border-l-purple-500">
                    <div className="flex items-center gap-1.5 mb-3">
                      <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                      <span className="text-xs font-medium text-purple-400">
                        GPT-4o Vision Analysis
                      </span>
                    </div>
                    <p className="text-sm text-text-primary leading-relaxed">
                      {result.vlm_result.classification}
                    </p>
                    {result.vlm_result.confidence !== undefined && (
                      <p className="text-xs text-text-tertiary mt-2">
                        Confidence: {(result.vlm_result.confidence * 100).toFixed(0)}%
                      </p>
                    )}
                  </Card>
                )}

                {/* Expert queue CTA */}
                {result.requires_expert_labeling && (
                  <Card padding="sm" className="border-l-2 border-l-kul-accent">
                    <div className="flex items-start gap-3">
                      <Tag className="w-4 h-4 text-kul-accent flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium text-text-primary">
                          Queued for Expert Review
                        </p>
                        <p className="text-xs text-text-tertiary mt-0.5">
                          This image requires labeling to improve Stage 2
                        </p>
                        <Link
                          href="/label"
                          className="inline-block mt-2 text-xs text-kul-accent hover:underline"
                        >
                          Open Annotation Studio →
                        </Link>
                      </div>
                    </div>
                  </Card>
                )}
              </>
            )}

            {/* Recent submissions */}
            {recent.length > 0 && (
              <Card padding="sm">
                <SectionHeader
                  title="Recent"
                  action={<Badge color="default">{recent.length}</Badge>}
                  className="mb-3"
                />
                <div className="space-y-2">
                  {recent.map((r, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <p className="text-xs text-text-secondary truncate flex-1 min-w-0">
                        {r.filename}
                      </p>
                      <AnomalyScoreBar score={r.anomaly_score} size="xs" className="w-16" />
                      <Badge color={r.is_anomaly ? 'alert' : 'pass'} className="text-[10px]">
                        {r.is_anomaly ? 'A' : 'N'}
                      </Badge>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
