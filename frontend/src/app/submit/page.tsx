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
  Layers,
} from 'lucide-react';
import { submitAndWait, type InferenceResult } from '@/lib/api';
import Card from '@/components/Card';
import GlassCard from '@/components/GlassCard';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import Badge from '@/components/Badge';
import SectionHeader from '@/components/SectionHeader';
import ErrorBanner from '@/components/ErrorBanner';

interface RecentEntry {
  filename: string;
  anomaly_score: number;
  is_anomaly: boolean;
  product_class: string | null;
}

export default function SubmitPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
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
    setJobId(null);
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
    setJobId(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);
    setJobId(null);

    try {
      const data = await submitAndWait(selectedFile, {
        pollMs: 1000,
        timeoutMs: 60_000,
      });
      setResult(data);
      setJobId(data.job_id);
      setRecent(prev => [{
        filename: selectedFile.name,
        anomaly_score: data.anomaly_score ?? 0,
        is_anomaly: !!data.is_anomaly,
        product_class: data.product_class ?? null,
      }, ...prev].slice(0, 5));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Submission failed. Is the API running on :3001?'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const stage2 = result?.stage2_verdict;
  const productLine = result?.product_class
    ? `${result.product_class}${result.product_confidence != null ? ` · ${(result.product_confidence * 100).toFixed(0)}% conf` : ''}`
    : null;

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-text-primary">Submit for Analysis</h1>
        <p className="text-sm text-text-tertiary mt-1">
          Upload an image — the worker identifies the product, runs PatchCore,
          and describes any defect via GPT-4o.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-5 gap-8">

          {/* Left: form */}
          <div className="col-span-3 space-y-4">
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
                  <p className="text-sm text-text-secondary">Drop image here</p>
                  <p className="text-xs text-text-disabled mt-1">or click to browse</p>
                  <p className="text-xs text-text-disabled mt-3">PNG · JPG · JPEG</p>
                </div>
              ) : (
                <div className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl!}
                    alt="Preview"
                    className="w-full max-h-72 object-contain rounded-t-2xl"
                  />
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

            {isSubmitting && jobId && (
              <Card padding="sm" className="border-l-2 border-l-kul-accent">
                <p className="text-xs font-mono text-text-secondary">
                  job_id={jobId} · polling /api/result/{jobId} …
                </p>
              </Card>
            )}

            <button
              type="submit"
              disabled={isSubmitting || !selectedFile}
              className="w-full h-11 bg-kul-blue hover:bg-kul-light text-white font-medium text-sm rounded-xl transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  Run Analysis
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
                  alert={!!result.is_anomaly}
                  className={result.is_anomaly ? '' : 'border-state-pass/20'}
                >
                  <div className="flex items-center justify-between mb-4">
                    <p className={[
                      'text-xl font-semibold',
                      result.is_anomaly ? 'text-state-alert' : 'text-state-pass',
                    ].join(' ')}>
                      {result.is_anomaly ? 'Anomaly Detected' : 'Normal'}
                    </p>
                    {result.confidence != null && (
                      <Badge color={result.is_anomaly ? 'alert' : 'pass'}>
                        {(result.confidence * 100).toFixed(0)}% conf.
                      </Badge>
                    )}
                  </div>

                  {productLine && (
                    <p className="text-xs text-text-tertiary mb-3">
                      Product: <span className="text-text-primary font-medium">{productLine}</span>
                    </p>
                  )}

                  <p className="text-xs text-text-tertiary mb-2">Anomaly Score</p>
                  <AnomalyScoreBar score={result.anomaly_score ?? 0} size="lg" showValue />

                  <div className="grid grid-cols-2 gap-3 mt-4">
                    {([
                      ['Model', result.model_used ?? '—'],
                      ['Route', result.routing_reason?.replace(/_/g, ' ') ?? '—'],
                      ['Time', result.processing_time_ms ? `${result.processing_time_ms}ms` : '—'],
                      ['Heatmap', result.stage1_output?.heatmap_available ? 'available' : '—'],
                    ] as [string, string][]).map(([label, val]) => (
                      <div key={label}>
                        <p className="text-xs text-text-tertiary">{label}</p>
                        <p className="text-xs font-mono text-text-primary">{val}</p>
                      </div>
                    ))}
                  </div>
                </Card>

                {/* GPT-4o description */}
                {result.natural_description && (
                  <Card padding="md" className="border-l-2 border-l-purple-500">
                    <div className="flex items-center gap-1.5 mb-3">
                      <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                      <span className="text-xs font-medium text-purple-400">
                        GPT-4o Vision Analysis
                      </span>
                      {result.vlm_model_used && (
                        <Badge color="default" className="text-[10px] ml-auto">
                          {result.vlm_model_used}
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-text-primary leading-relaxed">
                      {result.natural_description}
                    </p>
                    {(result.defect_type || result.defect_location || result.defect_severity) && (
                      <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-surface-border">
                        {([
                          ['Type', result.defect_type],
                          ['Location', result.defect_location],
                          ['Severity', result.defect_severity],
                        ] as [string, string | null | undefined][]).map(([label, val]) => (
                          <div key={label}>
                            <p className="text-[10px] text-text-tertiary uppercase">{label}</p>
                            <p className="text-xs text-text-primary">{val || '—'}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </Card>
                )}

                {/* Stage 2 verdict card */}
                {stage2 && (
                  <Card padding="md" className="border-l-2 border-l-kul-accent">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Layers className="w-3.5 h-3.5 text-kul-accent" />
                      <span className="text-xs font-medium text-kul-accent">
                        Stage 2 — Supervised Refiner
                      </span>
                      {result.stage2_confidence != null && (
                        <Badge color="default" className="text-[10px] ml-auto">
                          {(result.stage2_confidence * 100).toFixed(0)}% conf
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-text-primary">
                      Verdict: <span className="font-medium">{stage2.replace(/_/g, ' ')}</span>
                    </p>
                    {result.stage2_defect_class && (
                      <p className="text-xs text-text-tertiary mt-1">
                        Defect class: <span className="text-text-primary">{result.stage2_defect_class}</span>
                      </p>
                    )}
                  </Card>
                )}

                {/* Footer: cost + job id */}
                <Card padding="sm">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-text-tertiary truncate">job_id={result.job_id}</span>
                    {result.vlm_api_cost_estimate_usd != null && (
                      <span className="text-text-secondary">
                        VLM ${result.vlm_api_cost_estimate_usd.toFixed(5)}
                      </span>
                    )}
                  </div>
                </Card>

                {/* Expert queue CTA when ambiguous */}
                {(result.routing_reason === 'stage2_uncertain_kept' ||
                  result.routing_reason === 'unknown_product_zero_shot') && (
                  <Card padding="sm" className="border-l-2 border-l-kul-accent">
                    <div className="flex items-start gap-3">
                      <Tag className="w-4 h-4 text-kul-accent flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-sm font-medium text-text-primary">
                          Queued for Expert Review
                        </p>
                        <p className="text-xs text-text-tertiary mt-0.5">
                          This image benefits from a human label.
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
