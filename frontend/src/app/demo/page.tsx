'use client';

import { useState, useRef, useCallback } from 'react';
import { Upload, Loader2, Search, Brain } from 'lucide-react';
import { predictCascade, type CascadeResponse } from '@/lib/api';
import Card from '@/components/Card';
import AnomalyScoreBar from '@/components/AnomalyScoreBar';
import HeatmapOverlay from '@/components/HeatmapOverlay';
import Badge from '@/components/Badge';
import ErrorBanner from '@/components/ErrorBanner';

const CATEGORIES = [
  'bottle','cable','capsule','carpet','grid',
  'hazelnut','leather','metal_nut','pill','screw',
  'tile','toothbrush','transistor','wood','zipper',
];

type RoutingLabel = {
  label: string;
  detail: string;
  color: 'pass' | 'warn' | 'alert';
};

function routingLabel(routingCase: string): RoutingLabel {
  if (routingCase.includes('normal'))   return { label: 'Stage 1 — Normal', detail: 'PatchCore classified as normal with high confidence', color: 'pass' };
  if (routingCase.includes('anomaly'))  return { label: 'Stage 1 — Anomaly', detail: 'PatchCore flagged as anomalous with high confidence', color: 'alert' };
  return { label: 'Stage 1 → VLM Fallback', detail: 'Uncertain — routed to GPT-4o for zero-shot analysis', color: 'warn' };
}

export default function DemoPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl]     = useState<string | null>(null);
  const [isDragging, setIsDragging]     = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('bottle');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<CascadeResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) { setError('Please select an image file'); return; }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runInference = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictCascade(selectedFile, {
        normal_threshold: 0.2,
        anomaly_threshold: 0.8,
        use_vlm_fallback: true,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const routing = result ? routingLabel(result.routing_case) : null;

  return (
    <div>
      {/* Page header */}
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-semibold text-text-primary">Try Anomaly Detection</h1>
        <p className="text-sm text-text-tertiary mt-1">
          Upload an image to test the multi-stage detection pipeline
        </p>
      </div>

      {/* Category chips */}
      <div className="flex flex-wrap gap-2 justify-center mb-8">
        {CATEGORIES.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={[
              'rounded-full px-3 py-1 text-xs border transition-colors duration-150',
              selectedCategory === cat
                ? 'border-kul-accent bg-kul-blue/10 text-kul-accent'
                : 'border-surface-border text-text-tertiary hover:border-kul-accent/40 hover:text-kul-accent',
            ].join(' ')}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

        {/* Upload side */}
        <div className="space-y-4">
          {/* Drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onClick={() => fileInputRef.current?.click()}
            className={[
              'rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer',
              isDragging
                ? 'border-kul-accent bg-kul-blue/5'
                : 'border-surface-border hover:border-kul-accent/40 hover:bg-kul-blue/5',
            ].join(' ')}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFileSelect(f); }}
            />
            {previewUrl && result ? (
              <HeatmapOverlay
                imageSrc={previewUrl}
                heatmapBase64={null}
                className="max-h-72"
              />
            ) : previewUrl ? (
              <div className="p-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl} alt="Preview" className="max-h-64 mx-auto rounded-xl" />
                <p className="text-xs text-text-tertiary text-center mt-2">{selectedFile?.name}</p>
                <button
                  onClick={e => { e.stopPropagation(); setSelectedFile(null); setPreviewUrl(null); setResult(null); }}
                  className="block mx-auto mt-2 text-xs text-state-alert hover:underline"
                >
                  Remove image
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16">
                <Upload className="w-10 h-10 text-text-disabled mb-3" />
                <p className="text-sm text-text-secondary">Drop an image here</p>
                <p className="text-xs text-text-disabled mt-1">or click to browse</p>
              </div>
            )}
          </div>

          {error && <ErrorBanner message={error} onRetry={() => setError(null)} />}

          <button
            onClick={runInference}
            disabled={!selectedFile || loading}
            className="w-full h-11 bg-kul-blue hover:bg-kul-light text-white font-medium text-sm rounded-xl transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Run Detection
              </>
            )}
          </button>
        </div>

        {/* Results side */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Verdict */}
              <Card
                padding="md"
                alert={result.is_anomaly}
                className={result.is_anomaly ? '' : 'border-state-pass/20'}
              >
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className={[
                      'text-xl font-semibold',
                      result.is_anomaly ? 'text-state-alert' : 'text-state-pass',
                    ].join(' ')}>
                      {result.is_anomaly ? 'Anomaly Detected' : 'Normal Sample'}
                    </p>
                    <p className="text-xs text-text-tertiary mt-0.5">
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                  <p className="font-mono text-2xl font-bold text-text-primary">
                    {(result.anomaly_score * 100).toFixed(1)}%
                  </p>
                </div>

                <p className="text-xs text-text-tertiary mb-2">Anomaly Score</p>
                <AnomalyScoreBar score={result.anomaly_score} size="lg" showValue />

                <div className="grid grid-cols-2 gap-3 mt-4 text-xs">
                  <div>
                    <p className="text-text-tertiary">Model</p>
                    <p className="font-mono text-text-primary">{result.model_used}</p>
                  </div>
                  <div>
                    <p className="text-text-tertiary">Processing</p>
                    <p className="font-mono text-text-primary">
                      {result.processing_time_ms ? `${result.processing_time_ms}ms` : '—'}
                    </p>
                  </div>
                </div>
              </Card>

              {/* Stage routing */}
              {routing && (
                <Card padding="sm" className="border-l-2" style={{ borderLeftColor: routing.color === 'pass' ? '#34D399' : routing.color === 'alert' ? '#F87171' : '#FBBF24' }}>
                  <div className="flex items-start gap-3">
                    <Badge color={routing.color} dot className="mt-0.5 flex-shrink-0">
                      {routing.label}
                    </Badge>
                    <p className="text-xs text-text-tertiary leading-relaxed">{routing.detail}</p>
                  </div>
                </Card>
              )}

              {/* VLM result */}
              {result.vlm_result && (
                <Card padding="sm" className="border-l-2 border-l-purple-500">
                  <p className="text-xs font-medium text-purple-400 mb-1">GPT-4o Vision</p>
                  <p className="text-sm text-text-primary">{result.vlm_result.classification}</p>
                </Card>
              )}
            </>
          ) : (
            <Card padding="md" className="flex flex-col items-center justify-center min-h-64">
              <Brain className="w-10 h-10 text-surface-border mb-4" />
              <p className="text-sm text-text-tertiary">Ready to Analyze</p>
              <p className="text-xs text-text-disabled mt-1">
                Upload an image and click "Run Detection"
              </p>
            </Card>
          )}
        </div>
      </div>

      {/* How it works */}
      <Card padding="md" className="mt-12">
        <h3 className="text-base font-semibold text-text-primary mb-6">How It Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              num: '1',
              title: 'Feature Extraction',
              desc: 'WideResNet-50 extracts deep features from the input image',
              color: 'bg-kul-accent/10 text-kul-accent',
            },
            {
              num: '2',
              title: 'PatchCore Analysis',
              desc: 'Memory bank comparison using k-NN to detect anomalous patches',
              color: 'bg-purple-500/10 text-purple-400',
            },
            {
              num: '3',
              title: 'Stage 2 Refinement',
              desc: 'BGAD or Push-Pull refines detection using labeled examples',
              color: 'bg-state-passSubtle text-state-pass',
            },
          ].map(({ num, title, desc, color }) => (
            <div key={num} className="flex items-start gap-3">
              <div className={['w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-sm font-bold', color].join(' ')}>
                {num}
              </div>
              <div>
                <p className="text-sm font-medium text-text-primary">{title}</p>
                <p className="text-xs text-text-tertiary mt-1 leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
