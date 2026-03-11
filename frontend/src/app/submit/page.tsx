'use client';

/**
 * RETINA - Image Submission Page
 * ==============================
 *
 * Upload an image for anomaly detection via the cascade pipeline.
 * Supports:
 *  - Drag-and-drop or file picker image upload
 *  - Standard PatchCore inference (/inference/predict)
 *  - Cascade inference with VLM fallback (/api/predict/cascade)
 *  - Real-time result display with heatmap
 */

import { useState, useRef, useCallback } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type InferenceMode = 'standard' | 'cascade';

interface InferenceResult {
  anomaly_score: number;
  is_anomaly: boolean;
  confidence: number;
  processing_time_ms?: number;
  model_used: string;
  routing_case?: string;
  requires_expert_labeling?: boolean;
  heatmap_base64?: string;
  vlm_analysis?: string;
}

export default function SubmitPage() {
  // File state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Form state
  const [mode, setMode] = useState<InferenceMode>('cascade');
  const [category, setCategory] = useState('bottle');
  const [isDragOver, setIsDragOver] = useState(false);

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Recent results
  const [recentResults, setRecentResults] = useState<Array<InferenceResult & { filename: string }>>([]);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (PNG, JPG, etc.)');
      return;
    }
    setSelectedFile(file);
    setError(null);
    setResult(null);

    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile) {
      setError('Please select an image to analyze');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();

      if (mode === 'cascade') {
        formData.append('file', selectedFile);
        formData.append('normal_threshold', '0.2');
        formData.append('anomaly_threshold', '0.8');
        formData.append('use_vlm_fallback', 'true');

        const response = await fetch(`${API_URL}/api/predict/cascade`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          const detail = errData.detail;
          throw new Error(
            typeof detail === 'string'
              ? detail
              : Array.isArray(detail)
              ? detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
              : `Request failed: ${response.statusText}`
          );
        }

        const data = await response.json();
        setResult(data);
        setRecentResults(prev => [{ ...data, filename: selectedFile.name }, ...prev].slice(0, 5));
      } else {
        formData.append('image', selectedFile);
        formData.append('category', category);

        const response = await fetch(`${API_URL}/inference/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          const detail = errData.detail;
          throw new Error(
            typeof detail === 'string'
              ? detail
              : Array.isArray(detail)
              ? detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
              : `Request failed: ${response.statusText}`
          );
        }

        const data = await response.json();
        setResult(data);
        setRecentResults(prev => [{ ...data, filename: selectedFile.name }, ...prev].slice(0, 5));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Submit Image
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Upload an image for anomaly detection analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Submission Form */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
            New Submission
          </h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Inference Mode Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Inference Mode
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setMode('cascade')}
                  className={`p-3 rounded-lg border-2 text-left transition-all ${
                    mode === 'cascade'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white text-sm">
                    🔄 Cascade (4-Stage)
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    BGAD → VLM → Expert Queue
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setMode('standard')}
                  className={`p-3 rounded-lg border-2 text-left transition-all ${
                    mode === 'standard'
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                  }`}
                >
                  <div className="font-medium text-gray-900 dark:text-white text-sm">
                    🔍 Standard
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    PatchCore only
                  </div>
                </button>
              </div>
            </div>

            {/* Category (Standard mode only) */}
            {mode === 'standard' && (
              <div>
                <label
                  htmlFor="category"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                >
                  Product Category
                </label>
                <select
                  id="category"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  disabled={isSubmitting}
                >
                  <option value="bottle">Bottle</option>
                  <option value="cable">Cable</option>
                  <option value="capsule">Capsule</option>
                  <option value="carpet">Carpet</option>
                  <option value="grid">Grid</option>
                  <option value="hazelnut">Hazelnut</option>
                  <option value="leather">Leather</option>
                  <option value="metal_nut">Metal Nut</option>
                  <option value="pill">Pill</option>
                  <option value="screw">Screw</option>
                  <option value="tile">Tile</option>
                  <option value="toothbrush">Toothbrush</option>
                  <option value="transistor">Transistor</option>
                  <option value="wood">Wood</option>
                  <option value="zipper">Zipper</option>
                </select>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  MVTec AD category for PatchCore inference
                </p>
              </div>
            )}

            {/* File Upload / Drop Zone */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Image
              </label>
              {!selectedFile ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                    isDragOver
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                  }`}
                >
                  <div className="text-4xl mb-3">📸</div>
                  <p className="text-gray-700 dark:text-gray-300 font-medium">
                    Drop image here or click to browse
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Supports PNG, JPG, BMP, TIFF
                  </p>
                </div>
              ) : (
                <div className="relative border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
                  <img
                    src={previewUrl!}
                    alt="Preview"
                    className="w-full max-h-64 object-contain bg-gray-100 dark:bg-gray-900"
                  />
                  <div className="p-3 flex items-center justify-between bg-gray-50 dark:bg-gray-800">
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate max-w-[200px]">
                        {selectedFile.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={clearSelection}
                      className="text-sm text-red-600 hover:text-red-700 font-medium"
                    >
                      ✕ Remove
                    </button>
                  </div>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileSelect(file);
                }}
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-red-700 dark:text-red-300 text-sm">⚠️ {error}</p>
              </div>
            )}

            {/* Result Display */}
            {result && (
              <div className={`p-4 rounded-lg border ${
                result.is_anomaly
                  ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                  : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              }`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={`text-lg font-bold ${
                    result.is_anomaly
                      ? 'text-red-700 dark:text-red-300'
                      : 'text-green-700 dark:text-green-300'
                  }`}>
                    {result.is_anomaly ? '🚨 Anomaly Detected' : '✅ Normal'}
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {result.model_used}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Score</p>
                    <p className="font-mono font-bold text-gray-900 dark:text-white">
                      {result.anomaly_score.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Confidence</p>
                    <p className="font-mono font-bold text-gray-900 dark:text-white">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  {result.routing_case && (
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Route</p>
                      <p className="font-mono font-bold text-gray-900 dark:text-white">
                        {result.routing_case}
                      </p>
                    </div>
                  )}
                  {result.processing_time_ms !== undefined && (
                    <div>
                      <p className="text-gray-500 dark:text-gray-400">Time</p>
                      <p className="font-mono font-bold text-gray-900 dark:text-white">
                        {result.processing_time_ms}ms
                      </p>
                    </div>
                  )}
                </div>

                {result.requires_expert_labeling && (
                  <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-sm">
                    <span className="font-medium text-yellow-700 dark:text-yellow-300">
                      🏷️ Queued for expert labeling
                    </span>
                    <span className="text-yellow-600 dark:text-yellow-400 ml-1">
                      — go to the <a href="/label" className="underline">Label page</a> to annotate
                    </span>
                  </div>
                )}

                {result.vlm_analysis && (
                  <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded text-sm">
                    <p className="font-medium text-blue-700 dark:text-blue-300 mb-1">🤖 VLM Analysis:</p>
                    <p className="text-blue-600 dark:text-blue-400">{result.vlm_analysis}</p>
                  </div>
                )}

                {result.heatmap_base64 && (
                  <div className="mt-3">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Anomaly Heatmap:
                    </p>
                    <img
                      src={`data:image/png;base64,${result.heatmap_base64}`}
                      alt="Anomaly heatmap"
                      className="w-full max-h-48 object-contain rounded border border-gray-200 dark:border-gray-700"
                    />
                  </div>
                )}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting || !selectedFile}
              className="w-full py-3 px-4 rounded-lg font-medium text-white transition-all
                bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <span className="animate-spin inline-block mr-2">⏳</span>
                  Analyzing...
                </>
              ) : (
                <>
                  <span className="mr-2">🔍</span>
                  {mode === 'cascade' ? 'Run Cascade Analysis' : 'Run PatchCore Analysis'}
                </>
              )}
            </button>
          </form>
        </div>

        {/* Recent Results Sidebar */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Recent Results
          </h2>

          {recentResults.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              No results yet. Upload an image to analyze it.
            </p>
          ) : (
            <ul className="space-y-3">
              {recentResults.map((r, idx) => (
                <li
                  key={idx}
                  className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-900 dark:text-white truncate max-w-[150px]">
                      {r.filename}
                    </span>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                      r.is_anomaly
                        ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                        : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                    }`}>
                      {r.is_anomaly ? 'Anomaly' : 'Normal'}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Score: {r.anomaly_score.toFixed(4)} • {r.model_used}
                  </p>
                </li>
              ))}
            </ul>
          )}

          {/* Pipeline Info */}
          <div className="mt-6 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>Cascade mode:</strong> Images are scored by BGAD. Uncertain
              results (0.2–0.8) are routed to VLM, then to the expert annotation queue.
            </p>
          </div>
        </div>
      </div>

      {/* Information Panel */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="font-medium text-gray-900 dark:text-white mb-3">
          ℹ️ About Image Submission
        </h3>
        <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
          <p>
            Upload a product image for real-time anomaly detection. The system operates in two modes:
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>
              <strong>Cascade (4-Stage)</strong>: BGAD edge model screens images → uncertain cases
              route to GPT-4V → flagged images enter the annotation queue → nightly retraining
            </li>
            <li>
              <strong>Standard</strong>: Direct PatchCore inference with anomaly heatmap
            </li>
          </ul>
          <p className="mt-2">
            Results include anomaly scores, confidence levels, and heatmaps.
            Uncertain cascade results will appear on the{' '}
            <a href="/label" className="text-blue-600 dark:text-blue-400 hover:underline">
              Label page
            </a>{' '}
            for expert annotation.
          </p>
        </div>
      </div>
    </div>
  );
}
