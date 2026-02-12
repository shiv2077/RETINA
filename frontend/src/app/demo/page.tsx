'use client';

/**
 * RETINA - Live Inference Demo
 * ============================
 * 
 * Interactive demo for testing anomaly detection on uploaded images.
 * Supports drag & drop, file upload, and URL input.
 */

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface InferenceResult {
  anomaly_score: number;
  is_anomaly: boolean;
  confidence: number;
  heatmap_base64?: string;
  processing_time_ms: number;
  model_used: string;
  top_defect_type?: string;
}

export default function DemoPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('bottle');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const categories = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
  ];

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }
    
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const runInference = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('category', selectedCategory);

      const res = await fetch(`${API_URL}/inference/predict`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setResult(data);
      } else {
        const errorData = await res.json().catch(() => ({}));
        setError(errorData.detail || 'Inference failed');
      }
    } catch (e) {
      setError('Failed to connect to backend. Make sure the server is running.');
    } finally {
      setLoading(false);
    }
  };

  const loadSampleImage = async (category: string, type: 'good' | 'defect') => {
    try {
      // For demo purposes, we'll create a placeholder
      setSelectedCategory(category);
      setError(`Sample loading: Use your own images or connect to MVTec dataset`);
    } catch (e) {
      setError('Failed to load sample image');
    }
  };

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
                <p className="text-xs text-slate-400">Live Demo</p>
              </div>
            </Link>
          </div>

          <nav className="flex items-center space-x-6">
            <Link href="/" className="text-slate-400 hover:text-white transition">Dashboard</Link>
            <Link href="/label" className="text-slate-400 hover:text-white transition">Label</Link>
            <Link href="/results" className="text-slate-400 hover:text-white transition">Results</Link>
            <Link href="/demo" className="text-blue-400 font-medium">Demo</Link>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold">Try Anomaly Detection</h1>
          <p className="text-slate-400 mt-2">Upload an image to test the multi-stage detection pipeline</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            {/* Category Selection */}
            <div className="p-4 rounded-xl border border-slate-700 bg-slate-800/50">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Product Category</h3>
              <div className="flex flex-wrap gap-2">
                {categories.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition ${
                      selectedCategory === cat
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Upload Area */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`p-8 rounded-xl border-2 border-dashed transition-all cursor-pointer ${
                isDragging
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-slate-600 bg-slate-800/50 hover:border-slate-500'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                className="hidden"
              />
              
              <div className="text-center">
                {previewUrl ? (
                  <div className="space-y-4">
                    <img 
                      src={previewUrl} 
                      alt="Preview" 
                      className="max-h-64 mx-auto rounded-lg"
                    />
                    <p className="text-slate-400 text-sm">{selectedFile?.name}</p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedFile(null);
                        setPreviewUrl(null);
                        setResult(null);
                      }}
                      className="text-red-400 hover:text-red-300 text-sm"
                    >
                      Remove image
                    </button>
                  </div>
                ) : (
                  <>
                    <svg className="w-16 h-16 mx-auto text-slate-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="text-slate-300 font-medium mb-2">Drop an image here</p>
                    <p className="text-slate-500 text-sm">or click to browse</p>
                  </>
                )}
              </div>
            </div>

            {/* Run Inference Button */}
            <button
              onClick={runInference}
              disabled={!selectedFile || loading}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg transition flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <span>Run Detection</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400">
                {error}
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Main Result */}
                <div className={`p-6 rounded-xl border ${
                  result.is_anomaly 
                    ? 'border-red-500/50 bg-red-500/10' 
                    : 'border-green-500/50 bg-green-500/10'
                }`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <span className="text-4xl">{result.is_anomaly ? '⚠️' : '✅'}</span>
                      <div>
                        <h3 className="text-xl font-semibold">
                          {result.is_anomaly ? 'Anomaly Detected' : 'Normal Sample'}
                        </h3>
                        <p className="text-slate-400 text-sm">
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-3xl font-bold">
                        {(result.anomaly_score * 100).toFixed(1)}%
                      </p>
                      <p className="text-slate-400 text-sm">Anomaly Score</p>
                    </div>
                  </div>

                  {/* Score Bar */}
                  <div className="relative h-3 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`absolute inset-y-0 left-0 rounded-full transition-all ${
                        result.anomaly_score < 0.3 ? 'bg-green-500' :
                        result.anomaly_score < 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${result.anomaly_score * 100}%` }}
                    />
                    <div className="absolute inset-0 flex justify-between items-center px-2 text-xs text-white/70">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>

                {/* Heatmap Toggle */}
                {result.heatmap_base64 && (
                  <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold">Anomaly Heatmap</h3>
                      <button
                        onClick={() => setShowHeatmap(!showHeatmap)}
                        className={`px-3 py-1 rounded-lg text-sm ${
                          showHeatmap ? 'bg-purple-500' : 'bg-slate-700'
                        }`}
                      >
                        {showHeatmap ? 'Hide' : 'Show'}
                      </button>
                    </div>
                    {showHeatmap && (
                      <div className="relative">
                        <img
                          src={`data:image/png;base64,${result.heatmap_base64}`}
                          alt="Anomaly Heatmap"
                          className="w-full rounded-lg"
                        />
                        <p className="text-xs text-slate-400 mt-2 text-center">
                          Red regions indicate high anomaly probability
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Details */}
                <div className="p-6 rounded-xl border border-slate-700 bg-slate-800/50">
                  <h3 className="font-semibold mb-4">Detection Details</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-slate-400">Model Used</span>
                      <p className="font-medium">{result.model_used}</p>
                    </div>
                    <div>
                      <span className="text-slate-400">Processing Time</span>
                      <p className="font-medium">{result.processing_time_ms}ms</p>
                    </div>
                    {result.top_defect_type && (
                      <div className="col-span-2">
                        <span className="text-slate-400">Likely Defect Type</span>
                        <p className="font-medium capitalize">{result.top_defect_type}</p>
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="p-12 rounded-xl border border-slate-700 bg-slate-800/50 text-center">
                <svg className="w-20 h-20 mx-auto text-slate-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready to Analyze</h3>
                <p className="text-slate-500">Upload an image and click "Run Detection" to see results</p>
              </div>
            )}
          </div>
        </div>

        {/* Pipeline Info */}
        <div className="mt-12 p-6 rounded-xl border border-slate-700 bg-slate-800/30">
          <h3 className="text-lg font-semibold mb-4">🔬 How It Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-blue-400 font-bold">1</span>
              </div>
              <div>
                <h4 className="font-medium">Feature Extraction</h4>
                <p className="text-sm text-slate-400">
                  WideResNet-50 extracts deep features from the input image
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-purple-400 font-bold">2</span>
              </div>
              <div>
                <h4 className="font-medium">PatchCore Analysis</h4>
                <p className="text-sm text-slate-400">
                  Memory bank comparison using k-NN to detect anomalous patches
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-green-400 font-bold">3</span>
              </div>
              <div>
                <h4 className="font-medium">BGAD Refinement</h4>
                <p className="text-sm text-slate-400">
                  Push-pull learning refines detection using labeled data
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
