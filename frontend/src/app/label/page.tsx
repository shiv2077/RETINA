'use client';

/**
 * RETINA - Professional Image Annotation Interface
 * ================================================
 *
 * Roboflow-style labeling interface with:
 * - Canvas-based bounding box drawing
 * - Defect type classification
 * - Keyboard shortcuts for efficiency
 * - Real image preview with anomaly heatmap overlay
 * - Progress tracking
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Defect categories with colors
const DEFECT_CATEGORIES = [
  { name: 'scratch', color: '#ef4444', shortcut: '1' },
  { name: 'dent', color: '#f97316', shortcut: '2' },
  { name: 'contamination', color: '#eab308', shortcut: '3' },
  { name: 'crack', color: '#22c55e', shortcut: '4' },
  { name: 'discoloration', color: '#3b82f6', shortcut: '5' },
  { name: 'missing_part', color: '#8b5cf6', shortcut: '6' },
  { name: 'other', color: '#6b7280', shortcut: '7' },
];

interface BoundingBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  defect_type: string;
  confidence: number;
}

interface Sample {
  image_id: string;
  image_path: string;
  anomaly_score: number;
  uncertainty_score: number;
  heatmap_base64?: string;
}

interface Annotation {
  image_id: string;
  label: 'normal' | 'anomaly' | 'uncertain';
  bounding_boxes: BoundingBox[];
  defect_types: string[];
  notes: string;
  labeled_by: string;
  timestamp: string;
}

export default function LabelPage() {
  // State
  const [samples, setSamples] = useState<Sample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState('bottle');
  const [categories, setCategories] = useState<string[]>([]);

  // Annotation state
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([]);
  const [selectedDefect, setSelectedDefect] = useState<string>(DEFECT_CATEGORIES[0].name);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Partial<BoundingBox> | null>(null);
  const [notes, setNotes] = useState('');
  const [showHeatmap, setShowHeatmap] = useState(true);

  // Tool state
  const [activeTool, setActiveTool] = useState<'select' | 'box' | 'pan'>('box');
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  // Stats
  const [stats, setStats] = useState({ total: 0, normal: 0, anomaly: 0, uncertain: 0 });

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const currentSample = samples[currentIndex];

  // Fetch categories
  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const res = await fetch(`${API_URL}/categories`);
        if (res.ok) {
          const data = await res.json();
          setCategories(data.categories || []);
        }
      } catch (e) {
        console.error('Failed to fetch categories');
      }
    };
    fetchCategories();
  }, []);

  // Fetch samples for labeling
  useEffect(() => {
    const fetchSamples = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_URL}/labeling/queue?category=${selectedCategory}&limit=50`);
        if (res.ok) {
          const data = await res.json();
          setSamples(data.samples || []);
          setCurrentIndex(0);
          setBoundingBoxes([]);
        } else {
          setError('Failed to load samples');
        }
      } catch (e) {
        setError('Cannot connect to backend');
      } finally {
        setLoading(false);
      }
    };

    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_URL}/labeling/stats?category=${selectedCategory}`);
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch (e) {
        console.error('Failed to fetch stats');
      }
    };

    if (selectedCategory) {
      fetchSamples();
      fetchStats();
    }
  }, [selectedCategory]);

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentSample) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Load and draw image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      canvas.width = img.width;
      canvas.height = img.height;
      redrawCanvas();
    };
    
    // Try to load from backend or use placeholder
    img.src = `${API_URL}/images/${currentSample.image_path}`;
    img.onerror = () => {
      // Fallback to placeholder
      canvas.width = 512;
      canvas.height = 512;
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      ctx.font = '20px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Image Preview', canvas.width / 2, canvas.height / 2);
      ctx.font = '14px sans-serif';
      ctx.fillText(currentSample.image_id, canvas.width / 2, canvas.height / 2 + 30);
    };
  }, [currentSample]);

  // Redraw canvas with boxes
  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0);
    } else {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // Draw heatmap overlay if enabled
    if (showHeatmap && currentSample?.heatmap_base64) {
      const heatmapImg = new Image();
      heatmapImg.src = `data:image/png;base64,${currentSample.heatmap_base64}`;
      ctx.globalAlpha = 0.5;
      ctx.drawImage(heatmapImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }

    // Draw bounding boxes
    boundingBoxes.forEach((box) => {
      const category = DEFECT_CATEGORIES.find(c => c.name === box.defect_type);
      const color = category?.color || '#ffffff';
      
      ctx.strokeStyle = color;
      ctx.lineWidth = selectedBoxId === box.id ? 3 : 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Draw label
      ctx.fillStyle = color;
      ctx.fillRect(box.x, box.y - 20, ctx.measureText(box.defect_type).width + 10, 20);
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px sans-serif';
      ctx.fillText(box.defect_type, box.x + 5, box.y - 6);
    });

    // Draw current drawing box
    if (currentBox && drawStart) {
      const category = DEFECT_CATEGORIES.find(c => c.name === selectedDefect);
      ctx.strokeStyle = category?.color || '#ffffff';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(
        currentBox.x || 0,
        currentBox.y || 0,
        currentBox.width || 0,
        currentBox.height || 0
      );
      ctx.setLineDash([]);
    }
  }, [boundingBoxes, currentBox, drawStart, selectedBoxId, showHeatmap, currentSample, selectedDefect]);

  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas]);

  // Mouse handlers for drawing
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTool !== 'box') return;
    const coords = getCanvasCoords(e);
    setIsDrawing(true);
    setDrawStart(coords);
    setCurrentBox({ x: coords.x, y: coords.y, width: 0, height: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !drawStart || activeTool !== 'box') return;
    const coords = getCanvasCoords(e);
    setCurrentBox({
      x: Math.min(drawStart.x, coords.x),
      y: Math.min(drawStart.y, coords.y),
      width: Math.abs(coords.x - drawStart.x),
      height: Math.abs(coords.y - drawStart.y),
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox || !drawStart) return;
    setIsDrawing(false);

    // Only add if box is large enough
    if ((currentBox.width || 0) > 10 && (currentBox.height || 0) > 10) {
      const newBox: BoundingBox = {
        id: `box_${Date.now()}`,
        x: currentBox.x || 0,
        y: currentBox.y || 0,
        width: currentBox.width || 0,
        height: currentBox.height || 0,
        defect_type: selectedDefect,
        confidence: 1.0,
      };
      setBoundingBoxes([...boundingBoxes, newBox]);
    }

    setDrawStart(null);
    setCurrentBox(null);
  };

  // Delete selected box
  const deleteSelectedBox = () => {
    if (selectedBoxId) {
      setBoundingBoxes(boundingBoxes.filter(b => b.id !== selectedBoxId));
      setSelectedBoxId(null);
    }
  };

  // Submit annotation
  const submitAnnotation = async (label: 'normal' | 'anomaly' | 'uncertain') => {
    if (!currentSample) return;

    try {
      const annotation: Annotation = {
        image_id: currentSample.image_id,
        label,
        bounding_boxes: boundingBoxes,
        defect_types: [...new Set(boundingBoxes.map(b => b.defect_type))],
        notes,
        labeled_by: 'expert',
        timestamp: new Date().toISOString(),
      };

      const res = await fetch(`${API_URL}/labeling/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: selectedCategory, annotation }),
      });

      if (res.ok) {
        // Move to next sample
        if (currentIndex < samples.length - 1) {
          setCurrentIndex(currentIndex + 1);
          setBoundingBoxes([]);
          setNotes('');
          setStats(prev => ({
            ...prev,
            total: prev.total + 1,
            [label]: prev[label] + 1,
          }));
        }
      }
    } catch (e) {
      setError('Failed to submit annotation');
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case 'n':
        case 'N':
          submitAnnotation('normal');
          break;
        case 'a':
        case 'A':
          submitAnnotation('anomaly');
          break;
        case 'u':
        case 'U':
          submitAnnotation('uncertain');
          break;
        case 'Delete':
        case 'Backspace':
          deleteSelectedBox();
          break;
        case 'ArrowLeft':
          if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
            setBoundingBoxes([]);
          }
          break;
        case 'ArrowRight':
          if (currentIndex < samples.length - 1) {
            setCurrentIndex(currentIndex + 1);
            setBoundingBoxes([]);
          }
          break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
          const cat = DEFECT_CATEGORIES.find(c => c.shortcut === e.key);
          if (cat) setSelectedDefect(cat.name);
          break;
        case 'h':
        case 'H':
          setShowHeatmap(!showHeatmap);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentIndex, samples.length, boundingBoxes, selectedBoxId, showHeatmap]);

  const progressPercent = samples.length > 0 ? ((currentIndex + 1) / samples.length) * 100 : 0;

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-full mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <span className="font-bold">RETINA</span>
            </Link>
            <span className="text-slate-400">|</span>
            <h1 className="text-lg font-semibold">Annotation Studio</h1>
          </div>

          {/* Category Selector */}
          <div className="flex items-center space-x-4">
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
            >
              {categories.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>

            {/* Progress */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-slate-400">
                {currentIndex + 1} / {samples.length}
              </span>
              <div className="w-32 h-2 bg-slate-700 rounded-full">
                <div 
                  className="h-2 bg-blue-500 rounded-full transition-all"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span>{stats.normal} Normal</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-red-500"></span>
              <span>{stats.anomaly} Anomaly</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
              <span>{stats.uncertain} Uncertain</span>
            </div>
          </div>
        </div>
      </header>

      {loading ? (
        <div className="flex items-center justify-center h-[calc(100vh-60px)]">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : error ? (
        <div className="flex items-center justify-center h-[calc(100vh-60px)]">
          <div className="text-center">
            <p className="text-red-400 mb-4">{error}</p>
            <button 
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-500 rounded-lg hover:bg-blue-600"
            >
              Retry
            </button>
          </div>
        </div>
      ) : samples.length === 0 ? (
        <div className="flex items-center justify-center h-[calc(100vh-60px)]">
          <div className="text-center">
            <div className="text-6xl mb-4">📭</div>
            <p className="text-slate-400 mb-4">No samples available for labeling</p>
            <p className="text-sm text-slate-500">Train PatchCore first to generate uncertain samples</p>
            <Link href="/" className="mt-4 inline-block px-4 py-2 bg-blue-500 rounded-lg hover:bg-blue-600">
              Go to Dashboard
            </Link>
          </div>
        </div>
      ) : (
        <div className="flex h-[calc(100vh-60px)]">
          {/* Left Sidebar - Tools */}
          <div className="w-16 bg-slate-800 border-r border-slate-700 flex flex-col items-center py-4 space-y-2">
            <button
              onClick={() => setActiveTool('select')}
              className={`w-10 h-10 rounded-lg flex items-center justify-center transition ${
                activeTool === 'select' ? 'bg-blue-500' : 'bg-slate-700 hover:bg-slate-600'
              }`}
              title="Select (V)"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
              </svg>
            </button>
            <button
              onClick={() => setActiveTool('box')}
              className={`w-10 h-10 rounded-lg flex items-center justify-center transition ${
                activeTool === 'box' ? 'bg-blue-500' : 'bg-slate-700 hover:bg-slate-600'
              }`}
              title="Bounding Box (B)"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 010 2H6v3a1 1 0 01-2 0V5zM4 13a1 1 0 011 1v3h3a1 1 0 010 2H5a1 1 0 01-1-1v-4a1 1 0 011-1zM16 4h3a1 1 0 011 1v4a1 1 0 01-2 0V6h-2a1 1 0 010-2zM19 13a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 010-2h3v-3a1 1 0 011-1z" />
              </svg>
            </button>

            <div className="flex-1" />

            {/* Zoom controls */}
            <button
              onClick={() => setZoom(Math.min(zoom + 0.25, 3))}
              className="w-10 h-10 rounded-lg bg-slate-700 hover:bg-slate-600 flex items-center justify-center"
              title="Zoom In"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
            </button>
            <button
              onClick={() => setZoom(Math.max(zoom - 0.25, 0.5))}
              className="w-10 h-10 rounded-lg bg-slate-700 hover:bg-slate-600 flex items-center justify-center"
              title="Zoom Out"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
              </svg>
            </button>
            <span className="text-xs text-slate-400">{Math.round(zoom * 100)}%</span>
          </div>

          {/* Main Canvas Area */}
          <div className="flex-1 flex flex-col bg-slate-900">
            {/* Toolbar */}
            <div className="h-12 bg-slate-800 border-b border-slate-700 flex items-center px-4 space-x-4">
              <button
                onClick={() => setShowHeatmap(!showHeatmap)}
                className={`px-3 py-1.5 rounded text-sm flex items-center space-x-2 transition ${
                  showHeatmap ? 'bg-purple-500' : 'bg-slate-700 hover:bg-slate-600'
                }`}
              >
                <span>🔥</span>
                <span>Heatmap (H)</span>
              </button>
              
              <div className="h-6 w-px bg-slate-600" />
              
              <span className="text-sm text-slate-400">
                Anomaly Score: 
                <span className={`ml-2 font-semibold ${
                  (currentSample?.anomaly_score || 0) > 0.5 ? 'text-red-400' : 'text-green-400'
                }`}>
                  {((currentSample?.anomaly_score || 0) * 100).toFixed(1)}%
                </span>
              </span>

              <div className="flex-1" />

              <span className="text-sm text-slate-400">
                Boxes: <span className="font-semibold">{boundingBoxes.length}</span>
              </span>
            </div>

            {/* Canvas Container */}
            <div 
              ref={containerRef}
              className="flex-1 overflow-auto flex items-center justify-center p-4"
              style={{ backgroundColor: '#0f172a' }}
            >
              <canvas
                ref={canvasRef}
                className="border border-slate-600 rounded cursor-crosshair"
                style={{ 
                  transform: `scale(${zoom})`,
                  transformOrigin: 'center center',
                  maxWidth: '100%',
                  maxHeight: '100%'
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
            </div>

            {/* Bottom Action Bar */}
            <div className="h-20 bg-slate-800 border-t border-slate-700 flex items-center justify-center space-x-6 px-4">
              <button
                onClick={() => submitAnnotation('normal')}
                className="px-8 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold flex items-center space-x-2 transition"
              >
                <span>✅</span>
                <span>Normal</span>
                <kbd className="ml-2 px-2 py-0.5 bg-green-800 rounded text-xs">N</kbd>
              </button>
              
              <button
                onClick={() => submitAnnotation('anomaly')}
                className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold flex items-center space-x-2 transition"
              >
                <span>⚠️</span>
                <span>Anomaly</span>
                <kbd className="ml-2 px-2 py-0.5 bg-red-800 rounded text-xs">A</kbd>
              </button>
              
              <button
                onClick={() => submitAnnotation('uncertain')}
                className="px-8 py-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-semibold flex items-center space-x-2 transition"
              >
                <span>❓</span>
                <span>Uncertain</span>
                <kbd className="ml-2 px-2 py-0.5 bg-yellow-800 rounded text-xs">U</kbd>
              </button>

              <div className="h-10 w-px bg-slate-600" />

              {/* Navigation */}
              <button
                onClick={() => { setCurrentIndex(Math.max(0, currentIndex - 1)); setBoundingBoxes([]); }}
                disabled={currentIndex === 0}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg flex items-center space-x-1"
              >
                <span>←</span>
                <span>Prev</span>
              </button>
              <button
                onClick={() => { setCurrentIndex(Math.min(samples.length - 1, currentIndex + 1)); setBoundingBoxes([]); }}
                disabled={currentIndex === samples.length - 1}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg flex items-center space-x-1"
              >
                <span>Next</span>
                <span>→</span>
              </button>
            </div>
          </div>

          {/* Right Sidebar - Defect Categories & Annotations */}
          <div className="w-72 bg-slate-800 border-l border-slate-700 flex flex-col">
            {/* Defect Categories */}
            <div className="p-4 border-b border-slate-700">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Defect Type</h3>
              <div className="space-y-1">
                {DEFECT_CATEGORIES.map((cat) => (
                  <button
                    key={cat.name}
                    onClick={() => setSelectedDefect(cat.name)}
                    className={`w-full px-3 py-2 rounded-lg text-left text-sm flex items-center justify-between transition ${
                      selectedDefect === cat.name
                        ? 'bg-slate-600 ring-2 ring-blue-500'
                        : 'bg-slate-700 hover:bg-slate-600'
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <span 
                        className="w-3 h-3 rounded"
                        style={{ backgroundColor: cat.color }}
                      />
                      <span className="capitalize">{cat.name.replace('_', ' ')}</span>
                    </div>
                    <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-xs">{cat.shortcut}</kbd>
                  </button>
                ))}
              </div>
            </div>

            {/* Annotations List */}
            <div className="flex-1 p-4 overflow-auto">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Annotations ({boundingBoxes.length})</h3>
              {boundingBoxes.length === 0 ? (
                <p className="text-sm text-slate-500">
                  Draw bounding boxes around defects
                </p>
              ) : (
                <div className="space-y-2">
                  {boundingBoxes.map((box, idx) => {
                    const cat = DEFECT_CATEGORIES.find(c => c.name === box.defect_type);
                    return (
                      <div
                        key={box.id}
                        onClick={() => setSelectedBoxId(box.id)}
                        className={`p-2 rounded-lg cursor-pointer transition ${
                          selectedBoxId === box.id
                            ? 'bg-slate-600 ring-2 ring-blue-500'
                            : 'bg-slate-700 hover:bg-slate-600'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span 
                              className="w-2 h-2 rounded"
                              style={{ backgroundColor: cat?.color }}
                            />
                            <span className="text-sm capitalize">{box.defect_type.replace('_', ' ')}</span>
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setBoundingBoxes(boundingBoxes.filter(b => b.id !== box.id));
                            }}
                            className="text-red-400 hover:text-red-300"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                        <div className="text-xs text-slate-400 mt-1">
                          {Math.round(box.x)}, {Math.round(box.y)} • {Math.round(box.width)}×{Math.round(box.height)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Notes */}
            <div className="p-4 border-t border-slate-700">
              <h3 className="text-sm font-semibold text-slate-300 mb-2">Notes</h3>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add notes about this sample..."
                className="w-full h-20 bg-slate-700 border border-slate-600 rounded-lg p-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Keyboard Shortcuts */}
            <div className="p-4 border-t border-slate-700 bg-slate-850">
              <h3 className="text-sm font-semibold text-slate-300 mb-2">Shortcuts</h3>
              <div className="grid grid-cols-2 gap-1 text-xs text-slate-400">
                <div><kbd className="bg-slate-700 px-1 rounded">N</kbd> Normal</div>
                <div><kbd className="bg-slate-700 px-1 rounded">A</kbd> Anomaly</div>
                <div><kbd className="bg-slate-700 px-1 rounded">U</kbd> Uncertain</div>
                <div><kbd className="bg-slate-700 px-1 rounded">H</kbd> Heatmap</div>
                <div><kbd className="bg-slate-700 px-1 rounded">Del</kbd> Delete</div>
                <div><kbd className="bg-slate-700 px-1 rounded">←→</kbd> Navigate</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
