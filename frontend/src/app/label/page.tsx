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
 * - Cascade queue integration (auto-flagged anomalies)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import * as api from '@/lib/api';
import { 
  Loader2, 
  Maximize2, 
  ZoomIn, 
  ZoomOut, 
  Trash2,
  ChevronLeft,
  ChevronRight,
  Activity,
  TrendingUp,
  MousePointer2,
  SquareDashed,
  Inbox,
  Save,
  Check,
  HelpCircle,
  AlertTriangle
} from 'lucide-react';

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

interface Sample extends api.CascadeQueueItem {
  image_id: string;
  image_path: string;
  anomaly_score?: number;
  uncertainty_score?: number;
  heatmap_base64?: string;
  bgad_score: number;
  vlm_score?: number;
  routing_case: api.CascadeRoutingCase;
  created_at: string;
  metadata?: any;
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
  const [selectedCategory, setSelectedCategory] = useState('cascade'); // Default to cascade queue
  const [categories, setCategories] = useState<string[]>([]);
  const [isCascadeMode, setIsCascadeMode] = useState(true);

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
  const [stats, setStats] = useState({ total: 0, normal: 0, anomaly: 0, uncertain: 0, queueSize: 0 });
  const [cascadeStats, setCascadeStats] = useState<api.CascadeQueueStats | null>(null);

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
          setCategories(['cascade', ...(data.categories || [])]);
        }
      } catch (e) {
        console.error('Failed to fetch categories');
      }
    };
    fetchCategories();
  }, []);

  // Fetch cascade queue or standard samples
  useEffect(() => {
    const fetchSamples = async () => {
      setLoading(true);
      setError(null);
      try {
        if (selectedCategory === 'cascade') {
          // Fetch from cascade queue
          setIsCascadeMode(true);
          const response = await api.fetchAnnotationQueue(50);
          if (response.success) {
            setSamples(response.queue as unknown as Sample[]);
            setCurrentIndex(0);
            setBoundingBoxes([]);
            setStats(prev => ({
              ...prev,
              queueSize: response.stats.pending
            }));
            
            // Fetch cascade stats
            try {
              const cascadeStatsResponse = await api.getCascadeStats();
              setCascadeStats(cascadeStatsResponse);
            } catch (e) {
              console.error('Failed to fetch cascade stats');
            }
          }
        } else {
          // Fetch standard labeling queue
          setIsCascadeMode(false);
          const res = await fetch(`${API_URL}/labeling/queue?category=${selectedCategory}&limit=50`);
          if (res.ok) {
            const data = await res.json();
            setSamples(data.samples || []);
            setCurrentIndex(0);
            setBoundingBoxes([]);
          } else {
            setError('Failed to load samples');
          }
        }
      } catch (e) {
        setError('Cannot connect to backend');
        console.error(e);
      } finally {
        setLoading(false);
      }
    };

    fetchSamples();
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
      ctx.lineWidth = selectedBoxId === box.id ? 2 : 1.5;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Draw label background
      ctx.fillStyle = color;
      ctx.fillRect(box.x, box.y - 18, ctx.measureText(box.defect_type).width + 8, 18);
      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px monospace';
      ctx.fillText(box.defect_type.toUpperCase(), box.x + 4, box.y - 5);
    });

    // Draw current drawing box
    if (currentBox && drawStart) {
      const category = DEFECT_CATEGORIES.find(c => c.name === selectedDefect);
      ctx.strokeStyle = category?.color || '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
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
      if (isCascadeMode) {
        // Use cascade submission endpoint
        const defectTypes = Array.from(new Set(boundingBoxes.map(b => b.defect_type)));
        const submission: api.CascadeAnnotationSubmission = {
          image_id: currentSample.image_id,
          label: label === 'uncertain' ? 'anomaly' : label,  // Treat uncertain as anomaly
          bounding_boxes: boundingBoxes,
          defect_types: defectTypes,
          notes
        };

        const result = await api.submitCascadeAnnotation(submission);
        if (result.success) {
          // Move to next sample
          if (currentIndex < samples.length - 1) {
            setCurrentIndex(currentIndex + 1);
            setBoundingBoxes([]);
            setNotes('');
          } else {
            // Reload queue to get new items
            const response = await api.fetchAnnotationQueue(50);
            if (response.success && response.queue.length > 0) {
              setSamples(response.queue as unknown as Sample[]);
              setCurrentIndex(0);
              setBoundingBoxes([]);
            } else {
              setError('No more items in cascade queue');
            }
          }
        }
      } else {
        // Use standard labeling endpoint
        const annotation: Annotation = {
          image_id: currentSample.image_id,
          label,
          bounding_boxes: boundingBoxes,
          defect_types: Array.from(new Set(boundingBoxes.map(b => b.defect_type))),
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
              [label]: prev[label === 'uncertain' ? 'uncertain' : label] + 1,
            }));
          }
        }
      }
    } catch (e) {
      setError(`Failed to submit annotation: ${e}`);
      console.error(e);
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
  
  // Try to cleanly get VLM reasoning
  const vlmReasoning = currentSample?.metadata?.vlm_reasoning || (currentSample as any)?.vlm_reasoning || "VLM analysis unavailable.";

  return (
    <div className="h-screen w-screen bg-zinc-950 text-zinc-300 flex flex-col overflow-hidden font-sans">
      {/* Header - Professional dark with crisp borders */}
      <header className="h-14 bg-zinc-950 border-b border-zinc-800 flex items-center px-6 justify-between flex-shrink-0">
        <div className="flex items-center space-x-4">
          <Link href="/" className="flex items-center space-x-2 hover:text-indigo-400 transition-colors">
            <Activity className="w-5 h-5 text-indigo-500" />
            <span className="font-bold tracking-tight text-white">RETINA</span>
          </Link>
          <div className="w-px h-5 bg-zinc-800" />
          <h1 className="text-xs font-semibold tracking-widest text-zinc-400 uppercase">Annotation Studio</h1>
        </div>

        {/* Center: Progress & Queue Stats */}
        <div className="flex items-center space-x-6 text-[11px] font-mono text-zinc-500 uppercase tracking-wide">
          {samples.length > 0 && (
            <>
              <div className="flex items-center space-x-3">
                <span>{currentIndex + 1} / {samples.length}</span>
                <div className="w-32 h-1 bg-zinc-900 rounded-sm overflow-hidden">
                  <div 
                    className="h-full bg-indigo-500 transition-all"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
              
              {isCascadeMode && (
                <>
                  <div className="w-px h-3 bg-zinc-800" />
                  <div className="flex items-center space-x-2">
                    <span className="w-1.5 h-1.5 bg-amber-500 rounded-sm" />
                    <span>Queue: {stats.queueSize}</span>
                  </div>
                  {cascadeStats && (
                    <div className="flex items-center space-x-2">
                      <span className="w-1.5 h-1.5 bg-emerald-500 rounded-sm" />
                      <span>Labeled: {cascadeStats.labeled}</span>
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>

        {/* Right: Category selector */}
        <div className="flex items-center space-x-3">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-1.5 bg-zinc-900 border border-zinc-700 rounded-sm text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors text-zinc-300"
          >
            {categories.map(cat => (
              <option key={cat} value={cat} className="bg-zinc-950">{cat.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {loading ? (
          <div className="w-full flex flex-col items-center justify-center bg-zinc-950">
            <Loader2 className="w-8 h-8 text-indigo-500 animate-spin mb-4" />
            <p className="text-[11px] font-mono uppercase tracking-widest text-zinc-500">Connecting to Pipeline...</p>
          </div>
        ) : error ? (
          <div className="w-full flex flex-col items-center justify-center bg-zinc-950">
            <AlertTriangle className="w-8 h-8 text-red-500 mb-4" />
            <p className="text-sm text-zinc-300 font-mono mb-6">{error}</p>
            <button 
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-sm text-[11px] font-mono uppercase tracking-widest transition-colors"
            >
              Retry Connection
            </button>
          </div>
        ) : samples.length === 0 ? (
          <div className="w-full flex flex-col items-center justify-center bg-zinc-950">
            <div className="text-center">
              <Inbox className="w-12 h-12 text-zinc-800 mx-auto mb-6" />
              <p className="text-sm font-medium text-zinc-300 mb-2">No Samples Available</p>
              <p className="text-[11px] font-mono text-zinc-500 mb-8 max-w-sm mx-auto">
                The inference cascade queue is currently empty. Production edge models are monitoring live streams without uncertainty.
              </p>
              <Link href="/" className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-sm text-[11px] font-mono uppercase tracking-widest transition-colors inline-block">
                Return to Dashboard
              </Link>
            </div>
          </div>
        ) : (
          <>
            {/* COLUMN 1: LEFT SIDEBAR - Tools & Defect Classes */}
            <div className="w-64 bg-zinc-900 border-r border-zinc-800 flex flex-col flex-shrink-0">
              {/* Tools Section */}
              <div className="p-5 border-b border-zinc-800">
                <h3 className="text-[10px] uppercase tracking-widest text-zinc-500 font-bold mb-4">Tools</h3>
                <div className="space-y-1.5">
                  <button
                    onClick={() => setActiveTool('select')}
                    className={`w-full px-3 py-2 rounded-sm text-xs font-medium flex items-center transition-colors ${
                      activeTool === 'select' 
                        ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30' 
                        : 'bg-transparent text-zinc-400 border border-transparent hover:bg-zinc-800 hover:border-zinc-700'
                    }`}
                  >
                    <MousePointer2 className="w-3.5 h-3.5 mr-2" />
                    <span>Select Area</span>
                    <kbd className="ml-auto text-[9px] font-mono text-zinc-500 border border-zinc-700 rounded-sm px-1.5 py-0.5">S</kbd>
                  </button>
                  
                  <button
                    onClick={() => setActiveTool('box')}
                    className={`w-full px-3 py-2 rounded-sm text-xs font-medium flex items-center transition-colors ${
                      activeTool === 'box' 
                        ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30' 
                        : 'bg-transparent text-zinc-400 border border-transparent hover:bg-zinc-800 hover:border-zinc-700'
                    }`}
                  >
                    <SquareDashed className="w-3.5 h-3.5 mr-2" />
                    <span>Bounding Box</span>
                    <kbd className="ml-auto text-[9px] font-mono text-zinc-500 border border-zinc-700 rounded-sm px-1.5 py-0.5">B</kbd>
                  </button>
                </div>
              </div>

              {/* Defect Classes Section */}
              <div className="flex-1 overflow-auto p-5">
                <h3 className="text-[10px] uppercase tracking-widest text-zinc-500 font-bold mb-4">Ontology Classes</h3>
                <div className="space-y-1.5">
                  {DEFECT_CATEGORIES.map((cat) => (
                    <button
                      key={cat.name}
                      onClick={() => setSelectedDefect(cat.name)}
                      className={`w-full px-3 py-2 rounded-sm text-xs font-medium flex items-center transition-colors border-l-2 ${
                        selectedDefect === cat.name
                          ? 'bg-zinc-800 text-zinc-100'
                          : 'bg-transparent text-zinc-400 hover:bg-zinc-800/50'
                      }`}
                      style={{ borderLeftColor: cat.color }}
                    >
                      <span className="flex-1 text-left uppercase tracking-wide">{cat.name.replace('_', ' ')}</span>
                      <kbd className="text-[9px] font-mono text-zinc-500 border border-zinc-700 rounded-sm px-1.5 py-0.5">{cat.shortcut}</kbd>
                    </button>
                  ))}
                </div>
              </div>

              {/* Keyboard Shortcuts */}
              <div className="p-5 border-t border-zinc-800 bg-zinc-950/30">
                <h3 className="text-[10px] uppercase tracking-widest text-zinc-500 font-bold mb-3">Hotkeys</h3>
                <div className="grid grid-cols-2 gap-y-2 gap-x-1 text-[10px] font-mono text-zinc-500">
                  <div className="flex justify-between border-b border-zinc-800/50 pb-1"><span>Normal</span> <kbd>N</kbd></div>
                  <div className="flex justify-between border-b border-zinc-800/50 pb-1 pl-2"><span>Anomaly</span> <kbd>A</kbd></div>
                  <div className="flex justify-between border-b border-zinc-800/50 pb-1"><span>Unclear</span> <kbd>U</kbd></div>
                  <div className="flex justify-between border-b border-zinc-800/50 pb-1 pl-2"><span>Heatmap</span> <kbd>H</kbd></div>
                  <div className="flex justify-between"><span>Delete</span> <kbd>DEL</kbd></div>
                  <div className="flex justify-between pl-2"><span>Navigate</span> <kbd>←→</kbd></div>
                </div>
              </div>
            </div>

            {/* COLUMN 2: CENTER - Canvas Workspace */}
            <div className="flex-1 bg-[#121214] flex flex-col relative overflow-hidden" style={{
              backgroundImage: 'linear-gradient(0deg, rgba(39, 39, 42, 0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(39, 39, 42, 0.4) 1px, transparent 1px)',
              backgroundSize: '40px 40px'
            }}>
              {/* Canvas Container */}
              <div 
                ref={containerRef}
                className="flex-1 overflow-auto flex items-center justify-center p-8"
              >
                <div className="relative inline-block shadow-2xl border border-zinc-700 bg-zinc-900">
                  <canvas
                    ref={canvasRef}
                    className="block"
                    style={{ 
                      transform: `scale(${zoom})`,
                      transformOrigin: 'center center',
                      maxWidth: '100%',
                      maxHeight: '100%',
                      cursor: activeTool === 'box' ? 'crosshair' : 'default'
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                  />
                </div>
              </div>

              {/* Bottom Floating Toolbar */}
              <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 bg-zinc-900 border border-zinc-700 rounded-sm px-2 py-1.5 flex items-center space-x-1 shadow-2xl">
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={`px-3 py-1.5 rounded-sm text-[10px] uppercase tracking-widest font-semibold flex items-center space-x-1.5 transition-colors ${
                    showHeatmap 
                      ? 'bg-indigo-500/20 text-indigo-300' 
                      : 'bg-transparent text-zinc-400 hover:bg-zinc-800'
                  }`}
                >
                  <div className="w-2.5 h-2.5 bg-gradient-to-r from-blue-500 via-emerald-500 to-red-500 rounded-sm" />
                  <span>Heatmap Overlays</span>
                </button>

                <div className="w-px h-4 bg-zinc-700 mx-2" />

                <button
                  onClick={() => setZoom(Math.max(zoom - 0.25, 0.5))}
                  className="p-1.5 rounded-sm hover:bg-zinc-800 text-zinc-400 transition-colors"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <div className="w-12 text-center text-[11px] font-mono text-zinc-500">
                  {Math.round(zoom * 100)}%
                </div>
                <button
                  onClick={() => setZoom(Math.min(zoom + 0.25, 3))}
                  className="p-1.5 rounded-sm hover:bg-zinc-800 text-zinc-400 transition-colors"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoom(1)}
                  className="p-1.5 rounded-sm hover:bg-zinc-800 text-zinc-400 transition-colors ml-1"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* COLUMN 3: RIGHT SIDEBAR - Sample Intelligence & Queue */}
            <div className="w-80 bg-zinc-900 border-l border-zinc-800 flex flex-col flex-shrink-0">
              {/* Cascade Routing Info */}
              {isCascadeMode && currentSample && (
                <div className="p-5 border-b border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-[10px] uppercase tracking-widest text-zinc-500 font-bold">Inference Telemetry</h3>
                    <span className="px-2 py-0.5 bg-zinc-800 border border-zinc-700 rounded-sm text-[10px] font-mono text-zinc-400 uppercase tracking-widest">
                      {currentSample.routing_case.split('_').slice(-1)[0]}
                    </span>
                  </div>
                  
                  {/* Edge Net Score */}
                  <div className="mb-4 bg-zinc-950 border border-zinc-800 rounded-sm p-3">
                    <div className="flex items-end justify-between mb-2">
                      <span className="text-[11px] font-medium text-zinc-400">BGAD EDGE SCORE</span>
                      <span className={`text-[11px] font-mono font-bold ${
                        currentSample.bgad_score > 0.7 ? 'text-red-400' : 
                        currentSample.bgad_score > 0.4 ? 'text-amber-400' : 
                        'text-emerald-400'
                      }`}>
                        {currentSample.bgad_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="w-full h-1 bg-zinc-800 rounded-sm overflow-hidden">
                      <div 
                        className={`h-full opacity-80 ${
                          currentSample.bgad_score > 0.7 ? 'bg-red-500' : 
                          currentSample.bgad_score > 0.4 ? 'bg-amber-400' : 
                          'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(currentSample.bgad_score * 100, 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* VLM Fallback Reasoning */}
                  <div className="bg-zinc-950 border border-indigo-500/20 rounded-sm p-3 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500/50" />
                    <div className="flex items-end justify-between mb-2 pl-2">
                      <span className="text-[11px] font-medium text-indigo-400 uppercase tracking-wide">GPT-4V Fallback Analysis</span>
                      <span className="text-[10px] font-mono text-zinc-500">Zero-Shot</span>
                    </div>
                    <div className="pl-2">
                      <p className="text-[11px] leading-relaxed text-zinc-400 font-mono whitespace-pre-wrap">
                        {vlmReasoning}
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-4 flex flex-col">
                    <span className="text-[10px] text-zinc-600 uppercase tracking-widest mb-1">Blob Target ID</span>
                    <span className="text-[10px] font-mono text-zinc-500 truncate">{currentSample.image_id}</span>
                  </div>
                </div>
              )}

              {/* Annotations List */}
              <div className="flex-1 overflow-auto p-5 border-b border-zinc-800">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[10px] uppercase tracking-widest text-zinc-500 font-bold">Declared Regions</h3>
                  <span className="text-[10px] font-mono bg-zinc-800 px-1.5 py-0.5 rounded-sm">{boundingBoxes.length}</span>
                </div>
                
                {boundingBoxes.length === 0 ? (
                  <div className="h-24 border border-dashed border-zinc-700 rounded-sm flex items-center justify-center bg-zinc-950/50">
                    <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest">No Bounding Boxes DrawN</span>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {boundingBoxes.map((box) => {
                      const cat = DEFECT_CATEGORIES.find(c => c.name === box.defect_type);
                      return (
                        <div
                          key={box.id}
                          onClick={() => setSelectedBoxId(box.id)}
                          className={`p-3 rounded-sm cursor-pointer border transition-colors ${
                            selectedBoxId === box.id
                              ? 'bg-zinc-800 border-zinc-600'
                              : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <div 
                                className="w-1.5 h-1.5 rounded-sm" 
                                style={{ backgroundColor: cat?.color }}
                              />
                              <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-300">{box.defect_type.replace('_', ' ')}</span>
                            </div>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setBoundingBoxes(boundingBoxes.filter(b => b.id !== box.id));
                              }}
                              className="text-zinc-600 hover:text-red-400 transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                          <div className="text-[10px] text-zinc-500 font-mono bg-zinc-900 border border-zinc-800 rounded-sm px-2 py-1 inline-block">
                            X:{Math.round(box.x)} Y:{Math.round(box.y)} • W:{Math.round(box.width)} H:{Math.round(box.height)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Action Area */}
              <div className="p-5 flex-shrink-0 bg-zinc-950">
                <div className="mb-4">
                   <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Enter manual override notes..."
                    className="w-full h-16 bg-zinc-900 border border-zinc-800 rounded-sm p-3 text-xs resize-none focus:outline-none focus:border-indigo-500/50 text-zinc-300 font-mono"
                  />
                </div>

                <div className="flex gap-2 mb-3">
                  <button
                    onClick={() => { setCurrentIndex(Math.max(0, currentIndex - 1)); setBoundingBoxes([]); }}
                    disabled={currentIndex === 0}
                    className="flex-1 py-1.5 bg-zinc-900 border border-zinc-800 text-zinc-400 rounded-sm hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-mono text-[10px] uppercase tracking-widest flex items-center justify-center"
                  >
                    <ChevronLeft className="w-3 h-3 mr-1" />
                    Prev
                  </button>
                  <button
                    onClick={() => { setCurrentIndex(Math.min(samples.length - 1, currentIndex + 1)); setBoundingBoxes([]); }}
                    disabled={currentIndex === samples.length - 1}
                    className="flex-1 py-1.5 bg-zinc-900 border border-zinc-800 text-zinc-400 rounded-sm hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-mono text-[10px] uppercase tracking-widest flex items-center justify-center"
                  >
                    Next
                    <ChevronRight className="w-3 h-3 ml-1" />
                  </button>
                </div>

                <div className="grid grid-cols-3 gap-2 mb-3">
                  <button
                    onClick={() => submitAnnotation('normal')}
                    className="py-2.5 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 rounded-sm font-semibold text-[10px] uppercase tracking-widest transition-colors flex flex-col items-center justify-center space-y-1"
                  >
                    <Check className="w-4 h-4" />
                    <span>Approve</span>
                  </button>
                  
                  <button
                    onClick={() => submitAnnotation('uncertain')}
                    className="py-2.5 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-400 rounded-sm font-semibold text-[10px] uppercase tracking-widest transition-colors flex flex-col items-center justify-center space-y-1"
                  >
                    <HelpCircle className="w-4 h-4" />
                    <span>Unclear</span>
                  </button>

                  <button
                    onClick={() => submitAnnotation('anomaly')}
                    className="py-2.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 rounded-sm font-semibold text-[10px] uppercase tracking-widest transition-colors flex flex-col items-center justify-center space-y-1"
                  >
                    <AlertTriangle className="w-4 h-4" />
                    <span>Reject</span>
                  </button>
                </div>

                <button
                  onClick={() => submitAnnotation('anomaly')}
                  className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 rounded-sm font-bold text-xs uppercase tracking-widest text-white transition-colors flex items-center justify-center space-x-2"
                >
                  <Save className="w-4 h-4" />
                  <span>Submit Override Data</span>
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
