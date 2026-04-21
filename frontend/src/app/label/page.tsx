'use client';

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
  MousePointer2,
  SquareDashed,
  Inbox,
  Save,
  Check,
  HelpCircle,
  AlertTriangle,
  Sparkles,
  SkipForward,
  Spline,
} from 'lucide-react';

// Decospan Dutch canonical names per CLAUDE.md §6.2
const DEFECT_CATEGORIES = [
  { name: 'krassen',   englishLabel: 'Scratches',     color: '#EF4444', shortcut: '1' },
  { name: 'deuk',      englishLabel: 'Dent',           color: '#F97316', shortcut: '2' },
  { name: 'vlekken',   englishLabel: 'Stains',         color: '#EAB308', shortcut: '3' },
  { name: 'barst',     englishLabel: 'Crack',          color: '#22C55E', shortcut: '4' },
  { name: 'open fout', englishLabel: 'Open Defect',    color: '#3B82F6', shortcut: '5' },
  { name: 'open knop', englishLabel: 'Open Knot',      color: '#8B5CF6', shortcut: '6' },
  { name: 'snijfout',  englishLabel: 'Cutting Error',  color: '#6B7280', shortcut: '7' },
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

interface Polygon {
  id: string;
  vertices: { x: number; y: number }[];
  classKey: string;
  color: string;
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
  metadata?: Record<string, unknown>;
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
  const [samples, setSamples] = useState<Sample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([]);
  const [selectedDefect, setSelectedDefect] = useState<string>(DEFECT_CATEGORIES[0].name);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Partial<BoundingBox> | null>(null);
  const [notes, setNotes] = useState('');
  const [showHeatmap, setShowHeatmap] = useState(true);

  const [activeTool, setActiveTool] = useState<'select' | 'box' | 'polygon'>('box');
  const [polygons, setPolygons] = useState<Polygon[]>([]);
  const [currentPolygon, setCurrentPolygon] = useState<{ vertices: { x: number; y: number }[] } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const [stats, setStats] = useState({ total: 0, normal: 0, anomaly: 0, uncertain: 0, queueSize: 0 });
  const [sessionReviewCount, setSessionReviewCount] = useState(0);
  const [pendingLabel, setPendingLabel] = useState<'normal' | 'anomaly' | 'uncertain' | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const currentSample = samples[currentIndex];

  const loadPool = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getLabelPoolV2(50);
      const asSamples: Sample[] = res.pool.map((p) => ({
        image_id: p.image_id,
        image_path: p.image_id, // canvas URL builder uses this to fetch /api/images/{id}
        anomaly_score: p.anomaly_score ?? undefined,
        uncertainty_score: p.uncertainty_score ?? undefined,
        bgad_score: p.anomaly_score ?? p.score ?? 0,
        routing_case: 'C_uncertain_vlm_routed',
        status: 'pending',
        created_at: new Date().toISOString(),
      } as Sample));
      setSamples(asSamples);
      setCurrentIndex(0);
      setBoundingBoxes([]);
      setPolygons([]);
      setCurrentPolygon(null);
      setStats(prev => ({ ...prev, queueSize: res.count }));
    } catch (e) {
      setError(`Cannot connect to API: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPool();
  }, [loadPool]);

  // ─── Canvas draw logic (preserved exactly) ───────────────────────────────────

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0);
    } else {
      ctx.fillStyle = '#121214';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    if (showHeatmap && currentSample?.heatmap_base64) {
      const heatmapImg = new Image();
      heatmapImg.src = `data:image/png;base64,${currentSample.heatmap_base64}`;
      ctx.globalAlpha = 0.5;
      ctx.drawImage(heatmapImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }

    // GPT-4V bounding box overlay (purple dashed) per CLAUDE.md §5.4
    const vlmBbox = (currentSample?.metadata as Record<string, unknown> | undefined)?.vlm_bounding_box as
      { x: number; y: number; width: number; height: number } | undefined;
    if (vlmBbox) {
      ctx.strokeStyle = '#7C3AED';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(vlmBbox.x, vlmBbox.y, vlmBbox.width, vlmBbox.height);
      ctx.setLineDash([]);
    }

    boundingBoxes.forEach((box) => {
      const category = DEFECT_CATEGORIES.find(c => c.name === box.defect_type);
      const color = category?.color ?? '#ffffff';

      ctx.strokeStyle = color;
      ctx.lineWidth = selectedBoxId === box.id ? 2 : 1.5;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      ctx.fillStyle = color;
      ctx.fillRect(box.x, box.y - 18, ctx.measureText(box.defect_type).width + 8, 18);
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px monospace';
      ctx.fillText(box.defect_type.toUpperCase(), box.x + 4, box.y - 5);
    });

    if (currentBox && drawStart) {
      const category = DEFECT_CATEGORIES.find(c => c.name === selectedDefect);
      ctx.strokeStyle = category?.color ?? '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(currentBox.x ?? 0, currentBox.y ?? 0, currentBox.width ?? 0, currentBox.height ?? 0);
      ctx.setLineDash([]);
    }

    // ── Committed polygons ──────────────────────────────────────────────
    polygons.forEach((poly) => {
      if (poly.vertices.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(poly.vertices[0].x, poly.vertices[0].y);
      for (let i = 1; i < poly.vertices.length; i++) {
        ctx.lineTo(poly.vertices[i].x, poly.vertices[i].y);
      }
      ctx.closePath();
      ctx.strokeStyle = poly.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = poly.color + '33'; // ~20% alpha
      ctx.fill();
      ctx.fillStyle = poly.color;
      poly.vertices.forEach((v) => {
        ctx.beginPath();
        ctx.arc(v.x, v.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
      // Class label
      const first = poly.vertices[0];
      const labelText = poly.classKey.toUpperCase();
      ctx.font = '10px monospace';
      const tw = ctx.measureText(labelText).width + 8;
      ctx.fillStyle = poly.color;
      ctx.fillRect(first.x, first.y - 18, tw, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, first.x + 4, first.y - 5);
    });

    // ── In-progress polygon ─────────────────────────────────────────────
    if (currentPolygon && currentPolygon.vertices.length > 0) {
      const cat = DEFECT_CATEGORIES.find(c => c.name === selectedDefect);
      const color = cat?.color ?? '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      // solid edges between placed vertices
      ctx.beginPath();
      ctx.moveTo(currentPolygon.vertices[0].x, currentPolygon.vertices[0].y);
      for (let i = 1; i < currentPolygon.vertices.length; i++) {
        ctx.lineTo(currentPolygon.vertices[i].x, currentPolygon.vertices[i].y);
      }
      ctx.stroke();
      // dashed preview from last vertex to cursor
      if (mousePos) {
        const last = currentPolygon.vertices[currentPolygon.vertices.length - 1];
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(mousePos.x, mousePos.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // vertex dots (slightly larger while editable)
      ctx.fillStyle = color;
      currentPolygon.vertices.forEach((v) => {
        ctx.beginPath();
        ctx.arc(v.x, v.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [boundingBoxes, currentBox, drawStart, selectedBoxId, showHeatmap, currentSample, selectedDefect, polygons, currentPolygon, mousePos]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentSample) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      canvas.width = img.width;
      canvas.height = img.height;
      redrawCanvas();
    };
    img.src = `${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:3001'}/api/images/${currentSample.image_id}`;
    img.onerror = () => {
      canvas.width = 512;
      canvas.height = 512;
      ctx.fillStyle = '#121214';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#6E6E73';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Image Preview', canvas.width / 2, canvas.height / 2);
      ctx.font = '12px monospace';
      ctx.fillText(currentSample.image_id, canvas.width / 2, canvas.height / 2 + 24);
    };
  }, [currentSample, redrawCanvas]);

  useEffect(() => { redrawCanvas(); }, [redrawCanvas]);

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

  const closePolygon = useCallback(() => {
    if (!currentPolygon || currentPolygon.vertices.length < 3) return;
    const cat = DEFECT_CATEGORIES.find(c => c.name === selectedDefect);
    const id = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `poly_${Date.now()}`;
    setPolygons(prev => [...prev, {
      id,
      vertices: currentPolygon.vertices,
      classKey: selectedDefect,
      color: cat?.color ?? '#ffffff',
    }]);
    setCurrentPolygon(null);
  }, [currentPolygon, selectedDefect]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);

    if (activeTool === 'polygon') {
      if (e.button === 2) {
        // Right-click: remove last vertex (undo)
        e.preventDefault();
        setCurrentPolygon(prev => {
          if (!prev || prev.vertices.length === 0) return null;
          const next = prev.vertices.slice(0, -1);
          return next.length === 0 ? null : { vertices: next };
        });
        return;
      }
      if (e.button === 0) {
        // Left-click: add vertex (start a new polygon if none in progress)
        setCurrentPolygon(prev => ({
          vertices: [...(prev?.vertices ?? []), coords],
        }));
      }
      return;
    }

    if (activeTool !== 'box') return;
    setIsDrawing(true);
    setDrawStart(coords);
    setCurrentBox({ x: coords.x, y: coords.y, width: 0, height: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);
    if (activeTool === 'polygon' && currentPolygon) {
      setMousePos(coords);
      return;
    }
    if (!isDrawing || !drawStart || activeTool !== 'box') return;
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
    if ((currentBox.width ?? 0) > 10 && (currentBox.height ?? 0) > 10) {
      const newBox: BoundingBox = {
        id: `box_${Date.now()}`,
        x: currentBox.x ?? 0,
        y: currentBox.y ?? 0,
        width: currentBox.width ?? 0,
        height: currentBox.height ?? 0,
        defect_type: selectedDefect,
        confidence: 1.0,
      };
      setBoundingBoxes(prev => [...prev, newBox]);
    }
    setDrawStart(null);
    setCurrentBox(null);
  };

  const handleDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTool === 'polygon') {
      e.preventDefault();
      closePolygon();
    }
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTool === 'polygon') {
      e.preventDefault();
    }
  };

  // ─── Submission ───────────────────────────────────────────────────────────────

  const submitAnnotation = useCallback(async (label: 'normal' | 'anomaly' | 'uncertain') => {
    if (!currentSample) return;
    const selectedCat = DEFECT_CATEGORIES.find(c => c.name === selectedDefect);
    try {
      await api.submitLabelV2({
        image_id: currentSample.image_id,
        product_class: currentSample.routing_case ? 'unknown' : 'unknown',  // product name unavailable here — API stores what frontend sends
        label: label === 'uncertain' ? 'anomaly' : label,
        defect_class: selectedCat?.name ?? null,
        boxes: boundingBoxes.map(b => ({
          x: b.x, y: b.y, width: b.width, height: b.height,
          defect_type: b.defect_type, confidence: b.confidence,
        })),
        polygons: polygons.map(p => ({
          vertices: p.vertices,
          class: p.classKey,
        })),
        operator_id: 'expert',
        notes,
      });
      setSessionReviewCount(c => c + 1);
      setPendingLabel(null);
      if (currentIndex < samples.length - 1) {
        setCurrentIndex(currentIndex + 1);
        setBoundingBoxes([]);
        setPolygons([]);
        setCurrentPolygon(null);
        setNotes('');
      } else {
        // Reload the pool — we're at the end
        await loadPool();
      }
    } catch (e) {
      setError(`Failed to submit: ${(e as Error).message}`);
    }
  }, [currentSample, boundingBoxes, polygons, notes, currentIndex, samples.length, selectedDefect, loadPool]);

  const handleSkip = useCallback(async () => {
    if (!currentSample) return;
    if (currentIndex < samples.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setBoundingBoxes([]);
      setPolygons([]);
      setCurrentPolygon(null);
    }
  }, [currentSample, currentIndex, samples.length]);

  // ─── Keyboard shortcuts per CLAUDE.md ────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case 'Enter':
          // Polygon mode: Enter closes an in-progress polygon (>=3 verts).
          // Otherwise falls through to normal label submission.
          if (activeTool === 'polygon' && currentPolygon && currentPolygon.vertices.length >= 3) {
            closePolygon();
          } else if (pendingLabel) {
            submitAnnotation(pendingLabel);
          }
          break;
        case 'Escape':
          if (activeTool === 'polygon' && currentPolygon) {
            setCurrentPolygon(null);
          }
          break;
        case 's': case 'S': handleSkip(); break;
        case 'z': case 'Z':
          if (activeTool === 'polygon') {
            setPolygons(prev => prev.slice(0, -1));
          } else {
            setBoundingBoxes(prev => prev.slice(0, -1));
          }
          break;
        case 'b': case 'B': setActiveTool('box'); break;
        case 'v': case 'V': setActiveTool('select'); break;
        case 'p': case 'P': setActiveTool('polygon'); break;
        case 'h': case 'H': setShowHeatmap(v => !v); break;
        case 'ArrowLeft':
          if (currentIndex > 0) { setCurrentIndex(c => c - 1); setBoundingBoxes([]); setPolygons([]); setCurrentPolygon(null); }
          break;
        case 'ArrowRight':
          if (currentIndex < samples.length - 1) { setCurrentIndex(c => c + 1); setBoundingBoxes([]); setPolygons([]); setCurrentPolygon(null); }
          break;
        default: {
          const cat = DEFECT_CATEGORIES.find(c => c.shortcut === e.key);
          if (cat) setSelectedDefect(cat.name);
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [currentIndex, samples.length, pendingLabel, submitAnnotation, handleSkip, activeTool, currentPolygon, closePolygon]);

  const progressPct = samples.length > 0 ? ((currentIndex + 1) / samples.length) * 100 : 0;
  const vlmReasoning =
    (currentSample?.metadata as Record<string, unknown> | undefined)?.vlm_reasoning as string | undefined
    ?? 'VLM analysis unavailable.';

  // ─── Full-screen layout ───────────────────────────────────────────────────────
  return (
    <div className="fixed inset-0 bg-[#0C0C0E] text-text-primary flex flex-col overflow-hidden font-sans z-50">

      {/* Header */}
      <header className="h-14 bg-[#0C0C0E] border-b border-surface-border flex items-center px-6 justify-between flex-shrink-0">
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2 hover:text-kul-accent transition-colors">
            <div className="w-5 h-5 bg-kul-blue rounded-sm" />
            <span className="font-semibold text-sm text-text-primary">RETINA</span>
          </Link>
          <div className="w-px h-4 bg-surface-border" />
          <h1 className="text-xs font-medium tracking-widest text-text-tertiary uppercase">
            Annotation Studio
          </h1>
        </div>

        {/* Progress */}
        <div className="flex items-center gap-6 text-[11px] font-mono text-text-tertiary uppercase tracking-wide">
          {samples.length > 0 && (
            <>
              <span>{currentIndex + 1} / {samples.length}</span>
              <div className="w-32 h-0.5 bg-surface-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-kul-accent transition-all"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className="w-px h-3 bg-surface-border" />
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 bg-state-warn rounded-sm" />
                Queue: {stats.queueSize}
              </span>
            </>
          )}
        </div>

        {/* Right-side spacer (category dropdown removed — it was cascade-only) */}
        <div className="text-[10px] font-mono uppercase tracking-widest text-text-disabled">
          AL Pool · retina:al:pool
        </div>
      </header>

      {/* Body */}
      <div className="flex-1 flex overflow-hidden">
        {loading ? (
          <div className="w-full flex flex-col items-center justify-center">
            <Loader2 className="w-8 h-8 text-kul-accent animate-spin mb-4" />
            <p className="text-[11px] font-mono uppercase tracking-widest text-text-tertiary">
              Connecting to Pipeline...
            </p>
          </div>
        ) : error ? (
          <div className="w-full flex flex-col items-center justify-center gap-4">
            <AlertTriangle className="w-8 h-8 text-state-alert" />
            <p className="text-sm text-text-primary font-mono">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-surface-raised hover:bg-surface-overlay border border-surface-border rounded text-[11px] font-mono uppercase tracking-widest transition-colors"
            >
              Retry Connection
            </button>
          </div>
        ) : samples.length === 0 ? (
          <div className="w-full flex flex-col items-center justify-center text-center px-6">
            <Inbox className="w-12 h-12 text-surface-border mx-auto mb-6" />
            <p className="text-sm font-medium text-text-primary mb-2">Queue is empty</p>
            <p className="text-[11px] font-mono text-text-tertiary mb-8 max-w-md mx-auto leading-relaxed">
              Submit an image with a Stage 1 anomaly score in the uncertainty zone
              [0.5, 0.9) to populate this queue. Images that PatchCore flags as
              ambiguous land here for expert labels.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => loadPool()}
                className="px-6 py-2 bg-surface-raised hover:bg-surface-overlay border border-surface-border rounded text-[11px] font-mono uppercase tracking-widest transition-colors"
              >
                Refresh Queue
              </button>
              <Link
                href="/submit"
                className="px-6 py-2 bg-kul-blue hover:bg-kul-light text-white rounded text-[11px] font-mono uppercase tracking-widest transition-colors inline-block"
              >
                Submit Image
              </Link>
            </div>
          </div>
        ) : (
          <>
            {/* LEFT SIDEBAR */}
            <div className="w-64 bg-[#141416] border-r border-surface-border flex flex-col flex-shrink-0">

              {/* Thumbnail queue strip */}
              <div className="h-14 border-b border-surface-border flex items-center gap-1.5 px-2 overflow-x-auto flex-shrink-0">
                {samples.slice(Math.max(0, currentIndex - 1), currentIndex + 5).map((s, i) => {
                  const absIdx = Math.max(0, currentIndex - 1) + i;
                  return (
                    <button
                      key={s.image_id}
                      onClick={() => { setCurrentIndex(absIdx); setBoundingBoxes([]); }}
                      className={[
                        'w-10 h-10 rounded-md bg-surface-overlay flex-shrink-0 overflow-hidden',
                        'ring-1 transition-all',
                        absIdx === currentIndex ? 'ring-kul-accent' : 'ring-transparent hover:ring-surface-borderhover',
                      ].join(' ')}
                    >
                      <div className="w-full h-full flex items-center justify-center text-[8px] font-mono text-text-disabled">
                        {absIdx + 1}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Tools */}
              <div className="p-5 border-b border-surface-border">
                <h3 className="text-[10px] uppercase tracking-widest text-text-tertiary font-bold mb-4">
                  Tools
                </h3>
                <div className="space-y-1.5">
                  {([
                    { tool: 'select' as const,  icon: MousePointer2, label: 'Select Area', kbd: 'V' },
                    { tool: 'box' as const,     icon: SquareDashed,  label: 'Bounding Box', kbd: 'B' },
                    { tool: 'polygon' as const, icon: Spline,        label: 'Polygon',      kbd: 'P' },
                  ] as const).map(({ tool, icon: Icon, label, kbd }) => (
                    <button
                      key={tool}
                      onClick={() => setActiveTool(tool)}
                      className={[
                        'w-full px-3 py-2 rounded text-xs font-medium flex items-center transition-colors border',
                        activeTool === tool
                          ? 'bg-kul-accent/10 text-kul-accent border-kul-accent/30'
                          : 'bg-transparent text-text-tertiary border-transparent hover:bg-surface-overlay hover:border-surface-border',
                      ].join(' ')}
                    >
                      <Icon className="w-3.5 h-3.5 mr-2" />
                      <span>{label}</span>
                      <kbd className="ml-auto text-[9px] font-mono text-text-disabled border border-surface-border rounded px-1.5 py-0.5">
                        {kbd}
                      </kbd>
                    </button>
                  ))}
                </div>
              </div>

              {/* Decospan taxonomy */}
              <div className="flex-1 overflow-auto p-5">
                <h3 className="text-[10px] uppercase tracking-widest text-text-tertiary font-bold mb-4">
                  Decospan Taxonomy
                </h3>
                <div className="space-y-1.5">
                  {DEFECT_CATEGORIES.map(cat => (
                    <button
                      key={cat.name}
                      onClick={() => setSelectedDefect(cat.name)}
                      className={[
                        'w-full px-3 py-2 rounded text-xs font-medium flex items-center transition-colors border-l-2',
                        selectedDefect === cat.name
                          ? 'bg-surface-overlay text-text-primary'
                          : 'bg-transparent text-text-tertiary hover:bg-surface-overlay/50',
                      ].join(' ')}
                      style={{ borderLeftColor: cat.color }}
                    >
                      <span className="flex-1 text-left">
                        <span className="block uppercase tracking-wide">{cat.name}</span>
                        <span className="text-[10px] text-text-disabled normal-case tracking-normal">
                          {cat.englishLabel}
                        </span>
                      </span>
                      <kbd className="text-[9px] font-mono text-text-disabled border border-surface-border rounded px-1.5 py-0.5">
                        {cat.shortcut}
                      </kbd>
                    </button>
                  ))}
                </div>
              </div>

              {/* Hotkeys */}
              <div className="p-5 border-t border-surface-border bg-[#0C0C0E]/30">
                <h3 className="text-[10px] uppercase tracking-widest text-text-tertiary font-bold mb-3">
                  Hotkeys
                </h3>
                <div className="grid grid-cols-2 gap-y-2 gap-x-1 text-[10px] font-mono text-text-tertiary">
                  {[
                    ['Submit', 'Enter'], ['Skip', 'S'],
                    ['Undo',   'Z'],     ['Heatmap', 'H'],
                    ['Box',    'B'],     ['Select', 'V'],
                    ['Delete', 'Del'],   ['Navigate', '←→'],
                  ].map(([label, key], i) => (
                    <div key={i} className="flex justify-between border-b border-surface-border/50 pb-1">
                      <span>{label}</span>
                      <kbd>{key}</kbd>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* CENTER: Canvas */}
            <div
              className="flex-1 bg-[#121214] flex flex-col relative overflow-hidden"
              style={{
                backgroundImage:
                  'linear-gradient(0deg, rgba(42,42,46,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(42,42,46,0.4) 1px, transparent 1px)',
                backgroundSize: '40px 40px',
              }}
            >
              <div
                ref={containerRef}
                className="flex-1 overflow-auto flex items-center justify-center p-8"
              >
                <div className="relative inline-block shadow-2xl border border-surface-border bg-[#141416]">
                  {/* GPT-4V suggestion badge */}
                  {!!(currentSample?.metadata as Record<string, unknown> | undefined)?.vlm_bounding_box && (
                    <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 px-2 py-1 rounded bg-purple-500/10 border border-purple-500/30">
                      <Sparkles className="w-3 h-3 text-purple-400" />
                      <span className="text-[10px] text-purple-400 font-medium">GPT-4o Suggestion</span>
                    </div>
                  )}
                  <canvas
                    ref={canvasRef}
                    className="block"
                    style={{
                      transform: `scale(${zoom})`,
                      transformOrigin: 'center center',
                      maxWidth: '100%',
                      maxHeight: '100%',
                      cursor: activeTool === 'box' || activeTool === 'polygon' ? 'crosshair' : 'default',
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onDoubleClick={handleDoubleClick}
                    onContextMenu={handleContextMenu}
                  />
                </div>
              </div>

              {/* Floating toolbar */}
              <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-[#141416] border border-surface-border rounded px-2 py-1.5 flex items-center gap-1 shadow-2xl">
                <button
                  onClick={() => setShowHeatmap(v => !v)}
                  className={[
                    'px-3 py-1.5 rounded text-[10px] uppercase tracking-widest font-semibold flex items-center gap-1.5 transition-colors',
                    showHeatmap
                      ? 'bg-kul-accent/20 text-kul-accent'
                      : 'bg-transparent text-text-tertiary hover:bg-surface-overlay',
                  ].join(' ')}
                >
                  <div className="w-2.5 h-2.5 bg-gradient-to-r from-state-pass via-state-warn to-state-alert rounded-sm" />
                  Heatmap
                </button>
                <div className="w-px h-4 bg-surface-border mx-2" />
                <button
                  onClick={() => setZoom(z => Math.max(z - 0.25, 0.5))}
                  className="p-1.5 rounded hover:bg-surface-overlay text-text-tertiary transition-colors"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <div className="w-12 text-center text-[11px] font-mono text-text-tertiary">
                  {Math.round(zoom * 100)}%
                </div>
                <button
                  onClick={() => setZoom(z => Math.min(z + 0.25, 3))}
                  className="p-1.5 rounded hover:bg-surface-overlay text-text-tertiary transition-colors"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoom(1)}
                  className="p-1.5 rounded hover:bg-surface-overlay text-text-tertiary transition-colors ml-1"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* RIGHT SIDEBAR */}
            <div className="w-80 bg-[#141416] border-l border-surface-border flex flex-col flex-shrink-0">

              {/* Inference telemetry */}
              {currentSample && (
                <div className="p-5 border-b border-surface-border">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-[10px] uppercase tracking-widest text-text-tertiary font-bold">
                      Inference Telemetry
                    </h3>
                    <span className="px-2 py-0.5 bg-surface-overlay border border-surface-border rounded text-[10px] font-mono text-text-tertiary uppercase tracking-widest">
                      {currentSample.routing_case.split('_').slice(-1)[0]}
                    </span>
                  </div>

                  {/* BGAD score */}
                  <div className="mb-4 bg-[#0C0C0E] border border-surface-border rounded p-3">
                    <div className="flex items-end justify-between mb-2">
                      <span className="text-[11px] font-medium text-text-tertiary">BGAD Score</span>
                      <span className={[
                        'text-[11px] font-mono font-bold',
                        currentSample.bgad_score > 0.7 ? 'text-state-alert' :
                        currentSample.bgad_score > 0.4 ? 'text-state-warn' :
                        'text-state-pass',
                      ].join(' ')}>
                        {currentSample.bgad_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="w-full h-0.5 bg-surface-border rounded-full overflow-hidden">
                      <div
                        className={[
                          'h-full',
                          currentSample.bgad_score > 0.7 ? 'bg-state-alert' :
                          currentSample.bgad_score > 0.4 ? 'bg-state-warn' :
                          'bg-state-pass',
                        ].join(' ')}
                        style={{ width: `${Math.min(currentSample.bgad_score * 100, 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* GPT-4V reasoning */}
                  <div className="bg-[#0C0C0E] border border-purple-500/20 rounded p-3 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-0.5 h-full bg-purple-500/50" />
                    <div className="flex items-center gap-1.5 mb-2 pl-2">
                      <Sparkles className="w-3 h-3 text-purple-400" />
                      <span className="text-[11px] font-medium text-purple-400 uppercase tracking-wide">
                        GPT-4o Analysis
                      </span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-text-tertiary font-mono whitespace-pre-wrap pl-2">
                      {vlmReasoning}
                    </p>
                  </div>

                  <div className="mt-4">
                    <span className="text-[10px] text-text-disabled uppercase tracking-widest">
                      Image ID
                    </span>
                    <p className="text-[10px] font-mono text-text-tertiary truncate mt-0.5">
                      {currentSample.image_id}
                    </p>
                  </div>
                </div>
              )}

              {/* Declared regions */}
              <div className="flex-1 overflow-auto p-5 border-b border-surface-border">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-tertiary font-bold">
                    Declared Regions
                  </h3>
                  <span className="text-[10px] font-mono bg-surface-overlay px-1.5 py-0.5 rounded">
                    {boundingBoxes.length}
                  </span>
                </div>

                {boundingBoxes.length === 0 ? (
                  <div className="h-24 border border-dashed border-surface-border rounded flex items-center justify-center bg-[#0C0C0E]/50">
                    <span className="text-[10px] font-mono text-text-disabled uppercase tracking-widest">
                      No boxes drawn
                    </span>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {boundingBoxes.map(box => {
                      const cat = DEFECT_CATEGORIES.find(c => c.name === box.defect_type);
                      return (
                        <div
                          key={box.id}
                          onClick={() => setSelectedBoxId(box.id)}
                          className={[
                            'p-3 rounded cursor-pointer border transition-colors',
                            selectedBoxId === box.id
                              ? 'bg-surface-overlay border-surface-borderhover'
                              : 'bg-[#0C0C0E] border-surface-border hover:border-surface-borderhover',
                          ].join(' ')}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 rounded-sm" style={{ backgroundColor: cat?.color }} />
                              <span className="text-[11px] font-medium uppercase tracking-wide text-text-primary">
                                {box.defect_type}
                              </span>
                            </div>
                            <button
                              onClick={e => {
                                e.stopPropagation();
                                setBoundingBoxes(prev => prev.filter(b => b.id !== box.id));
                              }}
                              className="text-text-disabled hover:text-state-alert transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                          <div className="text-[10px] text-text-tertiary font-mono bg-surface-raised border border-surface-border rounded px-2 py-1 inline-block">
                            X:{Math.round(box.x)} Y:{Math.round(box.y)} · W:{Math.round(box.width)} H:{Math.round(box.height)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Action area */}
              <div className="p-5 flex-shrink-0 bg-[#0C0C0E] border-t border-surface-border">
                {/* Session counter */}
                <p className="text-[11px] text-text-tertiary mb-3">
                  {sessionReviewCount} reviewed · {Math.max(0, stats.queueSize - sessionReviewCount)} remaining
                </p>

                <textarea
                  value={notes}
                  onChange={e => setNotes(e.target.value)}
                  placeholder="Override notes..."
                  className="w-full h-14 bg-surface-raised border border-surface-border rounded p-3 text-xs resize-none focus:outline-none focus:border-kul-accent/50 text-text-primary font-mono mb-3"
                />

                {/* Prev / Skip / Next */}
                <div className="flex gap-2 mb-3">
                  <button
                    onClick={() => { setCurrentIndex(i => Math.max(0, i - 1)); setBoundingBoxes([]); }}
                    disabled={currentIndex === 0}
                    className="flex-1 py-1.5 bg-surface-raised border border-surface-border text-text-tertiary rounded hover:bg-surface-overlay disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-mono text-[10px] uppercase tracking-widest flex items-center justify-center gap-1"
                  >
                    <ChevronLeft className="w-3 h-3" /> Prev
                  </button>
                  <button
                    onClick={handleSkip}
                    className="flex-1 py-1.5 bg-surface-raised border border-surface-border text-text-tertiary rounded hover:bg-surface-overlay transition-colors font-mono text-[10px] uppercase tracking-widest flex items-center justify-center gap-1"
                  >
                    <SkipForward className="w-3 h-3" /> Skip
                  </button>
                  <button
                    onClick={() => { setCurrentIndex(i => Math.min(samples.length - 1, i + 1)); setBoundingBoxes([]); }}
                    disabled={currentIndex === samples.length - 1}
                    className="flex-1 py-1.5 bg-surface-raised border border-surface-border text-text-tertiary rounded hover:bg-surface-overlay disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-mono text-[10px] uppercase tracking-widest flex items-center justify-center gap-1"
                  >
                    Next <ChevronRight className="w-3 h-3" />
                  </button>
                </div>

                {/* Label buttons */}
                <div className="grid grid-cols-3 gap-2 mb-3">
                  {([
                    { label: 'normal' as const,    icon: Check,         colorClass: 'bg-state-passSubtle hover:bg-state-pass/20 border-state-pass/30 text-state-pass',   kbd: '↑' },
                    { label: 'uncertain' as const,  icon: HelpCircle,    colorClass: 'bg-state-warnSubtle hover:bg-state-warn/20 border-state-warn/30 text-state-warn',   kbd: '?' },
                    { label: 'anomaly' as const,    icon: AlertTriangle, colorClass: 'bg-state-alertSubtle hover:bg-state-alert/20 border-state-alert/30 text-state-alert', kbd: '!' },
                  ] as const).map(({ label, icon: Icon, colorClass }) => (
                    <button
                      key={label}
                      onClick={() => setPendingLabel(label)}
                      className={[
                        'py-2.5 border rounded font-semibold text-[10px] uppercase tracking-widest transition-colors flex flex-col items-center justify-center gap-1',
                        colorClass,
                        pendingLabel === label ? 'ring-1 ring-current' : '',
                      ].join(' ')}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{label === 'uncertain' ? 'Unclear' : label === 'normal' ? 'Approve' : 'Reject'}</span>
                    </button>
                  ))}
                </div>

                {/* Pending label indicator */}
                {pendingLabel && (
                  <p className="text-[10px] text-text-tertiary uppercase tracking-wider mb-2 text-center">
                    Will submit as:{' '}
                    <span className={[
                      'font-bold',
                      pendingLabel === 'normal' ? 'text-state-pass' :
                      pendingLabel === 'uncertain' ? 'text-state-warn' : 'text-state-alert',
                    ].join(' ')}>
                      {pendingLabel.toUpperCase()}
                    </span>
                  </p>
                )}

                <button
                  onClick={() => pendingLabel && submitAnnotation(pendingLabel)}
                  disabled={!pendingLabel}
                  className="w-full py-3 bg-kul-blue hover:bg-kul-light rounded font-bold text-xs uppercase tracking-widest text-white transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Save className="w-4 h-4" />
                  Submit Annotation
                  <kbd className="ml-1 text-[9px] border border-white/20 rounded px-1">↵</kbd>
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
