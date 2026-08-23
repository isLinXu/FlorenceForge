import { useRef, useEffect, useState, useCallback } from 'react';
import type { StepRecord } from '../types/agentic';

interface VisualOverlayProps {
  imageSrc: string;
  step: StepRecord;
}

const INTENT_COLORS: Record<string, string> = {
  detect: '#ef4444',
  read_text: '#22c55e',
  locate: '#3b82f6',
  describe: '#a855f7',
  default: '#78716c',
};

function getIntentColor(intent: string): string {
  return INTENT_COLORS[intent] ?? INTENT_COLORS.default;
}

function scaleBox(box: number[], imgW: number, imgH: number): [number, number, number, number] {
  const [x1, y1, x2, y2] = box;
  return [
    (x1 / 999) * imgW,
    (y1 / 999) * imgH,
    (x2 / 999) * imgW,
    (y2 / 999) * imgH,
  ];
}

export function VisualOverlay({ imageSrc, step }: VisualOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showOverlay, setShowOverlay] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageSize, setImageSize] = useState({ w: 0, h: 0 });

  const loadImage = useCallback(() => {
    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      imageRef.current = img;
      setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
      setImageLoaded(true);
      setScale(1);
      setPan({ x: 0, y: 0 });
    };
    img.onerror = () => {
      console.error('Failed to load image:', imageSrc.slice(0, 60));
    };
  }, [imageSrc]);

  useEffect(() => {
    loadImage();
  }, [loadImage]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !imageRef.current || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const cssW = rect.width;
    const cssH = rect.height;

    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = '#fafaf9';
    ctx.fillRect(0, 0, cssW, cssH);

    const img = imageRef.current;
    const imgW = img.naturalWidth;
    const imgH = img.naturalHeight;

    const scaleX = cssW / imgW;
    const scaleY = cssH / imgH;
    const baseScale = Math.min(scaleX, scaleY);

    const totalScale = baseScale * scale;

    const drawW = imgW * totalScale;
    const drawH = imgH * totalScale;
    const offsetX = (cssW - drawW) / 2 + pan.x;
    const offsetY = (cssH - drawH) / 2 + pan.y;

    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(totalScale, totalScale);

    ctx.drawImage(img, 0, 0, imgW, imgH);

    if (showOverlay && step.parsed) {
      const color = getIntentColor(step.intent);
      ctx.strokeStyle = color;
      ctx.fillStyle = color + '20';
      ctx.lineWidth = 2.5 / totalScale;
      ctx.lineJoin = 'round';

      const parsed = step.parsed as Record<string, unknown>;

      if (step.intent === 'detect' && Array.isArray(parsed.objects)) {
        for (const obj of parsed.objects) {
          if (obj && typeof obj === 'object' && 'box' in obj) {
            const box = (obj as Record<string, unknown>).box as number[];
            if (Array.isArray(box) && box.length >= 4) {
              const [sx1, sy1, sx2, sy2] = scaleBox(box, imgW, imgH);
              ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
              ctx.fillRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
              const label = (obj as Record<string, unknown>).label as string | undefined;
              if (label) {
                ctx.fillStyle = color + 'e0';
                ctx.font = `${Math.max(12, 14 / totalScale)}px sans-serif`;
                ctx.fillText(label, sx1 + 4, sy1 - 4);
                ctx.fillStyle = color + '20';
              }
            }
          }
        }
      }

      if (step.intent === 'read_text' && Array.isArray(parsed.regions)) {
        for (const region of parsed.regions) {
          if (region && typeof region === 'object' && 'polygon' in region) {
            const polygon = (region as Record<string, unknown>).polygon as number[];
            if (Array.isArray(polygon) && polygon.length >= 6) {
              ctx.beginPath();
              const pxs = polygon.filter((_, i) => i % 2 === 0).map((n) => (n as number / 999) * imgW);
              const pys = polygon.filter((_, i) => i % 2 === 1).map((n) => (n as number / 999) * imgH);
              ctx.moveTo(pxs[0], pys[0]);
              for (let i = 1; i < pxs.length; i++) {
                ctx.lineTo(pxs[i], pys[i]);
              }
              ctx.closePath();
              ctx.stroke();
              ctx.fill();
            }
          }
        }
      }

      if (step.intent === 'locate' && 'box' in parsed) {
        const box = parsed.box as number[];
        if (Array.isArray(box) && box.length >= 4) {
          const [sx1, sy1, sx2, sy2] = scaleBox(box, imgW, imgH);
          ctx.setLineDash([6 / totalScale, 4 / totalScale]);
          ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
          ctx.setLineDash([]);
          ctx.fillStyle = color + 'e0';
          ctx.font = `${Math.max(12, 14 / totalScale)}px sans-serif`;
          ctx.fillText(step.tool_call.task_name, sx1 + 4, sy1 - 4);
        }
      }

      if (step.intent === 'describe' && 'regions' in parsed) {
        const regions = parsed.regions as Array<Record<string, unknown>>;
        if (Array.isArray(regions)) {
          for (const region of regions) {
            if ('box' in region) {
              const box = region.box as number[];
              if (Array.isArray(box) && box.length >= 4) {
                const [sx1, sy1, sx2, sy2] = scaleBox(box, imgW, imgH);
                ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
              }
            }
          }
        }
      }
    }

    ctx.restore();
  }, [imageSrc, step, scale, pan, showOverlay, imageLoaded]);

  useEffect(() => {
    let rafId: number;
    const loop = () => {
      draw();
      rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [draw]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setScale((prev) => Math.max(0.2, Math.min(5, prev * delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const resetView = useCallback(() => {
    setScale(1);
    setPan({ x: 0, y: 0 });
  }, []);

  return (
    <div className="panel flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: getIntentColor(step.intent) }}
          />
          <h3 className="text-sm font-semibold text-warm-800 tracking-wide">
            {step.tool_call.task_name}
          </h3>
          <span className="text-xs text-warm-400 font-mono">
            #{step.sub_task_index}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlay((v) => !v)}
            className="text-xs px-2.5 py-1 rounded-md border border-warm-200 bg-warm-50 text-warm-600 hover:bg-warm-100 transition-colors"
          >
            {showOverlay ? '隐藏 Overlay' : '显示 Overlay'}
          </button>
          <button
            onClick={resetView}
            className="text-xs px-2.5 py-1 rounded-md border border-warm-200 bg-warm-50 text-warm-600 hover:bg-warm-100 transition-colors"
          >
            重置视图
          </button>
        </div>
      </div>
      <div
        ref={containerRef}
        className="relative w-full h-96 rounded-lg overflow-hidden border border-warm-200 bg-warm-50 cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        {!imageLoaded && (
          <div className="absolute inset-0 flex items-center justify-center text-warm-400">
            <div className="flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-2 border-warm-300 border-t-primary-500 rounded-full animate-spin" />
              <span className="text-xs">加载图像中...</span>
            </div>
          </div>
        )}
      </div>
      <div className="flex items-center justify-between text-xs text-warm-500">
        <span className="font-mono">
          缩放: {(scale * 100).toFixed(0)}% | 平移: ({pan.x.toFixed(0)}, {pan.y.toFixed(0)})
        </span>
        <span className="font-mono">
          {imageSize.w}×{imageSize.h} px
        </span>
      </div>
    </div>
  );
}
