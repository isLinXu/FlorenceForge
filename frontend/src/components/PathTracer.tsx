import React, { useState, useEffect, useRef, useCallback } from 'react';

// ─── Types ─────────────────────────────────────────────────────────
interface PathData {
  points: Array<[number, number]>;
  imageSrc?: string;
}

interface PathTracerProps {
  pathData: PathData;
  className?: string;
}

// ─── Playback State ────────────────────────────────────────────────
type PlaybackState = 'playing' | 'paused' | 'finished';

// ─── Simple SVG Icons ────────────────────────────────────────────
const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M8 5v14l11-7z" />
  </svg>
);
const PauseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
  </svg>
);
const RewindIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M11 18V6l-8.5 6L11 18zm8.5-6L11 6v12l8.5-6z" />
  </svg>
);

// ─── Color interpolation: blue → red ─────────────────────────────
function interpolateColor(t: number): string {
  // t: 0 = blue, 1 = red
  const r = Math.round(14 + t * (239 - 14));
  const g = Math.round(165 + t * (68 - 165));
  const b = Math.round(233 + t * (68 - 233));
  return `rgb(${r}, ${g}, ${b})`;
}

// ─── Component ─────────────────────────────────────────────────────
export const PathTracer: React.FC<PathTracerProps> = ({ pathData, className = '' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const [playback, setPlayback] = useState<PlaybackState>('paused');
  const [progress, setProgress] = useState(0); // 0 - 100
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 500 });

  const { points, imageSrc } = pathData;
  const totalPoints = points.length;

  // Load background image
  useEffect(() => {
    if (!imageSrc) {
      setImageLoaded(false);
      imageRef.current = null;
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      setImageLoaded(true);
      // Adjust canvas size to image aspect ratio (with max bounds)
      const maxW = 900;
      const maxH = 600;
      const ratio = Math.min(maxW / img.width, maxH / img.height, 1);
      setCanvasSize({
        width: Math.round(img.width * ratio),
        height: Math.round(img.height * ratio),
      });
    };
    img.src = imageSrc;
  }, [imageSrc]);

  // Compute point bounds and scaling
  const pointMetrics = React.useMemo(() => {
    if (points.length === 0) return { scale: 1, offsetX: 0, offsetY: 0, minX: 0, minY: 0, maxX: 0, maxY: 0 };
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    const pad = 40;
    const availW = canvasSize.width - pad * 2;
    const availH = canvasSize.height - pad * 2;
    const dataW = maxX - minX || 1;
    const dataH = maxY - minY || 1;
    const scale = Math.min(availW / dataW, availH / dataH, 1);
    const offsetX = (canvasSize.width - dataW * scale) / 2 - minX * scale;
    const offsetY = (canvasSize.height - dataH * scale) / 2 - minY * scale;
    return { scale, offsetX, offsetY, minX, minY, maxX, maxY };
  }, [points, canvasSize.width, canvasSize.height]);

  // Drawing function
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvasSize.width;
    const h = canvasSize.height;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    // Background image
    if (imageRef.current && imageLoaded) {
      ctx.drawImage(imageRef.current, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#fafaf9'; // warm-50
      ctx.fillRect(0, 0, w, h);
    }

    if (points.length < 2) return;

    const { scale, offsetX, offsetY } = pointMetrics;
    const floatIndex = (progress / 100) * (totalPoints - 1);
    const currentIdx = Math.floor(floatIndex);
    const frac = floatIndex - currentIdx;

    // Draw completed path segments (gradually fading in)
    const completedSegments = Math.floor(floatIndex);
    for (let i = 0; i < completedSegments && i + 1 < totalPoints; i++) {
      const [x1, y1] = points[i];
      const [x2, y2] = points[i + 1];
      const sx1 = x1 * scale + offsetX;
      const sy1 = y1 * scale + offsetY;
      const sx2 = x2 * scale + offsetX;
      const sy2 = y2 * scale + offsetY;

      const segmentProgress = i / Math.max(totalPoints - 1, 1);
      const color = interpolateColor(segmentProgress);
      const alpha = 0.3 + 0.7 * (i / Math.max(completedSegments, 1));

      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw current partial segment
    if (currentIdx + 1 < totalPoints) {
      const [x1, y1] = points[currentIdx];
      const [x2, y2] = points[currentIdx + 1];
      const sx1 = x1 * scale + offsetX;
      const sy1 = y1 * scale + offsetY;
      const sx2 = x2 * scale + offsetX;
      const sy2 = y2 * scale + offsetY;

      const curX = sx1 + (sx2 - sx1) * frac;
      const curY = sy1 + (sy2 - sy1) * frac;

      const segmentProgress = currentIdx / Math.max(totalPoints - 1, 1);
      const color = interpolateColor(segmentProgress);

      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(curX, curY);
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw current point with blinking highlight
      const pulse = (Date.now() % 800) / 800;
      const radius = 6 + pulse * 6;
      const glowAlpha = 0.4 + pulse * 0.4;

      ctx.fillStyle = `rgba(255, 255, 255, ${glowAlpha})`;
      ctx.beginPath();
      ctx.arc(curX, curY, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(curX, curY, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(curX, curY, 5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw all points as small dots (faded for future, full for past)
    for (let i = 0; i < totalPoints; i++) {
      const [x, y] = points[i];
      const sx = x * scale + offsetX;
      const sy = y * scale + offsetY;
      const pointProgress = i / Math.max(totalPoints - 1, 1);
      const color = interpolateColor(pointProgress);

      const isPast = i <= currentIdx;
      const isCurrent = i === currentIdx;

      if (isCurrent) continue; // already drawn above

      const alpha = isPast ? 0.8 : 0.2;
      const radius = isPast ? 3 : 2;

      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Draw start (blue) and end (red) labels
    const [sx, sy] = points[0];
    const [ex, ey] = points[totalPoints - 1];
    const startX = sx * scale + offsetX;
    const startY = sy * scale + offsetY;
    const endX = ex * scale + offsetX;
    const endY = ey * scale + offsetY;

    // Start label
    ctx.fillStyle = '#0ea5e9';
    ctx.beginPath();
    ctx.arc(startX, startY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('S', startX, startY + 1);

    // End label
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(endX, endY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.fillText('E', endX, endY + 1);
  }, [points, totalPoints, progress, canvasSize, imageLoaded, pointMetrics]);

  // Animation loop
  useEffect(() => {
    let running = true;

    function loop(ts: number) {
      if (!running) return;
      draw();

      if (playback === 'playing') {
        const elapsed = lastTimeRef.current ? ts - lastTimeRef.current : 0;
        const stepSize = 0.15; // progress increment per frame at 1x
        if (elapsed > 16) { // ~60fps
          setProgress((prev) => {
            const next = prev + stepSize;
            if (next >= 100) {
              setPlayback('finished');
              return 100;
            }
            return next;
          });
          lastTimeRef.current = ts;
        }
      }

      animRef.current = requestAnimationFrame(loop);
    }

    animRef.current = requestAnimationFrame(loop);
    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
    };
  }, [playback, draw]);

  useEffect(() => {
    if (playback === 'playing') {
      lastTimeRef.current = 0;
    }
  }, [playback]);

  const handlePlay = () => {
    if (playback === 'finished') {
      setProgress(0);
      setPlayback('playing');
    } else {
      setPlayback('playing');
    }
  };
  const handlePause = () => setPlayback('paused');
  const handleReset = () => {
    setPlayback('paused');
    setProgress(0);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPlayback('paused');
    setProgress(parseFloat(e.target.value));
  };

  return (
    <div className={`panel ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-warm-900">Path Trace</h3>
        <div className="flex items-center gap-1.5 text-xs text-warm-600">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-primary-500" /> Start
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-500 ml-2" /> End
        </div>
      </div>

      <div className="overflow-auto rounded-lg border border-warm-200 bg-warm-50">
        <canvas ref={canvasRef} className="block mx-auto" />
      </div>

      {/* Controls */}
      <div className="mt-4">
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-warm-100 text-warm-700 hover:bg-warm-200 transition-colors"
            title="Reset"
          >
            <RewindIcon />
          </button>
          {playback === 'playing' ? (
            <button
              onClick={handlePause}
              className="p-2.5 rounded-lg bg-primary-600 text-white hover:bg-primary-700 transition-colors"
              title="Pause"
            >
              <PauseIcon />
            </button>
          ) : (
            <button
              onClick={handlePlay}
              className="p-2.5 rounded-lg bg-primary-600 text-white hover:bg-primary-700 transition-colors"
              title="Play"
            >
              <PlayIcon />
            </button>
          )}
          <span className="text-xs text-warm-500 font-medium ml-2">
            {playback === 'playing' ? 'Playing' : playback === 'finished' ? 'Finished' : 'Paused'}
          </span>
        </div>

        {/* Timeline slider */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-warm-500 font-mono w-10 text-right">0%</span>
          <input
            type="range"
            min={0}
            max={100}
            step={0.1}
            value={progress}
            onChange={handleSliderChange}
            className="flex-1 h-1.5 bg-warm-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
            style={{ accentColor: '#0ea5e9' }}
          />
          <span className="text-xs text-warm-500 font-mono w-12">{progress.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};

export default PathTracer;
