import React, { useState, useEffect, useRef, useCallback } from 'react';

// ─── Types ─────────────────────────────────────────────────────────
interface MazeData {
  grid: boolean[][];      // true = wall, false = passage
  start: [number, number];
  end: [number, number];
  path: Array<[number, number]>; // solution path
}

interface MazePlayerProps {
  mazeData: MazeData;
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
const StepForwardIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z" />
  </svg>
);
const StepBackIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M11 18V6l-8.5 6L11 18zm8.5-6L11 6v12l8.5-6z" />
  </svg>
);
const RewindIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M11 18V6l-8.5 6L11 18zm8.5-6L11 6v12l8.5-6z" />
  </svg>
);

// ─── Component ─────────────────────────────────────────────────────
export const MazePlayer: React.FC<MazePlayerProps> = ({ mazeData, className = '' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  const [playback, setPlayback] = useState<PlaybackState>('paused');
  const [stepIndex, setStepIndex] = useState(0);
  const [speed, setSpeed] = useState(1); // 1x, 2x, 4x
  const [cellSize, setCellSize] = useState(24);
  const [padding, setPadding] = useState({ x: 20, y: 20 });

  const { grid, start, end, path } = mazeData;
  const rows = grid.length;
  const cols = grid[0]?.length || 0;

  // Generate DFS exploration trace for animation
  const explorationTrace = React.useMemo(() => {
    const trace: Array<{
      pos: [number, number];
      type: 'try' | 'dead' | 'final';
      from?: [number, number];
    }> = [];

    if (path.length === 0) return trace;

    const key = (r: number, c: number) => `${r},${c}`;
    const visitedInDFS = new Set<string>();

    function dfs(r: number, c: number): boolean {
      const k = key(r, c);
      visitedInDFS.add(k);
      trace.push({ pos: [r, c], type: 'try' });

      if (r === end[0] && c === end[1]) return true;

      const dirs = [
        [0, 1], [1, 0], [0, -1], [-1, 0],
      ];
      for (const [dr, dc] of dirs) {
        const nr = r + dr;
        const nc = c + dc;
        const nk = key(nr, nc);
        if (
          nr >= 0 && nr < rows &&
          nc >= 0 && nc < cols &&
          !grid[nr][nc] &&
          !visitedInDFS.has(nk)
        ) {
          if (dfs(nr, nc)) return true;
          trace.push({ pos: [nr, nc], type: 'dead' });
        }
      }
      return false;
    }

    dfs(start[0], start[1]);

    // Mark final path
    const finalPath = new Set<string>();
    for (const p of path) finalPath.add(key(p[0], p[1]));

    // Re-annotate: anything in final path that was tried is 'final'
    for (const t of trace) {
      if (finalPath.has(key(t.pos[0], t.pos[1]))) {
        t.type = 'final';
      }
    }

    return trace;
  }, [grid, start, end, path, rows, cols]);

  const totalSteps = explorationTrace.length;

  // Compute cell size based on canvas/container size
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const maxW = parent.clientWidth - 40;
    const maxH = 500;
    const cs = Math.min(Math.floor(maxW / cols), Math.floor(maxH / rows), 40);
    setCellSize(Math.max(cs, 12));
    setPadding({
      x: Math.max(20, (maxW - cs * cols) / 2),
      y: 20,
    });
  }, [cols, rows]);

  // Drawing function
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = padding.x * 2 + cellSize * cols;
    const h = padding.y * 2 + cellSize * rows;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#fafaf9'; // warm-50
    ctx.fillRect(0, 0, w, h);

    // Draw grid
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = padding.x + c * cellSize;
        const y = padding.y + r * cellSize;
        if (grid[r][c]) {
          // Wall
          ctx.fillStyle = '#292524'; // warm-800
          ctx.fillRect(x, y, cellSize, cellSize);
        } else {
          // Passage
          ctx.fillStyle = '#f5f5f4'; // warm-100
          ctx.fillRect(x, y, cellSize, cellSize);
          ctx.strokeStyle = '#e7e5e4'; // warm-200
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, cellSize, cellSize);
        }
      }
    }

    // Draw exploration trace up to current step
    const currentTrace = explorationTrace.slice(0, stepIndex);
    const pathWidth = Math.max(2, cellSize * 0.15);

    for (let i = 0; i < currentTrace.length; i++) {
      const t = currentTrace[i];
      const cx = padding.x + t.pos[1] * cellSize + cellSize / 2;
      const cy = padding.y + t.pos[0] * cellSize + cellSize / 2;

      // Draw line segment to previous
      if (i > 0) {
        const prev = currentTrace[i - 1];
        const px = padding.x + prev.pos[1] * cellSize + cellSize / 2;
        const py = padding.y + prev.pos[0] * cellSize + cellSize / 2;

        let strokeColor = '#38bdf8'; // try - primary-400
        if (t.type === 'dead') strokeColor = '#ef4444'; // red
        if (t.type === 'final') strokeColor = '#22c55e'; // green
        if (prev.type === 'final' && t.type === 'final') strokeColor = '#22c55e';

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = pathWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(cx, cy);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Draw current point marker (pulsing)
    if (stepIndex > 0 && stepIndex <= explorationTrace.length) {
      const t = explorationTrace[stepIndex - 1];
      const cx = padding.x + t.pos[1] * cellSize + cellSize / 2;
      const cy = padding.y + t.pos[0] * cellSize + cellSize / 2;

      const pulse = (Date.now() % 1000) / 1000;
      const radius = (cellSize * 0.25) + (pulse * 3);
      const alpha = 0.6 + (pulse * 0.4);

      ctx.fillStyle = t.type === 'dead' ? 'rgba(239, 68, 68,' : t.type === 'final' ? 'rgba(34, 197, 94,' : 'rgba(56, 189, 248,';
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Point primitive (small dot)
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw start and end markers
    const [sr, sc] = start;
    const [er, ec] = end;
    const sx = padding.x + sc * cellSize + cellSize / 2;
    const sy = padding.y + sr * cellSize + cellSize / 2;
    const ex = padding.x + ec * cellSize + cellSize / 2;
    const ey = padding.y + er * cellSize + cellSize / 2;

    // Start marker
    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.arc(sx, sy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('S', sx, sy);

    // End marker
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(ex, ey, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.fillText('E', ex, ey);
  }, [grid, rows, cols, cellSize, padding, explorationTrace, stepIndex, start, end]);

  // Animation loop
  useEffect(() => {
    let running = true;

    function loop(ts: number) {
      if (!running) return;
      draw();

      if (playback === 'playing') {
        const elapsed = lastTimeRef.current ? ts - lastTimeRef.current : 0;
        const stepInterval = 400 / speed; // ms per step at 1x
        if (elapsed > stepInterval) {
          setStepIndex((prev) => {
            if (prev >= totalSteps) {
              setPlayback('finished');
              return prev;
            }
            return prev + 1;
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
  }, [playback, speed, totalSteps, draw]);

  // Reset lastTime when playback starts
  useEffect(() => {
    if (playback === 'playing') {
      lastTimeRef.current = 0;
    }
  }, [playback]);

  const handlePlay = () => {
    if (playback === 'finished') {
      setStepIndex(0);
      setPlayback('playing');
    } else {
      setPlayback('playing');
    }
  };
  const handlePause = () => setPlayback('paused');
  const handleStepForward = () => {
    setPlayback('paused');
    setStepIndex((p) => Math.min(p + 1, totalSteps));
  };
  const handleStepBack = () => {
    setPlayback('paused');
    setStepIndex((p) => Math.max(p - 1, 0));
  };
  const handleReset = () => {
    setPlayback('paused');
    setStepIndex(0);
  };

  return (
    <div className={`panel ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-warm-900">Maze Navigation</h3>
        <div className="flex items-center gap-1.5">
          {/* Legend */}
          <div className="flex items-center gap-1.5 text-xs text-warm-600 mr-3">
            <span className="inline-block w-3 h-0.5 bg-primary-400 rounded" /> Try
            <span className="inline-block w-3 h-0.5 bg-red-500 rounded ml-2" /> Dead
            <span className="inline-block w-3 h-0.5 bg-green-500 rounded ml-2" /> Final
          </div>
        </div>
      </div>

      <div className="overflow-auto rounded-lg border border-warm-200 bg-warm-50">
        <canvas ref={canvasRef} className="block mx-auto" />
      </div>

      {/* Controls */}
      <div className="mt-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            className="p-2 rounded-lg bg-warm-100 text-warm-700 hover:bg-warm-200 transition-colors"
            title="Reset"
          >
            <RewindIcon />
          </button>
          <button
            onClick={handleStepBack}
            className="p-2 rounded-lg bg-warm-100 text-warm-700 hover:bg-warm-200 transition-colors"
            title="Step Back"
          >
            <StepBackIcon />
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
          <button
            onClick={handleStepForward}
            className="p-2 rounded-lg bg-warm-100 text-warm-700 hover:bg-warm-200 transition-colors"
            title="Step Forward"
          >
            <StepForwardIcon />
          </button>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-warm-500 font-medium">Speed</span>
          <div className="flex items-center gap-1 bg-warm-100 rounded-lg p-0.5">
            {[1, 2, 4].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
                  speed === s
                    ? 'bg-white text-primary-700 shadow-sm'
                    : 'text-warm-500 hover:text-warm-700'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="mt-3">
        <div className="w-full h-1.5 bg-warm-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary-500 rounded-full transition-all duration-200"
            style={{ width: `${totalSteps > 0 ? (stepIndex / totalSteps) * 100 : 0}%` }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-xs text-warm-500">Step {stepIndex} / {totalSteps}</span>
          <span className="text-xs text-warm-500">
            {playback === 'playing' ? 'Playing' : playback === 'finished' ? 'Finished' : 'Paused'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default MazePlayer;
