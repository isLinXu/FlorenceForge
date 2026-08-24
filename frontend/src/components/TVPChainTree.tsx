import React, { useState, useEffect, useRef, useCallback } from 'react';

// ─── Types ─────────────────────────────────────────────────────────
interface TreeNode {
  tag: string;
  title: string;
  content: string;
  children: TreeNode[];
  expanded: boolean;
  depth: number;
}

interface TVPChainTreeProps {
  transcript: string;
  className?: string;
}

// ─── Parser ──────────────────────────────────────────────────────────
const TAG_COLORS: Record<string, string> = {
  PLAN: '#0284c7',    // primary-600
  ACT: '#0ea5e9',      // primary-500
  VERIFY: '#38bdf8',   // primary-400
  REFLECT: '#a855f7',  // purple-500
  DECIDE: '#22c55e',   // green-500
};

function parseTranscript(transcript: string): TreeNode {
  const regex = /<(PLAN|ACT|VERIFY|REFLECT|DECIDE)>([\s\S]*?)(?=<(?:PLAN|ACT|VERIFY|REFLECT|DECIDE)>|$)/g;
  const matches: Array<{ tag: string; content: string }> = [];
  let m;
  while ((m = regex.exec(transcript)) !== null) {
    matches.push({ tag: m[1], content: m[2].trim() });
  }

  // Fallback: if no tags found, wrap whole text as PLAN
  if (matches.length === 0) {
    return {
      tag: 'PLAN',
      title: 'PLAN',
      content: transcript.trim(),
      children: [],
      expanded: true,
      depth: 0,
    };
  }

  // Build tree: first PLAN is root, subsequent tags chain as children
  const rootMatch = matches[0];
  const root: TreeNode = {
    tag: rootMatch.tag,
    title: rootMatch.tag,
    content: rootMatch.content,
    children: [],
    expanded: true,
    depth: 0,
  };

  let current = root;
  for (let i = 1; i < matches.length; i++) {
    const child: TreeNode = {
      tag: matches[i].tag,
      title: matches[i].tag,
      content: matches[i].content,
      children: [],
      expanded: true,
      depth: current.depth + 1,
    };
    current.children.push(child);
    current = child;
  }
  return root;
}

// ─── Layout Engine ───────────────────────────────────────────────────
interface LayoutNode {
  id: string;
  tag: string;
  title: string;
  content: string;
  x: number;
  y: number;
  width: number;
  height: number;
  children: LayoutNode[];
  parent?: LayoutNode;
  expanded: boolean;
  depth: number;
}

const NODE_WIDTH = 180;
const NODE_HEIGHT = 44;
const H_GAP = 80;
const V_GAP = 80;

function buildLayout(
  node: TreeNode,
  parent?: LayoutNode,
  idPrefix = '0'
): LayoutNode {
  const ln: LayoutNode = {
    id: idPrefix,
    tag: node.tag,
    title: node.title,
    content: node.content,
    x: 0,
    y: node.depth * V_GAP,
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
    children: [],
    parent,
    expanded: node.expanded,
    depth: node.depth,
  };

  if (node.expanded && node.children.length > 0) {
    ln.children = node.children.map((child, i) =>
      buildLayout(child, ln, `${idPrefix}-${i}`)
    );
  }
  return ln;
}

function computePositions(root: LayoutNode): LayoutNode {
  // Simple horizontal layout: root at center, children fan out
  let leafCount = 0;

  function assignX(node: LayoutNode, baseX: number): number {
    if (!node.expanded || node.children.length === 0) {
      node.x = baseX + leafCount * (NODE_WIDTH + H_GAP);
      leafCount++;
      return node.x;
    }
    const childXs = node.children.map((c) => assignX(c, baseX));
    node.x = childXs.reduce((a, b) => a + b, 0) / childXs.length;
    return node.x;
  }

  assignX(root, 0);
  return root;
}

// ─── Component ───────────────────────────────────────────────────────
export const TVPChainTree: React.FC<TVPChainTreeProps> = ({
  transcript,
  className = '',
}) => {
  const [treeData, setTreeData] = useState<TreeNode>(() => parseTranscript(transcript));
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [animating, setAnimating] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const [svgSize, setSvgSize] = useState({ width: 800, height: 400 });

  useEffect(() => {
    setTreeData(parseTranscript(transcript));
  }, [transcript]);

  // Recompute layout whenever treeData changes
  const layout = React.useMemo(() => {
    const root = buildLayout(treeData);
    const laid = computePositions(root);
    // Compute bounding box
    let minX = Infinity, maxX = -Infinity, maxY = -Infinity;
    function visit(node: LayoutNode) {
      minX = Math.min(minX, node.x - NODE_WIDTH / 2);
      maxX = Math.max(maxX, node.x + NODE_WIDTH / 2);
      maxY = Math.max(maxY, node.y + NODE_HEIGHT / 2);
      if (node.expanded) node.children.forEach(visit);
    }
    visit(laid);
    const width = Math.max(800, maxX - minX + 80);
    const height = Math.max(300, maxY + 80);
    return { root: laid, width, height, offsetX: -minX + 40 };
  }, [treeData]);

  useEffect(() => {
    setSvgSize({ width: layout.width, height: layout.height });
  }, [layout.width, layout.height]);

  const toggleExpand = useCallback((nodeId: string) => {
    setAnimating(true);
    setTreeData((prev) => {
      function toggle(node: TreeNode): TreeNode {
        if (node.title === nodeId) {
          return { ...node, expanded: !node.expanded };
        }
        return { ...node, children: node.children.map(toggle) };
      }
      // Actually nodeId is the tag/title in this simplified tree
      // Since we only have one node per tag level, we match by tag
      return toggle(prev);
    });
    setTimeout(() => setAnimating(false), 300);
  }, []);

  // Collect all nodes for rendering
  const allNodes: LayoutNode[] = [];
  const allLinks: Array<{ from: LayoutNode; to: LayoutNode }> = [];
  function collect(node: LayoutNode) {
    allNodes.push(node);
    if (node.expanded) {
      node.children.forEach((child) => {
        allLinks.push({ from: node, to: child });
        collect(child);
      });
    }
  }
  collect(layout.root);

  const nodeColor = (tag: string) => TAG_COLORS[tag] || '#57534e';

  return (
    <div className={`panel overflow-auto ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-warm-900">Chain of Thought</h3>
        <p className="text-sm text-warm-500 mt-1">Click nodes to expand/collapse. Select to view content.</p>
      </div>

      <svg
        ref={svgRef}
        width={svgSize.width}
        height={svgSize.height}
        className={`transition-all duration-300 ${animating ? 'opacity-80' : 'opacity-100'}`}
        style={{ minWidth: svgSize.width, minHeight: svgSize.height }}
      >
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#a8a29e" />
          </marker>
        </defs>
        <g transform={`translate(${layout.offsetX}, 20)`}>
          {/* Links */}
          {allLinks.map((link, i) => {
            const sx = link.from.x;
            const sy = link.from.y + NODE_HEIGHT / 2;
            const ex = link.to.x;
            const ey = link.to.y - NODE_HEIGHT / 2;
            const c1x = sx;
            const c1y = sy + V_GAP / 2;
            const c2x = ex;
            const c2y = ey - V_GAP / 2;
            const d = `M ${sx} ${sy} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${ex} ${ey}`;
            return (
              <path
                key={`link-${i}`}
                d={d}
                fill="none"
                stroke="#d6d3d1"
                strokeWidth={2}
                markerEnd="url(#arrowhead)"
                className="transition-all duration-300"
              />
            );
          })}

          {/* Nodes */}
          {allNodes.map((node) => {
            const isSelected = selectedNode === node.id;
            const hasChildren = node.children.length > 0 || treeData.children.some(c => c.tag === node.tag);
            const color = nodeColor(node.tag);
            return (
              <g
                key={node.id}
                transform={`translate(${node.x - NODE_WIDTH / 2}, ${node.y - NODE_HEIGHT / 2})`}
                className="cursor-pointer transition-all duration-300"
                onClick={() => {
                  setSelectedNode(node.id);
                  if (hasChildren) toggleExpand(node.tag);
                }}
                style={{ transformOrigin: `${node.x}px ${node.y}px` }}
              >
                <rect
                  width={NODE_WIDTH}
                  height={NODE_HEIGHT}
                  rx={10}
                  ry={10}
                  fill={isSelected ? '#f0f9ff' : '#ffffff'}
                  stroke={isSelected ? color : '#e7e5e4'}
                  strokeWidth={isSelected ? 3 : 1.5}
                  className="transition-all duration-200"
                />
                <text
                  x={NODE_WIDTH / 2}
                  y={NODE_HEIGHT / 2 + 5}
                  textAnchor="middle"
                  fill={color}
                  fontSize={13}
                  fontWeight={600}
                  fontFamily="Inter, system-ui, sans-serif"
                >
                  {node.title}
                </text>
                {/* Expand indicator */}
                {node.children.length > 0 && (
                  <circle
                    cx={NODE_WIDTH - 12}
                    cy={NODE_HEIGHT / 2}
                    r={4}
                    fill={node.expanded ? color : '#a8a29e'}
                    className="transition-all duration-200"
                  />
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {/* Content Panel */}
      {selectedNode && (
        <div className="mt-4 p-4 bg-warm-50 rounded-lg border border-warm-200 animate-[fadeIn_0.3s_ease-out]">
          {allNodes
            .filter((n) => n.id === selectedNode)
            .map((node) => (
              <div key={node.id}>
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: nodeColor(node.tag) }}
                  />
                  <span className="text-sm font-semibold" style={{ color: nodeColor(node.tag) }}>
                    {node.tag}
                  </span>
                </div>
                <p className="text-sm text-warm-700 leading-relaxed whitespace-pre-wrap">
                  {node.content}
                </p>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};

export default TVPChainTree;
