/**
 * DrugAdrNetworkSigma — WebGL graph renderer for the Drug–ADR Network.
 *
 * Replaces the SVG rendering in GNNNetworkView with sigma.js v3.
 * Props: { graphData, height }
 * Ref: { zoomIn, zoomOut, resetView, refresh }
 */
import { useState, useEffect, useCallback, useRef, useMemo, forwardRef, useImperativeHandle } from 'react';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import { SigmaContainer, useSigma, useRegisterEvents, useLoadGraph, useSetSettings } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import { NodeCircleProgram } from 'sigma/rendering';
import NodeDiamondProgram from './NodeDiamondProgram';

/* ── Color palette (matches existing SVG) ────────────────────────── */
const DRUG_COLOR   = '#6366f1';
const ADR_COLOR    = '#f59e0b';
const SIGNAL_EDGE  = '#ef4444';
const NORMAL_EDGE  = '#4338ca';

const DIM_COLOR      = 'rgba(60,60,80,0.15)';
const DIM_EDGE_COLOR = 'rgba(60,60,80,0.05)';

/* ── Build graphology graph from API payload ─────────────────────── */
function buildGraph(graphData) {
  const graph = new Graph({ multi: true, type: 'undirected' });
  if (!graphData?.nodes?.length) return graph;

  const { nodes, edges } = graphData;

  // Separate drugs and ADRs, sorted by count (most reports first → center)
  const drugs = nodes.filter(n => n.type === 'drug').sort((a, b) => (b.count || 0) - (a.count || 0));
  const adrs  = nodes.filter(n => n.type === 'adr').sort((a, b) => (b.count || 0) - (a.count || 0));

  // ── Grid layout: drugs in center grid, ADRs in surrounding rings ──

  const SPACING = 80;  // space between node centers

  // Place drugs in a roughly-square grid (matches old SVG layout)
  const drugCols = Math.ceil(Math.sqrt(drugs.length));
  const drugRows = Math.ceil(drugs.length / drugCols);
  const drugW = (drugCols - 1) * SPACING;
  const drugH = (drugRows - 1) * SPACING;
  const drugOffX = -drugW / 2;
  const drugOffY = -drugH / 2;

  drugs.forEach((n, idx) => {
    const row = Math.floor(idx / drugCols);
    const col = idx % drugCols;
    // Size 8–12: matches old SVG proportions, same as ADR nodes
    const size = Math.max(8, Math.min(Math.sqrt(n.count || 1) * 1.0, 12));
    graph.addNode(n.id, {
      label: n.id,
      size,
      color: DRUG_COLOR,
      type: 'circle',
      x: drugOffX + col * SPACING,
      y: drugOffY + row * SPACING,
      nodeType: n.type,
      count: n.count || 0,
      roleCounts: n.role_counts || null,
    });
  });

  // Place ADRs in concentric rectangular rings around the drug grid
  const margin = SPACING * 1.2; // gap between drug grid edge and first ADR ring
  const adrSpacing = SPACING * 0.75;

  let placed = 0;
  let ring = 0;

  while (placed < adrs.length) {
    const halfW = drugW / 2 + margin + ring * adrSpacing;
    const halfH = drugH / 2 + margin + ring * adrSpacing;

    // Calculate positions along the perimeter of this ring
    const positions = [];

    // Top edge (left to right)
    const topCount = Math.floor(2 * halfW / adrSpacing) + 1;
    for (let i = 0; i < topCount; i++) {
      positions.push({ x: -halfW + i * (2 * halfW / Math.max(topCount - 1, 1)), y: -halfH });
    }
    // Right edge (top to bottom, skip corners)
    const rightCount = Math.max(Math.floor(2 * halfH / adrSpacing) - 1, 1);
    for (let i = 1; i <= rightCount; i++) {
      positions.push({ x: halfW, y: -halfH + i * (2 * halfH / (rightCount + 1)) });
    }
    // Bottom edge (right to left)
    for (let i = topCount - 1; i >= 0; i--) {
      positions.push({ x: -halfW + i * (2 * halfW / Math.max(topCount - 1, 1)), y: halfH });
    }
    // Left edge (bottom to top, skip corners)
    for (let i = rightCount; i >= 1; i--) {
      positions.push({ x: -halfW, y: -halfH + i * (2 * halfH / (rightCount + 1)) });
    }

    // Place ADRs along this ring
    for (let i = 0; i < positions.length && placed < adrs.length; i++) {
      const n = adrs[placed];
      // Size 8–12: same as drug nodes (matches old SVG where both are similar size)
      const size = Math.max(8, Math.min(Math.sqrt(n.count || 1) * 1.0, 12));
      graph.addNode(n.id, {
        label: n.id,
        size,
        color: ADR_COLOR,
        type: 'diamond',
        x: positions[i].x,
        y: positions[i].y,
        nodeType: n.type,
        count: n.count || 0,
        roleCounts: null,
      });
      placed++;
    }
    ring++;
  }

  // Add edges
  edges.forEach((e) => {
    if (graph.hasNode(e.source) && graph.hasNode(e.target)) {
      graph.addEdge(e.source, e.target, {
        size: Math.min(Math.max((e.count || 1) / 20, 0.3), 2),
        color: e.is_signal ? SIGNAL_EDGE : NORMAL_EDGE,
        isSignal: !!e.is_signal,
        count: e.count || 0,
      });
    }
  });

  console.log(`[sigma] Grid layout: ${graph.order} nodes (${drugs.length} drugs, ${adrs.length} ADRs), ${graph.size} edges`);
  return graph;
}

/* ── Inner components (must be inside SigmaContainer) ────────────── */

/** Loads graph data into sigma */
function GraphLoader({ graphData }) {
  const loadGraph = useLoadGraph();

  useEffect(() => {
    try {
      const graph = buildGraph(graphData);
      loadGraph(graph);
    } catch (err) {
      console.error('[sigma] Failed to load graph:', err);
    }
  }, [graphData, loadGraph]);

  return null;
}

/** Handles hover events + tooltip + reducers */
function GraphEvents({ onHoverNode, onLeaveNode, hoveredNode }) {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();
  const setSettings = useSetSettings();

  useEffect(() => {
    registerEvents({
      enterNode: (event) => {
        const graph = sigma.getGraph();
        const nodeId = event.node;
        const attrs = graph.getNodeAttributes(nodeId);
        const displayData = sigma.getNodeDisplayData(nodeId);
        if (!displayData) return;
        const canvasPos = sigma.framedGraphToViewport(displayData);

        const connectedEdges = graph.edges(nodeId);
        const signalCount = connectedEdges.filter(eid =>
          graph.getEdgeAttribute(eid, 'isSignal')
        ).length;

        onHoverNode({
          id: nodeId,
          type: attrs.nodeType,
          count: attrs.count,
          connections: connectedEdges.length,
          signals: signalCount,
          roleCounts: attrs.roleCounts,
          x: canvasPos.x,
          y: canvasPos.y,
        });
      },
      leaveNode: () => {
        onLeaveNode();
      },
    });
  }, [registerEvents, sigma, onHoverNode, onLeaveNode]);

  // Node/edge reducers for dimming non-connected items on hover
  useEffect(() => {
    if (!hoveredNode) {
      setSettings({
        nodeReducer: null,
        edgeReducer: null,
      });
      return;
    }

    const graph = sigma.getGraph();
    const neighbors = new Set();
    if (graph.hasNode(hoveredNode)) {
      graph.forEachNeighbor(hoveredNode, (neighbor) => neighbors.add(neighbor));
    }

    setSettings({
      nodeReducer: (node, data) => {
        if (node === hoveredNode) {
          return { ...data, highlighted: true, size: data.size * 1.5, zIndex: 2, forceLabel: true };
        }
        if (neighbors.has(node)) {
          // forceLabel: true ensures connected ADR labels are ALWAYS visible on hover
          return { ...data, highlighted: true, zIndex: 1, forceLabel: true };
        }
        return { ...data, color: DIM_COLOR, label: null, zIndex: -1 };
      },
      edgeReducer: (edge, data) => {
        const source = graph.source(edge);
        const target = graph.target(edge);
        if (source === hoveredNode || target === hoveredNode) {
          // Bright green for connected edges on hover (matches old SVG green/cyan)
          return { ...data, color: data.isSignal ? '#f87171' : '#10b981', size: Math.max(data.size, 2.5) };
        }
        return { ...data, color: 'rgba(30,30,50,0.03)', size: 0.2 };
      },
    });
  }, [hoveredNode, sigma, setSettings]);

  return null;
}

/* ── Tooltip component (HTML overlay) ────────────────────────────── */
function Tooltip({ nodeInfo }) {
  if (!nodeInfo) return null;

  const roleLabels = { PS: 'Primary Suspect', SS: 'Secondary Suspect', C: 'Concomitant', I: 'Interacting' };
  const roleColors = { PS: '#ef4444', SS: '#f59e0b', C: '#9ca3af', I: '#60a5fa' };
  const rc = nodeInfo.type === 'drug' && nodeInfo.roleCounts ? nodeInfo.roleCounts : null;
  const roleEntries = rc ? Object.entries(rc).sort((a, b) => b[1] - a[1]) : [];

  const x = nodeInfo.x + 15;
  const y = nodeInfo.y - 10;

  return (
    <div
      className="absolute z-50 pointer-events-none"
      style={{ left: x, top: y, maxWidth: 260 }}
    >
      <div
        style={{
          background: 'rgba(15,15,35,0.95)',
          border: '1px solid rgba(99,102,241,0.35)',
          borderRadius: 10,
          padding: '12px 16px',
          backdropFilter: 'blur(12px)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        {/* Name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
          <span style={{ fontSize: 14 }}>{nodeInfo.type === 'drug' ? '💊' : '⚠️'}</span>
          <span style={{
            color: nodeInfo.type === 'drug' ? '#a5b4fc' : '#fcd34d',
            fontSize: 13, fontWeight: 700, fontFamily: 'Inter, sans-serif',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200,
          }}>
            {nodeInfo.id}
          </span>
        </div>

        {/* Stats */}
        <div style={{ fontSize: 11, color: '#9ca3af', fontFamily: 'Inter, sans-serif', lineHeight: 1.8 }}>
          <div>Type: <span style={{ color: '#d1d5db', fontWeight: 500 }}>{nodeInfo.type}</span> · Reports: <span style={{ color: '#d1d5db', fontWeight: 500 }}>{(nodeInfo.count || 0).toLocaleString()}</span></div>
          <div>Connections: <span style={{ color: '#d1d5db', fontWeight: 500 }}>{nodeInfo.connections}</span></div>
          <div>Signals: <span style={{ color: '#ef4444', fontWeight: 600 }}>{nodeInfo.signals}</span></div>
        </div>

        {/* Role breakdown (drug nodes only) */}
        {roleEntries.length > 0 && (
          <>
            <div style={{ borderTop: '1px solid rgba(99,102,241,0.25)', margin: '8px 0' }} />
            <div style={{ fontSize: 11, fontFamily: 'Inter, sans-serif', lineHeight: 2 }}>
              {roleEntries.map(([code, count]) => (
                <div key={code} style={{ display: 'flex', justifyContent: 'space-between', color: roleColors[code] || '#9ca3af' }}>
                  <span>{roleLabels[code] || code} ({code}):</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{count.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────── */
const DrugAdrNetworkSigma = forwardRef(function DrugAdrNetworkSigma({ graphData, height }, ref) {
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [tooltipInfo, setTooltipInfo] = useState(null);
  const containerRef = useRef(null);
  const sigmaRef = useRef(null);

  // Expose camera controls to parent
  useImperativeHandle(ref, () => ({
    zoomIn: () => {
      const sigma = sigmaRef.current;
      if (sigma) sigma.getCamera().animatedZoom({ duration: 200 });
    },
    zoomOut: () => {
      const sigma = sigmaRef.current;
      if (sigma) sigma.getCamera().animatedUnzoom({ duration: 200 });
    },
    resetView: () => {
      const sigma = sigmaRef.current;
      if (sigma) sigma.getCamera().animatedReset({ duration: 300 });
    },
    refresh: () => {
      const sigma = sigmaRef.current;
      if (sigma) sigma.refresh();
    },
  }));

  const handleHoverNode = useCallback((info) => {
    setHoveredNodeId(info.id);
    setTooltipInfo(info);
  }, []);

  const handleLeaveNode = useCallback(() => {
    setHoveredNodeId(null);
    setTooltipInfo(null);
  }, []);

  // Custom label renderer: dark background + bright text (matches old SVG)
  const drawLabel = useCallback((context, data, settings) => {
    if (!data.label) return;

    const size = settings.labelSize;
    const font = `${settings.labelWeight} ${size}px ${settings.labelFont}`;
    context.font = font;

    const textWidth = context.measureText(data.label).width;
    const pad = 3;
    const bgX = data.x + data.size + 4;
    const bgY = data.y - size / 2 - pad;

    // Dark background pill
    context.fillStyle = 'rgba(10, 10, 25, 0.85)';
    context.beginPath();
    context.roundRect(bgX - pad, bgY, textWidth + pad * 2, size + pad * 2, 3);
    context.fill();

    // Text in node's color (indigo for drugs, amber for ADRs)
    context.fillStyle = data.color === DRUG_COLOR ? '#a5b4fc' : '#fcd34d';
    context.fillText(data.label, bgX, data.y + size / 3);
  }, []);

  // Custom hover renderer: dark panel with name + highlight ring
  const drawHover = useCallback((context, data, settings) => {
    // Draw highlight ring around hovered node
    const ringSize = data.size + 4;
    context.beginPath();
    context.arc(data.x, data.y, ringSize, 0, Math.PI * 2);
    context.strokeStyle = data.color === DRUG_COLOR ? '#818cf8' : '#fbbf24';
    context.lineWidth = 2.5;
    context.stroke();
    context.closePath();

    // Draw label with dark background
    if (!data.label) return;
    const size = settings.labelSize + 2;
    const font = `600 ${size}px ${settings.labelFont}`;
    context.font = font;

    const textWidth = context.measureText(data.label).width;
    const pad = 5;
    const bgX = data.x + data.size + 6;
    const bgY = data.y - size / 2 - pad;

    // Dark background
    context.fillStyle = 'rgba(10, 10, 25, 0.95)';
    context.beginPath();
    context.roundRect(bgX - pad, bgY, textWidth + pad * 2, size + pad * 2, 4);
    context.fill();

    // Border
    context.strokeStyle = data.color === DRUG_COLOR ? 'rgba(99,102,241,0.4)' : 'rgba(245,158,11,0.4)';
    context.lineWidth = 1;
    context.stroke();

    // Text
    context.fillStyle = data.color === DRUG_COLOR ? '#c7d2fe' : '#fde68a';
    context.fillText(data.label, bgX, data.y + size / 3);
  }, []);

  // Sigma settings with custom diamond program for ADR nodes
  const sigmaSettings = useMemo(() => ({
    nodeProgramClasses: {
      circle: NodeCircleProgram,
      diamond: NodeDiamondProgram,
    },
    renderLabels: true,
    labelRenderedSizeThreshold: 10,
    labelColor: { color: '#e5e7eb' },
    labelFont: 'Inter, system-ui, sans-serif',
    labelSize: 10,
    labelWeight: '500',
    defaultDrawNodeLabel: drawLabel,
    defaultDrawNodeHover: drawHover,
    defaultNodeColor: '#999',
    defaultEdgeColor: '#555',
    defaultNodeType: 'circle',
    allowInvalidContainer: true,
    autoRescale: true,
    autoCenter: true,
    stagePadding: 50,
    enableEdgeEvents: false,
    zoomDuration: 200,
  }), [drawLabel, drawHover]);

  /** Inner component to capture sigma ref */
  function SigmaRefCapture() {
    const sigma = useSigma();
    useEffect(() => {
      sigmaRef.current = sigma;
      return () => { sigmaRef.current = null; };
    }, [sigma]);
    return null;
  }

  if (!graphData?.nodes?.length) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6b7280' }}>
        No graph data
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative" style={{ height, width: '100%' }}>
      <SigmaContainer
        style={{
          height: '100%',
          width: '100%',
          background: '#0a0a19',
          borderRadius: 8,
        }}
        settings={sigmaSettings}
      >
        <SigmaRefCapture />
        <GraphLoader graphData={graphData} />
        <GraphEvents
          hoveredNode={hoveredNodeId}
          onHoverNode={handleHoverNode}
          onLeaveNode={handleLeaveNode}
        />
      </SigmaContainer>

      {/* Tooltip overlay */}
      <Tooltip nodeInfo={tooltipInfo} />

      {/* Legend overlay */}
      <div
        className="absolute bottom-3 left-3 z-10"
        style={{
          background: 'rgba(10,10,25,0.92)',
          border: '1px solid rgba(99,102,241,0.15)',
          borderRadius: 6,
          padding: '8px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          fontSize: 11,
          fontFamily: 'Inter, sans-serif',
          color: '#d1d5db',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 12, height: 12, borderRadius: '50%', background: DRUG_COLOR, display: 'inline-block' }} />
          Drug
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{
            width: 10, height: 10, background: ADR_COLOR, display: 'inline-block',
            transform: 'rotate(45deg)', borderRadius: 1,
          }} />
          ADR
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 20, height: 2.5, background: SIGNAL_EDGE, display: 'inline-block', borderRadius: 1 }} />
          Signal
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 16, height: 1.5, background: NORMAL_EDGE, display: 'inline-block', borderRadius: 1 }} />
          Link
        </span>
      </div>

      {/* Stats footer */}
      <div
        className="absolute bottom-3 right-3 z-10"
        style={{
          fontSize: 10,
          fontFamily: 'Inter, sans-serif',
          color: '#6b7280',
          background: 'rgba(10,10,25,0.8)',
          padding: '3px 8px',
          borderRadius: 4,
        }}
      >
        {graphData.nodes.filter(n => n.type === 'drug').length} drugs ·{' '}
        {graphData.nodes.filter(n => n.type === 'adr').length} ADRs ·{' '}
        {graphData.edges.filter(e => e.is_signal).length} signals ·{' '}
        {graphData.edges.length} edges
      </div>

      {/* Hint */}
      <div
        className="absolute bottom-3 z-10"
        style={{
          left: '50%',
          transform: 'translateX(-50%)',
          fontSize: 10,
          color: '#4b5563',
          background: 'rgba(10,10,25,0.8)',
          padding: '2px 8px',
          borderRadius: 4,
        }}
      >
        Scroll to zoom · Drag to pan
      </div>
    </div>
  );
});

export default DrugAdrNetworkSigma;
