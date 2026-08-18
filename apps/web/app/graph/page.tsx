'use client';

import { useState, useCallback, useEffect } from 'react';

interface GraphNode {
  node_id: string;
  node_type: string;
  label: string;
  metadata: Record<string, unknown>;
}

interface GraphEdge {
  edge_id: string;
  edge_type: string;
  source_id: string;
  target_id: string;
  confidence: number;
  evidence_span: string | null;
}

interface GraphData {
  graph_id: string;
  dataset_id: string;
  description: string;
  total_nodes: number;
  total_edges: number;
  node_types: Record<string, number>;
  edge_types: Record<string, number>;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

const NODE_COLORS: Record<string, string> = {
  dataset: '#6366f1',
  document: '#3b82f6',
  chunk: '#0ea5e9',
  entity: '#10b981',
  extracted_fact: '#f59e0b',
  claim: '#ef4444',
  topic: '#8b5cf6',
};

const EDGE_COLORS: Record<string, string> = {
  contains: '#475569',
  mentions: '#0ea5e9',
  supports: '#22c55e',
  refutes: '#ef4444',
  related_to: '#6b7280',
  semantic_similar_to: '#8b5cf6',
};

function NodeChip({ node, selected, onClick }: { node: GraphNode; selected: boolean; onClick: () => void }) {
  const color = NODE_COLORS[node.node_type] ?? '#6b7280';
  return (
    <div
      onClick={onClick}
      style={{
        padding: '10px 14px',
        background: selected ? `${color}22` : 'rgba(255,255,255,0.04)',
        border: `1px solid ${selected ? color : 'rgba(255,255,255,0.08)'}`,
        borderRadius: 10,
        cursor: 'pointer',
        transition: 'all 0.18s',
        marginBottom: 6,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span
          style={{
            width: 8, height: 8, borderRadius: '50%', background: color,
            flexShrink: 0, boxShadow: selected ? `0 0 8px ${color}` : 'none',
          }}
        />
        <span style={{ fontSize: 11, fontWeight: 600, color, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {node.node_type}
        </span>
      </div>
      <p style={{ margin: '4px 0 0', fontSize: 13, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {node.label}
      </p>
    </div>
  );
}

export default function GraphPage() {
  const [graphId, setGraphId] = useState('');
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [nodeTypeFilter, setNodeTypeFilter] = useState<string>('all');
  const [edgeTypeFilter, setEdgeTypeFilter] = useState<string>('all');

  const loadGraph = useCallback(async () => {
    if (!graphId.trim()) return;
    setLoading(true);
    setError(null);
    setSelectedNode(null);
    try {
      const resp = await fetch(`/api/v1/evidence/graph/${graphId}`);
      if (!resp.ok) {
        const data = await resp.json();
        throw new Error(data.message ?? 'Graph not found');
      }
      setGraphData(await resp.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load graph');
    } finally {
      setLoading(false);
    }
  }, [graphId]);

  const filteredNodes = graphData
    ? nodeTypeFilter === 'all'
      ? graphData.nodes
      : graphData.nodes.filter(n => n.node_type === nodeTypeFilter)
    : [];

  const filteredEdges = graphData
    ? edgeTypeFilter === 'all'
      ? graphData.edges
      : graphData.edges.filter(e => e.edge_type === edgeTypeFilter)
    : [];

  const selectedNodeEdges = selectedNode
    ? filteredEdges.filter(e => e.source_id === selectedNode.node_id || e.target_id === selectedNode.node_id)
    : [];

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
        color: '#f1f5f9',
        fontFamily: "'Inter', 'Segoe UI', sans-serif",
        padding: '32px 24px',
      }}
    >
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 36 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span style={{ fontSize: 28 }}>🕸️</span>
            <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, background: 'linear-gradient(135deg, #a78bfa, #ec4899)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Provenance Graph Explorer
            </h1>
          </div>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: 15 }}>
            Deterministic traceability graph. NOT a neural network — no trained weights.
            Nodes and edges are built from extraction records.
          </p>
        </div>

        {/* Load Graph */}
        <div
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 16,
            padding: 20,
            marginBottom: 24,
            display: 'flex',
            gap: 12,
            alignItems: 'flex-end',
          }}
        >
          <label style={{ flex: 1 }}>
            <span style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Graph ID
            </span>
            <input
              value={graphId}
              onChange={e => setGraphId(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && loadGraph()}
              placeholder="e.g. scifact_q1 or dataset_queryid"
              style={{
                width: '100%', boxSizing: 'border-box', padding: '10px 14px',
                background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 8, color: '#f1f5f9', fontSize: 14, outline: 'none',
              }}
            />
          </label>
          <button
            onClick={loadGraph}
            disabled={loading || !graphId.trim()}
            style={{
              padding: '10px 24px', height: 42,
              background: 'linear-gradient(135deg, #7c3aed, #6366f1)',
              border: 'none', borderRadius: 8, color: '#fff', fontWeight: 700,
              fontSize: 14, cursor: 'pointer',
            }}
          >
            {loading ? '⏳' : '🔍 Load'}
          </button>
        </div>

        {error && (
          <div style={{ marginBottom: 20, padding: 14, background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 10, color: '#fca5a5', fontSize: 14 }}>
            ❌ {error}
          </div>
        )}

        {graphData && (
          <>
            {/* Stats Bar */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: 24 }}>
              {[
                { label: 'Nodes', value: graphData.total_nodes, color: '#60a5fa' },
                { label: 'Edges', value: graphData.total_edges, color: '#a78bfa' },
                ...Object.entries(graphData.node_types).slice(0, 3).map(([type, count]) => ({
                  label: type.charAt(0).toUpperCase() + type.slice(1) + 's',
                  value: count,
                  color: NODE_COLORS[type] ?? '#6b7280',
                })),
              ].map(({ label, value, color }) => (
                <div key={label} style={{ padding: '14px 16px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, textAlign: 'center' }}>
                  <p style={{ margin: 0, fontSize: 24, fontWeight: 800, color }}>{value}</p>
                  <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</p>
                </div>
              ))}
            </div>

            {/* Description */}
            <div style={{ marginBottom: 24, padding: '12px 16px', background: 'rgba(168,85,247,0.08)', border: '1px solid rgba(168,85,247,0.2)', borderRadius: 10 }}>
              <p style={{ margin: 0, fontSize: 13, color: '#c4b5fd' }}>ℹ️ {graphData.description}</p>
            </div>

            {/* Main Panel */}
            <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 20 }}>
              {/* Node List */}
              <div style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16, padding: 16, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <div style={{ marginBottom: 12 }}>
                  <label style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Filter by Node Type
                  </label>
                  <select
                    value={nodeTypeFilter}
                    onChange={e => setNodeTypeFilter(e.target.value)}
                    style={{ width: '100%', marginTop: 6, padding: '7px 10px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6, color: '#f1f5f9', fontSize: 13, outline: 'none' }}
                  >
                    <option value="all">All ({graphData.nodes.length})</option>
                    {Object.entries(graphData.node_types).map(([type, count]) => (
                      <option key={type} value={type}>{type} ({count})</option>
                    ))}
                  </select>
                </div>

                <div style={{ overflow: 'auto', flex: 1 }}>
                  {filteredNodes.map(node => (
                    <NodeChip
                      key={node.node_id}
                      node={node}
                      selected={selectedNode?.node_id === node.node_id}
                      onClick={() => setSelectedNode(selectedNode?.node_id === node.node_id ? null : node)}
                    />
                  ))}
                </div>
              </div>

              {/* Detail Panel */}
              <div style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16, padding: 20 }}>
                {!selectedNode ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: 300, color: '#475569' }}>
                    <span style={{ fontSize: 48, marginBottom: 12 }}>🕸️</span>
                    <p style={{ fontSize: 14 }}>Select a node to explore its connections</p>
                  </div>
                ) : (
                  <div>
                    {/* Node Header */}
                    <div style={{ marginBottom: 20, paddingBottom: 16, borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                        <span style={{ width: 12, height: 12, borderRadius: '50%', background: NODE_COLORS[selectedNode.node_type] ?? '#6b7280', display: 'inline-block', boxShadow: `0 0 12px ${NODE_COLORS[selectedNode.node_type] ?? '#6b7280'}` }} />
                        <span style={{ fontSize: 13, fontWeight: 700, color: NODE_COLORS[selectedNode.node_type] ?? '#6b7280', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                          {selectedNode.node_type}
                        </span>
                      </div>
                      <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: '#f1f5f9' }}>{selectedNode.label}</h3>
                      <code style={{ fontSize: 11, color: '#6b7280' }}>{selectedNode.node_id}</code>
                    </div>

                    {/* Metadata */}
                    {Object.keys(selectedNode.metadata).length > 0 && (
                      <div style={{ marginBottom: 20 }}>
                        <h4 style={{ margin: '0 0 10px', fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Metadata</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          {Object.entries(selectedNode.metadata).map(([k, v]) => (
                            <div key={k} style={{ padding: '8px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8 }}>
                              <p style={{ margin: 0, fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{k}</p>
                              <p style={{ margin: '2px 0 0', fontSize: 13, color: '#e2e8f0', wordBreak: 'break-word' }}>
                                {Array.isArray(v) ? v.join(', ') : String(v ?? '—')}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Connected Edges */}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                        <h4 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Connected Edges ({selectedNodeEdges.length})
                        </h4>
                        <select
                          value={edgeTypeFilter}
                          onChange={e => setEdgeTypeFilter(e.target.value)}
                          style={{ padding: '4px 8px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 6, color: '#f1f5f9', fontSize: 12, outline: 'none' }}
                        >
                          <option value="all">All types</option>
                          {Object.keys(EDGE_COLORS).map(t => (
                            <option key={t} value={t}>{t}</option>
                          ))}
                        </select>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {selectedNodeEdges.slice(0, 20).map(edge => {
                          const isOut = edge.source_id === selectedNode.node_id;
                          const eColor = EDGE_COLORS[edge.edge_type] ?? '#6b7280';
                          return (
                            <div key={edge.edge_id} style={{ padding: '10px 14px', background: `${eColor}0a`, border: `1px solid ${eColor}30`, borderRadius: 8 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <span style={{ fontSize: 12, color: '#64748b' }}>{isOut ? '→' : '←'}</span>
                                <span style={{ padding: '2px 8px', background: `${eColor}20`, borderRadius: 10, fontSize: 11, fontWeight: 600, color: eColor }}>
                                  {edge.edge_type}
                                </span>
                                <code style={{ fontSize: 11, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                  {isOut ? edge.target_id : edge.source_id}
                                </code>
                                <span style={{ marginLeft: 'auto', fontSize: 11, color: '#64748b' }}>
                                  {(edge.confidence * 100).toFixed(0)}%
                                </span>
                              </div>
                              {edge.evidence_span && (
                                <p style={{ margin: '6px 0 0', fontSize: 12, color: '#94a3b8', fontStyle: 'italic' }}>
                                  "{edge.evidence_span.slice(0, 120)}{edge.evidence_span.length > 120 ? '…' : ''}"
                                </p>
                              )}
                            </div>
                          );
                        })}
                        {selectedNodeEdges.length > 20 && (
                          <p style={{ fontSize: 12, color: '#64748b', textAlign: 'center' }}>
                            ...and {selectedNodeEdges.length - 20} more edges
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
