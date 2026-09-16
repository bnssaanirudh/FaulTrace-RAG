'use client';

import { useEffect, useState } from 'react';
import { api, Run } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { StatusBadge } from '@/components/ui/status-badge';
import { ArtifactHash } from '@/components/ui/artifact-hash';
import { SkeletonCard, EmptyState } from '@/components/ui/skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';
import { JsonViewer } from '@/components/ui/json-viewer';
import {
  Microscope,
  RefreshCw,
  Database,
  HelpCircle,
  ArrowRight,
  Info,
} from 'lucide-react';
import Link from 'next/link';
import { formatMs } from '@/lib/utils';

interface Attribution {
  run_id: string;
  pipeline_id: string;
  is_correct: boolean;
  total_error: number;
  components: {
    component: string;
    pipeline_answer: string;
    oracle_answer: string;
    ref_score: number;
    shapley_value: number;
  }[];
  dominant_fault: string;
}

const COMPONENT_COLORS: Record<string, string> = {
  retrieval:   'bg-orange-500/15 text-orange-300 ring-orange-500/30',
  extraction:  'bg-brand-500/15 text-brand-300 ring-brand-500/30',
  aggregation: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30',
};

const LATTICE_LABELS = [
  { key: 'none', label: 'None', description: 'No oracle intervention' },
  { key: 'R',    label: 'R',    description: 'Oracle Retrieval only' },
  { key: 'E',    label: 'E',    description: 'Oracle Extraction only' },
  { key: 'A',    label: 'A',    description: 'Oracle Aggregation only' },
  { key: 'RE',   label: 'RE',   description: 'Oracle Retrieval + Extraction' },
  { key: 'RA',   label: 'RA',   description: 'Oracle Retrieval + Aggregation' },
  { key: 'EA',   label: 'EA',   description: 'Oracle Extraction + Aggregation' },
  { key: 'REA',  label: 'REA',  description: 'Full oracle (all three)' },
];

export default function OracleDiagnosticsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [attribution, setAttribution] = useState<Attribution | null>(null);
  const [attrLoading, setAttrLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError('');
    try {
      const res = await api.listRuns(undefined, undefined, 1, 50);
      // Only show runs that are completed and potentially incorrect
      setRuns(res.items.filter(r => r.status === 'completed'));
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function loadAttribution(run: Run) {
    setSelectedRun(run);
    setAttribution(null);
    setAttrLoading(true);
    try {
      const attr = await api.getRunAttribution(run.run_id);
      setAttribution(attr);
    } catch {
      setAttribution(null);
    } finally {
      setAttrLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  const phi = attribution?.components ?? [];
  const phiSum = phi.reduce((acc, c) => acc + c.shapley_value, 0);

  return (
    <div className="p-8 animate-fade-in text-slate-100 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Microscope className="h-5 w-5 text-orange-500" />
            <span className="text-xs font-semibold uppercase tracking-widest text-orange-500">WP7</span>
          </div>
          <h1 className="text-3xl font-bold text-white">Oracle Diagnostics</h1>
          <p className="mt-1 text-sm text-slate-400 max-w-2xl">
            Eight-state intervention lattice with exact Shapley (φ_R, φ_E, φ_A) attribution.
            Select a completed run to view modular counterfactual analysis.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={load} disabled={loading}>
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Interpretation notice */}
      <div className="rounded-xl border border-orange-500/20 bg-orange-600/5 p-4 flex gap-3">
        <Info className="h-4 w-4 text-orange-400 flex-shrink-0 mt-0.5" />
        <p className="text-xs text-slate-300 leading-relaxed">
          <strong className="text-orange-400">Interpretation note:</strong> Shapley values show how much error is
          recoverable by substituting each component with a deterministic oracle. They measure
          <em> within-pipeline fault attribution</em>, not unrestricted real-world causality.
          φ_R + φ_E + φ_A ≈ 1.0 when the oracle covers all faults (efficiency axiom).
        </p>
      </div>

      {error && (
        <div className="rounded-lg bg-red-500/10 p-3 text-sm text-red-300 ring-1 ring-red-500/25">
          {error} — ensure the backend is running
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Run selector */}
        <div className="col-span-5">
          <Card padding="none">
            <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white">Completed Runs</h2>
              <Badge variant="neutral">{runs.length}</Badge>
            </div>
            <div className="divide-y divide-white/[0.04] max-h-[60vh] overflow-y-auto scrollbar-thin">
              {loading && Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="px-4 py-3 animate-pulse flex gap-3">
                  <div className="h-4 w-4 rounded-full bg-white/[0.05]" />
                  <div className="flex-1 space-y-1.5">
                    <div className="h-3 w-32 rounded bg-white/[0.05]" />
                    <div className="h-2.5 w-20 rounded bg-white/[0.04]" />
                  </div>
                </div>
              ))}
              {!loading && runs.length === 0 && (
                <EmptyState
                  icon={<Database className="h-5 w-5" />}
                  title="No completed runs"
                  description="Seed the demo and run pipelines to see oracle diagnostics."
                />
              )}
              {!loading && runs.map(run => (
                <button
                  key={run.run_id}
                  onClick={() => loadAttribution(run)}
                  className={`w-full text-left px-4 py-3 transition-colors duration-150 flex items-start gap-3 ${
                    selectedRun?.run_id === run.run_id
                      ? 'bg-orange-600/10 border-l-2 border-orange-500'
                      : 'hover:bg-white/[0.03] border-l-2 border-transparent'
                  }`}
                >
                  <StatusBadge
                    status={run.is_correct ? 'CORRECT' : run.is_correct === false ? 'INCORRECT' : 'UNKNOWN'}
                    className="mt-0.5 flex-shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-mono text-brand-400 truncate">{run.run_id.slice(0, 16)}…</p>
                    <p className="text-[10px] text-slate-500 mt-0.5">{run.pipeline_id.split('-')[0]} · {formatMs(run.latency_ms)}</p>
                    {run.loss !== null && (
                      <p className="text-[10px] text-red-400 mt-0.5">loss = {run.loss.toFixed(4)}</p>
                    )}
                  </div>
                  <ArrowRight className="h-3.5 w-3.5 text-slate-600 flex-shrink-0 mt-1" />
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* Attribution panel */}
        <div className="col-span-7 space-y-6">
          {!selectedRun && (
            <Card>
              <EmptyState
                icon={<HelpCircle className="h-6 w-6" />}
                title="Select a run"
                description="Click a completed run on the left to view its Shapley attribution lattice."
              />
            </Card>
          )}

          {selectedRun && attrLoading && (
            <div className="grid grid-cols-3 gap-4">
              {[1, 2, 3].map(i => <SkeletonCard key={i} />)}
            </div>
          )}

          {selectedRun && !attrLoading && (
            <ErrorBoundary>
              {/* Run header */}
              <Card>
                <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
                  <div>
                    <p className="text-xs font-mono text-orange-400">{selectedRun.run_id}</p>
                    <p className="text-xs text-slate-500 mt-0.5">{selectedRun.pipeline_id}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <StatusBadge status={selectedRun.is_correct ? 'CORRECT' : selectedRun.is_correct === false ? 'INCORRECT' : 'UNKNOWN'} />
                    <Link href={`/runs/${selectedRun.run_id}/trace`}>
                      <Button size="sm" variant="ghost">Full Trace →</Button>
                    </Link>
                  </div>
                </div>
                {selectedRun.config_hash && (
                  <div className="px-5 py-2.5">
                    <ArtifactHash hash={selectedRun.config_hash} label="config" />
                  </div>
                )}
              </Card>

              {attribution === null && (
                <Card>
                  <EmptyState
                    title="Attribution unavailable"
                    description="Oracle attribution requires a completed run with a gold answer. Correct runs show zero attribution by design."
                  />
                </Card>
              )}

              {attribution && (
                <>
                  {/* Shapley value cards */}
                  <div>
                    <p className="section-heading">Shapley Attribution (φ_R, φ_E, φ_A)</p>
                    <div className="grid grid-cols-3 gap-4">
                      {attribution.components.map(comp => {
                        const pct = phiSum > 0 ? (comp.shapley_value / phiSum) * 100 : 0;
                        const colorClass = COMPONENT_COLORS[comp.component.toLowerCase()] ?? 'bg-slate-500/15 text-slate-300 ring-slate-500/30';
                        return (
                          <div key={comp.component} className="glass-card p-4">
                            <div className="flex items-center justify-between mb-3">
                              <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-bold ring-1 ${colorClass}`}>
                                {comp.component.slice(0, 1).toUpperCase()}
                              </span>
                              <span className="text-xs font-mono font-semibold text-white">
                                φ = {(comp.shapley_value * 100).toFixed(1)}%
                              </span>
                            </div>
                            <p className="text-xs font-semibold text-slate-300 capitalize mb-2">{comp.component}</p>

                            {/* φ bar */}
                            <div className="mb-2">
                              <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                                <span>Attribution share</span>
                                <span>{pct.toFixed(0)}%</span>
                              </div>
                              <div className="h-1.5 w-full bg-white/[0.06] rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-orange-500 rounded-full transition-all duration-500"
                                  style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                                />
                              </div>
                            </div>

                            {/* REF score bar */}
                            <div>
                              <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                                <span>REF score</span>
                                <span>{comp.ref_score.toFixed(3)}</span>
                              </div>
                              <div className="h-1.5 w-full bg-white/[0.06] rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-validated-600 rounded-full transition-all duration-500"
                                  style={{ width: `${Math.max(0, Math.min(100, comp.ref_score * 100))}%` }}
                                />
                              </div>
                            </div>

                            {comp.component.toLowerCase() === attribution.dominant_fault?.toLowerCase() && (
                              <div className="mt-3 text-[10px] text-orange-400 font-semibold flex items-center gap-1">
                                ⚡ Dominant fault source
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Intervention lattice */}
                  <div>
                    <p className="section-heading">Eight-State Intervention Lattice</p>
                    <Card padding="none">
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b border-white/[0.06]">
                              {['Subset', 'Intervention', 'Recoverable', 'Description'].map(h => (
                                <th key={h} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {LATTICE_LABELS.map(({ key, label, description }) => {
                              const isNone = key === 'none';
                              const isFull = key === 'REA';
                              return (
                                <tr key={key} className="table-row-hover">
                                  <td className="px-4 py-3">
                                    <code className="text-xs font-mono text-orange-300 bg-orange-500/10 rounded px-1.5 py-0.5">
                                      {key}
                                    </code>
                                  </td>
                                  <td className="px-4 py-3 text-xs text-slate-300">{label}</td>
                                  <td className="px-4 py-3">
                                    {isNone
                                      ? <Badge variant="neutral">baseline</Badge>
                                      : isFull
                                        ? <Badge variant="success">all errors</Badge>
                                        : <Badge variant="neutral">partial</Badge>
                                    }
                                  </td>
                                  <td className="px-4 py-3 text-xs text-slate-500">{description}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </Card>
                  </div>

                  {/* Raw JSON */}
                  <div>
                    <p className="section-heading">Raw Attribution Payload</p>
                    <JsonViewer data={attribution} initialDepth={2} />
                  </div>
                </>
              )}
            </ErrorBoundary>
          )}
        </div>
      </div>
    </div>
  );
}
