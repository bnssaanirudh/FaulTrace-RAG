'use client';

import { useCallback, useEffect, useState } from 'react';
import { api, Run, TraceEvent } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  CheckCircle2,
  Clock,
  Database,
  ShieldCheck,
  TrendingUp,
  Download,
  Terminal,
} from 'lucide-react';
import Link from 'next/link';

export default function TracePage({ runId }: { runId: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [attribution, setAttribution] = useState<any | null>(null);
  const [certificate, setCertificate] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selected trace stage tab
  const [selectedStage, setSelectedStage] = useState<string>('scope_enumerate');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const runData = await api.getRun(runId);
      const traceData = await api.getRunTrace(runId);
      setRun(runData);
      setEvents(traceData);

      if (runData.status === 'completed' && runData.gold_answer_value !== null) {
        const [attr, cert] = await Promise.all([
          api.getRunAttribution(runId).catch(() => null),
          api.getRunCertificate(runId).catch(() => null),
        ]);
        setAttribution(attr);
        setCertificate(cert);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch trace details');
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  if (loading) return <div className="p-8 text-slate-500">Loading trace info...</div>;
  if (error) return <div className="p-8 text-red-400">Error loading trace: {error}</div>;
  if (!run) return <div className="p-8 text-slate-500">Run not found</div>;

  const downloadRunBundle = () => {
    const bundle = {
      run,
      events,
      attribution,
      certificate,
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `run_bundle_${runId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Find trace event matching selected stage
  const activeEvent = events.find(ev => ev.stage === selectedStage);

  return (
    <div className="p-8 animate-fade-in text-slate-100 space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs">
            <Link href="/runs">
              <span className="text-slate-500 hover:text-slate-300 cursor-pointer">Run History</span>
            </Link>
            <span className="text-slate-600">/</span>
            <span className="text-orange-400 font-mono">Trace Inspector</span>
          </div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white font-sans">Trace & Diagnostics</h1>
            <Badge variant={run.is_correct ? 'success' : 'error'}>
              {run.is_correct ? 'Correct' : 'Incorrect'}
            </Badge>
            {certificate && (
              <Badge variant={certificate.policy_decision === 'certified' ? 'success' : 'warning'}>
                Policy: {certificate.policy_decision?.toUpperCase()}
              </Badge>
            )}
          </div>
          <p className="text-xs text-slate-500 font-mono">
            RUN ID: {run.run_id} | Config Hash: {run.config_hash?.slice(0, 16)}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={downloadRunBundle} className="gap-1 text-slate-400 hover:text-white">
            <Download className="h-3.5 w-3.5" /> Download Bundle
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card padding="sm" className="flex items-center gap-3">
          <Clock className="h-4 w-4 text-orange-400" />
          <div>
            <p className="text-lg font-bold text-white">{run.latency_ms?.toFixed(0)} ms</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Latency</p>
          </div>
        </Card>
        <Card padding="sm" className="flex items-center gap-3">
          <Database className="h-4 w-4 text-brand-400" />
          <div>
            <p className="text-lg font-bold text-white truncate max-w-[140px]">{run.pipeline_id.split('-')[0]}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Pipeline Config</p>
          </div>
        </Card>
        <Card padding="sm" className="flex items-center gap-3">
          <TrendingUp className="h-4 w-4 text-emerald-400" />
          <div>
            <p className="text-lg font-bold text-white font-mono">{String(run.answer || '—').slice(0, 12)}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Pipeline Output</p>
          </div>
        </Card>
        <Card padding="sm" className="flex items-center gap-3">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <div>
            <p className="text-lg font-bold text-white font-mono">{String(run.gold_answer_value || '—').slice(0, 12)}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Grounded Gold</p>
          </div>
        </Card>
      </div>

      {/* Stage Timeline */}
      <Card padding="none">
        <div className="px-5 py-3.5 border-b border-white/[0.06] flex items-center justify-between">
          <h2 className="text-xs font-semibold text-white uppercase tracking-wider">Trace Chronology</h2>
          <span className="text-[10px] text-slate-500">Click a stage to inspect active variables</span>
        </div>
        <div className="flex overflow-x-auto divide-x divide-white/[0.06]">
          {events.map((ev) => (
            <button
              key={ev.event_id}
              onClick={() => setSelectedStage(ev.stage)}
              className={`flex-1 min-w-[140px] p-4 text-left transition-all ${
                selectedStage === ev.stage ? 'bg-orange-600/10' : 'hover:bg-white/[0.01]'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <Badge variant={selectedStage === ev.stage ? 'brand' : 'neutral'} className="capitalize text-[10px]">
                  {ev.stage.replace('_', ' ')}
                </Badge>
                <span className="text-[10px] text-slate-500">{ev.duration_ms?.toFixed(0)} ms</span>
              </div>
              <p className="text-[11px] text-slate-400 line-clamp-1 leading-snug">{ev.message}</p>
            </button>
          ))}
        </div>
      </Card>

      {/* Stage Details Context Panels */}
      {activeEvent && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <div className="border-b border-white/[0.06] pb-3 mb-4">
              <h3 className="text-xs font-semibold text-orange-400 uppercase tracking-wider">Stage Context</h3>
            </div>
            <div className="space-y-3 text-xs leading-relaxed text-slate-300">
              <div>
                <span className="text-slate-500 block mb-0.5">Stage Name:</span>
                <span className="text-white capitalize font-semibold">{activeEvent.stage.replace('_', ' ')}</span>
              </div>
              <div>
                <span className="text-slate-500 block mb-0.5">Message:</span>
                <p className="text-slate-200">{activeEvent.message}</p>
              </div>
              {(activeEvent.record_count_in != null || activeEvent.record_count_out != null) && (
                <div className="flex gap-4">
                  {activeEvent.record_count_in != null && (
                    <div>
                      <span className="text-slate-500 block">Records In:</span>
                      <span className="font-bold text-white">{activeEvent.record_count_in}</span>
                    </div>
                  )}
                  {activeEvent.record_count_out != null && (
                    <div>
                      <span className="text-slate-500 block">Records Out:</span>
                      <span className="font-bold text-white">{activeEvent.record_count_out}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>

          <Card>
            <div className="border-b border-white/[0.06] pb-3 mb-4">
              <h3 className="text-xs font-semibold text-orange-400 uppercase tracking-wider flex items-center gap-1.5">
                <Terminal className="h-4 w-4 text-slate-400" />
                Raw Payload Metadata
              </h3>
            </div>
            {activeEvent.payload && Object.keys(activeEvent.payload).length > 0 ? (
              <pre className="p-3 rounded bg-white/[0.02] border border-white/[0.06] font-mono text-[10px] text-slate-400 max-h-[160px] overflow-y-auto scrollbar-thin">
                {JSON.stringify(activeEvent.payload, null, 2)}
              </pre>
            ) : (
              <div className="text-slate-600 text-xs py-4 text-center">No structural payload for this stage.</div>
            )}
          </Card>
        </div>
      )}

      {/* Oracle Diagnostics Lattice & Shapley Bars (WP7) */}
      {attribution && (
        <Card className="border-orange-500/20 bg-orange-600/[0.01]">
          <div className="px-5 py-4 border-b border-orange-500/20 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-orange-400 uppercase tracking-wider flex items-center gap-1.5">
                <TrendingUp className="h-4 w-4 text-orange-500" />
                Oracle Diagnostics Intervention Lattice
              </h3>
              <p className="text-[11px] text-orange-400/70 mt-0.5">
                Formally computes exact 3-player Shapley allocations over 8 counterfactual replacement subsets.
              </p>
            </div>
            {attribution.dominant_fault && (
              <Badge variant="orange" className="text-[10px]">
                Dominant Root Cause: {attribution.dominant_fault.toUpperCase()}
              </Badge>
            )}
          </div>
          <div className="p-5 space-y-6">
            {/* Grid of 8 subsets */}
            <div>
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">Counterfactual Intervention Nodes</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-3">
                {[
                  { key: 'none', label: 'None (Base)' },
                  { key: 'r', label: 'R (Scope)' },
                  { key: 'e', label: 'E (Facts)' },
                  { key: 'a', label: 'A (Agg)' },
                  { key: 're', label: 'R + E' },
                  { key: 'ra', label: 'R + A' },
                  { key: 'ea', label: 'E + A' },
                  { key: 'rea', label: 'REA (Oracle)' },
                ].map((node) => {
                  const isGold = node.key === 'rea';
                  const isBase = node.key === 'none';
                  const intervention = attribution.interventions?.[node.key];
                  return (
                    <div
                      key={node.key}
                      className={`p-3 rounded-lg border text-center transition-all ${
                        isGold
                          ? 'border-emerald-500/30 bg-emerald-500/5'
                          : isBase
                          ? 'border-red-500/30 bg-red-500/5'
                          : 'border-white/[0.06] bg-white/[0.02]'
                      }`}
                    >
                      <span className="block text-[10px] text-slate-500 uppercase font-semibold">{node.label}</span>
                      <span className="block font-mono text-xs font-bold text-white mt-1">
                        {intervention ? String(intervention.answer_value).slice(0, 8) : 'Unavailable'}
                      </span>
                      <Badge variant={isGold ? 'success' : isBase ? 'error' : 'neutral'} className="mt-2 text-[9px]">
                        {intervention ? `${Number(intervention.normalized_loss).toFixed(2)} loss` : 'not persisted'}
                      </Badge>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Shapley Bars Chart */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/[0.04]">
              <div>
                <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">Marginal Shapley Value Contributions ($\phi$)</p>
                <div className="space-y-3">
                  {attribution.components?.map((comp: any) => (
                    <div key={comp.component} className="text-xs">
                      <div className="flex justify-between mb-1">
                        <span className="capitalize text-slate-300 font-semibold">{comp.component}</span>
                        <span className="font-mono text-orange-400">{(comp.shapley_value * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-orange-500"
                          style={{ width: `${Math.max(0, comp.shapley_value) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Explainer /Drawer */}
              <div className="rounded-lg bg-orange-500/5 border border-orange-500/10 p-4 text-xs leading-relaxed text-slate-300">
                <span className="font-bold text-orange-400 block mb-1">Marginal Root-Cause Analysis</span>
                The Shapley attribution algorithm systematically swaps the gold components (Scope, Facts, Aggregation) in and out of the execution lattice.
                By evaluating the reduction in output loss across all combinations, we isolate the true error source:
                <p className="mt-2 text-slate-400 font-mono text-[10px]">
                  Dominant: {attribution.dominant_fault?.toUpperCase()} | Recovery Margin: {(attribution.total_error * 100 || 0).toFixed(0)}%
                </p>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Coverage Certificate matrix display (WP8) */}
      {certificate && (
        <Card className="border-violet-500/20 bg-violet-600/[0.01]">
          <div className="px-5 py-4 border-b border-violet-500/20">
            <h3 className="text-sm font-semibold text-violet-400 uppercase tracking-wider flex items-center gap-1.5">
              <ShieldCheck className="h-4 w-4 text-violet-400" />
              Selective Prediction Evidence Certificate
            </h3>
            <p className="text-[11px] text-violet-400/70 mt-0.5">
              Validates if the retrieved context records satisfy the operator-specific scope bounds before presenting answers.
            </p>
          </div>
          <div className="p-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Requirements matrix table */}
              <div>
                <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Requirements vs Observations Matrix</p>
                <div className="glass-card overflow-hidden">
                  <table className="w-full text-[11px] text-left">
                    <thead className="bg-white/[0.02] border-b border-white/[0.06] text-slate-500 uppercase">
                      <tr>
                        <th className="px-3 py-2">Metric</th>
                        <th className="px-3 py-2">Requirement</th>
                        <th className="px-3 py-2">Observation</th>
                        <th className="px-3 py-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-white/[0.04]">
                        <td className="px-3 py-2 font-semibold">Scope Coverage</td>
                        <td className="px-3 py-2">100% full bounds</td>
                        <td className="px-3 py-2">Not persisted</td>
                        <td className="px-3 py-2"><Badge variant="neutral">Unavailable</Badge></td>
                      </tr>
                      <tr className="border-b border-white/[0.04]">
                        <td className="px-3 py-2 font-semibold">Extraction Rows</td>
                        <td className="px-3 py-2">No missing values</td>
                        <td className="px-3 py-2">Not persisted</td>
                        <td className="px-3 py-2"><Badge variant="neutral">Unavailable</Badge></td>
                      </tr>
                      <tr>
                        <td className="px-3 py-2 font-semibold">Tie Resolution</td>
                        <td className="px-3 py-2">Resolved boundary</td>
                        <td className="px-3 py-2">Not persisted</td>
                        <td className="px-3 py-2"><Badge variant="neutral">Unavailable</Badge></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Certificate values details */}
              <div className="space-y-4 text-xs">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-slate-500 block">Certificate ID:</span>
                    <span className="font-mono text-slate-300 font-semibold">{certificate.certificate_id?.slice(0, 16)}…</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Decision Hash:</span>
                    <span className="font-mono text-slate-300 font-semibold">{certificate.certificate_hash?.slice(0, 16)}</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-slate-500 block">Presented Output:</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      {certificate.final_presented_answer !== null ? String(certificate.final_presented_answer) : 'ABSTAINED'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Abstention Codes:</span>
                    <span className="text-red-400 font-mono text-[10px]">
                      {certificate.abstention_reason || 'CERTIFIED'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
