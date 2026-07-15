'use client';

import { useEffect, useState } from 'react';
import { Activity, CheckCircle2, ChevronLeft, ChevronRight, Play, XCircle, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { api, Run } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatMs, formatDate } from '@/lib/utils';

export function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [traceEvents, setTraceEvents] = useState<Record<string, unknown>[]>([]);
  const [traceLoading, setTraceLoading] = useState(false);

  async function load(p = page) {
    setLoading(true);
    try {
      const res = await api.listRuns(undefined, undefined, p, 20);
      setRuns(res.items);
      setTotal(res.total);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function loadTrace(runId: string) {
    setSelectedRun(runId);
    setTraceLoading(true);
    try {
      const events = await api.getRunTrace(runId);
      setTraceEvents(events as unknown as Record<string, unknown>[]);
    } catch {
      setTraceEvents([]);
    } finally {
      setTraceLoading(false);
    }
  }

  useEffect(() => { load(page); }, [page]);

  const totalPages = Math.ceil(total / 20);
  const correctCount = runs.filter((r) => r.is_correct === true).length;
  const failedCount = runs.filter((r) => r.status === 'failed').length;

  return (
    <div className="p-8 animate-fade-in">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <Activity className="h-5 w-5 text-brand-400" />
          <span className="text-xs font-semibold uppercase tracking-widest text-brand-400">Execution History</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Pipeline Runs</h1>
        <p className="mt-1 text-sm text-slate-500">
          Traced execution history with gold comparison and artifact references
        </p>
      </div>

      {/* Quick stats */}
      <div className="mb-6 flex gap-4">
        <div className="glass-card px-4 py-3 flex items-center gap-3">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <div>
            <p className="text-lg font-bold text-white">{correctCount}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Correct</p>
          </div>
        </div>
        <div className="glass-card px-4 py-3 flex items-center gap-3">
          <XCircle className="h-4 w-4 text-red-400" />
          <div>
            <p className="text-lg font-bold text-white">{failedCount}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Failed</p>
          </div>
        </div>
        <div className="glass-card px-4 py-3 flex items-center gap-3">
          <Play className="h-4 w-4 text-brand-400" />
          <div>
            <p className="text-lg font-bold text-white">{total}</p>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Total</p>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Runs table */}
        <div className={selectedRun ? 'col-span-7' : 'col-span-12'}>
          <Card padding="none">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    {['Status', 'Run ID', 'Pipeline', 'Answer', 'Gold', 'Latency', 'Started'].map((h) => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loading && [...Array(8)].map((_, i) => (
                    <tr key={i} className="border-b border-white/[0.04]">
                      {[...Array(7)].map((_, j) => (
                        <td key={j} className="px-4 py-4">
                          <div className="h-4 rounded bg-white/[0.05] animate-pulse" />
                        </td>
                      ))}
                    </tr>
                  ))}
                  {!loading && runs.map((run) => (
                    <tr
                      key={run.run_id}
                      className={`table-row-hover cursor-pointer ${selectedRun === run.run_id ? 'bg-brand-600/10' : ''}`}
                      onClick={() => loadTrace(run.run_id)}
                    >
                      <td className="px-4 py-3.5">
                        {run.is_correct === true ? (
                          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                        ) : run.is_correct === false ? (
                          <XCircle className="h-4 w-4 text-red-400" />
                        ) : (
                          <div className="h-4 w-4 rounded-full bg-slate-600" />
                        )}
                      </td>
                      <td className="px-4 py-3.5">
                        <p className="mono text-xs text-brand-400">{run.run_id.slice(0, 10)}…</p>
                      </td>
                      <td className="px-4 py-3.5">
                        <Badge variant="neutral">{run.pipeline_id.split('-')[0]}</Badge>
                      </td>
                      <td className="px-4 py-3.5 mono text-xs text-slate-300">
                        {run.answer != null ? String(run.answer).slice(0, 16) : '—'}
                      </td>
                      <td className="px-4 py-3.5 mono text-xs text-emerald-400">
                        {run.gold_answer_value != null ? String(run.gold_answer_value).slice(0, 16) : '—'}
                      </td>
                      <td className="px-4 py-3.5 text-xs text-slate-400">
                        {formatMs(run.latency_ms)}
                      </td>
                      <td className="px-4 py-3.5 text-xs text-slate-600">
                        {formatDate(run.started_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between px-5 py-4 border-t border-white/[0.06]">
              <p className="text-xs text-slate-500">
                {total.toLocaleString()} total runs
              </p>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                  <ChevronLeft className="h-3.5 w-3.5" />
                </Button>
                <span className="text-xs text-slate-400">{page} / {totalPages || 1}</span>
                <Button size="sm" variant="ghost" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </Card>
        </div>

        {/* Trace panel */}
        {selectedRun && (
          <div className="col-span-5">
            <Card padding="none">
              <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Execution Trace</h3>
                <Button size="sm" variant="ghost" onClick={() => setSelectedRun(null)}>✕</Button>
              </div>
              <div className="overflow-y-auto max-h-[600px] p-4 space-y-2 scrollbar-thin">
                {traceLoading && (
                  <div className="text-center py-8 text-slate-500 text-sm">Loading trace…</div>
                )}
                {!traceLoading && traceEvents.map((ev: Record<string, unknown>, i) => (
                  <div key={String(ev.event_id) || i} className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-semibold text-brand-400 uppercase tracking-wider">
                        {String(ev.stage)}
                      </span>
                      <span className="text-[10px] text-slate-500">
                        {ev.duration_ms != null ? `${Number(ev.duration_ms).toFixed(1)}ms` : ''}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{String(ev.message)}</p>
                    {(ev.record_count_in != null || ev.record_count_out != null) && (
                      <div className="mt-1.5 flex items-center gap-3 text-[10px] text-slate-500">
                        {ev.record_count_in != null && <span>in: {String(ev.record_count_in)}</span>}
                        {ev.record_count_out != null && <span>out: {String(ev.record_count_out)}</span>}
                      </div>
                    )}
                  </div>
                ))}
                
                {!traceLoading && traceEvents.length > 0 && (
                  <div className="mt-4 flex justify-center pb-4">
                    <Link href={`/runs/${selectedRun}/trace`}>
                      <Button variant="ghost" size="sm" className="gap-2 text-brand-300 hover:text-brand-200">
                        View Full Trace & Attribution <ArrowRight className="h-3 w-3" />
                      </Button>
                    </Link>
                  </div>
                )}
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
