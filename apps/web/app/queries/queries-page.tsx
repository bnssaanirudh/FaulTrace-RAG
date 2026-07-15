'use client';

import { useEffect, useState } from 'react';
import { HelpCircle, Play, ChevronLeft, ChevronRight } from 'lucide-react';
import { api, Query, Run } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { familyBadgeClass } from '@/lib/utils';

const FAMILIES = ['', 'count', 'mean', 'proportion', 'comparison', 'top_k', 'trend'];

export function QueriesPage() {
  const [queries, setQueries] = useState<Query[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [family, setFamily] = useState('');
  const [loading, setLoading] = useState(true);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<Record<string, Run>>({});
  const [error, setError] = useState('');

  async function load(p = page, f = family) {
    setLoading(true);
    try {
      const res = await api.listQueries(undefined, f || undefined, p, 20);
      setQueries(res.items);
      setTotal(res.total);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(page, family); }, [page, family]);

  async function executeQuery(queryId: string) {
    setRunningId(queryId);
    try {
      const run = await api.createRun(queryId);
      setRunResult((prev) => ({ ...prev, [queryId]: run }));
    } catch (e: unknown) {
      console.error(e);
    } finally {
      setRunningId(null);
    }
  }

  const totalPages = Math.ceil(total / 20);

  return (
    <div className="p-8 animate-fade-in">
      <div className="mb-8 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <HelpCircle className="h-5 w-5 text-brand-400" />
            <span className="text-xs font-semibold uppercase tracking-widest text-brand-400">Query Bank</span>
          </div>
          <h1 className="text-3xl font-bold text-white">Procedural Queries</h1>
          <p className="mt-1 text-sm text-slate-500">
            {total.toLocaleString()} grounded queries across 6 aggregation families
          </p>
        </div>
      </div>

      {/* Family filter */}
      <div className="mb-6 flex flex-wrap gap-2">
        {FAMILIES.map((f) => (
          <button
            key={f || 'all'}
            onClick={() => { setFamily(f); setPage(1); }}
            className={`rounded-full px-3 py-1.5 text-xs font-semibold transition-all duration-150 ${
              family === f
                ? 'bg-brand-600 text-white shadow-glow-brand'
                : 'bg-white/[0.05] text-slate-400 hover:bg-white/[0.09] hover:text-white'
            }`}
          >
            {f === '' ? 'All Families' : f.toUpperCase()}
          </button>
        ))}
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      <Card padding="none">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/[0.06]">
                {['Family', 'Question', 'Template', 'Gold Answer', 'Actions'].map((h) => (
                  <th key={h} className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading &&
                [...Array(8)].map((_, i) => (
                  <tr key={i} className="border-b border-white/[0.04]">
                    {[...Array(5)].map((_, j) => (
                      <td key={j} className="px-5 py-4">
                        <div className="h-4 rounded bg-white/[0.05] animate-pulse" />
                      </td>
                    ))}
                  </tr>
                ))}
              {!loading && queries.map((q) => {
                const run = runResult[q.query_id];
                return (
                  <tr key={q.query_id} className="table-row-hover">
                    <td className="px-5 py-3.5">
                      <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${familyBadgeClass(q.family)}`}>
                        {q.family.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 max-w-[320px]">
                      <p className="text-xs text-slate-300 leading-relaxed line-clamp-2">
                        {q.natural_language_question}
                      </p>
                      <p className="mt-1 text-[10px] text-slate-600 mono">{q.query_id.slice(0, 12)}…</p>
                    </td>
                    <td className="px-5 py-3.5">
                      <Badge variant="neutral">{q.template_id}</Badge>
                    </td>
                    <td className="px-5 py-3.5">
                      {q.gold ? (
                        <div>
                          <p className="text-xs font-mono text-emerald-400 font-semibold">
                            {JSON.stringify(q.gold.answer_value).slice(0, 20)}
                          </p>
                          <p className="text-[10px] text-slate-600 mt-0.5">
                            {q.gold.agreement_status}
                          </p>
                        </div>
                      ) : (
                        <span className="text-slate-600 text-xs">—</span>
                      )}
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          loading={runningId === q.query_id}
                          onClick={() => executeQuery(q.query_id)}
                          disabled={runningId !== null}
                        >
                          <Play className="h-3 w-3" />
                          Run P0
                        </Button>
                        {run && (
                          <Badge variant={run.is_correct ? 'success' : 'error'}>
                            {run.is_correct ? '✓ correct' : '✗ wrong'}
                          </Badge>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between px-5 py-4 border-t border-white/[0.06]">
          <p className="text-xs text-slate-500">
            Showing {Math.min((page - 1) * 20 + 1, total)}–{Math.min(page * 20, total)} of {total}
          </p>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="text-xs text-slate-400">
              {page} / {totalPages}
            </span>
            <Button size="sm" variant="ghost" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
