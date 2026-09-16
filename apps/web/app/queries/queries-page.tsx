'use client';

import { useCallback, useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { HelpCircle, Play, ChevronLeft, ChevronRight, Download, Eye, Clipboard } from 'lucide-react';
import { api, Query, Run } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { familyBadgeClass } from '@/lib/utils';

const FAMILIES = ['', 'count', 'mean', 'proportion', 'comparison', 'top_k', 'trend'];
const DIFFICULTIES = ['', 'easy', 'medium', 'adversarial'];
const SELECTIVITIES = ['', 'broad', 'narrow', 'selective'];
const SPLITS = ['', 'dev', 'val', 'test'];

export function QueriesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Load filters from URL
  const initialFamily = searchParams.get('family') || '';
  const initialDifficulty = searchParams.get('difficulty') || '';
  const initialSplit = searchParams.get('split') || '';

  const [queries, setQueries] = useState<Query[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [family, setFamily] = useState(initialFamily);
  const [difficulty, setDifficulty] = useState(initialDifficulty);
  const [selectivity, setSelectivity] = useState('');
  const [split, setSplit] = useState(initialSplit);
  const [goldReadyOnly, setGoldReadyOnly] = useState(false);
  const [viewMode, setViewMode] = useState<'compact' | 'expanded'>('expanded');
  const [loading, setLoading] = useState(true);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<Record<string, Run>>({});
  const [error, setError] = useState('');

  // Selected query detail modal
  const [selectedQuery, setSelectedQuery] = useState<Query | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const load = useCallback(async (p: number, f: string, d: string, s: string) => {
    setLoading(true);
    try {
      const res = await api.listQueries(undefined, f || undefined, p, 20);
      
      // Client-side filtering for difficulty, split, and selectivity since backend paginates primarily on family
      let items = res.items;
      if (d) {
        items = items.filter((item: any) => item.spec?.difficulty === d);
      }
      if (s) {
        items = items.filter((item: any) => item.spec?.split === s);
      }
      if (selectivity) {
        items = items.filter((item: any) => item.spec?.selectivity === selectivity);
      }
      if (goldReadyOnly) {
        items = items.filter((item: any) => item.gold !== null);
      }

      setQueries(items);
      setTotal(res.total);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [goldReadyOnly, selectivity]);

  useEffect(() => {
    // Update URL params
    const params = new URLSearchParams();
    if (family) params.set('family', family);
    if (difficulty) params.set('difficulty', difficulty);
    if (split) params.set('split', split);
    
    const qs = params.toString();
    router.replace(`/queries${qs ? '?' + qs : ''}`);

    void load(page, family, difficulty, split);
  }, [page, family, difficulty, selectivity, split, goldReadyOnly, load, router]);

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

  function handleCopySpec(query: Query) {
    navigator.clipboard.writeText(JSON.stringify(query.spec, null, 2));
    setCopiedId(query.query_id);
    setTimeout(() => setCopiedId(null), 2000);
  }

  function handleExportSelected() {
    const specs = queries.map(q => q.spec);
    const blob = new Blob([JSON.stringify(specs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query_specs_${family || 'all'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const totalPages = Math.ceil(total / 20);

  return (
    <div className="p-8 animate-fade-in text-slate-100">
      <div className="mb-8 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <HelpCircle className="h-5 w-5 text-orange-500" />
            <span className="text-xs font-semibold uppercase tracking-widest text-orange-500">Query Bank</span>
          </div>
          <h1 className="text-3xl font-bold text-white font-sans">Query Library</h1>
          <p className="mt-1 text-sm text-slate-400">
            {total.toLocaleString()} grounded queries for measuring RAG errors across splits and categories.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={handleExportSelected} className="gap-1 text-slate-400 hover:text-white">
            <Download className="h-3.5 w-3.5" /> Export Specs
          </Button>
          <div className="flex rounded-lg border border-white/[0.06] p-0.5 bg-white/[0.02]">
            <button
              onClick={() => setViewMode('compact')}
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all ${
                viewMode === 'compact' ? 'bg-orange-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Compact
            </button>
            <button
              onClick={() => setViewMode('expanded')}
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all ${
                viewMode === 'expanded' ? 'bg-orange-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Expanded
            </button>
          </div>
        </div>
      </div>

      {/* Advanced Filters */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-5 gap-3 text-xs">
        <div>
          <label className="block text-slate-500 mb-1 font-semibold uppercase tracking-wider text-[10px]">Family</label>
          <select
            value={family}
            onChange={(e) => { setFamily(e.target.value); setPage(1); }}
            className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none"
          >
            <option value="">All Families</option>
            {FAMILIES.filter(Boolean).map(f => <option key={f} value={f}>{f.toUpperCase()}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-slate-500 mb-1 font-semibold uppercase tracking-wider text-[10px]">Difficulty</label>
          <select
            value={difficulty}
            onChange={(e) => { setDifficulty(e.target.value); setPage(1); }}
            className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none"
          >
            <option value="">All Difficulties</option>
            {DIFFICULTIES.filter(Boolean).map(d => <option key={d} value={d}>{d.toUpperCase()}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-slate-500 mb-1 font-semibold uppercase tracking-wider text-[10px]">Selectivity</label>
          <select
            value={selectivity}
            onChange={(e) => { setSelectivity(e.target.value); setPage(1); }}
            className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none"
          >
            <option value="">All Selectivity</option>
            {SELECTIVITIES.filter(Boolean).map(s => <option key={s} value={s}>{s.toUpperCase()}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-slate-500 mb-1 font-semibold uppercase tracking-wider text-[10px]">Split</label>
          <select
            value={split}
            onChange={(e) => { setSplit(e.target.value); setPage(1); }}
            className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none"
          >
            <option value="">All Splits</option>
            {SPLITS.filter(Boolean).map(s => <option key={s} value={s}>{s.toUpperCase()}</option>)}
          </select>
        </div>

        <div className="flex items-end pb-1.5">
          <label className="flex items-center gap-2 cursor-pointer text-slate-400">
            <input
              type="checkbox"
              checked={goldReadyOnly}
              onChange={(e) => setGoldReadyOnly(e.target.checked)}
              className="rounded bg-white/5 border-white/10 text-orange-600 focus:ring-0"
            />
            <span>Grounded Gold Ready Only</span>
          </label>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Table View */}
        <div className={selectedQuery ? 'col-span-8' : 'col-span-12'}>
          <Card padding="none">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/[0.06] text-slate-500 uppercase">
                    <th className="px-4 py-3">Family</th>
                    <th className="px-4 py-3">Question</th>
                    {viewMode === 'expanded' && (
                      <>
                        <th className="px-4 py-3">Difficulty</th>
                        <th className="px-4 py-3">Split</th>
                      </>
                    )}
                    <th className="px-4 py-3">Gold Answer</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {loading &&
                    [...Array(6)].map((_, i) => (
                      <tr key={i} className="border-b border-white/[0.04]">
                        {[...Array(viewMode === 'expanded' ? 6 : 4)].map((_, j) => (
                          <td key={j} className="px-4 py-4">
                            <div className="h-4 rounded bg-white/[0.05] animate-pulse" />
                          </td>
                        ))}
                      </tr>
                    ))}
                  {!loading && queries.map((q: any) => {
                    const run = runResult[q.query_id];
                    return (
                      <tr key={q.query_id} className="table-row-hover border-b border-white/[0.04]">
                        <td className="px-4 py-3.5">
                          <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${familyBadgeClass(q.family)}`}>
                            {q.family.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3.5 max-w-[360px]">
                          <p className="text-slate-200 leading-relaxed font-medium">
                            {q.natural_language_question}
                          </p>
                          {viewMode === 'expanded' && (
                            <p className="mt-1 text-[9px] text-slate-600 mono">{q.query_id}</p>
                          )}
                        </td>
                        {viewMode === 'expanded' && (
                          <>
                            <td className="px-4 py-3.5 capitalize">
                              <Badge variant={q.spec?.difficulty === 'adversarial' ? 'error' : q.spec?.difficulty === 'medium' ? 'warning' : 'brand'}>
                                {q.spec?.difficulty || 'manual'}
                              </Badge>
                            </td>
                            <td className="px-4 py-3.5 uppercase text-slate-400 font-semibold">{q.spec?.split || 'dev'}</td>
                          </>
                        )}
                        <td className="px-4 py-3.5">
                          {q.gold ? (
                            <div>
                              <p className="font-mono text-emerald-400 font-bold">
                                {JSON.stringify(q.gold.answer_value).slice(0, 24)}
                              </p>
                              <p className="text-[9px] text-slate-500 mt-0.5">
                                counts: {q.gold.eligible_record_count} records
                              </p>
                            </div>
                          ) : (
                            <span className="text-slate-600 font-mono">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3.5 text-right">
                          <div className="flex items-center justify-end gap-1.5">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setSelectedQuery(q)}
                              className="text-slate-400 hover:text-white"
                            >
                              <Eye className="h-3 w-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              loading={runningId === q.query_id}
                              onClick={() => executeQuery(q.query_id)}
                              disabled={runningId !== null}
                              className="text-orange-500 hover:bg-orange-600/10"
                            >
                              <Play className="h-3 w-3 fill-current" /> Run P0
                            </Button>
                            {run && (
                              <Badge variant={run.is_correct ? 'success' : 'error'}>
                                {run.is_correct ? '✓' : '✗'}
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

        {/* Selected Query Details Modal/Drawer */}
        {selectedQuery && (
          <div className="col-span-4 animate-fade-in">
            <Card>
              <div className="border-b border-white/[0.06] pb-3 mb-4 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Query Specifications</h3>
                  <p className="text-[10px] text-slate-500 mt-0.5">AST Structure & Evidence Rules</p>
                </div>
                <Button size="sm" variant="ghost" onClick={() => setSelectedQuery(null)}>✕</Button>
              </div>
              <div className="space-y-4 text-xs">
                <div>
                  <span className="text-slate-500 block mb-1">Question:</span>
                  <p className="text-slate-200 font-semibold leading-relaxed">{selectedQuery.natural_language_question}</p>
                </div>

                <div>
                  <span className="text-slate-500 block mb-1">Scope Predicate AST:</span>
                  <pre className="p-2.5 rounded bg-white/[0.02] border border-white/[0.06] font-mono text-[10px] text-slate-400 overflow-x-auto">
                    {JSON.stringify((selectedQuery.spec as any)?.scope_predicate, null, 2)}
                  </pre>
                </div>

                <div>
                  <span className="text-slate-500 block mb-1">Fact Fields:</span>
                  <div className="flex flex-wrap gap-1">
                    {(selectedQuery.spec as any)?.fact_spec?.fields?.map((f: string) => (
                      <Badge key={f} variant="neutral">{f}</Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <span className="text-slate-500 block mb-1">Evidence completeness rule:</span>
                  <p className="text-slate-300 font-medium">{(selectedQuery.spec as any)?.expected_evidence_requirement || 'All matching records'}</p>
                </div>

                <div className="flex gap-2 pt-2 border-t border-white/[0.04]">
                  <Button size="sm" variant="ghost" onClick={() => handleCopySpec(selectedQuery)} className="gap-1 flex-1">
                    <Clipboard className="h-3 w-3" />
                    {copiedId === selectedQuery.query_id ? 'Copied!' : 'Copy Spec JSON'}
                  </Button>
                  <Link href={`/run-lab?query_id=${selectedQuery.query_id}`} className="flex-1">
                    <Button size="sm" variant="primary" className="bg-orange-600 hover:bg-orange-700 text-white gap-1 w-full">
                      <Play className="h-3 w-3 fill-current" /> Launch Lab
                    </Button>
                  </Link>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
