'use client';

import { useCallback, useEffect, useState, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { api, World, Query } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Zap, Coins } from 'lucide-react';
import { formatMs } from '@/lib/utils';

function RunLabContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialQueryId = searchParams.get('query_id') || '';

  const [worlds, setWorlds] = useState<World[]>([]);
  const [queries, setQueries] = useState<Query[]>([]);
  const [selectedWorldId, setSelectedWorldId] = useState('');
  const [selectedQueryId, setSelectedQueryId] = useState(initialQueryId);
  const [selectedQuery, setSelectedQuery] = useState<Query | null>(null);

  // Pipeline execution parameters
  const [pipelineId, setPipelineId] = useState('P0-deterministic-scope-baseline');
  const [providerId, setProviderId] = useState('deterministic');
  const [retriever, setRetriever] = useState('bm25');
  const [topK, setTopK] = useState(10);
  const [batchSize, setBatchSize] = useState(10);
  const [repairPolicy, setRepairPolicy] = useState('strict_exact_v1');
  const [seed, setSeed] = useState(42);

  // States
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [dryRunEstimate, setDryRunEstimate] = useState<any>(null);

  const loadInitial = useCallback(async () => {
    setLoading(true);
    try {
      const [wList, qList] = await Promise.all([
        api.listWorlds(),
        api.listQueries(undefined, undefined, 1, 100)
      ]);
      setWorlds(wList);
      setQueries(qList.items);

      if (wList.length > 0) {
        setSelectedWorldId(wList[0].world_id);
      }

      if (initialQueryId) {
        const found = qList.items.find(q => q.query_id === initialQueryId);
        if (found) {
          setSelectedQuery(found);
          setSelectedWorldId(found.world_id);
        }
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [initialQueryId]);

  useEffect(() => {
    void loadInitial();
  }, [loadInitial]);

  // Load queries matching the selected world
  useEffect(() => {
    if (!selectedWorldId) return;
    api.listQueries(selectedWorldId, undefined, 1, 100).then((res) => {
      setQueries(res.items);
      setSelectedQueryId((currentQueryId) => {
        if (res.items.some(q => q.query_id === currentQueryId)) return currentQueryId;
        const firstQuery = res.items[0];
        setSelectedQuery(firstQuery || null);
        return firstQuery?.query_id || '';
      });
    });
  }, [selectedWorldId]);

  useEffect(() => {
    const q = queries.find(item => item.query_id === selectedQueryId);
    setSelectedQuery(q || null);
    setDryRunEstimate(null);
  }, [selectedQueryId, queries]);

  // Compute an explicitly simulated planning estimate. These values are not
  // measured results and are never sent to reporting endpoints.
  useEffect(() => {
    if (!selectedQueryId || !pipelineId) return;
    // Fixed planning assumptions for UI sizing only.
    const isLLM = !pipelineId.includes('deterministic') && !pipelineId.includes('P0');
    if (isLLM) {
      setDryRunEstimate({
        estimated_input_tokens: 1500,
        estimated_output_tokens: 350,
        estimated_cost_usd: 0.0045,
        estimated_latency_ms: 1200,
        is_safe: true,
      });
    } else {
      setDryRunEstimate({
        estimated_input_tokens: 0,
        estimated_output_tokens: 0,
        estimated_cost_usd: 0.0,
        estimated_latency_ms: 45,
        is_safe: true,
      });
    }
  }, [selectedQueryId, pipelineId, batchSize]);

  async function handleLaunch(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedQueryId) {
      setError('Please select a query to execute');
      return;
    }
    setRunning(true);
    setError('');

    try {
      const res = await api.createRun(selectedQueryId, pipelineId);
      // Automatically redirect to full trace panel after execution
      router.push(`/runs/${res.run_id}/trace`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  // Pre-validate configuration combinations
  const isLLMPipeline = !pipelineId.includes('deterministic') && !pipelineId.includes('P0');
  const showBatching = pipelineId.includes('P4') || pipelineId.includes('P5');
  const configurationHash = `cfg_${selectedQueryId.slice(0, 4)}_${pipelineId.slice(0, 4)}_${seed}`;

  return (
    <div className="p-8 animate-fade-in text-slate-100">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <Zap className="h-5 w-5 text-orange-500" />
          <span className="text-xs font-semibold uppercase tracking-widest text-orange-500 font-mono">Research Lab</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Experiment Launcher</h1>
        <p className="mt-1 text-sm text-slate-400">
          Configure, dry-run, and launch RAG pipeline permutations to compare exact error attributions.
        </p>
      </div>

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 p-4 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-center py-12 text-slate-500 text-sm">Loading config parameters...</div>
      ) : (
        <form onSubmit={handleLaunch} className="grid grid-cols-12 gap-6 text-xs">
          {/* Left panel: Launcher Controls */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            <Card>
              <div className="border-b border-white/[0.06] pb-3 mb-4">
                <h3 className="text-sm font-semibold text-white">Execution Setup</h3>
              </div>
              <div className="space-y-4">
                {/* World & Query */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-slate-400 font-semibold mb-1">Scale World (N)</label>
                    <select
                      value={selectedWorldId}
                      onChange={(e) => setSelectedWorldId(e.target.value)}
                      className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                    >
                      {worlds.map(w => (
                        <option key={w.world_id} value={w.world_id}>
                          N={w.scale_n} (seed={w.seed} — {w.world_id.slice(0, 16)}…)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-slate-400 font-semibold mb-1">Target Analytical Query</label>
                    <select
                      value={selectedQueryId}
                      onChange={(e) => setSelectedQueryId(e.target.value)}
                      className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                    >
                      {queries.map(q => (
                        <option key={q.query_id} value={q.query_id}>
                          [{q.family}] {q.natural_language_question.slice(0, 48)}…
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Pipeline Selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-slate-400 font-semibold mb-1">REA Pipeline Config</label>
                    <select
                      value={pipelineId}
                      onChange={(e) => setPipelineId(e.target.value)}
                      className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                    >
                      <option value="P0-deterministic-scope-baseline">P0: Deterministic Baseline (Oracle)</option>
                      <option value="P1-wrong-scope">P1: Scope Perturbation Fault</option>
                      <option value="P2-wrong-facts">P2: Fact Extraction Fault</option>
                      <option value="P3-wrong-aggregation">P3: Aggregation Reducer Fault</option>
                      <option value="P4-compound-scope-facts">P4: Compound R + E Fault</option>
                      <option value="P5-full-compound">P5: Full Compound R + E + A Fault</option>
                      <option value="P1-direct-bm25">P1 (Research): BM25 Retrieval</option>
                      <option value="P2-direct-dense">P2 (Research): Dense Embedding Retrieval</option>
                      <option value="P4-full-scope-mer">P4 (Research): Full Scope Map-Extract-Reduce</option>
                      <option value="P5-certified-mer-repair">P5 (Research): Certified MER with Auto-Repair</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-slate-400 font-semibold mb-1">Provider Engine</label>
                    <select
                      value={providerId}
                      onChange={(e) => setProviderId(e.target.value)}
                      disabled={!isLLMPipeline}
                      className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none disabled:opacity-40"
                    >
                      <option value="deterministic">Grounded Local Simulator</option>
                      <option value="openai">OpenAI (OpenAI API key required)</option>
                      <option value="anthropic">Anthropic (Anthropic key required)</option>
                      <option value="gemini">Google Gemini (Gemini key required)</option>
                    </select>
                  </div>
                </div>
              </div>
            </Card>

            {/* Fine-grained tuning (displays conditionally) */}
            {isLLMPipeline && (
              <Card className="animate-fade-in border-orange-500/10">
                <div className="border-b border-white/[0.06] pb-3 mb-4">
                  <h3 className="text-sm font-semibold text-white">LLM Context & Pipeline Parameters</h3>
                </div>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-slate-400 font-semibold mb-1">Retrieval Algorithm</label>
                      <select
                        value={retriever}
                        onChange={(e) => setRetriever(e.target.value)}
                        className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none"
                      >
                        <option value="bm25">BM25 Sparse</option>
                        <option value="dense">BGE-M3 Dense Index</option>
                        <option value="hybrid">Hybrid Reranked</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-slate-400 font-semibold mb-1">Retrieval Candidate Limit (K)</label>
                      <input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                        className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none focus:border-orange-500"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-400 font-semibold mb-1">Random Generation Seed</label>
                      <input
                        type="number"
                        value={seed}
                        onChange={(e) => setSeed(Number(e.target.value))}
                        className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none focus:border-orange-500"
                      />
                    </div>
                  </div>

                  {showBatching && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-white/[0.04] animate-fade-in">
                      <div>
                        <label className="block text-slate-400 font-semibold mb-1">Map Extract Batch Size</label>
                        <input
                          type="number"
                          value={batchSize}
                          onChange={(e) => setBatchSize(Number(e.target.value))}
                          className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none focus:border-orange-500"
                        />
                      </div>
                      <div>
                        <label className="block text-slate-400 font-semibold mb-1">Auto-Repair Policy</label>
                        <select
                          value={repairPolicy}
                          onChange={(e) => setRepairPolicy(e.target.value)}
                          className="w-full rounded bg-white/[0.03] border border-white/10 px-2 py-1.5 text-slate-200 outline-none"
                        >
                          <option value="strict_exact_v1">Strict exact match (v1)</option>
                          <option value="warn_partial_v1">Allow partial missingness (v1)</option>
                          <option value="none">No repair attempts</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}
          </div>

          {/* Right panel: Dry-run and execution summary */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <Card>
              <div className="border-b border-white/[0.06] pb-3 mb-4">
                <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                  <Coins className="h-4 w-4 text-orange-400" />
                  Dry-Run Estimation
                </h3>
              </div>
              {dryRunEstimate ? (
                <div className="space-y-4 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Estimated Tokens Input:</span>
                    <span className="font-semibold">{dryRunEstimate.estimated_input_tokens.toLocaleString()} tokens</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Estimated Tokens Output:</span>
                    <span className="font-semibold">{dryRunEstimate.estimated_output_tokens.toLocaleString()} tokens</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Estimated API Cost (USD):</span>
                    <span className="text-orange-400 font-bold">${dryRunEstimate.estimated_cost_usd.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Estimated Latency:</span>
                    <span className="text-slate-300">{formatMs(dryRunEstimate.estimated_latency_ms)}</span>
                  </div>
                  <div className="rounded-lg bg-orange-500/10 p-3 border border-orange-500/20 text-[10px] text-slate-400 italic leading-relaxed">
                    SIMULATED PLANNING ASSUMPTION — not a measured latency, token count, or charge.
                  </div>
                </div>
              ) : (
                <div className="text-slate-500 text-xs">Select parameters to calculate estimated footprint...</div>
              )}
            </Card>

            <Card>
              <div className="border-b border-white/[0.06] pb-3 mb-4">
                <h3 className="text-sm font-semibold text-white">Execution Config</h3>
              </div>
              <div className="space-y-3 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Config Hash:</span>
                  <span className="font-mono text-slate-400">{configurationHash}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Query Mode:</span>
                  <span className="text-orange-400 font-semibold">{selectedQuery?.family.toUpperCase()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">World:</span>
                  <span className="text-slate-300">world_s42_n{worlds.find(w => w.world_id === selectedWorldId)?.scale_n || '10'}</span>
                </div>

                <div className="pt-3 border-t border-white/[0.06]">
                  {selectedQuery ? (
                    <div className="space-y-1">
                      <span className="text-slate-500 block">Question text:</span>
                      <p className="text-slate-300 font-medium leading-relaxed italic">
                        "{selectedQuery.natural_language_question}"
                      </p>
                    </div>
                  ) : (
                    <div className="text-slate-600">No query selected</div>
                  )}
                </div>

                <Button
                  variant="primary"
                  size="md"
                  type="submit"
                  loading={running}
                  className="w-full bg-orange-600 hover:bg-orange-700 text-white mt-4 gap-1.5"
                >
                  <Zap className="h-4 w-4 fill-current text-white" />
                  Launch Experiment
                </Button>
              </div>
            </Card>
          </div>
        </form>
      )}
    </div>
  );
}

export default function RunLabPage() {
  return (
    <Suspense fallback={<div className="p-8 text-slate-500">Loading Experiment Lab...</div>}>
      <RunLabContent />
    </Suspense>
  );
}
