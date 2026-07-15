'use client';

import { useEffect, useState } from 'react';
import {
  BarChart3,
  CheckCircle2,
  Database,
  HelpCircle,
  Play,
  RefreshCw,
  Sparkles,
  XCircle,
  Zap,
} from 'lucide-react';
import { api, SystemStatus, World, Run } from '@/lib/api';
import { StatCard } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatMs, formatDate, statusBadgeClass, familyBadgeClass } from '@/lib/utils';

export function DashboardPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [worlds, setWorlds] = useState<World[]>([]);
  const [recentRuns, setRecentRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [seeding, setSeeding] = useState(false);
  const [seedMsg, setSeedMsg] = useState('');
  const [error, setError] = useState('');

  async function load() {
    setLoading(true);
    setError('');
    try {
      const [s, w, r] = await Promise.all([
        api.status(),
        api.listWorlds(),
        api.listRuns(undefined, undefined, 1, 10),
      ]);
      setStatus(s);
      setWorlds(w);
      setRecentRuns(r.items);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSeed() {
    setSeeding(true);
    setSeedMsg('');
    try {
      const res = await api.seedDemo(42, [10, 50, 200, 1000], false);
      setSeedMsg(`✓ Seeded ${res.world_ids.length} worlds · ${res.queries_generated} queries generated`);
      await load();
    } catch (e: unknown) {
      setSeedMsg(`✗ ${(e as Error).message}`);
    } finally {
      setSeeding(false);
    }
  }

  useEffect(() => { load(); }, []);

  const correctRuns = recentRuns.filter((r) => r.is_correct === true).length;
  const totalRuns = recentRuns.length;

  return (
    <div className="min-h-screen p-8 animate-fade-in">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-brand-400" />
            <span className="text-xs font-semibold uppercase tracking-widest text-brand-400">
              FaultTrace-RAG
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Analytics Dashboard
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Counterfactual fault localization for corpus-level LLM pipelines
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleSeed}
            loading={seeding}
            disabled={seeding}
          >
            <Zap className="h-3.5 w-3.5" />
            Seed Demo
          </Button>
        </div>
      </div>

      {/* Seed message */}
      {seedMsg && (
        <div className={`mb-6 rounded-lg px-4 py-3 text-sm font-medium ${
          seedMsg.startsWith('✓')
            ? 'bg-emerald-500/10 text-emerald-300 ring-1 ring-emerald-500/25'
            : 'bg-red-500/10 text-red-300 ring-1 ring-red-500/25'
        }`}>
          {seedMsg}
        </div>
      )}

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-300 ring-1 ring-red-500/25">
          API unreachable: {error} — Start the backend with <code className="mono">make api</code>
        </div>
      )}

      {/* Stat cards */}
      <div className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          label="Corpus Worlds"
          value={status?.counts.worlds ?? '—'}
          icon={<Database className="h-5 w-5" />}
          glow="brand"
        />
        <StatCard
          label="Queries"
          value={status?.counts.queries ?? '—'}
          icon={<HelpCircle className="h-5 w-5" />}
          glow="gold"
        />
        <StatCard
          label="Pipeline Runs"
          value={status?.counts.runs ?? '—'}
          icon={<Play className="h-5 w-5" />}
          glow="emerald"
        />
        <StatCard
          label="Accuracy (Recent)"
          value={totalRuns > 0 ? `${Math.round((correctRuns / totalRuns) * 100)}%` : '—'}
          icon={<BarChart3 className="h-5 w-5" />}
        />
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Worlds */}
        <div className="col-span-4">
          <p className="section-heading">Corpus Worlds</p>
          <div className="space-y-2">
            {worlds.length === 0 && !loading && (
              <div className="glass-card p-4 text-sm text-slate-500">
                No worlds yet. Click <strong>Seed Demo</strong> to generate.
              </div>
            )}
            {worlds.map((w) => (
              <a
                key={w.world_id}
                href={`/worlds/${w.world_id}`}
                className="glass-card-hover flex items-center justify-between p-4 cursor-pointer block"
              >
                <div>
                  <p className="text-sm font-semibold text-white">
                    N={w.scale_n.toLocaleString()}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500 mono">
                    {w.world_id.slice(0, 20)}…
                  </p>
                </div>
                <div className="text-right">
                  <Badge variant="brand">seed={w.seed}</Badge>
                  <p className="mt-1 text-[10px] text-slate-600">{w.creation_policy}</p>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Pipeline status */}
        <div className="col-span-4">
          <p className="section-heading">Pipeline Registry</p>
          <div className="space-y-2">
            {status && Object.entries(status.pipelines).map(([pid, state]) => (
              <div key={pid} className="glass-card flex items-center justify-between p-3.5">
                <div>
                  <p className="text-xs font-semibold text-white">{pid}</p>
                </div>
                <Badge variant={state === 'available' ? 'success' : 'neutral'}>
                  {state}
                </Badge>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Runs */}
        <div className="col-span-4">
          <p className="section-heading">Recent Runs</p>
          <div className="space-y-2">
            {recentRuns.length === 0 && !loading && (
              <div className="glass-card p-4 text-sm text-slate-500">
                No runs yet. Execute a query to see results here.
              </div>
            )}
            {recentRuns.map((run) => (
              <a
                key={run.run_id}
                href={`/runs/${run.run_id}`}
                className="glass-card-hover flex items-center gap-3 p-3.5 block cursor-pointer"
              >
                <div>
                  {run.is_correct === true ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  ) : run.is_correct === false ? (
                    <XCircle className="h-4 w-4 text-red-400" />
                  ) : (
                    <div className="h-4 w-4 rounded-full bg-slate-600" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate text-xs font-medium text-slate-300 mono">
                    {run.run_id.slice(0, 12)}…
                  </p>
                  <p className="text-[10px] text-slate-600">{run.pipeline_id.split('-')[0]}</p>
                </div>
                <div className="text-right">
                  <Badge variant={run.status === 'completed' ? 'success' : run.status === 'failed' ? 'error' : 'neutral'}>
                    {run.status}
                  </Badge>
                  <p className="mt-1 text-[10px] text-slate-600">{formatMs(run.latency_ms)}</p>
                </div>
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* System info */}
      {status && (
        <div className="mt-8 glass-card p-5">
          <p className="section-heading">System Components</p>
          <div className="flex flex-wrap gap-3">
            {Object.entries(status.components).map(([name, state]) => (
              <div
                key={name}
                className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium ring-1 ${
                  state === 'ok'
                    ? 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/25'
                    : 'bg-red-500/10 text-red-300 ring-red-500/25'
                }`}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${state === 'ok' ? 'bg-emerald-400' : 'bg-red-400'}`} />
                {name}: {state}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
