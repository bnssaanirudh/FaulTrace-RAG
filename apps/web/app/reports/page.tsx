'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/skeleton';
import {
  BarChart3,
  Download,
  RefreshCw,
  FileSpreadsheet,
  TrendingUp,
  Trophy,
  Activity,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { formatMs } from '@/lib/utils';

interface LeaderboardEntry {
  pipeline_id: string;
  total_runs: number;
  correct_runs: number;
  accuracy: number;
  mean_loss: number;
  mean_latency_ms: number;
}

interface LeaderboardData {
  leaderboard: LeaderboardEntry[];
  total_pipelines: number;
}

const PIPELINE_SHORT: Record<string, string> = {
  'P0-deterministic-scope-baseline': 'P0',
  'P1-wrong-scope': 'P1',
  'P1-direct-bm25': 'P1b',
  'P2-wrong-facts': 'P2',
  'P2-direct-dense': 'P2b',
  'P3-extract-aggregate': 'P3',
  'P3-wrong-aggregation': 'P3w',
  'P4-compound-scope-facts': 'P4',
  'P4-full-scope-mer': 'P4f',
  'P5-certified-mer': 'P5c',
  'P5-full-compound': 'P5f',
};

function shortLabel(pid: string) {
  return PIPELINE_SHORT[pid] ?? pid.split('-')[0];
}

export default function ReportsPage() {
  const [data, setData] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'accuracy' | 'latency'>('leaderboard');

  async function load() {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/v1/leaderboard');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  // Download leaderboard as JSON
  function downloadJSON() {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `faulttrace_leaderboard_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Download as CSV
  function downloadCSV() {
    if (!data) return;
    const header = ['pipeline_id', 'total_runs', 'correct_runs', 'accuracy', 'mean_loss', 'mean_latency_ms'].join(',');
    const rows = data.leaderboard.map(e =>
      [e.pipeline_id, e.total_runs, e.correct_runs, e.accuracy.toFixed(4), e.mean_loss.toFixed(4), e.mean_latency_ms.toFixed(1)].join(',')
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `faulttrace_leaderboard_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const chartData = data?.leaderboard.map(e => ({
    name: shortLabel(e.pipeline_id),
    accuracy: parseFloat((e.accuracy * 100).toFixed(1)),
    loss: parseFloat(e.mean_loss.toFixed(4)),
    latency: parseFloat(e.mean_latency_ms.toFixed(0)),
    runs: e.total_runs,
  })) ?? [];

  const tabs = [
    { key: 'leaderboard', label: 'Rankings Table', icon: Trophy },
    { key: 'accuracy', label: 'Accuracy Chart', icon: TrendingUp },
    { key: 'latency', label: 'Latency Chart', icon: Activity },
  ] as const;

  return (
    <div className="p-8 animate-fade-in text-slate-100 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="h-5 w-5 text-orange-500" />
            <span className="text-xs font-semibold uppercase tracking-widest text-orange-500">WP9</span>
          </div>
          <h1 className="text-3xl font-bold text-white">Reports &amp; Exports</h1>
          <p className="mt-1 text-sm text-slate-400">
            Pipeline leaderboard, accuracy vs. loss comparison, latency analysis, and data exports.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="ghost" size="sm" onClick={downloadCSV} disabled={!data}>
            <FileSpreadsheet className="h-3.5 w-3.5" />
            CSV
          </Button>
          <Button variant="ghost" size="sm" onClick={downloadJSON} disabled={!data}>
            <Download className="h-3.5 w-3.5" />
            JSON
          </Button>
        </div>
      </div>

      {/* Stats strip */}
      {data && (
        <div className="grid grid-cols-4 gap-4">
          <div className="glass-card p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Pipelines</p>
            <p className="text-2xl font-bold text-white">{data.total_pipelines}</p>
          </div>
          <div className="glass-card p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Total Runs</p>
            <p className="text-2xl font-bold text-white">
              {data.leaderboard.reduce((s, e) => s + e.total_runs, 0).toLocaleString()}
            </p>
          </div>
          <div className="glass-card p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Best Accuracy</p>
            <p className="text-2xl font-bold text-teal-400">
              {data.leaderboard.length > 0
                ? `${(Math.max(...data.leaderboard.map(e => e.accuracy)) * 100).toFixed(1)}%`
                : '—'}
            </p>
          </div>
          <div className="glass-card p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Fastest Pipeline</p>
            <p className="text-2xl font-bold text-orange-400">
              {data.leaderboard.length > 0
                ? formatMs(Math.min(...data.leaderboard.map(e => e.mean_latency_ms)))
                : '—'}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-500/10 p-3 text-sm text-red-300 ring-1 ring-red-500/25">
          {error} — ensure the backend is running and runs have been executed.
        </div>
      )}

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-white/[0.06] pb-0">
        {tabs.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-colors ${
              activeTab === key
                ? 'text-orange-400 border-b-2 border-orange-500 bg-orange-600/5'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Rankings Table */}
      {activeTab === 'leaderboard' && (
        <Card padding="none">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/[0.06]">
                  {['Rank', 'Pipeline', 'Accuracy', 'Correct / Total', 'Mean Loss', 'Mean Latency'].map(h => (
                    <th key={h} className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading && Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="border-b border-white/[0.04]">
                    {Array.from({ length: 6 }).map((_, j) => (
                      <td key={j} className="px-5 py-4">
                        <div className="h-4 rounded bg-white/[0.05] animate-pulse" />
                      </td>
                    ))}
                  </tr>
                ))}
                {!loading && data?.leaderboard.length === 0 && (
                  <tr>
                    <td colSpan={6}>
                      <EmptyState
                        icon={<BarChart3 className="h-6 w-6" />}
                        title="No runs yet"
                        description="Execute pipeline runs to see comparative rankings."
                      />
                    </td>
                  </tr>
                )}
                {!loading && data?.leaderboard.map((entry, idx) => (
                  <tr key={entry.pipeline_id} className="table-row-hover">
                    <td className="px-5 py-3.5">
                      <span className={`text-sm font-bold ${idx === 0 ? 'text-orange-400' : idx === 1 ? 'text-slate-300' : 'text-slate-500'}`}>
                        #{idx + 1}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div>
                        <p className="text-sm font-medium text-slate-200">{entry.pipeline_id}</p>
                        {entry.pipeline_id.includes('P0') && (
                          <Badge variant="neutral" className="mt-0.5 text-[10px]">Baseline</Badge>
                        )}
                      </div>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-2">
                        <div className="w-20 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-teal-500 rounded-full"
                            style={{ width: `${Math.min(100, entry.accuracy * 100)}%` }}
                          />
                        </div>
                        <span className="text-sm font-bold text-teal-400">
                          {(entry.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-5 py-3.5 text-sm text-slate-400">
                      <span className="text-emerald-400 font-semibold">{entry.correct_runs}</span>
                      <span className="text-slate-600 mx-1">/</span>
                      {entry.total_runs}
                    </td>
                    <td className="px-5 py-3.5 text-sm font-mono text-orange-400">
                      {entry.mean_loss.toFixed(4)}
                    </td>
                    <td className="px-5 py-3.5 text-sm text-slate-400">
                      {formatMs(entry.mean_latency_ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Accuracy Chart */}
      {activeTab === 'accuracy' && (
        <Card>
          <div className="px-5 py-4 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-white">Accuracy vs. Mean Loss by Pipeline</h3>
            <p className="text-xs text-slate-500 mt-0.5">n = total runs per pipeline. Higher accuracy / lower loss = better.</p>
          </div>
          <div className="p-5">
            {chartData.length === 0 ? (
              <EmptyState title="No data" description="Execute pipeline runs to generate chart data." />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis yAxisId="left" tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={v => `${v}%`} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: '#18191c', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                    labelStyle={{ color: '#f1f1f1', fontWeight: 600 }}
                    itemStyle={{ color: '#9ca3af' }}
                  />
                  <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 11 }} />
                  <Bar yAxisId="left" dataKey="accuracy" name="Accuracy (%)" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                  <Bar yAxisId="right" dataKey="loss" name="Mean Loss" fill="#ea580c" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      )}

      {/* Latency Chart */}
      {activeTab === 'latency' && (
        <Card>
          <div className="px-5 py-4 border-b border-white/[0.06]">
            <h3 className="text-sm font-semibold text-white">Mean Latency by Pipeline (ms)</h3>
            <p className="text-xs text-slate-500 mt-0.5">Lower = faster execution. Deterministic pipelines (P0) should be fastest.</p>
          </div>
          <div className="p-5">
            {chartData.length === 0 ? (
              <EmptyState title="No data" description="Execute pipeline runs to generate latency data." />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={v => `${v}ms`} />
                  <Tooltip
                    contentStyle={{ background: '#18191c', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                    labelStyle={{ color: '#f1f1f1', fontWeight: 600 }}
                    formatter={(v: number) => [`${v.toFixed(0)}ms`, 'Mean Latency']}
                  />
                  <Bar dataKey="latency" name="Latency (ms)" fill="#f97316" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
