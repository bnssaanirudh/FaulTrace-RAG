import type { Metadata } from 'next';
export const metadata: Metadata = { title: 'Settings' };

export default function SettingsPage() {
  return (
    <div className="p-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="mt-1 text-sm text-slate-500">Configuration and environment details</p>
      </div>
      <div className="glass-card p-6 max-w-2xl">
        <h2 className="text-sm font-semibold text-white mb-4">Environment</h2>
        <div className="space-y-3 text-sm">
          {[
            ['API Base', 'http://localhost:8000'],
            ['Version', '0.1.0 — Prompt 1 (~30% complete)'],
            ['Active Pipeline', 'P0-deterministic-scope-baseline'],
            ['Gold Engine', 'Pandas + DuckDB (dual agreement)'],
            ['Default Seed', '42'],
            ['Database', 'SQLite (data/faulttrace.db)'],
          ].map(([k, v]) => (
            <div key={k} className="flex items-start gap-4">
              <span className="w-36 flex-shrink-0 text-slate-500 text-xs uppercase tracking-wider pt-0.5">{k}</span>
              <span className="mono text-slate-300 text-xs">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-6 pt-4 border-t border-white/[0.06]">
          <p className="text-xs text-slate-600">
            Set <code className="mono text-slate-400">FAULTTRACE_DATA_ROOT</code>,{' '}
            <code className="mono text-slate-400">FAULTTRACE_DATABASE_URL</code>, and{' '}
            <code className="mono text-slate-400">NEXT_PUBLIC_API_URL</code> environment variables to override defaults.
          </p>
        </div>
      </div>
    </div>
  );
}
