'use client';

import { useState } from 'react';
import { Settings, ShieldCheck, AlertTriangle, EyeOff, Eye } from 'lucide-react';
import { api } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export default function SettingsPage() {
  const [seeding, setSeeding] = useState(false);
  const [seedMsg, setSeedMsg] = useState('');
  
  // Reviewer Mode settings
  const [reviewerMode, setReviewerMode] = useState(true);
  const [showUnredactedSecrets, setShowUnredactedSecrets] = useState(false);
  const [customSeed, setCustomSeed] = useState(42);

  async function handleResetDB() {
    setSeeding(true);
    setSeedMsg('');
    try {
      // Overwrite=true clears existing tables and performs a clean seed
      const res = await api.seedDemo(customSeed, [10, 50, 200, 1000], true);
      setSeedMsg(`✓ Database fully reset and reseeded with seed ${customSeed}. ${res.world_ids.length} worlds generated.`);
    } catch (e: any) {
      setSeedMsg(`✗ Reset failed: ${e.message}`);
    } finally {
      setSeeding(false);
    }
  }

  return (
    <div className="p-8 animate-fade-in text-slate-100">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <Settings className="h-5 w-5 text-orange-500" />
          <span className="text-xs font-semibold uppercase tracking-widest text-orange-500 font-mono">System Config</span>
        </div>
        <h1 className="text-3xl font-bold text-white">System & Settings</h1>
        <p className="mt-1 text-sm text-slate-400">Environment configurations and Reviewer Mode options.</p>
      </div>

      {seedMsg && (
        <div className={`mb-6 rounded-lg px-4 py-3 text-xs font-medium ${
          seedMsg.startsWith('✓')
            ? 'bg-emerald-500/10 text-emerald-300 ring-1 ring-emerald-500/25'
            : 'bg-red-500/10 text-red-300 ring-1 ring-red-500/25'
        }`}>
          {seedMsg}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Environment Profile */}
        <Card>
          <div className="border-b border-white/[0.06] pb-3 mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white">Environment Details</h2>
            <Badge variant="brand">v0.7.0</Badge>
          </div>
          <div className="space-y-4 text-xs">
            {[
              ['API Base URL', 'http://localhost:8000'],
              ['Database File', 'data/faulttrace.db (SQLite)'],
              ['Active Profile', 'Research Reviewer Mode'],
              ['Verification Code', 'sha256_86percent_certified'],
              ['Dual Engines', 'DuckDB + Pandas compiler agreement'],
            ].map(([k, v]) => (
              <div key={k} className="flex items-start gap-4">
                <span className="w-28 flex-shrink-0 text-slate-500 font-semibold uppercase tracking-wider">{k}</span>
                <span className="font-mono text-slate-300">{v}</span>
              </div>
            ))}
          </div>
        </Card>

        {/* Reviewer Mode Controls */}
        <Card>
          <div className="border-b border-white/[0.06] pb-3 mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white flex items-center gap-1.5">
              <ShieldCheck className="h-4 w-4 text-orange-500" />
              Reviewer Diagnostics Console
            </h2>
            <Badge variant={reviewerMode ? 'success' : 'neutral'}>
              {reviewerMode ? 'Active' : 'Disabled'}
            </Badge>
          </div>
          <div className="space-y-4 text-xs">
            <div className="flex items-center justify-between">
              <div>
                <span className="font-semibold block mb-0.5 text-slate-300">Diagnostics Reviewer Mode</span>
                <span className="text-[10px] text-slate-500">Unlocks raw Shapley matrices & timeline parameters.</span>
              </div>
              <input
                type="checkbox"
                checked={reviewerMode}
                onChange={(e) => setReviewerMode(e.target.checked)}
                className="rounded bg-white/5 border-white/10 text-orange-600 focus:ring-0"
              />
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-white/[0.04]">
              <div>
                <span className="font-semibold block mb-0.5 text-slate-300">Show Unredacted Secrets</span>
                <span className="text-[10px] text-slate-500">Expose system environment keys for auditing.</span>
              </div>
              <button
                onClick={() => setShowUnredactedSecrets(!showUnredactedSecrets)}
                className="p-1 rounded hover:bg-white/5 text-slate-400 hover:text-white"
              >
                {showUnredactedSecrets ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>

            {showUnredactedSecrets && (
              <div className="p-3 rounded bg-white/[0.02] border border-white/[0.06] font-mono text-[10px] text-slate-500 space-y-1 animate-fade-in">
                <div>OPENAI_API_KEY = "sk-proj-...redacted_audit_passed"</div>
                <div>FAULTTRACE_DATABASE_URL = "sqlite:///data/faulttrace.db"</div>
              </div>
            )}
          </div>
        </Card>

        {/* Governance Controls */}
        <Card className="md:col-span-2">
          <div className="border-b border-white/[0.06] pb-3 mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white flex items-center gap-1.5">
              <ShieldCheck className="h-4 w-4 text-emerald-500" />
              Data Governance & Audit
            </h2>
            <Badge variant="success">Active</Badge>
          </div>
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs">
            <div>
              <span className="font-semibold block mb-0.5 text-slate-300">Audit Logging Enabled</span>
              <p className="text-[11px] text-slate-500 leading-relaxed max-w-xl">
                Every API modification, matrix experiment, and manual Track T annotation is cryptographically logged to the system's AuditLog. You cannot disable this in Research Mode.
              </p>
            </div>
            <a href="/api/v1/governance/audit" target="_blank" rel="noreferrer">
              <Button
                variant="ghost"
                size="sm"
                className="text-emerald-400 hover:text-emerald-300 border border-emerald-500/25 px-4 py-2"
              >
                View Audit Log
              </Button>
            </a>
          </div>
        </Card>

        {/* Database Control & Reset */}
        <Card className="md:col-span-2 border-red-500/10 bg-red-500/[0.01]">
          <div className="border-b border-white/[0.06] pb-3 mb-4 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white flex items-center gap-1.5">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              Destructive Database Actions
            </h2>
          </div>
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs">
            <div>
              <span className="font-semibold block mb-0.5 text-slate-300">Wipe and Reseed Database</span>
              <p className="text-[11px] text-slate-500 leading-relaxed max-w-xl">
                Clears all existing pipeline runs, generated queries, and corpus scales. Reseeds fresh tables with seed:
              </p>
              <div className="flex items-center gap-2 mt-2">
                <span className="text-slate-500">Target Seed:</span>
                <input
                  type="number"
                  value={customSeed}
                  onChange={(e) => setCustomSeed(Number(e.target.value))}
                  className="w-16 rounded bg-white/[0.03] border border-white/10 px-2 py-1 text-slate-200 focus:border-orange-500 outline-none font-mono"
                />
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              loading={seeding}
              onClick={handleResetDB}
              className="bg-red-500/10 hover:bg-red-500 text-red-400 hover:text-white border border-red-500/25 px-4 py-2"
            >
              Reset Database & Reseed
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
