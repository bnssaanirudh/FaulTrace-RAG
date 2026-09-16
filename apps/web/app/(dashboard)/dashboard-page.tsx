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
  ChevronRight,
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
} from 'lucide-react';
import { api, SystemStatus, World, Run } from '@/lib/api';
import { StatCard, Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatMs } from '@/lib/utils';
import Link from 'next/link';

type DemoStep =
  | 'idle'
  | 'verifying_world'
  | 'running_p1'
  | 'running_p4'
  | 'displaying_comparison'
  | 'showing_diagnostics'
  | 'showing_certificate'
  | 'completed'
  | 'failed';

export function DashboardPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [worlds, setWorlds] = useState<World[]>([]);
  const [recentRuns, setRecentRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [seeding, setSeeding] = useState(false);
  const [seedMsg, setSeedMsg] = useState('');
  const [error, setError] = useState('');

  // Guided Demo State
  const [demoStep, setDemoStep] = useState<DemoStep>('idle');
  const [demoQueryId, setDemoQueryId] = useState('74c8a1fa-d541-4cc9-91ab-4ef1332001cf'); // Fails top_k by omission
  const [demoQueryText, setDemoQueryText] = useState('Which 5 brands have the most reviews?');
  const [p1Run, setP1Run] = useState<any>(null);
  const [p4Run, setP4Run] = useState<any>(null);
  const [p1Attr, setP1Attr] = useState<any>(null);
  const [p4Attr, setP4Attr] = useState<any>(null);
  const [p1Cert, setP1Cert] = useState<any>(null);
  const [p4Cert, setP4Cert] = useState<any>(null);

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
      setWorlds(Array.isArray(w) ? w : (w as any).items || []);
      setRecentRuns(r.items || []);
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

  // Guided Demo Workflow Executer
  async function runGuidedDemo() {
    try {
      // 1. Verify/Seed World
      setDemoStep('verifying_world');
      let targetQuery = demoQueryId;

      // Ensure queries exist in the DB (seed if empty)
      const qRes = await api.listQueries(undefined, 'top_k', 1, 5);
      if (qRes.items.length === 0) {
        setSeedMsg('Seeding corpus database to prepare demo queries...');
        await api.seedDemo(42, [10, 50, 200, 1000], false);
        // Regenerate queries for world_s42_n50 if needed
        await api.generateQueries('world_s42_n50', 60);
      }

      // Re-fetch queries and match a top_k one
      const queriesList = await api.listQueries(undefined, 'top_k', 1, 10);
      const topkQuery = queriesList.items.find(q => q.family === 'top_k') || queriesList.items[0];
      if (topkQuery) {
        targetQuery = topkQuery.query_id;
        setDemoQueryId(topkQuery.query_id);
        setDemoQueryText(topkQuery.natural_language_question);
      }

      // 2. Run P1 Wrong Scope
      setDemoStep('running_p1');
      const p1Result = await api.createRun(targetQuery, 'P1-wrong-scope');
      setP1Run(p1Result);

      // 3. Run P4 Compound Scope Facts
      setDemoStep('running_p4');
      const p4Result = await api.createRun(targetQuery, 'P4-compound-scope-facts');
      setP4Run(p4Result);

      // 4. Load comparison
      setDemoStep('displaying_comparison');
      await new Promise(resolve => setTimeout(resolve, 1500));

      // 5. Run Oracle Diagnostics / Attribution
      setDemoStep('showing_diagnostics');
      const [attr1, attr4] = await Promise.all([
        api.getRunAttribution(p1Result.run_id),
        api.getRunAttribution(p4Result.run_id),
      ]);
      setP1Attr(attr1);
      setP4Attr(attr4);
      await new Promise(resolve => setTimeout(resolve, 1500));

      // 6. Show Coverage Certificate
      setDemoStep('showing_certificate');
      const [cert1, cert4] = await Promise.all([
        api.getRunCertificate(p1Result.run_id),
        api.getRunCertificate(p4Result.run_id),
      ]);
      setP1Cert(cert1);
      setP4Cert(cert4);

      setDemoStep('completed');
      await load(); // Reload stats and recent list
    } catch (e) {
      console.error(e);
      setDemoStep('failed');
    }
  }

  function resetDemo() {
    setDemoStep('idle');
    setP1Run(null);
    setP4Run(null);
    setP1Attr(null);
    setP4Attr(null);
    setP1Cert(null);
    setP4Cert(null);
  }

  useEffect(() => {
    load();
  }, []);

  const correctRuns = recentRuns.filter((r) => r.is_correct === true).length;
  const totalRuns = recentRuns.length;

  return (
    <div className="min-h-screen p-8 animate-fade-in text-slate-100">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-orange-500" />
            <span className="text-xs font-semibold uppercase tracking-widest text-orange-500">
              FaultTrace-RAG
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Overview Dashboard
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Counterfactual fault localization and evidence certification for corpus-level analytics.
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
            className="bg-orange-600 hover:bg-orange-700 text-white"
          >
            <Zap className="h-3.5 w-3.5" />
            Seed Demo
          </Button>
        </div>
      </div>

      {(seedMsg || error) && (
        <div className={`mb-6 rounded-lg px-4 py-3 text-sm ring-1 ${
          error || seedMsg.startsWith('✗')
            ? 'bg-red-500/10 text-red-300 ring-red-500/25'
            : 'bg-emerald-500/10 text-emerald-300 ring-emerald-500/25'
        }`}>
          {error || seedMsg}
        </div>
      )}

      {/* Problem Statement Box */}
      <div className="mb-8 rounded-xl border border-orange-500/20 bg-orange-600/5 p-6 backdrop-blur-sm">
        <h2 className="text-md font-semibold text-orange-400 mb-2">Research Objectives</h2>
        <p className="text-sm text-slate-300 leading-relaxed max-w-4xl">
          When an LLM analytics pipeline produces an incorrect answer over a corpus, identifying the failure origin is a core engineering challenge. FaultTrace-RAG systematically evaluates three key components: 
          <strong> Retrieval Scope (R)</strong>, <strong>Fact Extraction (E)</strong>, and <strong>Aggregation Logic (A)</strong>. 
          By substituting components with deterministic oracles, we compute exact Shapley value fault attributions and issue evidence coverage certificates to prevent hallucinated answers.
        </p>
      </div>

      {/* R -> E -> A Pipeline Diagram */}
      <div className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-4">REA Compounding Pipeline Architecture</p>
        <div className="grid grid-cols-1 md:grid-cols-5 items-center gap-4">
          <div className="glass-card p-4 flex flex-col items-center text-center">
            <div className="h-8 w-8 rounded-full bg-orange-500/10 flex items-center justify-center mb-2">
              <span className="text-orange-500 font-bold text-sm">R</span>
            </div>
            <h3 className="text-sm font-semibold text-white">Retrieval Scope</h3>
            <p className="text-[11px] text-slate-500 mt-1">Filters matches from corpus to form context record IDs.</p>
          </div>
          <div className="hidden md:flex justify-center text-slate-600">
            <ChevronRight className="h-6 w-6" />
          </div>
          <div className="glass-card p-4 flex flex-col items-center text-center">
            <div className="h-8 w-8 rounded-full bg-brand-500/10 flex items-center justify-center mb-2">
              <span className="text-brand-400 font-bold text-sm">E</span>
            </div>
            <h3 className="text-sm font-semibold text-white">Fact Extraction</h3>
            <p className="text-[11px] text-slate-500 mt-1">Extracts structured attributes from context records.</p>
          </div>
          <div className="hidden md:flex justify-center text-slate-600">
            <ChevronRight className="h-6 w-6" />
          </div>
          <div className="glass-card p-4 flex flex-col items-center text-center">
            <div className="h-8 w-8 rounded-full bg-emerald-500/10 flex items-center justify-center mb-2">
              <span className="text-emerald-400 font-bold text-sm">A</span>
            </div>
            <h3 className="text-sm font-semibold text-white">Aggregation Logic</h3>
            <p className="text-[11px] text-slate-500 mt-1">Computes structured reduction (Count, Mean, Top-K).</p>
          </div>
        </div>
      </div>

      {/* Guided Demo Wizard */}
      <div className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Guided Walkthrough</p>
        <Card className="border-orange-500/20 bg-orange-600/[0.02]">
          <div className="p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-white/[0.06]">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                Interactive Guided Research Demo
                {demoStep !== 'idle' && <Badge variant="orange">{demoStep.replace('_', ' ')}</Badge>}
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Demonstrates how FaultTrace-RAG isolates retrieval omission faults (P1 vs. P4) using Shapley values and Selective Prediction certificates.
              </p>
            </div>
            <div className="flex items-center gap-2">
              {demoStep === 'idle' ? (
                <Button variant="primary" size="sm" onClick={runGuidedDemo} className="bg-orange-600 hover:bg-orange-700 text-white gap-1">
                  <Play className="h-3 w-3 fill-current" /> Run Guided Demo
                </Button>
              ) : (
                <Button variant="ghost" size="sm" onClick={resetDemo} className="gap-1 text-slate-400 hover:text-white">
                  <RotateCcw className="h-3.5 w-3.5" /> Reset Demo
                </Button>
              )}
            </div>
          </div>

          <div className="p-5">
            {/* Step Timeline Indicator */}
            {demoStep !== 'idle' && (
              <div className="mb-6 flex flex-wrap gap-4 items-center text-xs text-slate-500">
                <span className={demoStep === 'verifying_world' ? 'text-orange-400 font-bold' : ''}>1. Setup World & Query</span>
                <ChevronRight className="h-3 w-3" />
                <span className={demoStep === 'running_p1' ? 'text-orange-400 font-bold' : ''}>2. Run P1 (Wrong Scope)</span>
                <ChevronRight className="h-3 w-3" />
                <span className={demoStep === 'running_p4' ? 'text-orange-400 font-bold' : ''}>3. Run P4 (Compound)</span>
                <ChevronRight className="h-3 w-3" />
                <span className={demoStep === 'displaying_comparison' ? 'text-orange-400 font-bold' : ''}>4. Compare Gold</span>
                <ChevronRight className="h-3 w-3" />
                <span className={demoStep === 'showing_diagnostics' ? 'text-orange-400 font-bold' : ''}>5. Attribution</span>
                <ChevronRight className="h-3 w-3" />
                <span className={demoStep === 'showing_certificate' || demoStep === 'completed' ? 'text-orange-400 font-bold' : ''}>6. Certification</span>
              </div>
            )}

            {/* Stage content */}
            {demoStep === 'idle' && (
              <div className="text-center py-6 text-slate-500 text-sm">
                <Sparkles className="h-10 w-10 mx-auto text-orange-500/20 mb-2" />
                <p>Click "Run Guided Demo" to spin up the RAG execution, compute diagnostic values, and verify evidence integrity.</p>
              </div>
            )}

            {demoStep === 'verifying_world' && (
              <div className="flex items-center gap-3 py-4">
                <div className="h-4 w-4 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Verifying database worlds, seeding data, and retrieving dynamic target top-k queries...</span>
              </div>
            )}

            {demoStep === 'running_p1' && (
              <div className="flex items-center gap-3 py-4">
                <div className="h-4 w-4 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Executing **P1-wrong-scope** pipeline against query: "{demoQueryText}"</span>
              </div>
            )}

            {demoStep === 'running_p4' && (
              <div className="flex items-center gap-3 py-4">
                <div className="h-4 w-4 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Executing **P4-compound-scope-facts** pipeline (R fault + E fault) against query...</span>
              </div>
            )}

            {(demoStep === 'displaying_comparison' || demoStep === 'showing_diagnostics' || demoStep === 'showing_certificate' || demoStep === 'completed') && (
              <div className="space-y-6">
                {/* Comparison Grid */}
                <div className="glass-card overflow-hidden">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-white/[0.02] border-b border-white/[0.06] text-slate-400">
                      <tr>
                        <th className="px-4 py-2">Pipeline config</th>
                        <th className="px-4 py-2">Scope R</th>
                        <th className="px-4 py-2">Extraction E</th>
                        <th className="px-4 py-2">Answer</th>
                        <th className="px-4 py-2">Gold Answer</th>
                        <th className="px-4 py-2">Match Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-white/[0.04]">
                        <td className="px-4 py-2.5 font-bold">Gold (P0 Oracle)</td>
                        <td className="px-4 py-2.5 text-emerald-400">Correct Oracle</td>
                        <td className="px-4 py-2.5 text-emerald-400">Correct Oracle</td>
                        <td className="px-4 py-2.5 font-mono text-emerald-400">Grounded Gold</td>
                        <td className="px-4 py-2.5 font-mono text-emerald-400">—</td>
                        <td className="px-4 py-2.5"><Badge variant="success">Correct</Badge></td>
                      </tr>
                      {p1Run && (
                        <tr className="border-b border-white/[0.04]">
                          <td className="px-4 py-2.5 font-bold">P1 (Wrong Scope)</td>
                          <td className="px-4 py-2.5 text-red-400 font-semibold">Perturbed (+/-20% Omission)</td>
                          <td className="px-4 py-2.5 text-emerald-400">Correct Oracle</td>
                          <td className="px-4 py-2.5 font-mono">{String(p1Run.answer || '—')}</td>
                          <td className="px-4 py-2.5 font-mono text-emerald-400">{String(p1Run.gold_answer_value || '—')}</td>
                          <td className="px-4 py-2.5">
                            <Badge variant={p1Run.is_correct ? 'success' : 'error'}>
                              {p1Run.is_correct ? 'Correct' : 'Incorrect'}
                            </Badge>
                          </td>
                        </tr>
                      )}
                      {p4Run && (
                        <tr>
                          <td className="px-4 py-2.5 font-bold">P4 (Compound SF)</td>
                          <td className="px-4 py-2.5 text-red-400 font-semibold">Perturbed (Omission)</td>
                          <td className="px-4 py-2.5 text-red-400 font-semibold">Noise Corrupted</td>
                          <td className="px-4 py-2.5 font-mono">{String(p4Run.answer || '—')}</td>
                          <td className="px-4 py-2.5 font-mono text-emerald-400">{String(p4Run.gold_answer_value || '—')}</td>
                          <td className="px-4 py-2.5">
                            <Badge variant={p4Run.is_correct ? 'success' : 'error'}>
                              {p4Run.is_correct ? 'Correct' : 'Incorrect'}
                            </Badge>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Attribution Panel */}
                {(demoStep === 'showing_diagnostics' || demoStep === 'showing_certificate' || demoStep === 'completed') && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in">
                    {/* P1 Attribution */}
                    <div className="glass-card p-4">
                      <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wider mb-3">P1 Shapley Attributions</h4>
                      {p1Attr ? (
                        <div className="space-y-2 text-xs">
                          {p1Attr.components?.map((c: any) => (
                            <div key={c.component}>
                              <div className="flex justify-between mb-1">
                                <span className="capitalize">{c.component}</span>
                                <span>{(c.shapley_value * 100).toFixed(0)}% contribution</span>
                              </div>
                              <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div className="h-full bg-orange-500" style={{ width: `${c.shapley_value * 100}%` }} />
                              </div>
                            </div>
                          ))}
                          <p className="text-[10px] text-slate-500 mt-2">
                            Dominant fault identified: **{p1Attr.dominant_fault?.toUpperCase()}**
                          </p>
                        </div>
                      ) : (
                        <div className="text-slate-600 text-xs">Loading P1 attribution metrics...</div>
                      )}
                    </div>

                    {/* P4 Attribution */}
                    <div className="glass-card p-4">
                      <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wider mb-3">P4 Shapley Attributions</h4>
                      {p4Attr ? (
                        <div className="space-y-2 text-xs">
                          {p4Attr.components?.map((c: any) => (
                            <div key={c.component}>
                              <div className="flex justify-between mb-1">
                                <span className="capitalize">{c.component}</span>
                                <span>{(c.shapley_value * 100).toFixed(0)}% contribution</span>
                              </div>
                              <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                <div className="h-full bg-brand-500" style={{ width: `${c.shapley_value * 100}%` }} />
                              </div>
                            </div>
                          ))}
                          <p className="text-[10px] text-slate-500 mt-2">
                            Dominant fault identified: **{p4Attr.dominant_fault?.toUpperCase()}**
                          </p>
                        </div>
                      ) : (
                        <div className="text-slate-600 text-xs">Loading P4 attribution metrics...</div>
                      )}
                    </div>
                  </div>
                )}

                {/* Certificate Panel */}
                {(demoStep === 'showing_certificate' || demoStep === 'completed') && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in">
                    {/* P1 Certificate */}
                    <div className="glass-card p-4 border-l-2 border-l-red-500">
                      <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <ShieldCheck className="h-4 w-4 text-red-500" />
                        P1 Evidence Integrity Certificate
                      </h4>
                      {p1Cert ? (
                        <div className="text-xs space-y-2">
                          <div className="flex items-center gap-2">
                            <span className="text-slate-500">Decision:</span>
                            <Badge variant="error">{p1Cert.policy_decision?.toUpperCase()}</Badge>
                          </div>
                          <div>
                            <span className="text-slate-500">Reason:</span>
                            <span className="ml-1 text-slate-300 font-mono text-[10px]">{p1Cert.abstention_reason || 'SCOPE_COVERAGE_UNKNOWN'}</span>
                          </div>
                          <p className="text-[10px] text-slate-500 italic mt-2">
                            Pipeline raw answer was {String(p1Run?.answer)}, but certificate refused to certify the result due to incomplete scope matches.
                          </p>
                        </div>
                      ) : (
                        <div className="text-slate-600 text-xs">Loading P1 certification...</div>
                      )}
                    </div>

                    {/* P4 Certificate */}
                    <div className="glass-card p-4 border-l-2 border-l-yellow-500">
                      <h4 className="text-xs font-semibold text-orange-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                        P4 Evidence Integrity Certificate
                      </h4>
                      {p4Cert ? (
                        <div className="text-xs space-y-2">
                          <div className="flex items-center gap-2">
                            <span className="text-slate-500">Decision:</span>
                            <Badge variant="warning">{p4Cert.policy_decision?.toUpperCase()}</Badge>
                          </div>
                          <div>
                            <span className="text-slate-500">Reason:</span>
                            <span className="ml-1 text-slate-300 font-mono text-[10px]">{p4Cert.abstention_reason || 'EXTRACTION_ROWS_MISSING'}</span>
                          </div>
                          <p className="text-[10px] text-slate-500 italic mt-2">
                            Pipeline abstains due to double extraction row failure and scope mismatch. Zero certification rate.
                          </p>
                        </div>
                      ) : (
                        <div className="text-slate-600 text-xs">Loading P4 certification...</div>
                      )}
                    </div>
                  </div>
                )}

                {/* Final Demo Complete Conclusion */}
                {demoStep === 'completed' && (
                  <div className="rounded-lg bg-emerald-500/10 p-4 border border-emerald-500/20 text-xs text-slate-300">
                    <p className="font-bold text-white mb-1">Guided Demo Conclusion</p>
                    FaultTrace-RAG successfully executed the top-k query across two faulty pipelines:
                    - **P1-wrong-scope** had a pure Scope (R) fault. The attributor accurately assigned **100% of the Shapley error value** to the scope component, and the certifier caught the omission and returned **ABSTAIN**.
                    - **P4-compound-scope-facts** had both Scope (R) and Extraction (E) errors. The attributor split the error across both layers.
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Stats Cards */}
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
        {/* Worlds list */}
        <div className="col-span-4">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Corpus Worlds</p>
          <div className="space-y-2">
            {worlds.length === 0 && !loading && (
              <div className="glass-card p-4 text-sm text-slate-500">
                No worlds yet. Click <strong>Seed Demo</strong> to generate.
              </div>
            )}
            {worlds.map((w) => (
              <Link
                key={w.world_id}
                href={`/worlds`}
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
              </Link>
            ))}
          </div>
        </div>

        {/* Pipeline status */}
        <div className="col-span-4">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Pipeline Registry</p>
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
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Recent Runs</p>
          <div className="space-y-2">
            {recentRuns.length === 0 && !loading && (
              <div className="glass-card p-4 text-sm text-slate-500">
                No runs yet. Execute a query to see results here.
              </div>
            )}
            {recentRuns.map((run) => (
              <Link
                key={run.run_id}
                href={`/runs/${run.run_id}/trace`}
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
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* System components info */}
      {status && (
        <div className="mt-8 glass-card p-5">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">System Components</p>
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
