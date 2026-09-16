'use client';

import { useState } from 'react';
import { Play, Database, FileJson, Calculator, Zap, ShieldAlert, ShieldCheck, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

type PipelineState = 'idle' | 'retrieving' | 'extracting' | 'aggregating' | 'attributing' | 'complete';

export function DemoShell() {
  const [worldSize, setWorldSize] = useState('10');
  const [queryType, setQueryType] = useState('count');
  const [pipelineState, setPipelineState] = useState<PipelineState>('idle');
  const [traceLogs, setTraceLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);

  // Explicitly illustrative values for the offline UI walkthrough.
  const worlds = [
    { id: '10', label: 'N=10 (Micro)' },
    { id: '50', label: 'N=50 (Small)' },
    { id: '200', label: 'N=200 (Medium)' },
    { id: '1000', label: 'N=1000 (Large)' }
  ];

  const runSimulation = () => {
    setPipelineState('retrieving');
    setTraceLogs(['[SYSTEM] Initializing Trace Session...']);
    setProgress(0);

    // Sequence of state changes to mock the live API
    setTimeout(() => {
      setTraceLogs(prev => [...prev, `[RETRIEVAL] Connecting to World ${worldSize}...`, `[RETRIEVAL] Filtering out-of-scope documents. Selected ${Math.floor(Number(worldSize) * 0.4)} records.`]);
      setProgress(25);
    }, 1000);

    setTimeout(() => {
      setPipelineState('extracting');
      setTraceLogs(prev => [...prev, '[EXTRACTION] Running schema f_E over in-scope records...', '[EXTRACTION] WARNING: Model truncated output on row 14.', '[EXTRACTION] Fallback to chunked extraction successful.']);
      setProgress(50);
    }, 2500);

    setTimeout(() => {
      setPipelineState('aggregating');
      setTraceLogs(prev => [...prev, '[AGGREGATION] Computing target metric...', `[AGGREGATION] Final computed value: ${queryType === 'count' ? '42' : '4.7'}`]);
      setProgress(75);
    }, 4500);

    setTimeout(() => {
      setPipelineState('attributing');
      setTraceLogs(prev => [...prev, '[ORACLE] Executing Shapley component swaps...', '[ORACLE] R-Swap completed.', '[ORACLE] E-Swap completed.', '[ORACLE] A-Swap completed.']);
      setProgress(90);
    }, 6000);

    setTimeout(() => {
      setPipelineState('complete');
      setTraceLogs(prev => [...prev, '[TRACE] Component failure attribution finalized.', '[SYSTEM] Trace Complete.']);
      setProgress(100);
    }, 8000);
  };

  const isComplete = pipelineState === 'complete';
  
  // Illustrative chart values; never exported as experiment measurements.
  const chartData = [
    { name: 'Retrieval', value: queryType === 'count' ? 0.05 : 0.65 },
    { name: 'Extraction', value: queryType === 'count' ? 0.85 : 0.25 },
    { name: 'Aggregation', value: queryType === 'count' ? 0.10 : 0.10 },
  ];

  return (
    <div className="min-h-screen bg-surface-0 text-white font-sans pt-20">
      
      {/* Header */}
      <header className="border-b border-white/10 bg-black sticky top-0 z-40">
        <div className="container mx-auto px-4 max-w-7xl h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Activity className="w-5 h-5 text-neon" />
            <h1 className="font-bold type-mono tracking-tight text-lg">Interactive Trace</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-xs type-mono text-amber-400">Offline UI Simulation — No Measured Data</span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 max-w-7xl py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Controls & Pipeline */}
        <div className="lg:col-span-8 flex flex-col gap-8">
          
          {/* Controls Bar */}
          <div className="bg-surface-1 border border-white/10 rounded-xl p-6 flex flex-col sm:flex-row gap-6 justify-between items-end">
            <div className="flex-1 flex gap-4 w-full">
              <div className="flex-1">
                <label className="block text-xs type-mono text-slate-500 uppercase mb-2">Corpus World Scale</label>
                <select 
                  className="w-full bg-black border border-white/20 rounded p-2 text-sm text-white focus:border-neon focus:ring-1 focus:ring-neon outline-none"
                  value={worldSize}
                  onChange={(e) => setWorldSize(e.target.value)}
                  disabled={pipelineState !== 'idle' && pipelineState !== 'complete'}
                >
                  {worlds.map(w => <option key={w.id} value={w.id}>{w.label}</option>)}
                </select>
              </div>
              <div className="flex-1">
                <label className="block text-xs type-mono text-slate-500 uppercase mb-2">Query Template</label>
                <select 
                  className="w-full bg-black border border-white/20 rounded p-2 text-sm text-white focus:border-neon focus:ring-1 focus:ring-neon outline-none"
                  value={queryType}
                  onChange={(e) => setQueryType(e.target.value)}
                  disabled={pipelineState !== 'idle' && pipelineState !== 'complete'}
                >
                  <option value="count">Count (High Extraction Risk)</option>
                  <option value="mean">Mean (High Retrieval Risk)</option>
                  <option value="rate">Rate (Balanced Risk)</option>
                </select>
              </div>
            </div>
            
            <button
              onClick={runSimulation}
              disabled={pipelineState !== 'idle' && pipelineState !== 'complete'}
              className="bg-neon text-black px-6 py-2.5 rounded font-bold type-mono text-sm flex items-center gap-2 hover:bg-orange-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shrink-0 w-full sm:w-auto justify-center"
            >
              <Play className="w-4 h-4 fill-black" />
              {pipelineState === 'idle' || pipelineState === 'complete' ? 'Preview Simulation' : 'Simulating...'}
            </button>
          </div>

          {/* Pipeline Visualizer */}
          <div className="bg-black border border-white/10 rounded-xl p-8 relative overflow-hidden flex flex-col">
            <div className="text-xs type-mono text-slate-500 uppercase mb-8">Pipeline State</div>
            
            {/* Progress Bar */}
            <div className="absolute top-0 left-0 w-full h-1 bg-white/5">
              <div 
                className="h-full bg-neon transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="flex items-center justify-between relative z-10 w-full max-w-3xl mx-auto py-8">
              {/* R Node */}
              <div className={cn(
                "flex flex-col items-center gap-3 transition-all duration-500",
                pipelineState === 'retrieving' ? "scale-110" : "scale-100",
                (pipelineState === 'idle' || pipelineState === 'complete') && "opacity-50"
              )}>
                <div className={cn(
                  "w-16 h-16 rounded-xl flex items-center justify-center border-2 transition-colors duration-300",
                  pipelineState === 'retrieving' ? "bg-neon/20 border-neon shadow-glow-brand" : "bg-surface-2 border-white/10"
                )}>
                  <Database className={cn("w-6 h-6", pipelineState === 'retrieving' ? "text-neon" : "text-white/50")} />
                </div>
                <div className="type-mono text-xs text-center font-bold">R<br/><span className="text-[9px] font-normal text-slate-500">Retrieval</span></div>
              </div>
              
              <div className="flex-1 h-px bg-white/20 mx-4 relative">
                {pipelineState === 'extracting' && (
                  <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-0 w-2 h-2 bg-neon rounded-full animate-[slideRight_1s_ease-in-out_infinite]" />
                )}
              </div>

              {/* E Node */}
              <div className={cn(
                "flex flex-col items-center gap-3 transition-all duration-500",
                pipelineState === 'extracting' ? "scale-110" : "scale-100",
                (pipelineState === 'idle' || pipelineState === 'complete') && "opacity-50"
              )}>
                <div className={cn(
                  "w-16 h-16 rounded-xl flex items-center justify-center border-2 transition-colors duration-300",
                  pipelineState === 'extracting' ? "bg-neon/20 border-neon shadow-glow-brand" : "bg-surface-2 border-white/10"
                )}>
                  <FileJson className={cn("w-6 h-6", pipelineState === 'extracting' ? "text-neon" : "text-white/50")} />
                </div>
                <div className="type-mono text-xs text-center font-bold">E<br/><span className="text-[9px] font-normal text-slate-500">Extraction</span></div>
              </div>

              <div className="flex-1 h-px bg-white/20 mx-4 relative">
                 {pipelineState === 'aggregating' && (
                  <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-0 w-2 h-2 bg-neon rounded-full animate-[slideRight_1s_ease-in-out_infinite]" />
                )}
              </div>

              {/* A Node */}
              <div className={cn(
                "flex flex-col items-center gap-3 transition-all duration-500",
                pipelineState === 'aggregating' ? "scale-110" : "scale-100",
                (pipelineState === 'idle' || pipelineState === 'complete') && "opacity-50"
              )}>
                <div className={cn(
                  "w-16 h-16 rounded-xl flex items-center justify-center border-2 transition-colors duration-300",
                  pipelineState === 'aggregating' ? "bg-neon/20 border-neon shadow-glow-brand" : "bg-surface-2 border-white/10"
                )}>
                  <Calculator className={cn("w-6 h-6", pipelineState === 'aggregating' ? "text-neon" : "text-white/50")} />
                </div>
                <div className="type-mono text-xs text-center font-bold">A<br/><span className="text-[9px] font-normal text-slate-500">Aggregation</span></div>
              </div>
            </div>

            {/* Oracle Overlay */}
            {pipelineState === 'attributing' && (
              <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px] flex flex-col items-center justify-center animate-fade-in z-20">
                <Zap className="w-12 h-12 text-neon animate-pulse mb-4" />
                <div className="type-mono text-neon font-bold text-lg mb-2">Oracle Replacement Protocol Active</div>
                <div className="type-mono text-xs text-white/70">Computing Shapley value marginal contributions...</div>
              </div>
            )}
          </div>
          
          {/* Raw Log Viewer */}
          <div className="bg-surface-1 border border-white/10 rounded-xl overflow-hidden flex-1 flex flex-col min-h-[250px]">
             <div className="px-4 py-2 border-b border-white/10 bg-black/50 flex justify-between items-center">
              <span className="text-xs type-mono text-slate-500 uppercase">Live Trace Log</span>
              <span className="flex h-2 w-2 relative">
                {pipelineState !== 'idle' && pipelineState !== 'complete' && (
                   <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon opacity-75"></span>
                )}
                <span className={cn("relative inline-flex rounded-full h-2 w-2", pipelineState !== 'idle' && pipelineState !== 'complete' ? "bg-neon" : "bg-slate-700")}></span>
              </span>
             </div>
             <div className="p-4 overflow-y-auto type-mono text-[11px] font-medium text-slate-400 flex flex-col gap-1.5 h-full max-h-[300px]">
                {traceLogs.length === 0 && <span className="opacity-50">Waiting for execution...</span>}
                {traceLogs.map((log, i) => (
                  <div key={i} className={cn(
                    "animate-fade-in",
                    log.includes('WARNING') ? "text-red-400" : 
                    log.includes('[ORACLE]') ? "text-neon" : 
                    log.includes('Complete') ? "text-emerald-400" : ""
                  )}>
                    <span className="opacity-50 mr-2">{new Date().toISOString().split('T')[1].slice(0,8)}</span> 
                    {log}
                  </div>
                ))}
             </div>
          </div>

        </div>

        {/* Right Column: Results & Attribution */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          {/* Certificate */}
          <div className={cn(
            "rounded-xl border p-6 transition-colors duration-500 relative overflow-hidden",
            isComplete 
              ? queryType === 'count' 
                ? "bg-red-500/10 border-red-500/30" 
                : "bg-emerald-500/10 border-emerald-500/30"
              : "bg-surface-1 border-white/10"
          )}>
            <div className="text-xs type-mono text-slate-500 uppercase mb-4">Coverage Certificate</div>
            
            {isComplete ? (
              <div className="flex flex-col gap-4 animate-fade-in">
                <div className="flex items-center gap-3">
                  {queryType === 'count' ? (
                    <ShieldAlert className="w-8 h-8 text-red-500" />
                  ) : (
                    <ShieldCheck className="w-8 h-8 text-emerald-500" />
                  )}
                  <div>
                    <div className="font-bold text-white text-lg">
                      {queryType === 'count' ? 'Failure Isolated' : 'Trace Verified'}
                    </div>
                    <div className="text-xs text-slate-400 type-mono">
                      {queryType === 'count' ? 'Oracle deviation detected.' : 'Gold match confirmed.'}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-16 flex items-center justify-center border border-dashed border-white/20 rounded opacity-50">
                <span className="text-xs type-mono text-slate-500">Awaiting Run</span>
              </div>
            )}
          </div>

          {/* Attribution Chart */}
          <div className="bg-surface-1 border border-white/10 rounded-xl p-6 flex-1 flex flex-col min-h-[350px]">
            <div className="text-xs type-mono text-slate-500 uppercase mb-6 flex justify-between">
              <span>Fault Attribution</span>
              <span>(Shapley ϕ)</span>
            </div>
            
            <div className="flex-1 w-full h-full relative">
              {isComplete ? (
                <div className="absolute inset-0 animate-fade-in">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 30, bottom: 0 }}>
                      <XAxis type="number" hide domain={[0, 1]} />
                      <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12, fontFamily: 'monospace'}} />
                      <Tooltip 
                        cursor={{fill: 'rgba(255,255,255,0.05)'}}
                        contentStyle={{backgroundColor: '#0c0d0f', borderColor: '#334155', borderRadius: '8px'}}
                        formatter={(value: number) => [value.toFixed(2), 'Marginal Loss (ϕ)']}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.value > 0.5 ? '#FF6A2B' : '#334155'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="mt-6 p-4 bg-black/50 border border-white/5 rounded-lg text-sm text-slate-300">
                    {queryType === 'count' ? (
                      <>The <strong className="text-neon">Extraction</strong> step is responsible for 85% of the total error. The model hallucinated schema fields for 14 documents.</>
                    ) : (
                      <>The <strong className="text-neon">Retrieval</strong> step is responsible for 65% of the total error. The top-k window omitted 2 critical denominator records.</>
                    )}
                  </div>
                </div>
              ) : (
                <div className="absolute inset-0 flex items-center justify-center border border-dashed border-white/10 rounded-lg">
                  <span className="text-xs type-mono text-slate-500">Chart will populate upon completion</span>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes slideRight {
          0% { left: 0; opacity: 0; }
          50% { opacity: 1; }
          100% { left: 100%; opacity: 0; }
        }
      `}} />
    </div>
  );
}
