"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";

interface TraceEvent {
  event_id: string;
  stage: string;
  event_type: string;
  message: string;
  duration_ms: number;
  payload: any;
  timestamp: string;
}

interface RunInfo {
  run_id: string;
  query_id: string;
  pipeline_id: string;
  status: string;
  is_correct: boolean;
  loss: number;
  latency_ms: number;
  answer: string;
  gold_answer_value: string;
}

interface ComponentAttribution {
  component: string;
  pipeline_answer: string;
  oracle_answer: string;
  ref_score: number;
  shapley_value: number;
}

interface AttributionData {
  run_id: string;
  pipeline_id: string;
  is_correct: boolean;
  total_error: number;
  components: ComponentAttribution[];
  dominant_fault: string;
}

export default function TracePage({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunInfo | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [attribution, setAttribution] = useState<AttributionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [runRes, traceRes] = await Promise.all([
          fetch(`http://localhost:8000/api/runs/${runId}`),
          fetch(`http://localhost:8000/api/runs/${runId}/trace`),
        ]);
        
        if (!runRes.ok) throw new Error("Failed to fetch run info");
        
        const runData = await runRes.json();
        const traceData = await traceRes.json();
        
        setRun(runData);
        setEvents(traceData);

        // Fetch attribution if completed and has gold answer
        if (runData.status === "completed" && runData.gold_answer_value !== null) {
          try {
            const attrRes = await fetch(`http://localhost:8000/api/runs/${runId}/attribution`);
            if (attrRes.ok) {
              setAttribution(await attrRes.json());
            }
          } catch (e) {
            console.error("Attribution failed:", e);
          }
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [runId]);

  if (loading) return <div className="p-8">Loading trace for run {runId}...</div>;
  if (error) return <div className="p-8 text-red-500">Error: {error}</div>;
  if (!run) return <div className="p-8">Run not found.</div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link href={`/runs/${runId}`}>
          <Button size="sm">← Back to Run</Button>
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">Trace & Attribution</h1>
        {run.is_correct ? (
          <Badge variant="success">Correct</Badge>
        ) : (
          <Badge variant="error">Incorrect (Loss: {run.loss?.toFixed(3)})</Badge>
        )}
      </div>
      
      <p className="text-muted-foreground">
        Pipeline: <span className="font-mono text-foreground">{run.pipeline_id}</span> | 
        Total Latency: {run.latency_ms?.toFixed(0)} ms
      </p>

      {attribution && (
        <Card className="border-orange-500/50 shadow-sm bg-orange-500/5">
          <div className="px-5 py-4 border-b border-orange-500/20">
            <h3 className="text-lg font-semibold text-orange-400 flex items-center gap-2">
              Counterfactual Fault Attribution
              {attribution.dominant_fault && (
                <Badge variant="orange">
                  Dominant: {attribution.dominant_fault.toUpperCase()}
                </Badge>
              )}
            </h3>
            <p className="text-xs text-orange-400/80 mt-1">
              Using 3-player Shapley value approximation via Oracle component replacement.
              Shows how much error (REF) is recoverable by substituting the gold Oracle for each component.
            </p>
          </div>
          <div className="p-5">
            {attribution.is_correct ? (
              <div className="text-green-600 font-medium p-4 bg-green-500/10 rounded-md">
                The pipeline answered correctly! Counterfactual error attribution is zero.
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {attribution.components.map((comp) => (
                    <div key={comp.component} className="p-4 rounded-lg border bg-card">
                      <div className="font-semibold text-lg capitalize mb-1">{comp.component}</div>
                      
                      <div className="space-y-3 mt-4">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-muted-foreground">Shapley Value (φ)</span>
                            <span className="font-mono">{(comp.shapley_value * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-orange-500" 
                              style={{ width: `${Math.max(0, Math.min(100, comp.shapley_value * 100))}%` }} 
                            />
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-muted-foreground">REF Score</span>
                            <span className="font-mono">{comp.ref_score.toFixed(3)}</span>
                          </div>
                          <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-blue-500" 
                              style={{ width: `${Math.max(0, Math.min(100, comp.ref_score * 100))}%` }} 
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      <Card>
        <div className="px-5 py-4 border-b border-white/[0.06]">
          <h3 className="text-lg font-semibold text-white">Execution Trace</h3>
          <p className="text-xs text-slate-400 mt-1">Chronological events from the pipeline run.</p>
        </div>
        <div className="p-5">
          <div className="space-y-4 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-border before:to-transparent">
            {events.map((ev, idx) => (
              <div key={ev.event_id} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                {/* Timeline dot */}
                <div className="flex items-center justify-center w-10 h-10 rounded-full border bg-background shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow-sm z-10">
                  <span className="text-xs font-bold text-muted-foreground">{idx + 1}</span>
                </div>
                
                {/* Content */}
                <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-lg border border-white/[0.06] bg-white/[0.02] shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="neutral">{ev.stage}</Badge>
                    <span className="text-xs text-muted-foreground font-mono">
                      {ev.duration_ms.toFixed(1)} ms
                    </span>
                  </div>
                  <p className="text-sm font-medium leading-relaxed">{ev.message}</p>
                  
                  {ev.payload && Object.keys(ev.payload).length > 0 && (
                    <div className="mt-3 p-2 bg-muted/50 rounded-md overflow-x-auto">
                      <pre className="text-[10px] text-muted-foreground font-mono m-0">
                        {JSON.stringify(ev.payload, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
