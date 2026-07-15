"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

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

export default function LeaderboardPage() {
  const [data, setData] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("http://localhost:8000/api/leaderboard");
        if (!res.ok) throw new Error("Failed to fetch leaderboard");
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return <div className="p-8">Loading leaderboard...</div>;
  }

  if (error) {
    return <div className="p-8 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Leaderboard</h1>
        <Badge variant="neutral" className="text-sm">
          {data?.total_pipelines} pipelines
        </Badge>
      </div>
      <p className="text-muted-foreground">
        Comparison of pipeline performance based on exact-match correctness and relative error loss.
      </p>

      <Card>
        <div className="px-5 py-4 border-b border-white/[0.06]">
          <h3 className="text-lg font-semibold text-white">Pipeline Rankings</h3>
          <p className="text-xs text-slate-400 mt-1">Sorted by exact-match accuracy.</p>
        </div>
        <div className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-muted-foreground uppercase bg-muted/50">
                <tr>
                  <th className="px-4 py-3 rounded-tl-md">Rank</th>
                  <th className="px-4 py-3">Pipeline ID</th>
                  <th className="px-4 py-3 text-right">Accuracy</th>
                  <th className="px-4 py-3 text-right">Correct / Total</th>
                  <th className="px-4 py-3 text-right">Mean Loss</th>
                  <th className="px-4 py-3 text-right rounded-tr-md">Mean Latency</th>
                </tr>
              </thead>
              <tbody>
                {data?.leaderboard.map((entry, idx) => (
                  <tr key={entry.pipeline_id} className="border-b last:border-0 hover:bg-muted/50 transition-colors">
                    <td className="px-4 py-3 font-semibold text-muted-foreground">#{idx + 1}</td>
                    <td className="px-4 py-3 font-medium">
                      {entry.pipeline_id}
                      {entry.pipeline_id.includes("P0") && (
                        <Badge className="ml-2" variant="neutral">Baseline</Badge>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-bold text-green-600 dark:text-green-400">
                      {(entry.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-right text-muted-foreground">
                      {entry.correct_runs} / {entry.total_runs}
                    </td>
                    <td className="px-4 py-3 text-right text-orange-600 dark:text-orange-400">
                      {entry.mean_loss.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 text-right text-muted-foreground">
                      {entry.mean_latency_ms.toFixed(0)} ms
                    </td>
                  </tr>
                ))}
                {data?.leaderboard.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                      No runs recorded yet. Generate some queries and run pipelines first!
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </Card>
    </div>
  );
}
