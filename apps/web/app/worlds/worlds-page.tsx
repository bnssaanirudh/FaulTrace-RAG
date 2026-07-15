'use client';

import { useEffect, useState } from 'react';
import { Database, ExternalLink, Globe } from 'lucide-react';
import { api, World } from '@/lib/api';
import { Card, StatCard } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { formatDate } from '@/lib/utils';

export function WorldsPage() {
  const [worlds, setWorlds] = useState<World[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    api.listWorlds()
      .then(setWorlds)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-8 animate-fade-in">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <Globe className="h-5 w-5 text-brand-400" />
          <span className="text-xs font-semibold uppercase tracking-widest text-brand-400">Data Layer</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Corpus Worlds</h1>
        <p className="mt-1 text-sm text-slate-500">
          Deterministic nested corpus worlds generated from the Track M dataset specification.
        </p>
      </div>

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 p-4 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      {loading && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-8">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 rounded-xl bg-white/[0.03] animate-pulse" />
          ))}
        </div>
      )}

      {!loading && worlds.length === 0 && (
        <Card padding="lg">
          <div className="text-center py-8 text-slate-500">
            <Database className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p className="font-medium">No worlds generated yet</p>
            <p className="text-sm mt-1">Go to the Dashboard and click <strong>Seed Demo</strong></p>
          </div>
        </Card>
      )}

      {worlds.length > 0 && (
        <>
          {/* Scale overview */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-8">
            {worlds.map((w) => (
              <StatCard
                key={w.world_id}
                label={`Scale N=${w.scale_n}`}
                value={w.scale_n.toLocaleString()}
                glow="brand"
                icon={<Database className="h-5 w-5" />}
              />
            ))}
          </div>

          {/* Detail table */}
          <Card padding="none">
            <div className="px-5 py-4 border-b border-white/[0.06]">
              <h2 className="text-sm font-semibold text-white">World Registry</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    {['World ID', 'Scale N', 'Seed', 'Parent', 'Policy', 'Records Hash', 'Created'].map((h) => (
                      <th key={h} className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {worlds.map((w) => (
                    <tr key={w.world_id} className="table-row-hover">
                      <td className="px-5 py-3.5">
                        <a href={`/worlds/${w.world_id}`} className="flex items-center gap-1.5 text-brand-400 hover:text-brand-300 mono text-xs">
                          {w.world_id.slice(0, 20)}…
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="font-bold text-white">{w.scale_n.toLocaleString()}</span>
                      </td>
                      <td className="px-5 py-3.5">
                        <Badge variant="brand">{w.seed}</Badge>
                      </td>
                      <td className="px-5 py-3.5 text-slate-500 mono text-xs">
                        {w.parent_world_id ? w.parent_world_id.slice(0, 16) + '…' : '—'}
                      </td>
                      <td className="px-5 py-3.5">
                        <Badge variant="neutral">{w.creation_policy}</Badge>
                      </td>
                      <td className="px-5 py-3.5 text-slate-500 mono text-xs">
                        {w.record_ids_hash.slice(0, 12)}…
                      </td>
                      <td className="px-5 py-3.5 text-slate-500 text-xs">
                        {formatDate(w.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
