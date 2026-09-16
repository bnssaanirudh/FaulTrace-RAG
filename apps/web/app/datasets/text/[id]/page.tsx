'use client';

import { useCallback, useEffect, useState, use } from 'react';
import { api } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { FileText, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function TextDatasetPreviewPage(props: { params: Promise<{ id: string }> }) {
  const params = use(props.params);
  const [chunks, setChunks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const [totalChunks, setTotalChunks] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const preview = await api.getTextDatasetPreview(params.id, page, pageSize);
      setChunks(preview.chunks || []);
      setTotalChunks(preview.total_chunks || 0);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [page, params.id]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="p-8 animate-fade-in text-slate-100">
      <div className="mb-6 flex items-center gap-4">
        <Link href="/datasets/text">
          <Button variant="ghost" size="sm" className="h-8">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Datasets
          </Button>
        </Link>
      </div>

      <div className="mb-8 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <FileText className="h-5 w-5 text-blue-500" />
            <span className="text-xs font-semibold uppercase tracking-widest text-blue-500">Text Corpora Viewer</span>
          </div>
          <h1 className="text-2xl font-bold text-white">Snapshot Chunks</h1>
          <p className="mt-1 text-sm text-slate-400 font-mono">
            {params.id}
          </p>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 p-4 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-400">
            Showing chunks {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, totalChunks)} of {totalChunks}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage(p => p - 1)}
            >
              Previous
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={page * pageSize >= totalChunks}
              onClick={() => setPage(p => p + 1)}
            >
              Next
            </Button>
          </div>
        </div>

        {loading && chunks.length === 0 ? (
          <div className="text-center py-12 text-slate-500">Loading chunks...</div>
        ) : chunks.length === 0 ? (
          <div className="text-center py-12 text-slate-500">No chunks found in this snapshot.</div>
        ) : (
          <div className="grid gap-4">
            {chunks.map((chunk, i) => (
              <Card key={`${chunk.chunk_id}-${i}`} className="flex flex-col">
                <div className="flex justify-between items-start border-b border-white/[0.06] pb-2 mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs text-blue-400">{chunk.chunk_id}</span>
                    <Badge variant="neutral">Doc: {chunk.doc_id}</Badge>
                    <Badge variant="neutral">Idx: {chunk.chunk_index}</Badge>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    [{chunk.start_char} - {chunk.end_char}]
                  </span>
                </div>
                <div className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed font-serif bg-slate-900/50 p-4 rounded border border-white/5">
                  {chunk.text}
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
