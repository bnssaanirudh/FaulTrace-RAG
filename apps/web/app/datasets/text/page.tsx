'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { FileText, Database, ShieldCheck } from 'lucide-react';
import { formatDate } from '@/lib/utils';
import Link from 'next/link';

export default function TextDatasetsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSnapshot, setSelectedSnapshot] = useState<any | null>(null);
  const [error, setError] = useState('');

  // Ingestion Form State
  const [inputPath, setInputPath] = useState('');
  const [datasetId, setDatasetId] = useState('docs_corpus');
  const [sourceType, setSourceType] = useState('txt');
  const [licenseNote] = useState('');
  const [chunkSize, setChunkSize] = useState(1000);
  const [overlap, setOverlap] = useState(100);
  const [strictChunkDedup, setStrictChunkDedup] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');

  const selectSnapshot = useCallback(async (id: string) => {
    try {
      const snap = await api.getTextDatasetSnapshot(id);
      setSelectedSnapshot(snap);
    } catch (e: unknown) {
      console.error(e);
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await api.listTextDatasets();
      setDatasets(res.items || []);
      if (res.items && res.items.length > 0) {
        await selectSnapshot(res.items[0].snapshot_id);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [selectSnapshot]);

  async function handleIngest(e: React.FormEvent) {
    e.preventDefault();
    if (!inputPath) {
      setIngestStatus('✗ Please specify a local file or directory path');
      return;
    }
    setIngesting(true);
    setIngestStatus('Ingesting and processing text corpus...');
    try {
      const payload = {
        input_path: inputPath,
        dataset_id: datasetId,
        source_type: sourceType,
        license_note: licenseNote,
        chunk_size: chunkSize,
        overlap: overlap,
        strict_chunk_dedup: strictChunkDedup
      };
      const res = await api.ingestTextDataset(payload);
      setIngestStatus(`✓ Successfully ingested snapshot: ${res.snapshot_id}`);
      setInputPath('');
      await load();
    } catch (e: any) {
      setIngestStatus(`✗ Ingestion failed: ${e.message}`);
    } finally {
      setIngesting(false);
    }
  }

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="p-8 animate-fade-in text-slate-100">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <FileText className="h-5 w-5 text-blue-500" />
          <span className="text-xs font-semibold uppercase tracking-widest text-blue-500">Text Corpora</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Unstructured Text Pipelines</h1>
        <p className="mt-1 text-sm text-slate-400">
          Ingest raw text, PDFs, HTML, or JSONL documents. Text is normalized, chunked, and deduplicated automatically.
        </p>
      </div>

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 p-4 text-sm text-red-300 ring-1 ring-red-500/25">
          {error}
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Left column: Registry list + Ingestion Form */}
        <div className="col-span-12 lg:col-span-7 space-y-6">
          {/* Snapshots Table */}
          <Card padding="none">
            <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white">Ingested Text Snapshots</h2>
              <span className="text-xs text-slate-500">{datasets.length} active</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-white/[0.06] text-slate-500 uppercase font-semibold">
                    <th className="px-4 py-3">Snapshot ID</th>
                    <th className="px-4 py-3">Dataset</th>
                    <th className="px-4 py-3">Format</th>
                    <th className="px-4 py-3">Docs</th>
                    <th className="px-4 py-3">Chunks</th>
                    <th className="px-4 py-3">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && [...Array(3)].map((_, idx) => (
                    <tr key={idx} className="border-b border-white/[0.04] animate-pulse">
                      <td colSpan={6} className="px-4 py-5 bg-white/[0.01]" />
                    </tr>
                  ))}
                  {!loading && datasets.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center text-slate-500">
                        No text corpora ingested yet.
                      </td>
                    </tr>
                  )}
                  {!loading && datasets.map((snap) => (
                    <tr
                      key={snap.snapshot_id}
                      onClick={() => selectSnapshot(snap.snapshot_id)}
                      className={`table-row-hover cursor-pointer border-b border-white/[0.04] ${
                        selectedSnapshot?.snapshot_id === snap.snapshot_id ? 'bg-blue-600/10' : ''
                      }`}
                    >
                      <td className="px-4 py-3.5 font-mono text-blue-400">{snap.snapshot_id.slice(0, 16)}…</td>
                      <td className="px-4 py-3.5 capitalize">{snap.dataset_id}</td>
                      <td className="px-4 py-3.5 font-mono uppercase">{snap.source_type}</td>
                      <td className="px-4 py-3.5 font-semibold">{snap.document_count?.toLocaleString()}</td>
                      <td className="px-4 py-3.5 font-semibold text-slate-300">{snap.chunk_count?.toLocaleString()}</td>
                      <td className="px-4 py-3.5 text-slate-500">{formatDate(snap.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Local Ingestion Form */}
          <Card>
            <div className="border-b border-white/[0.06] pb-3 mb-4">
              <h3 className="text-sm font-semibold text-white">Corpus Ingestion</h3>
              <p className="text-xs text-slate-500 mt-1">
                Process unstructured files into queryable text chunks.
              </p>
            </div>
            <form onSubmit={handleIngest} className="space-y-4 text-xs">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-slate-400 font-semibold mb-1">Source Path (Absolute)</label>
                  <input
                    type="text"
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    placeholder="e.g. C:\data\corpus or C:\data\docs.pdf"
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-blue-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Dataset ID</label>
                  <input
                    type="text"
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-blue-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Format</label>
                  <select
                    value={sourceType}
                    onChange={(e) => setSourceType(e.target.value)}
                    className="w-full rounded bg-slate-900 border border-white/10 px-3 py-2 text-slate-200 focus:border-blue-500 outline-none"
                  >
                    <option value="txt">TXT Directory / File</option>
                    <option value="pdf">PDF Directory / File</option>
                    <option value="jsonl">JSONL Lines</option>
                    <option value="csv">CSV Records</option>
                    <option value="html">HTML Directory / File</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Chunk Size (Chars)</label>
                  <input
                    type="number"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(Number(e.target.value))}
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-blue-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Chunk Overlap (Chars)</label>
                  <input
                    type="number"
                    value={overlap}
                    onChange={(e) => setOverlap(Number(e.target.value))}
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-blue-500 outline-none"
                  />
                </div>
              </div>

              <div className="flex items-center gap-2 mt-2">
                <input
                  type="checkbox"
                  id="strictDedup"
                  checked={strictChunkDedup}
                  onChange={(e) => setStrictChunkDedup(e.target.checked)}
                  className="rounded border-white/10 bg-slate-900"
                />
                <label htmlFor="strictDedup" className="text-slate-400 font-semibold">
                  Strict Chunk Deduplication (Drop exact chunk duplicates)
                </label>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-[10px] text-slate-500">Processing may take a while for large corpora.</span>
                <Button variant="primary" size="sm" loading={ingesting} type="submit" className="bg-blue-600 hover:bg-blue-700 text-white">
                  Ingest Corpus
                </Button>
              </div>
              {ingestStatus && (
                <div className={`mt-2 p-2.5 rounded text-xs ${
                  ingestStatus.startsWith('✓') ? 'bg-emerald-500/10 text-emerald-300' : 'bg-red-500/10 text-red-300'
                }`}>
                  {ingestStatus}
                </div>
              )}
            </form>
          </Card>
        </div>

        {/* Right column: Detail view */}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          {selectedSnapshot ? (
            <>
              {/* Snapshot Details */}
              <Card>
                <div className="border-b border-white/[0.06] pb-3 mb-4 flex justify-between items-start">
                  <div>
                    <h3 className="text-sm font-semibold text-white">Snapshot Details</h3>
                    <p className="font-mono text-[10px] text-slate-500 mt-1">{selectedSnapshot.snapshot_id}</p>
                  </div>
                  <Link href={`/datasets/text/${selectedSnapshot.snapshot_id}`}>
                    <Button variant="ghost" size="sm" className="h-8">View Chunks</Button>
                  </Link>
                </div>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Dataset ID:</span>
                    <span className="font-semibold text-slate-200 capitalize">{selectedSnapshot.dataset_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Original Format:</span>
                    <span className="font-mono text-slate-300 uppercase">{selectedSnapshot.source_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Documents Processed:</span>
                    <span className="text-emerald-400 font-bold">{selectedSnapshot.document_count?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Total Chunks:</span>
                    <span className="text-blue-400 font-bold">{selectedSnapshot.chunk_count?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Rejected / Malformed:</span>
                    <span className={selectedSnapshot.malformed_documents > 0 ? 'text-red-400 font-bold' : 'text-slate-400'}>
                      {selectedSnapshot.malformed_documents?.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Duplicate Documents:</span>
                    <span className={selectedSnapshot.duplicate_documents > 0 ? 'text-orange-400 font-bold' : 'text-slate-400'}>
                      {selectedSnapshot.duplicate_documents?.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Exact Chunk Dupes:</span>
                    <span className="text-slate-400">{selectedSnapshot.exact_duplicate_chunks?.toLocaleString()}</span>
                  </div>
                  
                  {selectedSnapshot.languages && Object.keys(selectedSnapshot.languages).length > 0 && (
                    <div className="pt-2 border-t border-white/[0.04]">
                      <span className="text-slate-500 block mb-1">Language Distribution:</span>
                      <div className="flex gap-2">
                        {Object.entries(selectedSnapshot.languages).map(([lang, count]) => (
                          <Badge key={lang} variant="neutral">{lang}: {String(count)}</Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedSnapshot.size_distribution && (
                    <div className="pt-2 border-t border-white/[0.04]">
                      <span className="text-slate-500 block mb-1">Doc Size Distribution:</span>
                      <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                        <div className="bg-white/5 rounded p-1">
                          <div className="text-slate-400">&lt;1KB</div>
                          <div className="font-semibold">{selectedSnapshot.size_distribution["<1KB"] || 0}</div>
                        </div>
                        <div className="bg-white/5 rounded p-1">
                          <div className="text-slate-400">1-10KB</div>
                          <div className="font-semibold">{selectedSnapshot.size_distribution["1KB-10KB"] || 0}</div>
                        </div>
                        <div className="bg-white/5 rounded p-1">
                          <div className="text-slate-400">&gt;10KB</div>
                          <div className="font-semibold">{selectedSnapshot.size_distribution[">10KB"] || 0}</div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="pt-2 border-t border-white/[0.04]">
                    <span className="text-slate-500 block mb-1">Canonical Parquet Path:</span>
                    <p className="font-mono text-slate-400 text-[10px] break-all">{selectedSnapshot.parquet_root}</p>
                  </div>
                </div>
              </Card>

              {/* Cryptographic Validation */}
              <Card>
                <div className="border-b border-white/[0.06] pb-3 mb-3">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                    <ShieldCheck className="h-4 w-4 text-emerald-400" />
                    Provenance Integrity
                  </h3>
                </div>
                <div className="text-xs space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Config Hash:</span>
                    <span className="font-mono text-slate-400 text-[10px]">{selectedSnapshot.ingestion_config_hash?.slice(0, 16)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Source Path Fingerprint:</span>
                    <span className="font-mono text-slate-400 text-[10px]">{selectedSnapshot.source_path_fingerprint?.slice(0, 16)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Canonical Content Hash:</span>
                    <span className="font-mono text-slate-400 text-[10px]">{selectedSnapshot.canonical_content_hash?.slice(0, 16)}</span>
                  </div>
                </div>
              </Card>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 text-xs">
              <Database className="h-12 w-12 mx-auto mb-2 opacity-20" />
              Select a snapshot to inspect details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
