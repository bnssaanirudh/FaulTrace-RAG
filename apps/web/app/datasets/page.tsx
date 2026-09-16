'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Database, ShieldCheck, FileSpreadsheet, CheckCircle2, AlertTriangle } from 'lucide-react';
import { formatDate } from '@/lib/utils';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSnapshot, setSelectedSnapshot] = useState<any | null>(null);
  const [missingness, setMissingness] = useState<any | null>(null);
  const [validationReport, setValidationReport] = useState<any | null>(null);
  const [error, setError] = useState('');

  // Ingestion Form State
  const [inputPath, setInputPath] = useState('');
  const [datasetId, setDatasetId] = useState('amazon_demo');
  const [licenseNote, setLicenseNote] = useState('Amazon Review Synthetic Subset');
  const [maxBytesMb, setMaxBytesMb] = useState(10);
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');

  const selectSnapshot = useCallback(async (id: string) => {
    try {
      const snap = await api.getDatasetSnapshot(id);
      setSelectedSnapshot(snap);
      setValidationReport(null);

      const miss = await api.getDatasetMissingness(id);
      setMissingness(miss);

      const val = await api.validateDatasetSnapshot(id);
      setValidationReport(val);
    } catch (e: unknown) {
      console.error(e);
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await api.listDatasets();
      setDatasets(res.snapshots || []);
      if (res.snapshots && res.snapshots.length > 0) {
        await selectSnapshot(res.snapshots[0].snapshot_id);
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
      setIngestStatus('✗ Please specify a local file path');
      return;
    }
    setIngesting(true);
    setIngestStatus('Ingesting local file...');
    try {
      const res = await api.ingestDataset(inputPath, datasetId, licenseNote, maxBytesMb);
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
          <Database className="h-5 w-5 text-orange-500" />
          <span className="text-xs font-semibold uppercase tracking-widest text-orange-500">Data Layer</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Datasets Provenance</h1>
        <p className="mt-1 text-sm text-slate-400">
          Inspect, validate, and ingest Amazon-style reviews or SEC EDGAR XBRL corpuses into canonical Parquet schemas.
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
              <h2 className="text-sm font-semibold text-white">Ingested Snapshots</h2>
              <span className="text-xs text-slate-500">{datasets.length} snapshots active</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-white/[0.06] text-slate-500 uppercase font-semibold">
                    <th className="px-4 py-3">Snapshot ID</th>
                    <th className="px-4 py-3">Dataset</th>
                    <th className="px-4 py-3">Rows</th>
                    <th className="px-4 py-3">Validation</th>
                    <th className="px-4 py-3">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && [...Array(3)].map((_, idx) => (
                    <tr key={idx} className="border-b border-white/[0.04] animate-pulse">
                      <td colSpan={5} className="px-4 py-5 bg-white/[0.01]" />
                    </tr>
                  ))}
                  {!loading && datasets.length === 0 && (
                    <tr>
                      <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                        No custom ingested snapshots. Seed example datasets to start.
                      </td>
                    </tr>
                  )}
                  {!loading && datasets.map((snap) => (
                    <tr
                      key={snap.snapshot_id}
                      onClick={() => selectSnapshot(snap.snapshot_id)}
                      className={`table-row-hover cursor-pointer border-b border-white/[0.04] ${
                        selectedSnapshot?.snapshot_id === snap.snapshot_id ? 'bg-orange-600/10' : ''
                      }`}
                    >
                      <td className="px-4 py-3.5 font-mono text-orange-400">{snap.snapshot_id.slice(0, 16)}…</td>
                      <td className="px-4 py-3.5 capitalize">{snap.dataset_id}</td>
                      <td className="px-4 py-3.5 font-semibold">{snap.row_count.toLocaleString()}</td>
                      <td className="px-4 py-3.5">
                        <Badge variant={snap.active ? 'success' : 'neutral'}>
                          {snap.active ? 'Validated' : 'Inactive'}
                        </Badge>
                      </td>
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
              <h3 className="text-sm font-semibold text-white">Local File Ingestion</h3>
              <p className="text-xs text-slate-500 mt-1">
                Ingest local files safely. Path restrictions block paths outside data directories.
              </p>
            </div>
            <form onSubmit={handleIngest} className="space-y-4 text-xs">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Local File Absolute Path</label>
                  <input
                    type="text"
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    placeholder="e.g. C:\data\reviews.jsonl"
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Dataset Identifier</label>
                  <input
                    type="text"
                    value={datasetId}
                    onChange={(e) => setDatasetId(e.target.value)}
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">License/Provenance Note</label>
                  <input
                    type="text"
                    value={licenseNote}
                    onChange={(e) => setLicenseNote(e.target.value)}
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 font-semibold mb-1">Max Bytes Limit (MB)</label>
                  <input
                    type="number"
                    value={maxBytesMb}
                    onChange={(e) => setMaxBytesMb(Number(e.target.value))}
                    className="w-full rounded bg-white/[0.03] border border-white/10 px-3 py-2 text-slate-200 focus:border-orange-500 outline-none"
                  />
                </div>
              </div>
              <div className="flex items-center justify-between pt-2">
                <span className="text-[10px] text-slate-500">Security check active: Refuses uncompressed files &gt;500MB</span>
                <Button variant="primary" size="sm" loading={ingesting} type="submit" className="bg-orange-600 hover:bg-orange-700 text-white">
                  Ingest Snapshot
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

        {/* Right column: Detail view + Null Matrix */}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          {selectedSnapshot ? (
            <>
              {/* Snapshot Details */}
              <Card>
                <div className="border-b border-white/[0.06] pb-3 mb-4">
                  <h3 className="text-sm font-semibold text-white">Snapshot Details</h3>
                  <p className="font-mono text-[10px] text-slate-500 mt-1">{selectedSnapshot.snapshot_id}</p>
                </div>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Dataset ID:</span>
                    <span className="font-semibold text-slate-200 capitalize">{selectedSnapshot.dataset_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Source Format:</span>
                    <span className="font-mono text-slate-300">{selectedSnapshot.source_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Accepted Records:</span>
                    <span className="text-emerald-400 font-bold">{selectedSnapshot.accepted_count?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Rejected / Malformed:</span>
                    <span className={selectedSnapshot.rejected_count > 0 ? 'text-red-400 font-bold' : 'text-slate-400'}>
                      {selectedSnapshot.rejected_count?.toLocaleString()} / {selectedSnapshot.malformed_count?.toLocaleString()}
                    </span>
                  </div>
                  {selectedSnapshot.license_note && (
                    <div className="pt-2 border-t border-white/[0.04]">
                      <span className="text-slate-500 block mb-1">Provenance Note:</span>
                      <p className="text-slate-300 text-[11px] leading-relaxed">{selectedSnapshot.license_note}</p>
                    </div>
                  )}
                </div>
              </Card>

              {/* Cryptographic Validation */}
              <Card>
                <div className="border-b border-white/[0.06] pb-3 mb-4">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                    <ShieldCheck className="h-4 w-4 text-emerald-400" />
                    Cryptographic Integrity
                  </h3>
                </div>
                {validationReport ? (
                  <div className="text-xs space-y-3">
                    <div className="flex items-center justify-between">
                      <span>Status:</span>
                      <Badge variant={validationReport.status === 'valid' ? 'success' : 'error'}>
                        {validationReport.status?.toUpperCase()}
                      </Badge>
                    </div>
                    <div className="space-y-1">
                      {validationReport.checks?.map((c: any, idx: number) => (
                        <div key={idx} className="flex items-center gap-2 text-[10px] text-slate-400">
                          {c.status === 'pass' ? (
                            <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                          ) : (
                            <AlertTriangle className="h-3 w-3 text-red-400" />
                          )}
                          <span>{c.check}: {c.status}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-slate-500 text-xs">Computing checks...</div>
                )}
              </Card>

              {/* Null Matrix / Missingness Overview */}
              <Card>
                <div className="border-b border-white/[0.06] pb-3 mb-4">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                    <FileSpreadsheet className="h-4 w-4 text-violet-400" />
                    Missingness Null Matrix
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">Ratio of null fields across all rows in snapshot</p>
                </div>
                {missingness ? (
                  <div className="space-y-4 text-xs">
                    <div className="space-y-2">
                      {Object.entries(missingness.null_count_by_field || {}).map(([field, count]: any) => {
                        const ratio = count / missingness.row_count;
                        return (
                          <div key={field}>
                            <div className="flex justify-between text-[11px] mb-1">
                              <span className="font-mono text-slate-300">{field}</span>
                              <span className="text-slate-500">{ratio > 0 ? `${(ratio * 100).toFixed(1)}% null` : '0%'}</span>
                            </div>
                            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                              <div
                                className={`h-full ${ratio > 0 ? 'bg-orange-500' : 'bg-emerald-400'}`}
                                style={{ width: `${ratio * 100}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="text-slate-500 text-xs">Loading null values...</div>
                )}
              </Card>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 text-xs">
              <Database className="h-12 w-12 mx-auto mb-2 opacity-20" />
              Select a snapshot to inspect field profiles
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
