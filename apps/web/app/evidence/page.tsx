'use client';

import { useState, useCallback } from 'react';

interface ExtractionResult {
  doc_id: string;
  dataset_id: string;
  provider: string;
  model_version: string | null;
  support_status: string;
  cited_doc_ids: string[];
  supporting_quote: string | null;
  abstention_reason: string | null;
  total_facts: number;
  repair_log: Array<{ attempt: number; success: boolean; error: string | null; duration_ms: number }>;
  schema_version: string;
}

const STATUS_CONFIG: Record<string, { label: string; color: string; bg: string; dot: string }> = {
  supported: {
    label: 'Supported',
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.12)',
    dot: '#22c55e',
  },
  partially_supported: {
    label: 'Partially Supported',
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.12)',
    dot: '#f59e0b',
  },
  unsupported: {
    label: 'Unsupported',
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.12)',
    dot: '#ef4444',
  },
  conflicting: {
    label: 'Conflicting',
    color: '#a855f7',
    bg: 'rgba(168,85,247,0.12)',
    dot: '#a855f7',
  },
  insufficient_evidence: {
    label: 'Insufficient Evidence',
    color: '#6b7280',
    bg: 'rgba(107,114,128,0.12)',
    dot: '#6b7280',
  },
};

function SupportBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG['insufficient_evidence'];
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '4px 12px',
        borderRadius: '20px',
        fontSize: '13px',
        fontWeight: 600,
        color: cfg.color,
        background: cfg.bg,
        border: `1px solid ${cfg.color}40`,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: cfg.dot,
          display: 'inline-block',
        }}
      />
      {cfg.label}
    </span>
  );
}

function HighlightedText({ text, quote }: { text: string; quote: string | null }) {
  if (!quote || !text.includes(quote)) {
    return (
      <p style={{ margin: 0, lineHeight: 1.7, color: '#d1d5db', fontSize: 14 }}>
        {text}
      </p>
    );
  }

  const idx = text.indexOf(quote);
  const before = text.slice(0, idx);
  const highlighted = text.slice(idx, idx + quote.length);
  const after = text.slice(idx + quote.length);

  return (
    <p style={{ margin: 0, lineHeight: 1.7, color: '#d1d5db', fontSize: 14 }}>
      {before}
      <mark
        style={{
          background: 'rgba(251,191,36,0.3)',
          color: '#fbbf24',
          borderRadius: 3,
          padding: '1px 3px',
        }}
      >
        {highlighted}
      </mark>
      {after}
    </p>
  );
}

export default function EvidencePage() {
  const [docId, setDocId] = useState('');
  const [docText, setDocText] = useState('');
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [validateDocIds, setValidateDocIds] = useState('');
  const [knownDocIds, setKnownDocIds] = useState('');
  const [validateResult, setValidateResult] = useState<{
    is_valid: boolean;
    violations: string[];
    invented_doc_ids: string[];
    citation_integrity_status: string;
  } | null>(null);

  const handleExtract = useCallback(async () => {
    if (!docId.trim() || !docText.trim() || !query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch('/api/v1/evidence/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, doc_text: docText, query }),
      });
      if (!resp.ok) throw new Error((await resp.json()).message ?? 'Extraction failed');
      setResult(await resp.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [docId, docText, query]);

  const handleValidate = useCallback(async () => {
    const cited = validateDocIds.split(',').map(s => s.trim()).filter(Boolean);
    const known = knownDocIds.split(',').map(s => s.trim()).filter(Boolean);
    try {
      const resp = await fetch('/api/v1/evidence/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_id: 'ui_test',
          dataset_id: 'unknown',
          split: 'test',
          cited_doc_ids: cited,
          known_doc_ids: known,
          support_status: 'supported',
        }),
      });
      setValidateResult(await resp.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Validation error');
    }
  }, [validateDocIds, knownDocIds]);

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
        color: '#f1f5f9',
        fontFamily: "'Inter', 'Segoe UI', sans-serif",
        padding: '32px 24px',
      }}
    >
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span style={{ fontSize: 28 }}>🔍</span>
            <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, background: 'linear-gradient(135deg, #60a5fa, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Evidence Inspector
            </h1>
          </div>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: 15 }}>
            Extract evidence from documents, assess support status, and validate citation integrity.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          {/* Left Panel — Extraction */}
          <div
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 16,
              padding: 24,
            }}
          >
            <h2 style={{ margin: '0 0 20px', fontSize: 18, fontWeight: 700, color: '#60a5fa' }}>
              Extract Evidence
            </h2>

            <label style={{ display: 'block', marginBottom: 12 }}>
              <span style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Document ID</span>
              <input
                value={docId}
                onChange={e => setDocId(e.target.value)}
                placeholder="e.g. doc_4983"
                style={{
                  width: '100%', boxSizing: 'border-box', padding: '10px 14px',
                  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 8, color: '#f1f5f9', fontSize: 14, outline: 'none',
                }}
              />
            </label>

            <label style={{ display: 'block', marginBottom: 12 }}>
              <span style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Query / Claim</span>
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="e.g. Does aspirin reduce cardiovascular risk?"
                style={{
                  width: '100%', boxSizing: 'border-box', padding: '10px 14px',
                  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 8, color: '#f1f5f9', fontSize: 14, outline: 'none',
                }}
              />
            </label>

            <label style={{ display: 'block', marginBottom: 20 }}>
              <span style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Document Text</span>
              <textarea
                value={docText}
                onChange={e => setDocText(e.target.value)}
                rows={6}
                placeholder="Paste document text here..."
                style={{
                  width: '100%', boxSizing: 'border-box', padding: '10px 14px',
                  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 8, color: '#f1f5f9', fontSize: 14, outline: 'none',
                  resize: 'vertical', fontFamily: 'inherit',
                }}
              />
            </label>

            <button
              onClick={handleExtract}
              disabled={loading || !docId || !docText || !query}
              style={{
                width: '100%', padding: '12px 20px',
                background: loading ? 'rgba(96,165,250,0.3)' : 'linear-gradient(135deg, #3b82f6, #6366f1)',
                border: 'none', borderRadius: 10, color: '#fff', fontWeight: 700,
                fontSize: 14, cursor: loading ? 'wait' : 'pointer', transition: 'all 0.2s',
                letterSpacing: '0.02em',
              }}
            >
              {loading ? '⏳ Extracting...' : '⚡ Extract Evidence'}
            </button>

            {error && (
              <div style={{ marginTop: 16, padding: 12, background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, color: '#fca5a5', fontSize: 13 }}>
                ❌ {error}
              </div>
            )}
          </div>

          {/* Right Panel — Results */}
          <div
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 16,
              padding: 24,
            }}
          >
            <h2 style={{ margin: '0 0 20px', fontSize: 18, fontWeight: 700, color: '#a78bfa' }}>
              Extraction Result
            </h2>

            {!result && !loading && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: '#475569', fontSize: 14 }}>
                Run extraction to see results
              </div>
            )}

            {result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Support Status */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>Support Status:</span>
                  <SupportBadge status={result.support_status} />
                </div>

                {/* Supporting Quote with highlighting */}
                {result.supporting_quote && (
                  <div>
                    <p style={{ margin: '0 0 8px', fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      Supporting Quote
                    </p>
                    <div style={{ padding: '10px 14px', background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)', borderRadius: 8, borderLeft: '3px solid #fbbf24' }}>
                      <HighlightedText text={docText} quote={result.supporting_quote} />
                    </div>
                  </div>
                )}

                {result.abstention_reason && (
                  <div style={{ padding: '10px 14px', background: 'rgba(107,114,128,0.08)', border: '1px solid rgba(107,114,128,0.2)', borderRadius: 8 }}>
                    <p style={{ margin: 0, fontSize: 13, color: '#9ca3af' }}>
                      <strong>Abstention:</strong> {result.abstention_reason}
                    </p>
                  </div>
                )}

                {/* Metadata */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {[
                    { label: 'Doc ID', value: result.doc_id },
                    { label: 'Provider', value: result.provider },
                    { label: 'Facts', value: result.total_facts },
                    { label: 'Schema', value: result.schema_version },
                  ].map(({ label, value }) => (
                    <div key={label} style={{ padding: '8px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8 }}>
                      <p style={{ margin: 0, fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</p>
                      <p style={{ margin: '2px 0 0', fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>{String(value)}</p>
                    </div>
                  ))}
                </div>

                {/* Cited Doc IDs */}
                {result.cited_doc_ids.length > 0 && (
                  <div>
                    <p style={{ margin: '0 0 6px', fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      Cited Doc IDs
                    </p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {result.cited_doc_ids.map(id => (
                        <span key={id} style={{ padding: '3px 10px', background: 'rgba(96,165,250,0.12)', border: '1px solid rgba(96,165,250,0.3)', borderRadius: 20, fontSize: 12, color: '#93c5fd' }}>
                          {id}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Citation Validator */}
        <div
          style={{
            marginTop: 24,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 16,
            padding: 24,
          }}
        >
          <h2 style={{ margin: '0 0 20px', fontSize: 18, fontWeight: 700, color: '#34d399' }}>
            Citation Integrity Validator
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <label>
              <span style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Cited Doc IDs (comma separated)
              </span>
              <input
                value={validateDocIds}
                onChange={e => setValidateDocIds(e.target.value)}
                placeholder="doc1, doc2, INVENTED_ID"
                style={{
                  width: '100%', boxSizing: 'border-box', padding: '10px 14px',
                  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 8, color: '#f1f5f9', fontSize: 14, outline: 'none',
                }}
              />
            </label>
            <label>
              <span style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Known Doc IDs (comma separated)
              </span>
              <input
                value={knownDocIds}
                onChange={e => setKnownDocIds(e.target.value)}
                placeholder="doc1, doc2, doc3"
                style={{
                  width: '100%', boxSizing: 'border-box', padding: '10px 14px',
                  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 8, color: '#f1f5f9', fontSize: 14, outline: 'none',
                }}
              />
            </label>
          </div>
          <button
            onClick={handleValidate}
            style={{
              padding: '10px 24px',
              background: 'linear-gradient(135deg, #059669, #10b981)',
              border: 'none', borderRadius: 8, color: '#fff', fontWeight: 700,
              fontSize: 14, cursor: 'pointer',
            }}
          >
            🔒 Validate Citations
          </button>

          {validateResult && (
            <div style={{ marginTop: 16, padding: 16, background: validateResult.is_valid ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)', border: `1px solid ${validateResult.is_valid ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`, borderRadius: 10 }}>
              <p style={{ margin: '0 0 8px', fontWeight: 700, fontSize: 14, color: validateResult.is_valid ? '#22c55e' : '#ef4444' }}>
                {validateResult.is_valid ? '✅ Citations Valid' : '❌ Citation Integrity Violated'}
              </p>
              {validateResult.invented_doc_ids.length > 0 && (
                <div>
                  <p style={{ margin: '0 0 6px', fontSize: 13, color: '#fca5a5' }}>Invented IDs found:</p>
                  {validateResult.invented_doc_ids.map(id => (
                    <code key={id} style={{ display: 'block', fontSize: 12, color: '#fca5a5', background: 'rgba(239,68,68,0.1)', padding: '2px 8px', borderRadius: 4, marginBottom: 4 }}>
                      {id}
                    </code>
                  ))}
                </div>
              )}
              {validateResult.violations.map((v, i) => (
                <p key={i} style={{ margin: '4px 0', fontSize: 12, color: '#fca5a5' }}>{v}</p>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
