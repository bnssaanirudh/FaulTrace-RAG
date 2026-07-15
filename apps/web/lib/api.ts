/**
 * API client for the FaultTrace-RAG backend.
 * All requests use Next.js rewrites proxy to avoid CORS issues.
 */

const API_BASE = '/api/v1';

export interface ApiError {
  error: string;
  message: string;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'unknown', message: res.statusText }));
    throw new Error(err.message || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ---- Types ----
export interface World {
  world_id: string;
  dataset_id: string;
  seed: number;
  scale_n: number;
  parent_world_id: string | null;
  creation_policy: string;
  record_ids_hash: string;
  manifest_path: string;
  created_at: string;
  schema_version: string;
}

export interface Query {
  query_id: string;
  world_id: string;
  family: string;
  natural_language_question: string;
  template_id: string;
  version: string;
  spec: Record<string, unknown>;
  gold: GoldAnswer | null;
  created_at: string;
}

export interface GoldAnswer {
  query_id: string;
  world_id: string;
  answer_value: unknown;
  eligible_record_count: number;
  contributing_record_ids: string[];
  evidence_hash: string;
  agreement_status: string;
  tolerance: number;
  pandas_result?: unknown;
  duckdb_result?: unknown;
}

export interface Run {
  run_id: string;
  query_id: string;
  pipeline_id: string;
  provider_id: string;
  status: string;
  answer: string | null;
  gold_answer_value: string | null;
  is_correct: boolean | null;
  loss: number | null;
  latency_ms: number | null;
  error_message: string | null;
  config_hash: string | null;
  artifact_refs: Record<string, string>;
  started_at: string;
  completed_at: string | null;
}

export interface TraceEvent {
  event_id: string;
  run_id: string;
  parent_event_id: string | null;
  stage: string;
  event_type: string;
  message: string;
  record_count_in: number | null;
  record_count_out: number | null;
  duration_ms: number | null;
  payload: Record<string, unknown>;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

export interface SystemStatus {
  status: string;
  timestamp: string;
  version: string;
  milestone: string;
  components: Record<string, string>;
  counts: { worlds: number; queries: number; runs: number };
  settings: Record<string, unknown>;
  pipelines: Record<string, string>;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

// ---- API calls ----

export const api = {
  health: () => apiFetch<HealthResponse>('/health'),

  status: () => apiFetch<SystemStatus>('/system/status'),

  seedDemo: (seed = 42, scales = [10, 50, 200, 1000], overwrite = false) =>
    apiFetch<{ status: string; world_ids: string[]; queries_generated: number }>('/demo/seed', {
      method: 'POST',
      body: JSON.stringify({ seed, scales, overwrite }),
    }),

  // Worlds
  listWorlds: () => apiFetch<World[]>('/worlds'),
  getWorld: (id: string) => apiFetch<World>(`/worlds/${id}`),
  listWorldRecords: (worldId: string, page = 1, pageSize = 50, filters?: Record<string, string>) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (filters) Object.entries(filters).forEach(([k, v]) => v && params.set(k, v));
    return apiFetch<PaginatedResponse<Record<string, unknown>>>(`/worlds/${worldId}/records?${params}`);
  },

  // Queries
  listQueries: (worldId?: string, family?: string, page = 1, pageSize = 20) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (worldId) params.set('world_id', worldId);
    if (family) params.set('family', family);
    return apiFetch<PaginatedResponse<Query>>(`/queries?${params}`);
  },
  getQuery: (id: string) => apiFetch<Query>(`/queries/${id}`),
  generateQueries: (worldId: string, count = 60) =>
    apiFetch<{ generated: number; stored: number }>('/queries/generate', {
      method: 'POST',
      body: JSON.stringify({ world_id: worldId, count }),
    }),

  // Gold
  getGold: (queryId: string) => apiFetch<GoldAnswer>(`/gold/${queryId}`),

  // Runs
  createRun: (queryId: string, pipelineId = 'P0-deterministic-scope-baseline') =>
    apiFetch<Run>('/runs', {
      method: 'POST',
      body: JSON.stringify({ query_id: queryId, pipeline_id: pipelineId }),
    }),
  listRuns: (queryId?: string, pipelineId?: string, page = 1, pageSize = 20) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (queryId) params.set('query_id', queryId);
    if (pipelineId) params.set('pipeline_id', pipelineId);
    return apiFetch<PaginatedResponse<Run>>(`/runs?${params}`);
  },
  getRun: (id: string) => apiFetch<Run>(`/runs/${id}`),
  getRunTrace: (id: string) => apiFetch<TraceEvent[]>(`/runs/${id}/trace`),
};
