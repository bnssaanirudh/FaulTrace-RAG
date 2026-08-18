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
  listWorlds: async () => {
    const res = await apiFetch<PaginatedResponse<World>>('/worlds');
    return res.items;
  },
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
  getRunAttribution: (id: string) => apiFetch<any>(`/runs/${id}/attribution`),
  getRunCertificate: (id: string) => apiFetch<any>(`/runs/${id}/certificate`),
  batchEvaluatePolicy: (policyId = 'strict_exact_v1') =>
    apiFetch<any>(`/runs/batch-evaluate-policy?policy_id=${policyId}`),

  // Policies
  listPolicies: () => apiFetch<any[]>('/policies'),
  getPolicy: (id: string) => apiFetch<any>(`/policies/${id}`),

  // Datasets
  listDatasets: (datasetId?: string, activeOnly = true) => {
    const params = new URLSearchParams({ active_only: String(activeOnly) });
    if (datasetId) params.set('dataset_id', datasetId);
    return apiFetch<any>(`/datasets?${params}`).then(res => ({ count: res.total || res.count, snapshots: res.items || res.snapshots }));
  },
  getDatasetSnapshot: (snapshotId: string) => apiFetch<any>(`/datasets/${snapshotId}`),
  validateDatasetSnapshot: (snapshotId: string) => apiFetch<any>(`/datasets/${snapshotId}/validate`),
  getDatasetMissingness: (snapshotId: string) => apiFetch<any>(`/datasets/${snapshotId}/missingness`),
  ingestDataset: (inputPath: string, datasetId: string, licenseNote = "", maxBytesMb = 500) =>
    apiFetch<any>('/datasets/ingest', {
      method: 'POST',
      body: JSON.stringify({ input_path: inputPath, dataset_id: datasetId, license_note: licenseNote, max_bytes_mb: maxBytesMb }),
    }),

  // Text Datasets
  listTextDatasets: (datasetId?: string, activeOnly = true) => {
    const params = new URLSearchParams({ active_only: String(activeOnly) });
    if (datasetId) params.set('dataset_id', datasetId);
    return apiFetch<{ total: number; items: any[] }>(`/datasets/text?${params}`);
  },
  getTextDatasetSnapshot: (snapshotId: string) => apiFetch<any>(`/datasets/text/${snapshotId}`),
  getTextDatasetPreview: (snapshotId: string, page = 1, pageSize = 10) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    return apiFetch<any>(`/datasets/text/${snapshotId}/preview?${params}`);
  },
  ingestTextDataset: (payload: any) =>
    apiFetch<any>('/datasets/text/ingest', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Experiments
  planExperiment: (spec: any) =>
    apiFetch<any>('/experiments/plan', {
      method: 'POST',
      body: JSON.stringify(spec),
    }),
  runExperiment: (spec: any) =>
    apiFetch<any>('/experiments/run', {
      method: 'POST',
      body: JSON.stringify(spec),
    }),
  listExperiments: () => apiFetch<any>('/experiments'),
  getExperiment: (id: string) => apiFetch<any>(`/experiments/${id}`),
  cancelExperiment: (id: string) =>
    apiFetch<any>(`/experiments/${id}/cancel`, {
      method: 'POST',
    }),
  compareExperiments: (id1: string, id2: string) =>
    apiFetch<any>('/experiments/compare', {
      method: 'POST',
      body: JSON.stringify({ experiment_id_1: id1, experiment_id_2: id2 }),
    }),

  // Annotations
  listAnnotationTasks: () => apiFetch<any[]>('/annotations/tasks'),
  assignAnnotationTask: (taskId: string, userId: string) =>
    apiFetch<any>(`/annotations/assignments?task_id=${taskId}&user_id=${userId}`, {
      method: 'POST',
    }),
  submitAnnotation: (assignmentId: string, result: any, timeSpent: number) =>
    apiFetch<any>(`/annotations/assignments/${assignmentId}/submit?time_spent=${timeSpent}`, {
      method: 'POST',
      body: JSON.stringify(result),
    }),
};
