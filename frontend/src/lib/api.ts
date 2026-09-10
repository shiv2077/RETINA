/**
 * RETINA Frontend - API Client
 * ============================
 *
 * Type-safe API client for the FastAPI wrapper in api/main.py.
 *
 * All API calls go through this module to ensure:
 * - Consistent error handling
 * - Type safety with TypeScript
 * - Centralized configuration
 */

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// =============================================================================
// Types
// =============================================================================

/** Model types available for inference */
export type ModelType = 'patchcore' | 'padim' | 'winclip' | 'gpt4v' | 'pushpull';

/** Pipeline stage */
export type PipelineStage = 1 | 2;

/** Job status */
/**
 * Job lifecycle status. Three terminal states, and the difference between
 * the last two is operational: `needs_review` means the pipeline ran fine
 * and declined to decide (a human should look), `failed` means the
 * pipeline broke (someone should be paged). See docs/DECISIONS.md #18.
 */
export type JobStatus =
  | 'pending'
  | 'queued'
  | 'processing'
  | 'completed'
  | 'needs_review'
  | 'failed';

// -----------------------------------------------------------------------------
// Request Types
// -----------------------------------------------------------------------------

/** Request to submit an image for inference */
export interface SubmitImageRequest {
  image_id: string;
  model_type?: ModelType;
  priority?: number;
  source?: string;
  /** Absolute path to image on shared volume (set after uploading via /api/images/upload) */
  image_path?: string;
}

// -----------------------------------------------------------------------------
// Response Types
// -----------------------------------------------------------------------------

/** Health check response */
export interface HealthResponse {
  status: string;
  timestamp: string;
  gpu_available: boolean;
  gpu_name: string | null;
}

/** Image submission response */
export interface SubmitImageResponse {
  job_id: string;
  status: JobStatus;
  queue_position: number | null;
  message: string;
}

/** Stage 1 specific outputs */
export interface Stage1Output {
  heatmap_available: boolean;
  heatmap_key?: string;
  feature_distance?: number;
  clip_similarity?: number;
}

/** Stage 2 specific outputs */
export interface Stage2Output {
  defect_category?: string;
  category_probabilities?: Record<string, number>;
  embedding_distance?: number;
}

/** Active learning metadata */
export interface ActiveLearningMeta {
  uncertainty_score: number;
  in_labeling_pool: boolean;
  labeled: boolean;
}

/** Inference result — matches shared/schemas/result.json */
export interface InferenceResult {
  job_id: string;
  image_id: string;
  status: JobStatus;
  created_at: string;
  completed_at?: string;
  model_used?: ModelType;
  stage: PipelineStage;
  anomaly_score?: number;
  is_anomaly?: boolean;
  confidence?: number;
  stage1_output?: Stage1Output;
  stage2_output?: Stage2Output;
  active_learning: ActiveLearningMeta;
  error?: { code: string; message: string };
  processing_time_ms?: number;
  /** Legacy GPT-4V description, kept for backward compatibility */
  defect_description?: string | null;
  defect_location?: string | null;
  gpt4v_reasoning?: string | null;
  /** VLM router outputs (added in the Stage 1/2 integration) */
  product_class?: string | null;
  product_confidence?: number | null;
  natural_description?: string | null;
  defect_severity?: 'minor' | 'moderate' | 'severe' | null;
  defect_type?: string | null;
  routing_reason?:
    | 'patchcore_normal'
    | 'patchcore_confirmed_anomaly'
    | 'stage2_confirmed'
    | 'stage2_rejected'
    | 'stage2_uncertain_kept'
    | 'unknown_product_zero_shot'
    | string
    | null;
  vlm_model_used?: 'gpt-4o' | 'gpt-4o-mini' | string | null;
  vlm_api_cost_estimate_usd?: number | null;
  /** Stage 2 supervised refiner outputs */
  stage2_verdict?:
    | 'confirmed_anomaly'
    | 'rejected_false_positive'
    | 'uncertain'
    | null;
  stage2_defect_class?: string | null;
  stage2_confidence?: number | null;
}

/** Result query response */
export interface ResultResponse {
  found: boolean;
  result?: InferenceResult;
  message: string;
}

// =============================================================================
// API Error
// =============================================================================

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// =============================================================================
// Fetch Helper
// =============================================================================

async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || errorData.error?.message || `Request failed: ${response.statusText}`,
      response.status,
      errorData.error?.code
    );
  }

  return response.json();
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Check the health of the backend service.
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health');
}

/**
 * Submit an image file for anomaly detection via the FastAPI wrapper.
 * Backend route: POST /api/submit   (multipart/form-data)
 *
 * The backend returns only { job_id }. Callers should poll getResult(job_id).
 */
export async function submitImage(
  file: File,
  productClass?: string | null,
): Promise<{ job_id: string; image_id: string; product_class: string | null }> {
  const form = new FormData();
  form.append('file', file);
  // Sent only when the caller actually knows it. The backend rejects an
  // unknown value with 400 rather than falling back to VLM identification,
  // so a wrong value fails loudly instead of quietly costing an API call.
  if (productClass) form.append('product_class', productClass);
  const response = await fetch(`${API_BASE_URL}/api/submit`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Submit failed: ${response.statusText}`,
      response.status,
    );
  }
  return response.json();
}

/**
 * Get the InferenceResult for a job_id. Returns null if not ready yet (404).
 * Backend route: GET /api/result/{job_id}
 */
export async function getResult(job_id: string): Promise<InferenceResult | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/result/${encodeURIComponent(job_id)}?wait=0`,
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Result fetch failed: ${response.statusText}`,
      response.status,
    );
  }
  return response.json();
}

/**
 * Submit + poll until a result comes back. Resolves to the InferenceResult or
 * throws ApiError on timeout. Convenience wrapper for the submit page.
 */
export async function submitAndWait(
  file: File,
  opts: { pollMs?: number; timeoutMs?: number; productClass?: string | null } = {},
): Promise<InferenceResult> {
  const { pollMs = 1000, timeoutMs = 60_000, productClass = null } = opts;
  const { job_id } = await submitImage(file, productClass);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await getResult(job_id);
    if (result) return result;
    await new Promise(r => setTimeout(r, pollMs));
  }
  throw new ApiError(
    `Inference timed out after ${timeoutMs}ms (job_id=${job_id})`,
    504,
  );
}

/** Active learning pool item returned by the FastAPI wrapper. */
export interface PoolItem {
  image_id: string;
  score: number;
  image_url: string;
  anomaly_score?: number | null;
  uncertainty_score?: number | null;
  product_class?: string | null;
}

/**
 * FastAPI labels pool (used by the annotation studio).
 * Matches api/main.py /api/labels/pool response.
 */
export async function getLabelPoolV2(limit = 20): Promise<{ pool: PoolItem[]; count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/labels/pool?limit=${limit}`);
  if (!response.ok) {
    throw new ApiError(`labels pool fetch failed: ${response.statusText}`, response.status);
  }
  return response.json();
}

/** Label submission payload accepted by POST /api/labels/submit (FastAPI). */
export interface LabelSubmissionV2 {
  image_id: string;
  product_class: string;
  label: 'anomaly' | 'normal';
  defect_class?: string | null;
  polygons?: Array<{ vertices: Array<{ x: number; y: number }>; class: string | null }> | null;
  boxes?: Array<{
    x: number; y: number; width: number; height: number;
    defect_type: string; confidence?: number;
  }> | null;
  operator_id?: string | null;
  notes?: string | null;
}

/** Simple alias: flat array of pool items (matches user-facing contract). */
export async function getLabelPool(limit = 50): Promise<PoolItem[]> {
  const res = await getLabelPoolV2(limit);
  return res.pool;
}

export interface CustomTaxonomyEntry {
  key: string;
  name: string;
  color: string;
  shortcut: string;
  custom?: boolean;
}

/** Fetch operator-added categories for a product. */
export async function fetchCustomTaxonomy(product_class: string): Promise<CustomTaxonomyEntry[]> {
  const r = await fetch(`${API_BASE_URL}/api/taxonomy/${encodeURIComponent(product_class)}`);
  if (!r.ok) throw new ApiError(`taxonomy fetch failed: ${r.statusText}`, r.status);
  const data = await r.json();
  return data.custom ?? [];
}

/** Append a custom category for a product. Returns the full updated list. */
export async function addCustomTaxonomyEntry(
  product_class: string,
  entry: CustomTaxonomyEntry,
): Promise<CustomTaxonomyEntry[]> {
  const r = await fetch(`${API_BASE_URL}/api/taxonomy/${encodeURIComponent(product_class)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new ApiError(err.detail || `taxonomy post failed: ${r.statusText}`, r.status);
  }
  const data = await r.json();
  return data.custom ?? [];
}

export async function submitLabelV2(
  body: LabelSubmissionV2,
): Promise<{ ok: boolean; labels_count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/labels/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new ApiError(
      err.detail || `label submit failed: ${response.statusText}`,
      response.status,
    );
  }
  return response.json();
}

/**
 * Product categories with a trained PatchCore checkpoint — exactly the set
 * POST /api/submit will accept for product_class. Fetched rather than
 * hardcoded so the UI cannot offer a category the backend would reject.
 */
export async function fetchTrainedCategories(): Promise<string[]> {
  const r = await fetch(`${API_BASE_URL}/api/categories`);
  if (!r.ok) throw new ApiError(`categories fetch failed: ${r.statusText}`, r.status);
  const data = await r.json();
  return data.categories ?? [];
}
