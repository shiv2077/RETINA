/**
 * RETINA Frontend - API Client
 * ============================
 *
 * Type-safe API client for communicating with the Rust backend.
 *
 * All API calls go through this module to ensure:
 * - Consistent error handling
 * - Type safety with TypeScript
 * - Centralized configuration
 */

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// =============================================================================
// Types
// =============================================================================

/** Model types available for inference */
export type ModelType = 'patchcore' | 'padim' | 'winclip' | 'pushpull';

/** Pipeline stage */
export type PipelineStage = 1 | 2;

/** Job status */
export type JobStatus = 'pending' | 'queued' | 'processing' | 'completed' | 'failed';

/** Label confidence level */
export type LabelConfidence = 'low' | 'medium' | 'high';

// -----------------------------------------------------------------------------
// Request Types
// -----------------------------------------------------------------------------

/** Request to submit an image for inference */
export interface SubmitImageRequest {
  image_id: string;
  model_type?: ModelType;
  priority?: number;
  source?: string;
}

/** Request to submit a label */
export interface SubmitLabelRequest {
  image_id: string;
  is_anomaly: boolean;
  defect_category?: string;
  confidence: LabelConfidence;
  labeled_by: string;
  notes?: string;
}

// -----------------------------------------------------------------------------
// Response Types
// -----------------------------------------------------------------------------

/** Health check response */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  redis_connected: boolean;
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

/** Inference result */
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
}

/** Result query response */
export interface ResultResponse {
  found: boolean;
  result?: InferenceResult;
  message: string;
}

/** Label submission response */
export interface SubmitLabelResponse {
  success: boolean;
  image_id: string;
  total_labels: number;
  labels_for_stage2: number;
  stage2_available: boolean;
  message: string;
}

/** Unlabeled sample for active learning */
export interface UnlabeledSample {
  image_id: string;
  anomaly_score: number;
  uncertainty_score: number;
  added_at: string;
}

/** Labeling pool response */
export interface LabelingPoolResponse {
  samples: UnlabeledSample[];
  pool_size: number;
  labels_collected: number;
  stage2_threshold: number;
}

/** System statistics */
export interface SystemStats {
  jobs_submitted: number;
  jobs_completed: number;
  queue_length: number;
  labels_collected: number;
  labeling_pool_size: number;
}

/** Active models information */
export interface ActiveModels {
  stage1_model: string;
  stage2_model?: string;
  models_loaded: boolean;
}

/** System status response */
export interface SystemStatusResponse {
  status: string;
  current_stage: number;
  stage2_available: boolean;
  labels_for_stage2: number;
  stage2_progress: number;
  stats: SystemStats;
  active_models: ActiveModels;
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

  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.error?.message || `Request failed: ${response.statusText}`,
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
 * Get the current system status.
 */
export async function getSystemStatus(): Promise<SystemStatusResponse> {
  return apiFetch<SystemStatusResponse>('/api/system/status');
}

/**
 * Submit an image for anomaly detection.
 */
export async function submitImage(
  request: SubmitImageRequest
): Promise<SubmitImageResponse> {
  return apiFetch<SubmitImageResponse>('/api/images/submit', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get the inference result for a job or image.
 *
 * @param id - Either a job_id (UUID) or image_id
 */
export async function getResult(id: string): Promise<ResultResponse> {
  return apiFetch<ResultResponse>(`/api/images/${encodeURIComponent(id)}/result`);
}

/**
 * Submit a label for an image (active learning).
 */
export async function submitLabel(
  request: SubmitLabelRequest
): Promise<SubmitLabelResponse> {
  return apiFetch<SubmitLabelResponse>('/api/labels/submit', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get samples from the active learning labeling pool.
 */
export async function getLabelingPool(): Promise<LabelingPoolResponse> {
  return apiFetch<LabelingPoolResponse>('/api/labels/pool');
}
