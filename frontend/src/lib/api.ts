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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// =============================================================================
// Types
// =============================================================================

/** Model types available for inference */
export type ModelType = 'patchcore' | 'padim' | 'winclip' | 'gpt4v' | 'pushpull';

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
  /** Absolute path to image on shared volume (set after uploading via /api/images/upload) */
  image_path?: string;
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
  /** Human-readable defect description from GPT-4o (null for non-VLM results) */
  defect_description?: string | null;
  /** Spatial description of defect location from GPT-4o */
  defect_location?: string | null;
  /** One-sentence GPT-4o reasoning for the prediction */
  gpt4v_reasoning?: string | null;
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

// ============================================================================
// CASCADE INFERENCE & ACTIVE LEARNING TYPES
// ============================================================================

/** Cascade route case */
export type CascadeRoutingCase = 
  | 'A_confident_normal'
  | 'B_confident_anomaly'
  | 'C_uncertain_vlm_routed';

/** Cascade prediction response */
export interface CascadeResponse {
  model_used: 'bgad' | 'ensemble' | 'vlm';
  anomaly_score: number;
  is_anomaly: boolean;
  confidence: number;
  routing_case: CascadeRoutingCase;
  requires_expert_labeling: boolean;
  vlm_result?: {
    classification: string;
    confidence?: number;
  };
  bgad_score?: number;
  vlm_score?: number;
  timestamp: string;
  processing_time_ms?: number;
  queue_info?: {
    success: boolean;
    image_id: string;
    queue_position: number;
    queue_size: number;
  };
}

/** Cascade queue item */
export interface CascadeQueueItem {
  image_id: string;
  image_path: string;
  bgad_score: number;
  vlm_score?: number;
  routing_case: CascadeRoutingCase;
  status: 'pending' | 'labeled' | 'skipped';
  created_at: string;
  metadata?: Record<string, any>;
}

/** Cascade queue response */
export interface CascadeQueueResponse {
  success: boolean;
  queue: CascadeQueueItem[];
  queue_size: number;
  stats: {
    total_in_queue: number;
    pending: number;
    labeled: number;
    skipped: number;
  };
}

/** Cascade annotation submission */
export interface CascadeAnnotationSubmission {
  image_id: string;
  label: 'normal' | 'anomaly';
  bounding_boxes: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    defect_type: string;
    confidence?: number;
  }>;
  defect_types: string[];
  notes?: string;
}

/** Cascade queue statistics */
export interface CascadeQueueStats {
  total_queued: number;
  pending: number;
  labeled: number;
  skipped: number;
  avg_bgad_score: number;
  annotation_store_stats: Record<string, any>;
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
 * Get the current system status.
 * Backend route: GET /api/system/status
 */
export async function getSystemStatus(): Promise<SystemStatusResponse> {
  return apiFetch<SystemStatusResponse>('/api/system/status');
}

/**
 * Submit an image for anomaly detection.
 * Backend route: POST /api/images/submit
 */
export async function submitImage(
  request: SubmitImageRequest
): Promise<SubmitImageResponse> {
  return apiFetch<SubmitImageResponse>('/api/images/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

/**
 * Get the inference result for a job or image.
 * Backend route: GET /api/images/{id}/result
 *
 * @param id - Either a job_id (UUID) or image_id
 */
export async function getResult(id: string): Promise<ResultResponse> {
  return apiFetch<ResultResponse>(`/api/images/${encodeURIComponent(id)}/result`);
}

/**
 * Submit a label for an image (active learning).
 * Backend route: POST /api/labels/submit
 */
export async function submitLabel(
  request: SubmitLabelRequest
): Promise<SubmitLabelResponse> {
  return apiFetch<SubmitLabelResponse>('/api/labels/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

/**
 * Get samples from the active learning labeling pool.
 * Backend route: GET /api/labels/pool
 */
export async function getLabelingPool(): Promise<LabelingPoolResponse> {
  return apiFetch<LabelingPoolResponse>('/api/labels/pool');
}

// =============================================================================
// CASCADE INFERENCE & ACTIVE LEARNING API
// =============================================================================

/**
 * Run cascade prediction on an image.
 * Routes between BGAD (fast) and VLM (fallback) based on confidence.
 * Automatically queues for annotation if uncertain.
 */
export async function predictCascade(
  imageFile: File,
  options?: {
    normal_threshold?: number;
    anomaly_threshold?: number;
    use_vlm_fallback?: boolean;
  }
): Promise<CascadeResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  if (options?.normal_threshold !== undefined) {
    formData.append('normal_threshold', options.normal_threshold.toString());
  }
  if (options?.anomaly_threshold !== undefined) {
    formData.append('anomaly_threshold', options.anomaly_threshold.toString());
  }
  if (options?.use_vlm_fallback !== undefined) {
    formData.append('use_vlm_fallback', options.use_vlm_fallback.toString());
  }

  const response = await fetch(`${API_BASE_URL}/api/predict/cascade`, {
    method: 'POST',
    body: formData,
    // Don't set Content-Type header - let browser set it with boundary
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Cascade prediction failed: ${response.statusText}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Fetch pending images from the cascade annotation queue.
 * These are images flagged by the cascade router for human review.
 */
export async function fetchAnnotationQueue(
  limit?: number
): Promise<CascadeQueueResponse> {
  const params = new URL(`${API_BASE_URL}/api/labeling/cascade/queue`);
  if (limit !== undefined) {
    params.searchParams.append('limit', limit.toString());
  }

  const response = await fetch(params.toString(), {
    method: 'GET',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Failed to fetch annotation queue: ${response.statusText}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Submit an annotation for a cascade queue item.
 */
export async function submitCascadeAnnotation(
  submission: CascadeAnnotationSubmission
): Promise<{ success: boolean; image_id: string; remaining_in_queue: number }> {
  const formData = new FormData();
  formData.append('image_id', submission.image_id);
  formData.append('label', submission.label);
  formData.append('bounding_boxes', JSON.stringify(submission.bounding_boxes));
  formData.append('defect_types', JSON.stringify(submission.defect_types));
  if (submission.notes) {
    formData.append('notes', submission.notes);
  }

  const response = await fetch(`${API_BASE_URL}/api/labeling/cascade/submit`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Failed to submit annotation: ${response.statusText}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Skip a cascade queue item without labeling.
 */
export async function skipCascadeItem(
  image_id: string
): Promise<{ success: boolean; image_id: string; remaining_in_queue: number }> {
  const response = await fetch(
    `${API_BASE_URL}/api/labeling/cascade/skip/${encodeURIComponent(image_id)}`,
    {
      method: 'POST',
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Failed to skip item: ${response.statusText}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Get cascade queue statistics.
 */
export async function getCascadeStats(): Promise<CascadeQueueStats> {
  const response = await fetch(
    `${API_BASE_URL}/api/labeling/cascade/stats`,
    {
      method: 'GET',
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Failed to fetch stats: ${response.statusText}`,
      response.status
    );
  }

  return response.json();
}
