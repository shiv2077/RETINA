# RETINA — Design Decisions

## 1. PatchCore owns the numeric verdict; GPT-4o only explains it.
**Status:** Implemented
**Context:** A defect verdict needs to be calibrated and reproducible across runs so operators can trust a threshold. GPT-4o's output shifts with prompt framing and shows priming bias — telling it an image was flagged makes it more likely to report a defect even on clean images.
**Decision:** `worker.py:306` computes `is_anomaly = anomaly_score > settings.anomaly_threshold` from PatchCore's score alone. GPT-4o is only invoked afterward, in `describe_defect` (`worker.py:323`) and `stage2_refine` (`worker.py:347`), to explain or refine a verdict PatchCore already produced.
**Alternatives rejected:**
- *VLM as primary detector* — no calibrated score grounded in the product's normal distribution, and repeated calls on the same image can return different verdicts.
- *VLM as a second scorer, averaged with PatchCore* — mixes a calibrated score with an uncalibrated one; a single bad VLM read degrades a good PatchCore read instead of being caught by it.
**Consequences:** Any defect visible to the VLM but invisible to PatchCore's memory bank is missed entirely — the pipeline never asks the VLM to independently flag something PatchCore didn't already flag.
**Revisit when:** a VLM produces a reproducible calibrated score (same image, same score, run twice) on a held-out set.
**Evidence:** worker.py:306,316-333,340-357; docs/vlm_router_design.md:1-13.

## 2. PatchCore for Stage 1, not PaDiM or WinCLIP.
**Status:** Implemented
**Context:** Stage 1 needs a one-class detector trained per product category from normal images only, running across 15 MVTec categories on a single 6 GB GPU.
**Decision:** `patchcore_registry.py` loads per-category `anomalib.models.Patchcore` checkpoints. Measured on MVTec: mean image AUROC 0.9829 across all 15 categories, ranging from 0.9194 (toothbrush) to 1.0000 (bottle, hazelnut, leather, tile).
**Alternatives rejected:**
- *PaDiM* — scored 0.884 AUROC against PatchCore's 0.895 in the Flanders Make project evaluation, while carrying the identical one-class per-category training requirement. Lower accuracy for no portability gain — nothing about the deployment gets easier by switching.
- *WinCLIP* — zero-shot, no per-category training needed, but scored weakest of the three at 0.856 in that same evaluation. The zero-shot advantage doesn't offset being the lowest scorer in a role that owns the primary verdict.
**Consequences:** Per-category training is required before a new product line can be inspected, and the checkpoints/ directory (gitignored) totals 7.4 GB. The 0.895/0.884/0.856 comparison is the Flanders Make project's own evaluation, run on project data (Decospan/Deprez) — not re-benchmarked in this repo, and not the MVTec dataset the 0.9829 figure above comes from; the two numbers are not comparable.
**Revisit when:** GPU VRAM becomes the binding constraint before detection accuracy does.
**Evidence:** results/mvtec_stage1_results.csv (this repo, MVTec); Flanders Make project evaluation deck (PatchCore/PaDiM/WinCLIP on Decospan/Deprez data — not checked into this repo).

## 3. Stage 2 is a GPT-4o in-context refiner, not a trained supervised model.
**Status:** Implemented
**Context:** The prior architecture called for a trained Stage 2 model — BGAD or a custom Push-Pull network — to classify confirmed defects. Both need labeled training data per product category.
**Decision:** `stage2_refine` (`vlm_router.py:242-322`) sends the image plus up to 5 operator-labeled examples fetched from `retina:labels:*` as in-context few-shot examples to `gpt-4o`, no model training step.
**Alternatives rejected:**
- *BGAD* — scored 0.93 AUROC in the Flanders Make project evaluation, but requires per-category pixel-level defect masks and a training run; a new customer or product line needs a fresh masked dataset and model before Stage 2 works at all.
- *Custom Push-Pull* — scored 0.86 in that same evaluation, no masks required and fewer samples than BGAD (300: 100 normal, 200 anomaly), but still needs per-category training; same onboarding delay, smaller.
**Consequences:** Traded classification accuracy for zero training cost, and added a hard dependency on an external API being reachable for every ambiguous image. BGAD stays in `research/supervised/BGAD/`, unwired. The 0.93/0.86 figures are the Flanders Make project's evaluation on project data, not a benchmark run in this repo.
**Revisit when:** a single customer's per-category label volume approaches the ~300 samples Custom Push-Pull trained on — at that point in-context refinement stops being the cheaper option.
**Evidence:** worker.py:340-357; vlm_router.py:242-322; Flanders Make project evaluation deck (not checked into this repo); research/supervised/Custom_Model_Push_Pull/outputs/dataset_info.json.

## 4. Stage 2 only sees the uncertainty band, not every flagged image.
**Status:** Implemented
**Context:** The prior architecture (CLAUDE.md §1.1) ran Stage 2 on every image Stage 1 flagged as anomalous — calling GPT-4o on images PatchCore already scored with high confidence.
**Decision:** `should_run_stage2` (`vlm_router.py:236-240`) gates the call to scores in `[0.5, 0.9)`. `worker.py:340` only calls `stage2_refine` when this returns true, on top of the `is_anomaly` check that already triggered `describe_defect`.
**Alternatives rejected:**
- *Run Stage 2 on every flagged image* — the CLAUDE.md-era design. Spends an API call refining images PatchCore is already confident about, where the refiner has the least to add.
**Consequences:** A PatchCore score below 0.5 is never checked by Stage 2 — if PatchCore is confidently wrong on a genuine defect, nothing downstream catches it. Stage 2 can only correct scores PatchCore was already unsure about.
**Revisit when:** measured false-negative rate on scores below 0.5 exceeds what a customer's QC line will tolerate.
**Evidence:** vlm_router.py:67-68,236-240; worker.py:340; CLAUDE.md:112.

## 5. No separate multi-class defect classifier.
**Status:** Deliberately not built
**Context:** The prior architecture specified a third model producing calibrated per-class defect probabilities; CLAUDE.md lists it as a planned, unbuilt component.
**Decision:** `defect_class` is a single string field on `Stage2Verdict` (`vlm_router.py:57-61`), populated by asking GPT-4o to name one class in its JSON response (`worker.py:354`). No classifier model exists.
**Alternatives rejected:**
- *Train a dedicated multi-class classifier* — needs a labeled multi-class dataset per product category, the same per-customer labeling and training budget problem that ruled out BGAD in entry 3.
**Consequences:** The system reports a single label with no calibrated confidence distribution across classes — an operator gets "crack" or "contamination," not a probability that it might be one of several.
**Revisit when:** a customer asks for per-class defect rate reporting — there is currently no counting mechanism downstream of the single `defect_class` string to support it.
**Evidence:** CLAUDE.md:32,501; vlm_router.py:57-61; worker.py:354.

## 6. Added identify_product as a routing layer in front of PatchCore.
**Status:** Implemented
**Context:** The prior architecture assumed one configured product type per deployment (`GPT4V_PRODUCT_TYPE` env var, CLAUDE.md §7.1). This worker instead serves all 15 trained categories from one process and needs to know which checkpoint applies to each image.
**Decision:** `identify_product` (`vlm_router.py:102-159`) calls `gpt-4o-mini` to classify the image into one of `KNOWN_PRODUCT_CATEGORIES` (15 entries, `vlm_router.py:27-31`) before any PatchCore call.
**Alternatives rejected:**
- *Single configured product type per deployment* — the prior assumption; breaks the goal of pointing the camera at any of the 15 trained categories without a config change.
- *A trained product classifier* — another model to train, version, and ship alongside 15 PatchCore checkpoints, for a problem gpt-4o-mini already solves adequately.
**Consequences:** Every image with no session-cache hit costs a paid API call on the inference critical path, and gpt-4o-mini has a known failure mode: it over-classifies unrecognized objects into one of the 15 known categories instead of returning `unknown`.
**Revisit when:** an operator or evaluation run shows gpt-4o-mini returning a known category with high confidence for an object that is not actually one of the 15 trained products — the trigger for the confidence gate README.md already flags as needed.
**Evidence:** vlm_router.py:27-31,102-159; CLAUDE.md:346,545; README.md:371-375.

## 7. Product identity is session-cached with a 1-hour TTL.
**Status:** Implemented
**Context:** Calling `identify_product` on every image dominates per-image cost when the same product runs for an extended period, the common case on a QC line.
**Decision:** `worker.py:457-464` stores the identified product class in `retina:session:product_class` with a 3600-second TTL (`SESSION_TTL_S`, `worker.py:76`); subsequent images reuse it until expiry.
**Alternatives rejected:**
- *Identify every image* — correct on every frame, but the dominant cost driver at any real image volume.
- *Cache indefinitely, no TTL* — never re-identifies, so a genuine product changeover is never detected.
**Consequences:** This is a real, currently unmitigated bug: if an operator switches products mid-session without clearing the cache, every subsequent image runs through the previous product's PatchCore checkpoint and scores unreliably until the TTL expires or someone manually deletes the key.
**Revisit when:** a deployment needs two product lines running concurrently against the same worker — the current global session key can't distinguish them.
**Evidence:** worker.py:74-76,457-464; README.md:393-398.

## 8. zero_shot_detect is the fallback for products with no trained checkpoint.
**Status:** Implemented
**Context:** Any product `identify_product` doesn't recognize, or recognizes but has no PatchCore checkpoint for, has no calibrated detector available.
**Decision:** `worker.py:272-301` routes these images to `zero_shot_detect` (`vlm_router.py:324-368`), which asks `gpt-4o` to judge the image against general defect principles with no product-specific training.
**Alternatives rejected:**
- *Refuse to process unrecognized products* — simpler, but means anything outside the 15 trained categories produces nothing, which blocks demonstrating the system on a new object and blocks onboarding a new product line on day one.
- *Auto-train a checkpoint on first sighting* — there is no clean set of confirmed-normal images for a product the moment it's first seen; training on unverified images risks baking defects into the memory bank as "normal."
**Consequences:** Best-effort only — no calibrated score, and blind to reference-dependent defects (a part that's physically intact but assembled wrong, e.g. `cable_swap`), which zero-shot visual reasoning cannot catch without a known-correct reference to compare against. This specific miss is locked in as a passing test rather than something later prompt tuning is expected to fix.
**Revisit when:** a reference image or few-shot example per part number becomes available to inject into the prompt — the mitigation already named in docs/vlm_router_design.md.
**Evidence:** worker.py:272-301; vlm_router.py:324-368; scripts/test_vlm_router.py:89-97; docs/vlm_router_design.md:32-42.

## 9. Redis Streams for the job queue, not Celery, RQ, or SQS.
**Status:** Implemented
**Context:** One GPU-bound worker process needs to consume jobs with at-least-once delivery and be safely restartable without losing an in-flight job.
**Decision:** `redis_client.py:50-51` defines the queue as a Redis Stream (`retina:jobs:queue`) with consumer group `workers`, consumed via `XREADGROUP` and acknowledged via `XACK` (`redis_client.py:96-115,192-215`).
**Alternatives rejected:**
- *Celery* — needs a broker, a result backend, and a worker pool; for one GPU and one consumer that's operational surface with no throughput benefit.
- *RQ* — simpler than Celery, but has no consumer-group concept, so a crashed worker's in-flight job isn't automatically redeliverable the way an unacked stream entry is.
- *SQS* — solves the same delivery problem, but pulls in a cloud dependency for a system meant to run next to a camera and PLC on a factory floor, not call out to AWS.
**Consequences:** Redis is now a single point of failure for both the queue and every piece of state (results, labels, session cache) — if Redis goes down, the whole pipeline stops, not just the queue.
**Revisit when:** the deployment needs multiple GPU workers behind one queue at a scale where a managed broker's operational tooling (dead-letter queues, delayed retry, monitoring) outweighs Streams' simplicity.
**Evidence:** redis_client.py:12-29,50-51,89-115,192-215.

## 10. Redis with TTL is the only persistence layer; no Postgres.
**Status:** Implemented
**Context:** The prior architecture specified Postgres for durable storage (CLAUDE.md §3.2, §7.1). Labels here feed a GPT-4o few-shot prompt, not a training pipeline that needs a queryable historical table.
**Decision:** Labels, results, and job state all live in Redis hashes/sorted sets with a 7-day TTL on results and labels. No database, schema, or migration exists in the current demo path.
**Alternatives rejected:**
- *Postgres, as originally specified* — correct for a system that needs to survive restarts and support audit queries, but adds a schema, a migration story, and a second service to operate for a demo where every label already round-trips through Redis in the request path.
**Consequences:** This is the largest gap between this build and a deployable system: labels expire after 7 days, there is no audit trail, no export, no label versioning, and nothing to retrain a model from six months from now.
**Revisit when:** a label needs to survive past its TTL to feed a retraining run.
**Evidence:** CLAUDE.md §3.2, §7.1 (POSTGRES_PASSWORD); README.md:245-247; redis_client.py:1-29.

## 11. Two-model LRU cache for PatchCore checkpoints, not all 15 loaded at once.
**Status:** Implemented
**Context:** The target GPU is a 6 GB RTX 3060 Laptop. Each PatchCore checkpoint occupies roughly 1.5 GB resident (backbone plus coreset memory bank) once loaded.
**Decision:** `PatchCoreRegistry` (`patchcore_registry.py:31-118`) keeps at most 2 models resident, evicting least-recently-used on a cache miss (`patchcore_registry.py:110-115`).
**Alternatives rejected:**
- *Load all 15 categories at startup* — at ~1.5 GB each, exceeds the 6 GB budget well before reaching category 5.
- *Load per request, discard after* — avoids the memory ceiling, but a cold checkpoint load takes multiple seconds (`patchcore_registry.py:77-108` times and logs each load), unacceptable latency for every single image.
- *CPU-only inference* — removes the VRAM constraint, but wasn't viable at production line speed.
**Consequences:** A product-changeover pattern that alternates between more than 2 categories thrashes the cache, paying the multi-second checkpoint load cost repeatedly instead of once.
**Revisit when:** more VRAM is available, or a smaller backbone reduces the per-model footprint enough to raise `max_cached` beyond 2.
**Evidence:** patchcore_registry.py:22-28,57-118.

## 12. Single-threaded worker, one image processed at a time.
**Status:** Implemented
**Context:** GPU inference for one image already occupies the GPU; concurrent Python threads submitting work would still queue behind each other on the same device.
**Decision:** `worker_concurrency` defaults to 1 (`config.py:54`) and the main loop (`worker.py:146-154`) processes exactly one job start to finish before pulling the next.
**Alternatives rejected:**
- *A thread pool submitting concurrent inference calls* — adds lock contention around the single shared GPU and the 2-model LRU cache, without changing how many images per second the one GPU can process.
- *Micro-batch several images before running PatchCore* — trades latency for throughput; without a measured production line speed and the physical distance to the reject actuator, there's no basis for choosing a batch size or flush timeout that wouldn't risk a reject decision arriving too late to act on.
**Consequences:** The GPU sits idle between the end of one image's inference and the next image being pulled off the queue — kernel launch and preprocessing overhead is paid per image with no batching to amortize it.
**Revisit when:** a real production line speed and actuator distance exist — then batch on whichever of size or timeout fires first, grouped by product_class so a mixed batch doesn't force a checkpoint swap mid-batch.
**Evidence:** config.py:24-25,54; worker.py:146-154.

## 13. No authentication on the API.
**Status:** Deliberately not built
**Context:** This is a single-operator demo system, one FastAPI process and one Redis instance, not a multi-tenant deployment.
**Decision:** `api/main.py` has no auth middleware; `operator_id` on a label submission (`api/main.py:164,178`) is a self-reported optional string with no verification.
**Alternatives rejected:**
- *JWT auth at the API edge* — a well-understood pattern, but adds a login flow in front of a demo whose purpose is showing the inference pipeline. The prior Rust backend already had a working JWT implementation; it went unused along with the rest of that codebase (entry 14).
**Consequences:** Not multi-tenant deployable as-is — no verified operator identity on any label, no per-customer isolation of checkpoints, Redis keys, or defect taxonomies, and anyone with network access to the API can submit images or labels.
**Revisit when:** a second customer or a second operator role is added — auth belongs at the API edge only, with the worker staying behind the network boundary.
**Evidence:** api/main.py:56-65,157-187.

## 14. Deleted the Rust/Axum backend and the archived FastAPI backend.
**Status:** Implemented
**Context:** `backend/` (Rust/Axum, 3,562 LOC) accumulated compile errors during a prior reorganization with no user-visible behavior behind it — every demo feature runs through `api/main.py`. `legacy/fastapi_backend/` (3,114 LOC) was fully superseded, zero runtime imports elsewhere in the repo.
**Decision:** Both directories were removed, along with the `backend` service in `docker-compose.yml` and the now-dangling `depends_on: backend` in `frontend`.
**Alternatives rejected:**
- *Repair backend/* — `cargo check` reported 7 mechanical errors (private field access in `alerts.rs`; missing struct fields in `db/models.rs`), fixable, but fixing them changes nothing a demo user sees.
- *Keep both, unrepaired, as reference material* — the JWT auth, Postgres schema, and route structure represent real design work. Rejected: a compose file still building a broken service, plus two unused backend implementations in the tree, reads as unresolved indecision — worse than rebuilding this scaffolding from git history if ever needed.
**Consequences:** The JWT auth, Postgres schema, and Rust route structure are gone from the working tree. A future deployment needing Rust-level throughput or that auth implementation has to rebuild it, not repair it in place.
**Revisit when:** measured p99 latency on `/api/submit` or `/api/result` exceeds what a customer's line tolerates and FastAPI/uvicorn's single-process model is the bottleneck — decide fresh rather than resurrecting deleted code.
**Evidence:** docker-compose.yml (backend service removed); README.md:203-210,385-388 (original rationale, superseded by this deletion).

## 15. Stage 2 runs before describe_defect, not independently of it.
**Status:** Implemented
**Context:** For a score in the uncertainty band [0.5, 0.9), the worker used to call both `describe_defect` (fires for any score above 0.5) and `stage2_refine` (fires for the same band) on every mid-band image — including ones Stage 2 went on to reject, where the description was wasted.
**Decision:** The mid-band branch now calls `stage2_refine` first. `describe_defect` (factored into a `_describe_defect` helper) only fires afterward if the image is still anomalous once Stage 2 has run. A Stage 2 rejection (`rejected_false_positive`, confidence >= 0.7) skips the description call entirely. Scores at or above 0.9 are unchanged — `describe_defect` runs directly; Stage 2 has nothing to add above the band.
**Alternatives rejected:**
- *Leave both calls unconditional* — the prior behavior. Simpler, but paid for a description on every mid-band image even when Stage 2 found no defect.
**Consequences:** A mid-band image Stage 2 confirms as genuine still costs two GPT-4o calls — that pairing was never the problem. The saving only applies to images Stage 2 rejects; those no longer pay for a description of a non-defect.
**Revisit when:** cost tracking shows the post-Stage-2 describe_defect call is still a material share of spend on confirmed mid-band defects — at that point, have Stage 2 produce the operator-facing description directly instead of a second call.
**Evidence:** worker.py (`_run_inference` steps 3-4, `_describe_defect` helper).

## 16. Redis requires a password, and the ports are loopback-only.
**Status:** Implemented
**Context:** Redis and Postgres published their ports on all interfaces with no password. Anything on the host — and on the LAN if the host firewall allowed it — could read or write every job, result and label with raw Redis commands, bypassing the API entirely. Entry 13 covers the deliberate absence of API auth; this is a different hole, and one nothing had decided on.
**Decision:** Both port mappings bind to `127.0.0.1`. `redis-server` starts with `--requirepass` sourced from `REDIS_PASSWORD`, using compose's `${VAR:?err}` form so a missing password fails the stack loudly instead of silently starting an unauthenticated Redis. Every connection string carries the credential; `api/main.py` reads `REDIS_URL` from the environment rather than hardcoding `redis://localhost:6379`.
**Alternatives rejected:**
- *Leave it, since it is a single-operator demo* — the same reasoning as entry 13, but the blast radius is different: no-API-auth exposes an API with a known shape, whereas an open Redis exposes the queue and every stored label to anything that can open a socket.
- *Bind to loopback without a password* — protects against the LAN but not against any other process or container on the same host, and provides nothing if the port is ever republished.
**Consequences:** Adding auth silently broke two healthchecks, which is the part worth remembering. The Redis healthcheck was `redis-cli ping`, which returns `NOAUTH Authentication required` once a password exists, and the worker's healthcheck dialled a hardcoded `redis://redis:6379` with no credential. Both would have reported the stack unhealthy for a reason unrelated to the change's intent. The healthcheck now authenticates via `REDISCLI_AUTH` (keeping the secret off the process command line) and the worker's reads its own `REDIS_URL`. A security change's blast radius extends to everything that was quietly relying on the absence of that security.
**Revisit when:** the deployment gains a second service that needs Redis — at that point per-service ACL users are worth more than one shared password.
**Evidence:** docker-compose.yml (redis `command`, `REDISCLI_AUTH`, both `ports` blocks, worker `healthcheck`); api/main.py (`REDIS_URL` from `os.getenv`); .env.example (`REDIS_PASSWORD`).

## 17. product_class is declared by the submitter, not inferred per image.
**Status:** Implemented
**Context:** `identify_product` (gpt-4o-mini) ran on every job whose session cache was cold, before PatchCore was consulted at all — so a paid external API sat on the hot path of every request, deciding something the submitting station already knows. On a production line the product class is line metadata: a station is pointed at one product for a shift, and the changeover is a scheduled event, not a per-image discovery. CLAUDE.md §1.1 states the VLM is "the cold-start fallback only"; the code contradicted that, and entry 6's own consequences line admitted the cost.
**Decision:** `POST /api/submit` accepts an optional `product_class` form field. When supplied, the API validates it against the checkpoints actually on disk (`config.available_categories`, shared with `PatchCoreRegistry` so the two cannot disagree) and rejects an unknown value with 400. The worker then routes straight to that checkpoint and never calls `identify_product` (`worker.py`, step 1). Omitting the field preserves the old inference path, which is now what it was documented to be: the cold-start case. `GET /api/categories` exposes the accepted set so the UI offers exactly what the backend will take.
**Alternatives rejected:**
- *Keep inferring, and cache harder* — a longer session TTL reduces the call count but not the dependency: the first image of every session still cannot be scored without OpenAI reachable, and entry 7 already records that a wrong cached guess misroutes silently for an hour.
- *Accept an unknown product_class and let the worker fall back to zero-shot* — this is what the code did implicitly. It converts an operator typo into a silently degraded inference that still returns a confident-looking score, which is worse than a 400 at submit time.
- *Configure one product per deployment* — the prior design (entry 6 rejected it). One worker serves all 15 categories; a per-deployment constant cannot express a changeover.
**Consequences:** A declared class is deliberately not written to the session cache: it belongs to its job, and caching it would steer later jobs that declared nothing. `product_confidence` is null for declared classes rather than a fabricated 1.0, so a stated fact stays distinguishable from a confident guess. The API now needs read access to the checkpoint directory, which it previously did not — a containerized API would need it mounted, or `product_class` becomes unusable and returns 503 rather than silently ignoring the field.
**Revisit when:** a deployment has more than one station per worker, at which point the class belongs on a per-station queue rather than a per-request field.
**Evidence:** api/main.py (`submit`, `categories`); worker.py (step 1 branch); config.py (`available_categories`); schemas.py (`InferenceJob.product_class`); shared/schemas/job.json.

## 18. NEEDS_REVIEW is a terminal status, separate from FAILED.
**Status:** Implemented
**Context:** `JobStatus` had one terminal error state, and it absorbed two unrelated events: the pipeline breaking (Redis unreachable, a model failing to load, an unhandled exception) and the pipeline running correctly but declining to reach a verdict (zero-shot on a product with no calibrated detector, a Stage 2 pass that came back `uncertain`). Those need opposite responses. The first is an alarm: something is wrong and someone should be paged. The second is normal operation on a hard image: a human should look at it.
**Decision:** `JobStatus.NEEDS_REVIEW` (`schemas.py`) is a third terminal state. The worker sets it when a zero-shot score sits too close to the decision boundary to call (`_is_abstention`, `abstain_uncertainty`, default 0.5) or when Stage 2 returns `uncertain`. A NEEDS_REVIEW result is added to the labeling pool unconditionally in `_record_for_active_learning` — the status *is* the request for a human, so the uncertainty arithmetic does not get to overrule it. FAILED keeps its meaning and still carries an `InferenceError`. The terminal log line reports the status rather than saying "Job completed" for all three.
**Alternatives rejected:**
- *Keep one terminal error state and distinguish by inspecting `routing_reason`* — the information exists in the payload, so this is technically sufficient. Rejected because every consumer then has to reimplement the distinction correctly: an alerting rule on `status=failed` would page on hard images, and the failure rate would move with image difficulty rather than system health, which makes the number meaningless as a health signal.
- *Report abstention as COMPLETED with low confidence* — hides it entirely. A confident-looking result with a mediocre confidence figure is exactly how an abstention gets acted on as a verdict.
**Consequences:** Abstention volume and failure volume are now separately countable, which is what makes a circuit breaker possible: an open breaker can degrade to NEEDS_REVIEW and route work to humans without inflating the failure rate that would otherwise be its own trigger (entry 19). Zero-shot abstention is derived from distance to the decision boundary rather than a model-reported confidence, because the zero-shot prompt does not ask for one and adding it would change the prompt for something the score already expresses. The frontend renders the three states distinctly.
**Revisit when:** operators disagree with the abstention band often enough to measure — at which point the band should be fitted to their agreement rate rather than asserted (see the calibration gap entries).
**Evidence:** schemas.py (`JobStatus`); worker.py (`_is_abstention`, stage-2 branch, `_record_for_active_learning`); shared/schemas/result.json; frontend/src/lib/api.ts, frontend/src/app/submit/page.tsx.

## 19. The API sheds load above a queue-depth ceiling.
**Status:** Implemented
**Context:** `retina:jobs:queue` accepts submissions from any number of producers while exactly one worker drains it, one image at a time (entry 12). Nothing connected the two: a submitter got a `job_id` at the same speed whether the queue was empty or thousands deep, and the only signal that the system was behind was a result arriving minutes late — or never, once the stream cap (F9) began trimming. An unbounded producer against a single consumer with no backpressure is not a queue, it is a place where the mismatch accumulates out of sight.
**Decision:** `POST /api/submit` reads `XLEN retina:jobs:queue` before accepting work. At or above `RETINA_JOB_QUEUE_MAX_DEPTH` (default 1000) it refuses with 503 and a message naming the depth and the ceiling. Every successful response carries `queue_depth` (including the job just enqueued, so the first submitter of a burst does not read zero) and `queue_ceiling`, and the submit page surfaces both, warning once the backlog passes 80% of the ceiling.
**Alternatives rejected:**
- *Let the queue grow and rely on the MAXLEN cap* — the cap protects Redis memory, not the submitter. Trimming silently discards the oldest jobs, so unbounded acceptance converts a visible refusal into invisible data loss.
- *Block the submitter until the queue drains* — turns a fast failure into a slow one and holds a connection per waiting producer, which the single-process API can least afford (entry 20 is about exactly that).
- *Scale workers instead of shedding* — the right answer eventually, and orthogonal: a ceiling is still needed at whatever throughput the pool reaches, and worker scale-out has its own unresolved problems (consumer identity, GPU contention).
**Consequences:** `XLEN` counts entries still resident in the stream, which includes acknowledged ones until MAXLEN trims them, so the depth is a conservative over-estimate of outstanding work — the ceiling triggers earlier than strictly necessary. A precise figure would need `XPENDING` plus the unread tail, which is two more round trips on the hot path for a number used only as a threshold. The ceiling is a static constant; it does not adapt to how fast the worker is currently draining.
**Revisit when:** the depth signal is used for anything beyond shedding — autoscaling or an operator SLA would need the precise outstanding count, not the over-estimate.
**Evidence:** api/main.py (`JOB_QUEUE_MAX_DEPTH`, `submit`); frontend/src/lib/api.ts (`SubmitResponse`), frontend/src/app/submit/page.tsx.

## 20. OpenAI calls are timed, retried a bounded number of times, and deadlined.
**Status:** Implemented
**Context:** `OpenAI(api_key=key)` was constructed with no timeout and none of the four call sites passed one, so a hung request stalled the worker for as long as the SDK's own default allowed — minutes. The worker processes one job at a time (entry 12), so that is not one slow job, it is the entire pipeline stopped. There was also no retry: a single 429 during a burst failed the job outright, despite CLAUDE.md §4.2 specifying exponential backoff with a retry cap.
**Decision:** The client is built with an explicit per-request timeout (`openai_timeout_s`, default 30s) and `max_retries=0`, and every call goes through `_call_with_retry`. That helper retries only transient failures — 429, timeout, connection error, 5xx — with `2**attempt` backoff up to `gpt4v_max_retries`, and bounds the whole logical call with `openai_total_deadline_s` (default 90s). Exhaustion raises `VLMUnavailableError`, distinct from a parse or schema failure.
**Alternatives rejected:**
- *Use the SDK's built-in `max_retries`* — it retries the same classes, but its budget is invisible to us and composes badly: our three attempts times its two is six calls and six times the backoff. One retry layer, ours, so the numbers in config are the numbers that happen.
- *Per-attempt timeouts only, no total deadline* — three attempts that each just miss a 30s timeout cost the loop 90s plus backoff before failing. The deadline is what actually bounds a single job's claim on the worker.
- *Retry everything* — a 400 or an auth failure fails identically on the next attempt; retrying it only spends the budget that a genuinely transient failure needed.
**Consequences:** A failing OpenAI region now costs a bounded amount of loop time per job instead of an unbounded one, but every job still pays that bound before failing — which is why the breaker in entry 21 is the necessary companion, not an optional extra. `VLMUnavailableError` exists specifically so the breaker can count "the API did not answer" without also counting malformed responses, which are a different problem with a different fix.
**Revisit when:** the deadline is being hit routinely rather than exceptionally, which would mean the timeout is mistuned rather than the API being unhealthy.
**Evidence:** vlm_router.py (`_call_with_retry`, client construction); config.py (`openai_timeout_s`, `openai_total_deadline_s`, `gpt4v_max_retries`); worker/tests/test_vlm_retry.py.
