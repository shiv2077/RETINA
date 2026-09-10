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

## 21. A circuit breaker around identify_product, degrading to NEEDS_REVIEW.
**Status:** Implemented
**Context:** `identify_product` runs on every job that did not declare a product class (entry 17), before PatchCore is consulted at all. An OpenAI outage therefore failed 100% of such jobs, and after entry 20 each one first spent its full retry budget re-establishing a fact the previous job had already established. Worse, each was recorded FAILED — an operations alarm for a dependency operations cannot fix — so the outage arrived as a flood of pages rather than one signal.
**Decision:** An in-process `CircuitBreaker` guards that call. Three consecutive `VLMUnavailableError`s open it; while open, calls are refused immediately without an attempt. After `vlm_breaker_cooldown_s` it half-opens and admits exactly one probe — success closes it, failure re-opens it for another cooldown. A refused or failed identification returns a NEEDS_REVIEW result with `routing_reason="vlm_unavailable_needs_review"` and a neutral 0.5 score, so the image goes to an operator instead of being scored by a detector that never ran.
**Alternatives rejected:**
- *Share breaker state in Redis across replicas* — a shared view sounds better and is worse here: it adds a network round trip to the fast path the breaker exists to protect, and couples replicas so one replica's local network problem silences the others. Each worker's own recent experience is the correct input to its own routing.
- *Reorder PatchCore ahead of the identifier so the outage stops mattering* — the real fix, and out of scope for a patch. It changes which model owns the verdict for an unidentified image, needs its own evaluation, and entry 17 has already removed the need for most identify calls by making the class declarable. Recorded as a gap rather than smuggled in here.
- *Count all VLM failures, not just identification* — the other three calls run after Stage 1 has produced a score, so their failure degrades an explanation rather than blocking a verdict. Tripping the same breaker on them would open it for outages that were not on the critical path.
**Consequences:** During an outage the first three jobs pay the full retry budget and the rest fail fast, all landing in the labeling pool as NEEDS_REVIEW. That is a deliberate trade: the operator queue absorbs the outage, which is visible and actionable, rather than the failure rate absorbing it. The breaker is per-process, so with N replicas up to N×3 jobs pay full cost before all of them have opened. It only protects the identify path; a Stage 2 or describe outage still fails per call, bounded by entry 20.
**Revisit when:** PatchCore runs first for undeclared products, at which point an identifier outage stops being on the critical path and the breaker's job shrinks to suppressing retry cost.
**Evidence:** circuit_breaker.py; worker.py (identify branch); config.py (`vlm_breaker_*`); worker/tests/test_circuit_breaker.py.

## 22. The code contradicted CLAUDE.md §1.1, and D1 made the router optional rather than removing it.
**Status:** Recorded, partially addressed
**Context:** CLAUDE.md §1.1 states as an invariant that the VLM "is NOT a co-equal Stage 1 model — it is the cold-start fallback only. Once PatchCore has a memory bank, PatchCore runs first." The code did the opposite: `identify_product` ran first on every job with a cold session cache, and PatchCore was gated behind its result. A reader of the architecture document would have drawn the dependency graph backwards.
**Context — how it happened:** the router was added (entry 6) to solve a real problem the original design did not have. The prior architecture assumed one configured product type per deployment; serving all 15 categories from one worker requires knowing which checkpoint applies, and there was no other source for that. Rather than adding an input, identification was inferred from the image, which put a paid external API in front of every inference. Each step was locally reasonable; the stated invariant was never revisited against the result.
**Decision:** Entry 17 makes `product_class` a submitted field, so the router is skipped entirely when the caller knows the answer — which on a production line is always. That restores the documented relationship for the normal path without reordering the models. The router was **not** removed: it remains the only mechanism for a genuinely unknown product, which is what the invariant calls a cold start.
**Not done:** PatchCore still does not run first for an *undeclared* product. Doing so means scoring an image against a checkpoint chosen by some other means — a default category, a per-station config, or every checkpoint in turn — and each of those is a different system with different failure modes and its own evaluation. That is an architectural change, not a patch, and it needs its own ADR and measurement rather than being folded into a fix for the symptom.
**Consequences:** The invariant is now true for declared submissions and still false for undeclared ones. Anyone reading §1.1 should read this entry alongside it. Entry 21's breaker exists precisely because the undeclared path still has the dependency.
**Revisit when:** a deployment sends undeclared images at volume, which would make the remaining gap operational rather than theoretical.
**Evidence:** CLAUDE.md §1.1; worker.py (step 1); DECISIONS.md 6, 17, 21.

## 23. Consumer name is derived from the hostname.
**Status:** Implemented
**Context:** `consumer_name` defaulted to the literal string `"worker-1"` with no environment override, while docker-compose.yml documented `docker compose up --scale worker=4` as the way to add throughput. Following that instruction launched four replicas that all joined the `workers` group under one identity. Redis Streams still deliver each entry to one caller, so nothing was duplicated and nothing failed — which is the problem: the misconfiguration was invisible.
**Decision:** `consumer_name` defaults to `socket.gethostname()`, which Docker sets per container, with a random suffix only for the degenerate case where the hostname is empty or the call raises. `WORKER_CONSUMER_NAME` overrides it.
**Alternatives rejected:**
- *A random uuid per process* — unique, but a restarted worker gets a new identity and orphans its predecessor's pending entries under a name nothing will ever reclaim. The hostname is stable across a container restart.
- *Require the operator to set it per replica* — correct and unused: the failure mode is someone running the documented scale command, and a default that only works when overridden is not a default.
**Consequences:** Pending entries are now attributable to a specific replica, which is what makes the reclaim sweep in entry 24 meaningful. Two workers on the same host outside Docker would still collide; the override exists for that.
**Evidence:** config.py (`_default_consumer_name`); docker-compose.yml (scale comment); worker/tests/test_redis_client.py.

## 24. Unparseable jobs are dead-lettered; stalled ones are reclaimed.
**Status:** Implemented
**Context:** `read_job` decoded each entry inside a `try` that caught only `redis.RedisError`, so a `JSONDecodeError` or a Pydantic `ValidationError` propagated out uncaught — and the entry, already claimed by `XREADGROUP`, stayed in the pending entries list. The empty-`job_data` branch returned `None` without acknowledging, with the same result. Nothing in the codebase called `XCLAIM`, `XPENDING` or `XAUTOCLAIM`, so a claimed entry was never redelivered to anyone. CLAUDE.md §8 warns that an unacknowledged message is "redelivered forever"; the truth was the opposite and worse — it was delivered exactly once, to a worker that could not process it, and then lost.
**Decision:** `_parse_entry` catches decode and validation failures separately from transport failures. A payload that cannot be parsed is acknowledged and copied to `retina:jobs:dlq` with the reason and a timestamp — it will never become parseable, so holding the slot open serves nothing. Separately, each poll iteration runs an `XAUTOCLAIM` sweep for entries idle longer than `job_reclaim_idle_ms`, and an entry past `job_max_deliveries` is dead-lettered rather than reclaimed again.
**Alternatives rejected:**
- *Leave the entry pending and alert on PEL depth* — preserves the payload for inspection, which is the DLQ's job anyway, and meanwhile the entry counts against the consumer forever with no path back.
- *Retry indefinitely* — a malformed payload is poison, not bad luck. The delivery cap is what separates the two.
**Consequences:** Two failure modes that were indistinguishable are now separate: poison payloads land in the DLQ immediately, crashed-worker jobs are retried up to a cap and then land there too. The DLQ has no consumer and no retention policy; it is an inspection surface, and it will grow.
**Evidence:** redis_client.py (`_parse_entry`, `reclaim_stale_jobs`, `JOB_DLQ_STREAM`); worker.py (`_process_next_job`); §1.3.

## 25. The completed result is written before any bookkeeping.
**Status:** Implemented
**Context:** `store_result` for a COMPLETED result sat in the same `try` as the labeling-pool insert and the stats counter. A transient Redis error in either — a connection reset during `zadd`, say — fell through to `except Exception`, which constructed a fresh `InferenceResult` with `status=FAILED` and stored *that*, overwriting a correct, already-durable result. The inference had succeeded and been paid for; a bookkeeping hiccup erased it and replaced it with an alarm.
**Decision:** The COMPLETED write happens first and outside the bookkeeping block. The pool insert and the counter run in their own `try`, which logs `post_processing_failed` and continues. A verdict, once durable, is never downgraded by something that happened after it.
**Alternatives rejected:**
- *Make the whole thing atomic with MULTI/EXEC* — the result write and the pool insert are not one transaction in any meaningful sense: the result is the output, the pool entry is an optimisation. Coupling them makes the important write fail whenever the unimportant one would.
- *Retry the bookkeeping inline* — extends the time the job holds the loop for work that can be recomputed from the result later.
**Consequences:** A failed pool insert now means a sample quietly missing from the labeling queue, logged at warning. That is the right trade against destroying a result, but it is silent shrinkage of the active-learning input and nothing reconciles it.
**Evidence:** worker.py (`_handle_job`, `_record_for_active_learning`); worker/tests/test_worker_loop.py.

## 26. The API uses the async Redis client.
**Status:** Implemented
**Context:** Every route handler was `async def` while every Redis call inside them was synchronous, on a process launched as a single uvicorn worker. Each call blocked the event loop for its whole round trip, so requests served strictly one at a time regardless of being written as coroutines. `GET /api/result` made it visible: its `time.sleep(0.5)` polling loop held the loop for up to 60 seconds, during which nothing else — including `/health` — could be served.
**Decision:** `redis.asyncio` throughout, every call awaited, and `asyncio.sleep` in the polling loop.
**Alternatives rejected:**
- *Run uvicorn with `--workers N`* — the obvious fix and an incomplete one. It adds parallelism across processes but changes nothing within one: a single client polling `/api/result` with a long `wait` still occupies its worker entirely, and N clients doing so still exhaust all N. It treats a concurrency bug as a capacity problem.
- *Move the polling to a background task and return immediately* — a larger API change that pushes the wait to the client, which is already what `wait=0` allows callers to choose.
**Consequences:** Measured before and after: two concurrent 3-second polls took 6.01s before and 3.01s after, and `/health` issued mid-poll went from 5011ms to 4ms. The API is still one process, so it remains a single point of failure and CPU-bound work would still block it — there is none today.
**Evidence:** api/main.py; verified against a live instance.

## 27. Checkpoint and image paths resolve from config, never from the working directory.
**Status:** Implemented
**Context:** `PatchCoreRegistry` hardcoded `Path("./checkpoints")`. Launched from the repo root it found 15 categories; launched from anywhere else it found zero and reported no trained categories — not an error, just a silently empty registry. Meanwhile `PATCHCORE_CHECKPOINT_PATH` was declared in config and read by nothing, and `.env.example` and docker-compose.yml disagreed about its value, both wrong. The same class of bug had already been documented for image paths in CLAUDE.md §8.
**Decision:** `Settings.resolved_checkpoint_dir()` and `Settings.resolved_image_root()` own resolution, in one order: explicit argument, then environment variable, then a default derived from the module's own location. Never the process working directory.
**Alternatives rejected:**
- *Resolve relative paths against the repo root instead of the CWD* — still a relative path in the payload, and still wrong the moment the two processes disagree about where the root is, which is exactly what a container boundary guarantees.
- *Require the environment variable, with no default* — every developer then configures it identically by hand, and forgetting it produces a startup failure rather than a working default.
**Consequences:** Both processes resolve the same logical location independently, which is what makes the native-API/containerized-worker split work at all. The anchor is derived by path arithmetic on `__file__`, which is layout-dependent — see entry 28.
**Evidence:** config.py; patchcore_registry.py; worker/tests/test_checkpoint_path.py.

## 28. Path arithmetic on __file__ must survive a flatter layout.
**Status:** Implemented
**Context:** Entry 27's default anchor was `Path(__file__).resolve().parents[3]`, which is the repo root for the native layout `worker/src/retina_worker/config.py`. The container installs the package at `/app/retina_worker/`, which has fewer parents than that, so the index raised `IndexError` at import and the worker container died on startup before logging anything of its own.
**Decision:** The index is guarded — `parents[3] if len(parents) > 3 else parent` — and a regression test asserts the module imports under a container-shaped path. The defaults are unused in the container anyway, because compose sets both paths explicitly, but the module still has to import.
**Alternatives rejected:**
- *Use `importlib.resources` or a packaging anchor* — correct for locating package data, but these are locations *outside* the package (a checkpoints directory beside the repo), which package anchors deliberately do not address.
- *Require the env var in the container and skip the default* — that is in fact the deployed configuration; it does not help, because the failure was at import time, before any value was read.
**Consequences worth recording:** this bug was introduced by the fix for entry 27 — a fix for native/container divergence that was itself verified only natively, reproducing the same class of bug inside itself. The lesson is not about `parents[]`: it is that a fix for an environment-specific failure has to be exercised in both environments, or it inherits the assumption it was written to remove.
**Evidence:** config.py; worker/tests/test_checkpoint_path.py (`TestShallowLayoutImport`).

## 29. Images are content-addressed; the payload carries no filesystem path.
**Status:** Implemented
**Context:** `api/main.py` wrote a host-absolute path into the job payload and the worker opened it verbatim. That worked only while both processes shared a filesystem. With the worker containerized the path did not exist, and the failure was silent: a missing image simply produced `None`. CLAUDE.md §8 names this exact anti-pattern, written for the system this one replaced.
**Decision:** `image_id` is the sha256 of the image bytes. Each process joins `image_relpath(image_id)` onto its own root (entry 27), so the relative layout below the root is identical on both sides and nothing host-specific crosses the boundary. The `image_path` field is deleted from `InferenceJob` outright, making the regression structurally impossible rather than merely avoided. The compose mount is a bind mount of `./data/images`, not a named volume, because the API runs natively and a named volume would be invisible to it.
**Alternatives rejected:**
- *Send the image bytes through Redis with the job* — no shared filesystem needed, and it turns the queue into a blob store: a 5MB image per entry against a stream nothing trims aggressively.
- *Keep the path but make it relative* — still requires both sides to agree on a root, which is the same problem with an extra assumption.
- *Object storage (S3/MinIO)* — the right answer for a real deployment and disproportionate for a demo that runs on one host; content addressing is a step toward it, since the id is already the key.
**Consequences:** Identical bytes now deduplicate to one file, so a resubmission overwrites itself. Ids are sharded two characters deep because a flat directory accumulates every image ever submitted. `get_image` collapsed from five lookup paths to one derivation — those fallbacks existed only because the id did not tell you where the bytes were.
**Evidence:** config.py (`image_relpath`); api/main.py; worker.py (`_load_image_bytes`); docker-compose.yml; worker/tests/test_image_addressing.py.

## 30. The worker image had never built, and its ML stack was a major version adrift.
**Status:** Implemented
**Context:** `docker compose build worker` failed at metadata generation: `pyproject.toml` declared `readme = "README.md"`, `worker/README.md` did not exist, and the Dockerfile never copied it. Because the frontend builds first and failed too (a `COPY` of a `public/` directory that has never existed), a plain `docker compose up` never reached the worker at all. Once building, the image installed **anomalib 1.2.0** — the constraint was `>=1.0.0,<2.0.0` — against **2.3.3** natively, which is the version that produced every checkpoint in `checkpoints/` and every AUROC figure in README.md. It also lacked `lightning` entirely, because anomalib 2.x ships a minimal base install and nothing declared it; `torch` was present only because this `pyproject.toml` names it directly. Then `cv2`, imported transitively by anomalib, failed on missing X11 libraries in the slim base image.
**Decision:** Add the README and copy it in the dependency layer; pin `anomalib==2.3.3` and declare `lightning`; install the OpenCV runtime libraries; drop the frontend `public/` COPY and probe `127.0.0.1` rather than `localhost` in its healthcheck.
**The finding, not the fix:** the containerized path had never run end to end. Every prior statement about the system's behaviour was a statement about the native environment, and the container differed in its ML stack's major version, its Python package set, its system libraries and its filesystem layout. Two of the bugs fixed in this pass (entries 28 and 29) exist only across that boundary, and neither was findable without crossing it.
**Alternatives rejected:**
- *Create an empty `frontend/public/`* — invents a directory to satisfy a COPY for assets that do not exist; nothing in `src/` references a public-path asset.
- *Pin anomalib to the 1.x the container happened to resolve* — the checkpoints and the published metrics come from 2.3.3. Pinning to what was accidentally installed would make the container's results incomparable to every number in the README.
**Consequences:** `docker compose up` now brings all three services to healthy. The ML dependencies are pinned exactly, because checkpoint loading is version-sensitive; the cost is that upgrades are now deliberate.
**Evidence:** worker/Dockerfile, worker/README.md, worker/pyproject.toml, frontend/Dockerfile; verified by a full `docker compose up`.

## 31. One anomaly threshold spans fifteen categories with different score distributions.
**Status:** Known gap, not addressed
**Current design:** `anomaly_threshold` is a single float (default 0.5) applied to every PatchCore score regardless of which category produced it (`config.py`, `worker.py`). The Stage 2 band is anchored to the same constant at its lower edge.
**The constraint that produced it:** the threshold predates the per-category registry. When one model served one deployment, one threshold was the only thing there was to configure, and the registry was added underneath without revisiting it.
**The flaw:** PatchCore scores are normalised distances to a per-category coreset, and nothing makes them comparable across categories. The memory bank for `screw` and the one for `carpet` were built from different image statistics; a score of 0.6 does not mean the same thing to each. `results/mvtec_stage1_results.csv` shows per-category AUROC from 0.919 to 1.000, which is direct evidence that the score distributions differ — a threshold tuned to the middle of that range is simultaneously too strict for some categories and too permissive for others, and the resulting false-positive rate varies per product with no way to see it in aggregate.
**The fix:** fit a threshold per category on a per-category validation split held out from training, choosing the operating point from the precision/recall tradeoff the line actually wants, and store it in the checkpoint's metadata so it travels with the model it was fitted to. `Settings.anomaly_threshold` becomes the fallback for a category with no fitted value.
**Why not now:** it requires a validation split that does not currently exist — the training script uses MVTec's train/test split with no third partition — and a stated operating-point preference from the customer. Guessing either would produce fifteen numbers with the same evidential standing as the one they replaced, at fifteen times the cost of maintaining them.
**Evidence:** config.py (`anomaly_threshold`); results/mvtec_stage1_results.csv; scripts/train_anomalib.py.

## 32. The Stage 2 band is asserted, and Stage 2's accuracy inside it is unmeasured.
**Status:** Known gap, not addressed
**Current design:** scores in `[anomaly_threshold, stage2_trigger_max)` — by default `[0.5, 0.9)` — are sent to `stage2_refine`, whose verdict can override Stage 1 (entries 4, 17).
**The constraint that produced it:** the band was chosen to express a plausible intuition — below the flag threshold nothing was flagged, above 0.9 PatchCore is confident — and both edges are now configurable. Configurable is not the same as derived.
**The flaw:** two separate claims are unevidenced. First, that `[0.5, 0.9)` is where Stage 1 is actually unreliable: that is an empirical property of the per-bin error rate, and nobody has computed it. Second, and more important, that Stage 2 is *better* than Stage 1 inside that band. The cascade only pays for itself if the refiner's accuracy on those images exceeds the detector's, and no measurement of that exists at all. A Stage 2 that agrees with Stage 1 adds cost and nothing else; one that disagrees at random makes the system worse while looking sophisticated, because the override direction is asymmetric — a `rejected_false_positive` at confidence ≥0.7 silently flips a flag off.
**The fix:** bin a validation split by Stage 1 score, compute the error rate per bin, and set the band to the region where it exceeds the accepted rate. Then run Stage 2 over held-out images from that band and compare its verdicts against operator labels, per class. Only if it wins is the cascade justified; if it wins only on some defect types, the band should be conditioned on the predicted type.
**Why not now:** the comparison needs operator labels on band images, and the labels the system has are unversioned, TTL-bounded, and collected without a sampling protocol (entries 33, 34). Measuring against them would produce a number that cannot be reproduced.
**Evidence:** vlm_router.py (`should_run_stage2`); worker.py (stage 2 branch); config.py (`stage2_trigger_max`).

## 33. The active-learning pool has no consumer.
**Status:** Known gap, not addressed
**Current design:** uncertain and NEEDS_REVIEW samples are added to `retina:al:pool`, operators label them through `/label`, and labels are written to `retina:labels:*` and indexed in `retina:labels_index`. Stage 2 reads the most recent few as few-shot context (entry 3). That is the entire loop.
**The flaw:** nothing retrains anything. The pool is a queue whose consumer was never built, so the "active learning" in the architecture is sample *selection* with no learning attached. Labels accumulate, feed at most five in-context examples per Stage 2 call, and expire after seven days. The uncertainty sampling, the pool trimming, the labeling UI and the index are all machinery for a loop that has no second half.
**The loop that would close it:** (1) a retraining trigger — N new labels for a category, or a measured drift in its score distribution; (2) a training run that produces a *candidate* checkpoint, versioned and tagged with the label set and code revision that produced it, never overwriting the incumbent; (3) a validation gate comparing candidate against incumbent on a frozen held-out split for that category, with a required margin, since a candidate trained on operator-selected hard cases will look worse on a random split and better on a hard one; (4) promotion by atomic registry pointer swap, with the incumbent retained; (5) rollback on the same pointer, triggered by the same metric that gated promotion; (6) an audit record tying each production inference to the checkpoint version that produced it.
**Why not now:** every step depends on durable labels and a frozen validation split, neither of which exists (entries 31, 34). Building the retraining half against seven-day labels would produce checkpoints that cannot be reproduced or explained after the labels expire.
**Evidence:** redis_client.py (`add_to_labeling_pool`, `recent_labels`); worker.py (`_record_for_active_learning`); frontend/src/app/label/page.tsx.

## 34. Operator labels have the durability of a cache entry.
**Status:** Known gap, not addressed
**Current design:** labels live in `retina:labels:{image_id}` hashes with a seven-day TTL, in the same Redis instance as the job queue, the result cache and the session cache (entry 10).
**The constraint that produced it:** entry 10 chose Redis-only deliberately, for a demo where every label round-trips through Redis in the request path anyway, and it recorded this as the largest gap between the build and a deployable system. This entry specifies the split it deferred.
**The flaw:** labels are the most expensive artifact the system produces — an expert's judgment on a hard image, unreproducible once the operator moves on — and they are stored with the same guarantees as a cached score that can be recomputed by rerunning a model. A `FLUSHALL`, an eviction under `maxmemory-policy allkeys-lru` (which is configured), or simply seven days of elapsed time destroys them. The policy is the sharper problem than the TTL: under memory pressure Redis will evict labels to make room for results, which is exactly backwards.
**The fix:** split by durability requirement rather than by access pattern. Redis keeps the queue, the result cache, the session cache and the AL pool — all reconstructible. A durable store (Postgres, the one entry 10 removed) holds labels with their operator, timestamp, image content hash and the model version that produced the prediction being corrected; the fitted thresholds from entry 31; and checkpoint metadata and promotion history from entry 33. Those three are the things that cannot be regenerated by rerunning the pipeline.
**Why not now:** it reintroduces the service entry 10 removed, and doing it properly means a schema, migrations and a backup story. It is the first thing to build before any retraining work, and nothing in the retraining loop is safe to build before it.
**Evidence:** api/main.py (`labels_submit`, 7-day expire); docker-compose.yml (`maxmemory-policy allkeys-lru`); DECISIONS.md 10.

## 35. The two-model LRU treats a scheduling problem as a caching problem.
**Status:** Known gap, not addressed
**Current design:** `PatchCoreRegistry` keeps at most two checkpoints resident and evicts least-recently-used (entry 11), sized for a 6GB GPU at roughly 1.5GB resident per model.
**The flaw:** an LRU cache is the right structure when the access sequence is given and unknowable. Here it is neither: jobs sit in a queue the system controls, and their categories are visible before any of them runs — especially now that `product_class` is declared at submit time (entry 17). Round-robin traffic across three categories evicts on every job and pays a multi-second checkpoint load each time, so throughput collapses to the load time while the GPU idles. The cache is absorbing a cost that scheduling could remove entirely.
**The fix:** partition the queue by category and drain it in bursts — hold a checkpoint resident and run every queued job for that category, then swap. With `product_class` on the job this needs no inference to route. Fairness requires a bound: a maximum burst length or a maximum age before a starved category is serviced regardless of queue depth, or a steady stream of one category starves the others indefinitely.
**The tradeoff, stated plainly:** this trades latency for throughput. A job whose category is not currently resident waits for the burst to end rather than triggering an immediate swap, so worst-case per-image latency rises to roughly the burst bound while checkpoint loads per hour fall toward the number of distinct categories. That is the right trade for batch inspection and the wrong one for a line that must reject a part within a fixed distance of the camera. Which regime applies is a customer fact nobody has stated, which is why this is a gap and not a decision.
**Evidence:** patchcore_registry.py; DECISIONS.md 11, 12.

## 36. The system models each image as independent when the real input is a stream.
**Status:** Known gap, not addressed
**Current design:** every image is a self-contained job carrying its own id and optional product class, scored without reference to any other image.
**The flaw:** real inference is a fixed camera at a fixed station photographing a sequence of nominally identical parts under lighting that drifts slowly. Almost every difficulty in this document follows from discarding that structure. Per-image product identification (entry 17) exists only because the job does not know it belongs to a run of one product. A global threshold (entry 31) is unavoidable when scores cannot be normalised against the recent local distribution. Checkpoint thrashing (entry 35) is a scheduling problem created by treating a category-ordered stream as an arbitrary sequence. Drift detection has nowhere to live, because there is no series to detect drift in.
**What the session cache actually is:** entry 7's `retina:session:product_class` with a one-hour TTL is an incomplete gesture at exactly this. It admits that consecutive images share a product, but expresses it as a global mutable string with a timeout rather than as a session with an identity, a start, an end and members. Hence its known failure mode — a changeover mid-hour silently misroutes every subsequent image — which is not a bug in the cache so much as the absence of the concept it is approximating.
**The fix:** make the run a first-class object. A job belongs to a session identified by station and product; the session carries the declared class, the fitted threshold, a rolling score distribution for drift detection, and its own labeled examples. Scores normalise against the session's recent distribution rather than a global constant. Checkpoint residency follows the session, so entry 35's scheduling problem disappears rather than being managed.
**Why not now:** it is a change to the central data model, touching the job schema, the queue, the worker's routing and the frontend. Every entry above would need revisiting against it. It is the right shape and it is not a patch.
**Evidence:** worker.py (session cache); schemas.py (`InferenceJob`); DECISIONS.md 7, 31, 35.

## 37. Per-inference cost is estimated in constants, never accounted.
**Status:** Known gap, not addressed
**Current design:** `worker.py` carries four hardcoded constants (`COST_IDENTIFY`, `COST_DESCRIBE`, `COST_ZERO_SHOT`, `COST_STAGE2_REFINE`) and sums them into `vlm_api_cost_estimate_usd` on each result.
**The flaw:** the central design choice of this system is substituting an API model for a trained detector — entry 3 replaced supervised Stage 2 with in-context GPT-4o, entry 8 made zero-shot the fallback for untrained products. The case for that rests on cost per inference against the cost of training and maintaining a model, and the system cannot report the number the argument depends on. The constants are per-call guesses, not measurements: they ignore actual token counts, which vary with image size, prompt length and the number of few-shot examples injected; nothing reads back `usage` from the API response, which returns exact token counts on every call. The figures are not aggregated anywhere either, so there is no cost per image, per category, per day, or per operator-hour saved.
**The fix:** record `response.usage.prompt_tokens` and `completion_tokens` per call, price them from a table keyed by model, and store the real figure on the result. Aggregate per category and per day. The interesting derived number is cost per *avoided* operator review, which the labeling pool makes computable once entry 33's loop exists.
**Why not now:** small and genuinely deferrable — but it is the measurement that would justify or refute entries 3 and 8, so it should not be deferred long.
**Evidence:** worker.py (`COST_*` constants); vlm_router.py (`usage` unread on every response).

## 38. mypy is a baseline, not a gate, and the baseline moved.
**Status:** Known gap, not addressed
**Current design:** `pyproject.toml` sets `strict = true`, and `worker/TYPING_BASELINE.txt` records a count. Nothing runs mypy in CI, and CLAUDE.md §2.2 lists it as a required check.
**The flaw:** a recorded count is not a gate. During the correctness pass the count went from 36 to 40 — and the delta is not what it looks like: seven new errors appeared and three disappeared, because the code carrying them moved from `worker.py` into `redis_client.py`. Nothing rejected either change. The declared `strict = true` describes an aspiration the codebase has never met, which is worse than an honest lower setting, because a reader takes it as an invariant.
**Also worth stating:** all seven new errors are the same third-party artifact — redis-py types its sync client's returns as `Awaitable[T] | T`, since one class body serves both clients — so none is a real defect. That does not make the situation fine: a baseline that mixes real findings with unavoidable stub noise is one nobody reads, which is how a genuine `call-arg` error (a FAILED-path constructor missing five required arguments) sat in it unnoticed until it was read by hand.
**The fix:** one narrowing seam at the client boundary — a `cast` on `from_url`'s result or a thin typed wrapper — removes the stub noise. Then ratchet: fail CI if the count exceeds the committed baseline, and lower the baseline as errors are fixed. Never raise it silently.
**Why not now:** the cast is small, but making mypy a gate means the remaining ~33 errors must be triaged or explicitly baselined first, and a gate that is immediately overridden teaches people to override gates.
**Evidence:** worker/pyproject.toml (`strict = true`); worker/TYPING_BASELINE.txt; CLAUDE.md §2.2.
