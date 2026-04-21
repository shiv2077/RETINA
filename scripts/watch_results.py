"""Poll Redis for inference results and pretty-print them."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import redis

RESULT_KEY_FMT = "retina:results:{job_id}"
RESULT_KEY_PATTERN = "retina:results:*"
POLL_INTERVAL_S = 0.5


# ── ANSI helpers ────────────────────────────────────────────────────────
def _use_color() -> bool:
    return sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def red(t: str) -> str: return _paint(t, "31;1")
def green(t: str) -> str: return _paint(t, "32;1")
def yellow(t: str) -> str: return _paint(t, "33;1")
def dim(t: str) -> str: return _paint(t, "2")


def fmt(v: Any, default: str = "—") -> str:
    return default if v is None else str(v)


def pretty_print(r: dict) -> None:
    job_id = r.get("job_id", "?")[:6]
    product_class = r.get("product_class") or "unknown"
    product_conf = r.get("product_confidence")
    routing_reason = r.get("routing_reason") or "—"
    score = r.get("anomaly_score")
    is_anomaly = bool(r.get("is_anomaly"))
    threshold = 0.50
    proc_ms = r.get("processing_time_ms")
    image_id = r.get("image_id", "?")
    vlm_cost = r.get("vlm_api_cost_estimate_usd")
    vlm_model = r.get("vlm_model_used")
    natural = r.get("natural_description")
    defect_type = r.get("defect_type")
    defect_loc = r.get("defect_location")
    defect_sev = r.get("defect_severity")
    s1 = r.get("stage1_output") or {}
    heatmap_available = s1.get("heatmap_available", False)

    bar = "═" * 63
    sep = "─" * 63
    conf_frag = (
        f" (VLM conf {product_conf:.2f})" if isinstance(product_conf, (int, float))
        else ""
    )

    print(bar)

    # ── zero-shot branch (no PatchCore) ────────────────────────────
    if routing_reason == "unknown_product_zero_shot":
        header = f"Job {job_id} — {product_class} {yellow('(zero-shot)')}"
        print(header)
        print(f"Image ref:      {image_id}")
        verdict = (
            f"{red('ANOMALY')} (score {score:.2f})" if is_anomaly
            else f"{green('NORMAL')} (score {score:.2f})"
        )
        print(f"VLM zero-shot verdict: {verdict}")
        if natural:
            print(f"Reasoning: {natural!r}")
        print(sep)
        print(f"Models used:    gpt-4o (no PatchCore checkpoint for this product)")
        print(f"Processing:     {fmt(proc_ms)}ms")
        if vlm_cost is not None:
            print(f"VLM cost:       ${vlm_cost:.5f}")
        print(bar)
        return

    # ── normal (PatchCore, not anomalous) ──────────────────────────
    if not is_anomaly:
        header = f"Job {job_id} — {product_class}{conf_frag}"
        print(header)
        verdict = f"{score:.2f}  →  {green('NORMAL')} (threshold {threshold:.2f})"
        print(f"Anomaly score:  {verdict}")
        print(f"Processing:     {fmt(proc_ms)}ms  {dim('(no VLM describe call)')}")
        if vlm_cost is not None:
            print(f"VLM cost:       ${vlm_cost:.5f}")
        print(bar)
        return

    # ── anomaly branch (PatchCore + VLM description) ───────────────
    header = f"Job {job_id} — {product_class}{conf_frag}"
    print(header)
    print(f"Image ref:      {image_id}")
    print(sep)
    print(f"Routing:        {routing_reason}")
    print(f"Anomaly score:  {score:.4f}  (threshold {threshold:.2f})")
    print(f"Verdict:        {red('ANOMALY DETECTED')}")
    heatmap_str = "available (256×256)" if heatmap_available else "not produced"
    print(f"Heatmap:        {heatmap_str}")
    if natural:
        print("VLM description:")
        for line in str(natural).splitlines() or [str(natural)]:
            print(f"  {line}")
    print(f"Defect type:    {fmt(defect_type)}")
    print(f"Location:       {fmt(defect_loc)}")
    print(f"Severity:       {fmt(defect_sev)}")
    print(sep)
    print(f"Processing:     {fmt(proc_ms)}ms")
    if vlm_cost is not None:
        print(f"VLM cost:       ${vlm_cost:.5f}")
    models_bits = [f"patchcore_{product_class}.ckpt"]
    if vlm_model:
        models_bits.append(f"{vlm_model} (describe)")
    print(f"Models used:    {' + '.join(models_bits)}")
    print(bar)


def decode(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[warn] malformed result JSON: {e}", file=sys.stderr)
        print(raw, file=sys.stderr)
        return None


def watch_one(client: redis.Redis, job_id: str, timeout_s: int) -> int:
    key = RESULT_KEY_FMT.format(job_id=job_id)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = client.hget(key, "result_data")
        result = decode(raw)
        if result is not None:
            pretty_print(result)
            return 0
        time.sleep(POLL_INTERVAL_S)
    print(f"[timeout] no result after {timeout_s}s for {job_id}", file=sys.stderr)
    return 2


def watch_all(client: redis.Redis, follow: bool) -> int:
    seen: set[str] = set()
    # Seed with existing results so we only print new ones.
    for key in client.scan_iter(match=RESULT_KEY_PATTERN):
        seen.add(key)
    print("[watching] retina:results:* — Ctrl+C to stop", file=sys.stderr)
    try:
        while True:
            new_keys = []
            for key in client.scan_iter(match=RESULT_KEY_PATTERN):
                if key not in seen:
                    seen.add(key)
                    new_keys.append(key)
            for key in new_keys:
                result = decode(client.hget(key, "result_data"))
                if result is not None:
                    pretty_print(result)
                    if not follow:
                        return 0
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", help="job_id to watch for")
    p.add_argument("--all", action="store_true", help="watch for any new result")
    p.add_argument("--redis-url", default="redis://localhost:6379")
    p.add_argument("--timeout", type=int, default=60, help="seconds (--job mode)")
    p.add_argument("--follow", action="store_true", help="keep running (--all mode)")
    args = p.parse_args()

    if not args.job and not args.all:
        p.error("either --job <id> or --all is required")

    try:
        client = redis.from_url(args.redis_url, decode_responses=True)
        client.ping()
    except redis.RedisError as e:
        raise SystemExit(
            f"[error] Redis not reachable at {args.redis_url}: {e}"
        ) from None

    rc = watch_all(client, args.follow) if args.all else watch_one(
        client, args.job, args.timeout,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
