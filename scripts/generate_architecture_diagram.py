#!/usr/bin/env python3
"""Emit docs/figures/architecture.excalidraw from the system as it exists.

Generated rather than drawn so it can be regenerated when the pipeline
changes, and so every element is traceable to a line of code. Each node
below carries the file it was verified against; nothing is drawn that the
code does not do.

Run: python scripts/generate_architecture_diagram.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "architecture.excalidraw"

# Palette (matches the app's dark theme; Excalidraw renders on light canvas).
INK = "#1e1e1e"
API = "#1971c2"          # api/main.py
WORKER = "#2f9e44"       # worker
REDIS = "#e8590c"        # redis keys
VLM = "#9c36b5"          # external OpenAI calls
TERMINAL = "#495057"
GAP = "#e03131"          # the missing edge

FILL = {
    API: "#a5d8ff",
    WORKER: "#b2f2bb",
    REDIS: "#ffd8a8",
    VLM: "#eebefa",
    TERMINAL: "#dee2e6",
    GAP: "#ffc9c9",
    INK: "#ffffff",
}

_seed = random.Random(20260910)


def _base(eid: str, x: float, y: float, w: float, h: float, stroke: str) -> dict:
    return {
        "id": eid,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": FILL.get(stroke, "#ffffff"),
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": _seed.randint(1, 2**31),
        "version": 1,
        "versionNonce": _seed.randint(1, 2**31),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1757500000000,
        "link": None,
        "locked": False,
    }


elements: list[dict] = []


def node(eid, x, y, w, h, label, stroke=INK, dashed=False, font=14, shape="rectangle"):
    box = _base(eid, x, y, w, h, stroke)
    box["type"] = shape
    if dashed:
        box["strokeStyle"] = "dashed"
        box["strokeWidth"] = 2
    box["boundElements"] = [{"id": f"{eid}_t", "type": "text"}]
    elements.append(box)

    lines = label.count("\n") + 1
    text = _base(f"{eid}_t", x, y + (h - lines * (font + 4)) / 2, w, lines * (font + 4), stroke)
    text.update({
        "type": "text",
        "text": label,
        "originalText": label,
        "fontSize": font,
        "fontFamily": 2,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": eid,
        "lineHeight": 1.25,
        "backgroundColor": "transparent",
    })
    elements.append(text)


def arrow(eid, x1, y1, x2, y2, stroke=INK, dashed=False, label=None, bend=None):
    pts = [[0, 0]]
    if bend:
        pts.append([bend[0] - x1, bend[1] - y1])
    pts.append([x2 - x1, y2 - y1])
    a = _base(eid, x1, y1, abs(x2 - x1), abs(y2 - y1), stroke)
    a.update({
        "type": "arrow",
        "points": pts,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "startBinding": None,
        "endBinding": None,
        "lastCommittedPoint": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
    })
    if dashed:
        a["strokeStyle"] = "dashed"
        a["strokeWidth"] = 2
    elements.append(a)

    if label:
        mx, my = ((x1 + x2) / 2, (y1 + y2) / 2) if not bend else bend
        t = _base(f"{eid}_l", mx - 70, my - 22, 140, 18, stroke)
        t.update({
            "type": "text",
            "text": label,
            "originalText": label,
            "fontSize": 11,
            "fontFamily": 2,
            "textAlign": "center",
            "verticalAlign": "top",
            "containerId": None,
            "lineHeight": 1.25,
            "backgroundColor": "transparent",
        })
        elements.append(t)


def caption(eid, x, y, w, text, stroke=INK, size=12, align="left"):
    t = _base(eid, x, y, w, size * 2, stroke)
    t.update({
        "type": "text",
        "text": text,
        "originalText": text,
        "fontSize": size,
        "fontFamily": 2,
        "textAlign": align,
        "verticalAlign": "top",
        "containerId": None,
        "lineHeight": 1.25,
        "backgroundColor": "transparent",
    })
    elements.append(t)


# ── Title ────────────────────────────────────────────────────────────────
caption("title", 40, 20, 900,
        "RETINA — inference pipeline as built", INK, 20)
caption("subtitle", 40, 48, 1000,
        "Generated from code by scripts/generate_architecture_diagram.py. "
        "Every edge verified against api/main.py, worker.py, redis_client.py.", INK, 11)

# ── Submit path ──────────────────────────────────────────────────────────
node("submit", 40, 100, 210, 74,
     "POST /api/submit\nfile + product_class?", API)
caption("submit_c", 40, 180, 210,
        "api/main.py — 503 above queue\nceiling, 400 on unknown class", API, 10)

node("store", 40, 230, 210, 58,
     "data/images/<ab>/<sha256>.png", REDIS)
caption("store_c", 40, 292, 210,
        "content-addressed, ADR 29", REDIS, 10)

node("stream", 300, 100, 200, 74,
     "retina:jobs:queue\nXADD MAXLEN ~100k", REDIS)

node("dlq", 300, 230, 200, 58, "retina:jobs:dlq", REDIS)
caption("dlq_c", 300, 292, 200, "unparseable + over\ndelivery cap, ADR 24", REDIS, 10)

node("reclaim", 300, 350, 200, 58, "XAUTOCLAIM sweep", WORKER)
caption("reclaim_c", 300, 412, 200, "stalled entries, each poll", WORKER, 10)

arrow("a_submit_store", 145, 174, 145, 230)
arrow("a_submit_stream", 250, 132, 300, 132, label="XADD")
arrow("a_stream_dlq", 380, 174, 380, 230, GAP, label="parse fail → XACK")
arrow("a_reclaim_stream", 420, 350, 420, 174, WORKER)
arrow("a_reclaim_dlq", 340, 350, 340, 288, GAP, label="past cap")

# ── Worker: routing ──────────────────────────────────────────────────────
node("read", 560, 100, 200, 58, "read_job (XREADGROUP)", WORKER)

node("declared", 560, 195, 200, 74,
     "product_class\ndeclared?", WORKER, shape="diamond")

node("session", 830, 195, 190, 58, "retina:session:*\n1h TTL", REDIS, font=12)

node("identify", 830, 285, 190, 74,
     "identify_product\ngpt-4o-mini", VLM)
node("breaker", 830, 375, 190, 58, "circuit breaker", GAP, dashed=True, font=12)
caption("breaker_c", 830, 437, 200,
        "3 fails → open → NEEDS_REVIEW\nADR 21", GAP, 10)

node("registry", 560, 320, 200, 74,
     "PatchCoreRegistry\n2-model LRU", WORKER)
caption("registry_c", 560, 400, 210, "15 checkpoints, ADR 11/35", WORKER, 10)

node("zeroshot", 300, 480, 200, 74, "zero_shot_detect\ngpt-4o", VLM)
caption("zeroshot_c", 300, 560, 210, "no checkpoint for class", VLM, 10)

node("band", 560, 480, 200, 74,
     "score in\n[thr, 0.9)?", WORKER, shape="diamond")

node("stage2", 830, 480, 190, 74, "stage2_refine\ngpt-4o + few-shot", VLM)

arrow("a_stream_read", 500, 118, 560, 118, label="claim")
arrow("a_read_declared", 660, 158, 660, 195)
arrow("a_dec_reg", 620, 269, 620, 320, WORKER, label="yes — skip VLM")
arrow("a_dec_sess", 760, 224, 830, 224, WORKER, label="no")
arrow("a_sess_id", 925, 253, 925, 285, VLM, label="cold")
arrow("a_id_reg", 830, 320, 760, 350, WORKER)
arrow("a_id_break", 925, 359, 925, 375, GAP)
arrow("a_reg_zero", 560, 375, 500, 500, VLM, label="no checkpoint")
arrow("a_reg_band", 660, 394, 660, 480, WORKER, label="score")
arrow("a_band_s2", 760, 517, 830, 517, VLM, label="yes")

# ── Terminal states ──────────────────────────────────────────────────────
node("completed", 300, 660, 200, 58, "COMPLETED", TERMINAL)
node("needsreview", 560, 660, 200, 58, "NEEDS_REVIEW", TERMINAL)
node("failed", 830, 660, 190, 58, "FAILED", TERMINAL)
caption("term_c", 300, 724, 740,
        "COMPLETED: verdict stands.  NEEDS_REVIEW: ran, declined to decide (normal).  "
        "FAILED: pipeline broke (alarm).  ADR 18", TERMINAL, 10)

arrow("a_band_done", 640, 554, 560, 660, TERMINAL, label="no")
arrow("a_s2_review", 900, 554, 700, 660, TERMINAL, label="uncertain")
arrow("a_s2_done", 860, 554, 480, 660, TERMINAL, label="confirmed/rejected")
arrow("a_zero_review", 430, 554, 600, 660, TERMINAL, label="near boundary")
arrow("a_zero_done", 380, 554, 380, 660, TERMINAL)
arrow("a_break_review", 1010, 404, 760, 675, GAP, label="open → review")

node("results", 300, 790, 460, 58, "retina:results:{job_id}  ·  7d TTL", REDIS)
arrow("a_c_res", 400, 718, 430, 790, REDIS)
arrow("a_n_res", 660, 718, 620, 790, REDIS)
arrow("a_f_res", 900, 718, 720, 790, REDIS)

# ── Active learning ──────────────────────────────────────────────────────
node("pool", 830, 790, 190, 58, "retina:al:pool", REDIS)
arrow("a_review_pool", 760, 689, 830, 800, REDIS, label="always pooled")

node("labelui", 1080, 790, 200, 58, "operator /label UI", API)
arrow("a_pool_ui", 1020, 815, 1080, 815, API, label="GET labels/pool")

node("labels", 1080, 660, 200, 58, "retina:labels:*\n+ labels_index", REDIS, font=12)
arrow("a_ui_labels", 1180, 790, 1180, 718, REDIS, label="POST submit")

arrow("a_labels_s2", 1080, 660, 1000, 545, VLM, label="few-shot examples")

# ── The missing edge ─────────────────────────────────────────────────────
node("retrain", 1080, 480, 200, 74,
     "retraining /\ncheckpoint promotion", GAP, dashed=True)
arrow("a_labels_retrain", 1180, 660, 1180, 554, GAP, dashed=True)
caption("gap_c", 1080, 400, 260,
        "✕ DOES NOT EXIST\nNo retraining trigger, no checkpoint\n"
        "versioning, no validation gate, no\npromotion or rollback. Labels expire\n"
        "after 7 days. ADR 33.", GAP, 11)

caption("legend", 40, 860, 600,
        "Dashed red = does not exist in code.  Purple = external OpenAI call.  "
        "Orange = Redis key.  Green = worker.  Blue = API.", INK, 11)

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "scripts/generate_architecture_diagram.py",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(doc, indent=2) + "\n")
print(f"wrote {OUT} ({len(elements)} elements)")
