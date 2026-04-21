"""GPT-4o orchestration layer for RETINA Stage 1.

Three responsibilities:
  1. identify_product  — classify the image into one of the trained PatchCore
                         categories (or "unknown"). Used to route to the right
                         checkpoint.
  2. describe_defect   — natural-language defect description for the operator
                         UI; called when Stage 1 flags an anomaly.
  3. zero_shot_detect  — anomaly detection for products not in the trained
                         set; gives day-one capability on new product lines.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json

import structlog
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

logger = structlog.get_logger()


KNOWN_PRODUCT_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


class ProductIdentification(BaseModel):
    product_class: str
    confidence: float
    reasoning: str
    is_known_category: bool


class DefectDescription(BaseModel):
    has_defect: bool
    defect_type: str | None
    location: str | None
    severity: str | None
    natural_description: str
    confidence: float


class ZeroShotAnomaly(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    reasoning: str
    suggested_defect_type: str | None


class Stage2Verdict(BaseModel):
    verdict: str  # "confirmed_anomaly" | "rejected_false_positive" | "uncertain"
    defect_class: str | None
    confidence: float
    reasoning: str


class VLMRouter:
    """Orchestrates GPT-4o calls for product routing and defect description."""

    STAGE2_TRIGGER_MIN = 0.5
    STAGE2_TRIGGER_MAX = 0.9

    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise ValueError(
                "VLMRouter requires an OpenAI API key. Set OPENAI_API_KEY in "
                ".env and ensure it's passed through Settings.openai_api_key."
            )
        self.client = OpenAI(api_key=api_key)
        logger.info(
            "vlm_router_initialized",
            identify_model="gpt-4o-mini",
            describe_model="gpt-4o",
            key_prefix=api_key[:10] + "...",
        )
        self._product_cache: dict[str, ProductIdentification] = {}

    @staticmethod
    def _encode_image(image_bytes: bytes, max_side: int = 1024) -> str:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _image_hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def identify_product(self, image_bytes: bytes) -> ProductIdentification:
        img_hash = self._image_hash(image_bytes)
        if img_hash in self._product_cache:
            return self._product_cache[img_hash]

        b64 = self._encode_image(image_bytes)
        known_list = ", ".join(KNOWN_PRODUCT_CATEGORIES)

        system_prompt = (
            "You identify industrial products in images for a manufacturing "
            "quality control system. Respond ONLY with valid JSON matching "
            "the schema. No markdown, no prose."
        )
        user_prompt = f"""Identify the product in this image.

Known trained categories: {known_list}

If the image shows one of these, return that exact category name.
If it shows something else entirely (e.g., fabric, pastry, a window frame,
a random object), return "unknown".

Respond with JSON:
{{
  "product_class": "bottle" | "cable" | ... | "unknown",
  "confidence": 0.0-1.0,
  "reasoning": "one short sentence",
  "is_known_category": true | false
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    }},
                ]},
            ],
            max_tokens=150,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)

        if data["product_class"] not in KNOWN_PRODUCT_CATEGORIES:
            data["product_class"] = "unknown"
            data["is_known_category"] = False

        result = ProductIdentification(**data)
        self._product_cache[img_hash] = result
        logger.info(
            "product_identified",
            product=result.product_class,
            confidence=result.confidence,
        )
        return result

    def describe_defect(
        self,
        image_bytes: bytes,
        product_class: str,
        anomaly_score: float | None = None,
    ) -> DefectDescription:
        """Describe a defect in an image that has ALREADY BEEN FLAGGED BY PATCHCORE.

        Do NOT call this on images with low anomaly scores — GPT-4o will
        hallucinate defects due to priming bias. The caller is responsible
        for only invoking this when anomaly_score > threshold (recommend 0.7).
        """
        if anomaly_score is not None and anomaly_score < 0.3:
            logger.warning(
                "describe_defect called on low-score image — skipping VLM call",
                anomaly_score=anomaly_score,
                product_class=product_class,
            )
            return DefectDescription(
                has_defect=False,
                defect_type=None,
                location=None,
                severity=None,
                natural_description=(
                    f"No defect detected (anomaly score {anomaly_score:.3f} "
                    "below threshold)."
                ),
                confidence=1.0 - anomaly_score,
            )

        b64 = self._encode_image(image_bytes)
        score_note = (
            f" Our detector scored this at {anomaly_score:.3f} anomaly likelihood."
            if anomaly_score is not None
            else ""
        )

        system_prompt = (
            "You are a quality control inspector describing product defects "
            "to factory operators. Be specific, concise, and actionable. "
            "Respond ONLY with valid JSON."
        )
        user_prompt = f"""This is a {product_class} that our anomaly detector flagged.{score_note}

Examine the image and describe any visible defect. If you don't see any
defect and think the detector made a false positive, say so.

Respond with JSON:
{{
  "has_defect": true | false,
  "defect_type": "scratch" | "crack" | "contamination" | "dent" | "discoloration" | "missing_part" | "deformation" | "other" | null,
  "location": "upper-left" | "center" | "rim" | "edge" | etc. | null,
  "severity": "minor" | "moderate" | "severe" | null,
  "natural_description": "One sentence an operator would read (max 20 words).",
  "confidence": 0.0-1.0
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    }},
                ]},
            ],
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return DefectDescription(**data)

    def should_run_stage2(self, anomaly_score: float) -> bool:
        """True when the Stage 1 score falls in the uncertainty zone
        [0.5, 0.9). Above 0.9 PatchCore is confident anomalous; below 0.5
        it was not flagged. Narrows Stage 2 spend to the hard cases."""
        return self.STAGE2_TRIGGER_MIN <= anomaly_score < self.STAGE2_TRIGGER_MAX

    def stage2_refine(
        self,
        image_bytes: bytes,
        product_class: str,
        stage1_score: float,
        labeled_examples: list[dict] | None = None,
    ) -> Stage2Verdict:
        """Supervised Stage 2 refiner: verdict on an ambiguous Stage 1 flag.

        Uses text-only few-shot context from operator labels when available;
        falls back to zero-shot reasoning otherwise. Image bytes for
        examples are not embedded in the prompt (doubles cost per call).
        """
        b64_query = self._encode_image(image_bytes)

        examples_count = len(labeled_examples) if labeled_examples else 0
        examples_context = ""
        if labeled_examples:
            lines = [
                f"\n\nFor reference, here are {examples_count} operator-"
                "labeled examples from this product line:"
            ]
            for i, ex in enumerate(labeled_examples[:5], 1):
                defect = ex.get("defect_class")
                suffix = f" (defect: {defect})" if defect else ""
                lines.append(f"  Example {i}: label={ex['label']}{suffix}")
            examples_context = "\n".join(lines)

        system_prompt = (
            "You are the Stage 2 supervised anomaly refiner in a two-stage "
            "industrial inspection pipeline. Stage 1 (PatchCore) flagged "
            "this image as potentially anomalous but its confidence is "
            "moderate. Your job: using both visual evidence AND the "
            "operator-labeled examples provided (when available), decide "
            "whether this is a genuine defect or a false positive from "
            "Stage 1. Respond ONLY with valid JSON."
        )
        user_prompt = (
            f"This is a {product_class}. Stage 1 anomaly score: "
            f"{stage1_score:.3f} (uncertainty zone 0.5-0.9).{examples_context}\n\n"
            "Assess the image and decide:\n"
            "1. Is this a GENUINE defect? (confirmed_anomaly)\n"
            "2. Is Stage 1 wrong — no real defect visible? (rejected_false_positive)\n"
            "3. Are you unsure? (uncertain)\n\n"
            "If confirmed, name the defect class (e.g., \"scratch\", \"crack\", "
            "\"contamination\", \"dent\", \"discoloration\"). If you cannot tell, "
            "use null.\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "verdict": "confirmed_anomaly" | "rejected_false_positive" | "uncertain",\n'
            '  "defect_class": "scratch" | "crack" | ... | null,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "reasoning": "one short sentence citing visual evidence"\n'
            "}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_query}",
                    }},
                ]},
            ],
            max_tokens=300,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        logger.info(
            "stage2_refined",
            product_class=product_class,
            stage1_score=stage1_score,
            verdict=data["verdict"],
            defect_class=data.get("defect_class"),
            examples_used=examples_count,
        )
        return Stage2Verdict(**data)

    def zero_shot_detect(
        self,
        image_bytes: bytes,
        product_description: str = "industrial product",
    ) -> ZeroShotAnomaly:
        b64 = self._encode_image(image_bytes)

        system_prompt = (
            "You inspect industrial products for defects. You receive products "
            "you have never seen before and must judge whether they appear "
            "defective based on general principles: surface uniformity, "
            "structural integrity, consistent coloring, absence of foreign "
            "matter. Respond ONLY with valid JSON."
        )
        user_prompt = f"""This is a {product_description}. Does it show any defect?

Consider: scratches, cracks, dents, discoloration, contamination, missing
parts, deformation, or anything unusual compared to what a normal unit
should look like.

Respond with JSON:
{{
  "is_anomaly": true | false,
  "anomaly_score": 0.0-1.0,
  "reasoning": "one short sentence explaining the visual evidence",
  "suggested_defect_type": "scratch" | "crack" | ... | null
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    }},
                ]},
            ],
            max_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return ZeroShotAnomaly(**data)
