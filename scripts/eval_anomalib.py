"""Evaluate an existing anomalib checkpoint on an MVTec category."""
import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore


MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", default="patchcore", choices=list(MODEL_REGISTRY))
    args = p.parse_args()

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls.load_from_checkpoint(str(args.checkpoint))
    datamodule = MVTecAD(
        root="./mvtec",
        category=args.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )
    engine = Engine(accelerator="gpu", devices=1, precision="32-true")
    results = engine.test(model=model, datamodule=datamodule)
    r = results[0] if results else {}
    img = r.get("image_AUROC", r.get("image_auroc", float("nan")))
    pix = r.get("pixel_AUROC", r.get("pixel_auroc", float("nan")))
    print(f"CATEGORY={args.category} IMAGE_AUROC={img:.4f} PIXEL_AUROC={pix:.4f}")


if __name__ == "__main__":
    main()
