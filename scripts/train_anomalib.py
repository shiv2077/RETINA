"""Train an anomalib model on an MVTec category and save the checkpoint."""
import argparse
import sys
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore


MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}

# EfficientAd hard-requires train_batch_size=1 and an out-of-distribution
# image dir (e.g. Imagenette) for its penalty loss. Other models use default bs.
EFFICIENTAD_BS = 1
DEFAULT_BS = 32
IMAGENET_DIR = Path("./datasets/imagenette")


def build_model(model_name: str):
    if model_name == "efficientad":
        return EfficientAd(imagenet_dir=str(IMAGENET_DIR))
    return MODEL_REGISTRY[model_name]()


def build_datamodule(category: str, train_bs: int, eval_bs: int) -> MVTecAD:
    return MVTecAD(
        root="./mvtec",
        category=category,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_workers=4,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--model", default="patchcore", choices=list(MODEL_REGISTRY))
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "efficientad" and not IMAGENET_DIR.is_dir():
        print(
            f"ERROR: {IMAGENET_DIR} does not exist. EfficientAd requires an "
            "ImageNet-like directory for its penalty loss.",
            file=sys.stderr,
        )
        sys.exit(1)

    bs = EFFICIENTAD_BS if args.model == "efficientad" else DEFAULT_BS
    model = build_model(args.model)
    datamodule = build_datamodule(args.category, train_bs=bs, eval_bs=bs)
    engine = Engine(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        default_root_dir=str(args.output_dir / "lightning_logs"),
    )
    engine.fit(model=model, datamodule=datamodule)
    ckpt_path = args.output_dir / f"{args.model}_{args.category}.ckpt"
    engine.trainer.save_checkpoint(str(ckpt_path))
    print(f"CHECKPOINT_SAVED={ckpt_path}")

    results = engine.test(model=model, datamodule=datamodule)
    if results:
        r = results[0]
        img = r.get("image_AUROC", r.get("image_auroc", float("nan")))
        pix = r.get("pixel_AUROC", r.get("pixel_auroc", float("nan")))
        print(f"CATEGORY={args.category} IMAGE_AUROC={img:.4f} PIXEL_AUROC={pix:.4f}")


if __name__ == "__main__":
    main()
