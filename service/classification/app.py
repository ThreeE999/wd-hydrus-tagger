import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
import onnxruntime as rt
from PIL import Image

MODEL_FILENAME = "model.onnx"
META_FILENAME = "meta.json"
FORCE_LOCAL_FILE = True

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def _get_onnx_provider() -> str:
    available = rt.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def _open_onnx_session(model_path: str) -> rt.InferenceSession:
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider = _get_onnx_provider()
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count() or 1
    logging.info("Model %r loaded with provider %r", model_path, provider)
    return rt.InferenceSession(model_path, options, [provider])


_PRE_DOWNSCALE_LONG_SIDE = 128


def _shrink_long_side(
    image: Image.Image,
    target_long_side: int,
    resample=Image.Resampling.BILINEAR,
) -> Image.Image:
    """按长边等比缩小到 target_long_side；已更小则原样返回。"""
    w, h = image.size
    long_side = max(w, h)
    if long_side <= target_long_side:
        return image
    scale = target_long_side / float(long_side)
    return image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        resample,
    )


def _img_encode(
    image: Image.Image,
    size: Tuple[int, int] = (384, 384),
    normalize: Optional[Tuple[float, float]] = (0.5, 0.5),
    pre_long_side: int = _PRE_DOWNSCALE_LONG_SIDE,
) -> np.ndarray:
    """先缩小再拉伸到模型尺寸，RGB CHW float32，再可选归一化。

    1) 长边缩到约 pre_long_side（默认见 _PRE_DOWNSCALE_LONG_SIDE，保比例）
    2) 再拉伸到 size（由模型 input shape 决定，通常 384x384）
    """
    image = image.convert("RGB")
    if pre_long_side:
        image = _shrink_long_side(image, pre_long_side, Image.Resampling.BILINEAR)
    image = image.resize(size, Image.Resampling.BILINEAR)
    data = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_], dtype=np.float32).reshape((-1, 1, 1))
        std = np.asarray([std_], dtype=np.float32).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


class ImageTypeClassifier:
    """Anime image type classifier (deepghs/anime_classification)."""

    def __init__(self):
        self.model = None
        self.labels = None
        self.last_loaded_key = None
        self.model_input_size = (384, 384)

    def _resolve_paths(self, repo: str, model_name: str) -> Tuple[str, str]:
        if FORCE_LOCAL_FILE:
            model_dir = os.path.join("models", repo, model_name)
            model_path = os.path.join(model_dir, MODEL_FILENAME)
            meta_path = os.path.join(model_dir, META_FILENAME)
            if os.path.isfile(model_path) and os.path.isfile(meta_path):
                return model_path, meta_path

        import huggingface_hub

        model_path = huggingface_hub.hf_hub_download(
            repo,
            f"{model_name}/{MODEL_FILENAME}",
            token=HF_TOKEN or None,
        )
        meta_path = huggingface_hub.hf_hub_download(
            repo,
            f"{model_name}/{META_FILENAME}",
            token=HF_TOKEN or None,
        )
        return model_path, meta_path

    def load_model(self, repo: str, model_name: str) -> None:
        key = (repo, model_name)
        if key == self.last_loaded_key and self.model is not None:
            return

        model_path, meta_path = self._resolve_paths(repo, model_name)
        with open(meta_path, "r", encoding="utf-8") as f:
            labels = json.load(f)["labels"]

        self.model = _open_onnx_session(model_path)
        _, _, height, width = self.model.get_inputs()[0].shape
        if isinstance(height, int) and isinstance(width, int):
            self.model_input_size = (width, height)
        else:
            self.model_input_size = (384, 384)
        self.labels = labels
        self.last_loaded_key = key

    def predict(
        self,
        image,
        repo: str,
        model_name: str,
        pre_long_side: int = _PRE_DOWNSCALE_LONG_SIDE,
    ) -> dict:
        """Return mapping of class label -> score."""
        self.load_model(repo, model_name)

        if not isinstance(image, Image.Image):
            image = Image.open(image)
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        else:
            image = image.convert("RGB")

        input_ = _img_encode(
            image,
            size=self.model_input_size,
            pre_long_side=pre_long_side,
        )[None, ...]
        output, = self.model.run(["output"], {"input": input_})
        return dict(zip(self.labels, map(lambda x: float(x.item()), output[0])))


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="单文件图片类型分类测试")
    parser.add_argument("image", help="图片路径")
    parser.add_argument(
        "--repo",
        default="deepghs/anime_classification",
        help="Hugging Face 仓库名",
    )
    parser.add_argument(
        "--model",
        default="mobilenetv3_v1.4_dist",
        help="模型变体名",
    )
    parser.add_argument(
        "--pre-long-side",
        type=int,
        default=_PRE_DOWNSCALE_LONG_SIDE,
        help="预缩小长边像素（先缩小再拉伸到模型输入）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"文件不存在: {args.image}", file=sys.stderr)
        sys.exit(1)

    clf = ImageTypeClassifier()
    scores = clf.predict(
        image=args.image,
        repo=args.repo,
        model_name=args.model,
        pre_long_side=args.pre_long_side,
    )
    best = max(scores, key=scores.get)
    print(f"image: {args.image}")
    print(f"best:  {best} ({scores[best]:.4f})")
    print("scores:")
    for label, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {label}: {score:.4f}")

