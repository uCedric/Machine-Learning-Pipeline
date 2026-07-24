"""Export DINOv2 ViT-S/14 to ONNX for the inference-service clustering stage.

Dev-only utility — run it on a machine with internet access (torch.hub
downloads the model code and weights on first use); it is not part of the
service image. Upload the output to the MinIO models bucket at
``models/dinov2/{version}/`` alongside ``pca.joblib``, ``umap.joblib`` and
``hdbscan.joblib`` (see the README activation runbook).

Prerequisites (torch.onnx.export needs the ``onnx`` package, which the service
image deliberately does not carry):

    pip install torch==2.3.1 torchvision==0.18.1 onnx onnxruntime

    python scripts/export_dinov2_onnx.py --output dinov2_vits14.onnx
"""
from __future__ import annotations

import argparse
import os

# Force the plain-attention path: xformers ops are not ONNX-exportable.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DINOv2 ViT-S/14 to ONNX")
    parser.add_argument("--output", default="dinov2_vits14.onnx")
    # opset 17 is the ceiling of torch 2.3's TorchScript exporter and enough
    # for the hub model (built with interpolate_antialias=False).
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()

    # Fixed spatial size (the positional-encoding interpolation is traced for
    # it — TracerWarnings are expected); only the batch axis stays dynamic.
    # forward() returns the 384-d CLS embedding.
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
    )

    # Numeric parity check: torch vs onnxruntime on the same input.
    import onnxruntime as ort

    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(dummy).numpy()
    actual = session.run(None, {"input": dummy.numpy()})[0]
    assert actual.shape == (1, 384), f"unexpected embedding shape {actual.shape}"
    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-4)
    print(
        f"OK: wrote {args.output} (opset {args.opset}); "
        f"max abs diff vs torch {np.abs(expected - actual).max():.2e}"
    )


if __name__ == "__main__":
    main()
