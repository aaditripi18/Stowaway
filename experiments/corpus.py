"""The model corpus, and helpers to move between torch state dicts, RawTensor
dicts (for embedding) and safetensors files (for scanning)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from stowaway.bits import FLOAT_LAYOUT, RawTensor
from stowaway.report import scan

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# name -> (torchvision builder attr, weights enum attr)
VISION = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    # Held out: never looked at while designing the checks; E0 validates on these.
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "shufflenet_v2_x0_5": ("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights"),
    "squeezenet1_1": ("squeezenet1_1", "SqueezeNet1_1_Weights"),
    "mnasnet0_5": ("mnasnet0_5", "MNASNet0_5_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "regnet_y_400mf": ("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
}
# Models whose results were looked at while the checks were designed.
DEV = {"resnet18", "mobilenet_v3_small"}


def _torch_home():
    import os
    os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))


def load_vision(name: str):
    """Return (model, weights) with ImageNet weights, from the local cache."""
    import torchvision.models as M
    _torch_home()
    builder, wname = VISION[name]
    weights = getattr(M, wname).IMAGENET1K_V1
    model = getattr(M, builder)(weights=weights)
    model.eval()
    return model, weights


def state_to_raw(state) -> dict[str, RawTensor]:
    """Float tensors of a torch state_dict as RawTensor, keyed by name."""
    out = {}
    for name, t in state.items():
        if not hasattr(t, "numpy"):
            continue
        arr = t.detach().cpu().numpy()
        if arr.dtype == np.float32:
            out[name] = RawTensor.from_float(np.ascontiguousarray(arr))
    return out


def raw_into_state(state, raw: dict[str, RawTensor]):
    """A copy of `state` with the float tensors replaced by `raw`'s bits."""
    import torch
    new = {k: v.clone() if hasattr(v, "clone") else v for k, v in state.items()}
    for name, rt in raw.items():
        new[name] = torch.from_numpy(rt.to_float32().copy())
    return new


def save_safetensors(raw: dict[str, RawTensor], extra: dict, path: Path) -> Path:
    """Write a float RawTensor dict (plus any non-float tensors in `extra`) to a
    safetensors file so the scanner can read it exactly as a real one."""
    from safetensors.numpy import save_file
    arrays = {n: rt.to_float32().astype(np.float32) for n, rt in raw.items()}
    for n, a in extra.items():
        arrays[n] = a
    save_file(arrays, str(path))
    return path


def scan_raw(raw: dict[str, RawTensor], base_raw: dict[str, RawTensor] | None = None) -> dict:
    """Scan an in-memory model by writing it to a temporary safetensors file."""
    with tempfile.TemporaryDirectory() as d:
        model_path = save_safetensors(raw, {}, Path(d) / "model.safetensors")
        base_path = None
        if base_raw is not None:
            base_path = save_safetensors(base_raw, {}, Path(d) / "base.safetensors")
        return scan(model_path, base=base_path)
