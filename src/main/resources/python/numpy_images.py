"""Read-only NumPy images and the shared XY[CZT]/[TZC]YX convention."""

from pathlib import Path

import numpy as np

DOCUMENTATION = "https://github.com/bioimage-io/JDLL/blob/main/docs/image-formats.md"


def candidates(shape, *, mask=False):
    if not 2 <= len(shape) <= 5 or any(size < 1 for size in shape):
        raise ValueError("Unsupported image type: expected 2 to 5 nonempty dimensions")
    first = ("XY", "XYC", "XYCZ", "XYCZT")[len(shape) - 2]
    last = ("YX", "CYX", "ZCYX", "TZCYX")[len(shape) - 2]
    choices = [first, last]
    if mask and len(shape) == 3:
        choices += ["XYZ", "ZYX"]
    if mask and len(shape) == 4:
        choices += ["XYZT", "TZYX"]
    ranked = []
    for priority, axes in enumerate(choices):
        x, y = shape[axes.index("X")], shape[axes.index("Y")]
        if mask or (min(x, y) > 100 and max(x, y) / min(x, y) < 3):
            ranked.append((x * y, priority, axes))
    if not ranked:
        raise ValueError(f"Cannot infer XY/C/Z/T axes from {shape}")
    return [axes for _, _, axes in sorted(ranked, reverse=True)]


def spatial_shape(shape, axes):
    sizes = dict(zip(axes, shape, strict=True))
    if sizes.get("T", 1) != 1:
        raise ValueError("Training requires individual time points; split the non-singleton T axis")
    return ((sizes["Z"],) if sizes.get("Z", 1) > 1 else ()) + (sizes["Y"], sizes["X"])


def infer_axes(shape, *, mask=False, spatial=None, dimensions=None):
    choices = candidates(shape, mask=mask)
    for axes in choices:
        actual = spatial_shape(shape, axes)
        if spatial is not None and tuple(spatial) != actual:
            continue
        if dimensions is not None and len(actual) != dimensions:
            continue
        return axes
    raise ValueError(f"NumPy shape {shape} does not match the required image/mask geometry")


class NumpyImage:
    """A mapped source; returned views retain ownership until their consumers release them."""

    def __init__(self, path):
        self.path = Path(path)
        self.array = None

    def __enter__(self):
        try:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
            dtype = self.array.dtype
            if dtype.fields is not None or dtype.kind not in "biuf" or (
                dtype.kind == "f" and dtype.itemsize not in (4, 8)
            ):
                raise ValueError(f"Unsupported NumPy dtype {dtype}; use plain real numeric arrays")
            if not 2 <= self.array.ndim <= 5 or any(size < 1 for size in self.array.shape):
                raise ValueError("Unsupported image type: expected 2 to 5 nonempty dimensions")
            return self
        except Exception as exc:
            self.array = None
            raise ValueError(f"{self.path}: {exc}. See {DOCUMENTATION}") from exc

    @property
    def shape(self):
        return self.array.shape

    def axes(self, *, mask=False, spatial=None, dimensions=None):
        try:
            return infer_axes(self.shape, mask=mask, spatial=spatial, dimensions=dimensions)
        except ValueError as exc:
            raise ValueError(f"{self.path}: {exc}. See {DOCUMENTATION}") from exc

    def __exit__(self, *_):
        # Do not close a mapping that a returned array/view still owns.
        self.array = None
