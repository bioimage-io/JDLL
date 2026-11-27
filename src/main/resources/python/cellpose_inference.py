"""Diameter resolution for the Cellpose 3 inference bridge."""

def resolve_cellpose_diameter(model, images, channels, requested, weights, task):
    from pathlib import Path
    import numpy as np
    source = "manual"
    selected = None
    warning = None
    diameter = requested
    if requested is None:
        path = Path(weights)
        official = {
            "cyto3": "size_cyto3.npy",
            "cyto2torch_0": "size_cyto2torch_0.npy",
            "cytotorch_0": "size_cytotorch_0.npy",
            "nucleitorch_0": "size_nucleitorch_0.npy",
        }
        expected = path.parent / official.get(path.name, path.name + "_size.npy")
        if expected.is_file():
            selected = expected
        if selected is not None:
            try:
                from cellpose.models import SizeModel

                cache = getattr(model, "_jdll_size_models", None)
                if cache is None:
                    cache = model._jdll_size_models = {}
                key = str(selected.resolve())
                if key not in cache:
                    cache[key] = SizeModel(pretrained_size=str(selected), cp_model=model)
                styles = model.eval(images, channels=channels, channel_axis=-1, diameter=None,
                                    rescale=1.0, resample=False, compute_masks=False)[2]
                estimate = np.asarray(cache[key]._size_estimation(styles[0])).reshape(-1)
                if estimate.size != 1 or not np.isfinite(estimate[0]) or estimate[0] <= 0:
                    raise ValueError("size estimator returned an invalid diameter")
                diameter = float(estimate[0])
                source = "estimated"
            except MemoryError:
                raise
            except Exception as error:
                warning = "Diameter estimation failed: " + str(error) + "."
        elif path.name in official:
            warning = "No matching size model found."
        if diameter is None:
            diameter = float(getattr(model, "diam_labels", model.diam_mean))
            source = "model_default"
            if warning:
                warning += " Using the segmentation model's default diameter."
    diameter = float(diameter)
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Cellpose diameter must be a finite, positive number of pixels")
    description = {"manual": "reference", "estimated": "automatically estimated",
                   "model_default": "model default"}[source]
    message = "Diameter: {:.3f} px ({})".format(diameter, description)
    if selected is not None:
        message += ", size model: " + selected.name
    if warning:
        message = "Warning: " + warning + " " + message
    task.update(message=message, info={"type": "cellpose_diameter", "diameter": diameter,
                "source": source, "size_model": str(selected) if selected else None,
                "warning": bool(warning)})
    return diameter


def restore_cellpose_plane(model, image, channels, diameter, task):
    """Restore the selected channels at input resolution using Cellpose 3 denoising."""
    import numpy as np
    from pathlib import Path
    from cellpose.denoise import DenoiseModel

    family = "nuclei" if model.diam_mean == 17 else "cyto3"
    if Path(model.pretrained_model).name in ("cyto2torch_0", "cytotorch_0"):
        family = "cyto2"
    key = ("denoise_" + family, channels[1] > 0)
    if getattr(model, "_jdll_restorer_key", None) != key:
        task.update(message="Loading Cellpose restoration model " + key[0] + ".",
                    info={"type": "cellpose_restoration"})
        model._jdll_restorer = DenoiseModel(model_type=key[0], device=model.device, chan2=key[1])
        model._jdll_restorer_key = key
    restored = model._jdll_restorer.eval(image, channels=channels, channel_axis=-1, diameter=diameter)
    # Cellpose drops a constant secondary channel. Keep the selected layout stable across Z and T.
    if key[1] and restored.shape[-1] == 1:
        restored = np.concatenate((restored, np.zeros_like(restored)), axis=-1)
    return restored
