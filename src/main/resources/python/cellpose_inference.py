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
        expected = path.parent / official[path.name] if path.name in official else None
        candidates = sorted(p for p in path.parent.glob("size_*.npy") if p.is_file())
        if expected is not None and expected.is_file():
            selected = expected
        elif expected is None and candidates:
            selected = candidates[0]
            if len(candidates) > 1:
                warning = "Multiple size models found; using " + selected.name + "."
        if selected is not None:
            try:
                from cellpose.models import SizeModel

                cache = getattr(model, "_jdll_size_models", None)
                if cache is None:
                    cache = model._jdll_size_models = {}
                key = str(selected.resolve())
                if key not in cache:
                    cache[key] = SizeModel(pretrained_size=str(selected), cp_model=model.cp)
                estimate = np.asarray(cache[key].eval(images, channels=channels, channel_axis=-1)[0]).reshape(-1)
                if estimate.size != 1 or not np.isfinite(estimate[0]) or estimate[0] <= 0:
                    raise ValueError("size estimator returned an invalid diameter")
                diameter = float(estimate[0])
                source = "estimated"
            except MemoryError:
                raise
            except Exception as error:
                warning = "Diameter estimation failed: " + str(error) + "."
        else:
            warning = "No matching size model found." if expected is not None else "No size model found."
        if diameter is None:
            diameter = float(getattr(model.cp, "diam_labels", model.cp.diam_mean))
            source = "model_default"
            warning = (warning or "") + " Using the segmentation model's default diameter."
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
