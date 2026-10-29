# Images and NumPy Arrays

## YOLO Training

YOLO training accepts its existing image layouts and annotations, plus numeric
`.npy` images and instance masks. Image sources are read-only. Generated NumPy or
floating-point image crops are references into their original sources, not
converted copies. Keep those sources available while training.
Keep the generated `jdll_samples.json` alongside `data.yaml`: JDLL's YOLO
adapter uses it to resolve image references and crops. Numeric validation
sources receive small PNG display assets, not converted training-image copies.

NumPy images support XY, XYC, XYCZ, XYCZT and their reverse conventions YX, CYX,
ZCYX, TZCYX. Both candidate spatial dimensions must exceed 100 pixels and their
aspect ratio must be less than 3. The largest qualifying spatial pair is used;
ties prefer trailing YX. Grayscale volumes need an explicit singleton channel
axis. Three-channel arrays use RGB channel order. YOLO training requires
individual 2D samples; non-singleton Z/T axes must be exported as separately
annotated images. Masks may omit channels and are
matched against their image geometry. Only their first channel is used.

Image arrays may use uint8, uint16, float32 or float64. Object/pickled arrays and
structured records are unsupported. Masks must contain finite, nonnegative
integer instance IDs fitting int32. An invalid mask stops preparation and names
the offending file; fix it before retrying.

Keep `cache=False`. Existing NumPy files are never overwritten as Ultralytics
image caches. If a conventional image and a NumPy array share a stem in the
same image directory, the conventional image takes precedence, avoiding stale
Ultralytics cache files being mistaken for additional training images.

YOLO floating-point images are scaled using the minimum and maximum of the
training images only, then clipped to byte range before augmentation. The saved
`config.json` records these parameters; validation and inference reuse them.
Constant training intensities map to zero. NaN/Inf are rejected. Uint16 images
use the upper eight bits, following current Ultralytics image loading. Source
files and host images are never modified. Legacy/pretrained models without
float statistics use the current input image's range, resolved before tiling.

## Dense Segmentation Training

StarDist, UNet and Cross-GOOSE accept `.npy` images and label masks in the same
paired folder layouts as their raster inputs. NumPy and TIFF/PNG can be mixed
within a dataset or an image/mask pair. Layout normalization links source files;
it does not convert or rewrite their pixels.
Raster images take precedence over same-stem NumPy caches in the same image
folder. This does not discard or resolve competing mask files; the existing
mask-pairing rules still apply.

The axis convention above also applies here. For grayscale volumes, use
`Z,1,Y,X` or `X,Y,1,Z`; masks can omit C (`Z,Y,X` or `X,Y,Z`) because the paired
image guides their geometry. UNet supports these sources for 2D, Fast 3D and
True 3D according to its existing dataset-eligibility rules. StarDist supports
2D or 3D datasets; Cross-GOOSE remains 2D. Singleton Z/T axes are supported;
non-singleton T must be split into separately annotated time points.

Sources are mapped read-only, including Fortran-ordered and big-endian arrays.
StarDist and UNet read patches from those mappings and retain their bounded
caches and existing normalization. Unlike YOLO, dense workflows do not convert
images to uint8. Label precision is retained; Java checks that IDs fit int32.
Only the first mask channel is used. Invalid NumPy masks stop preparation at
the first offending file, with a path and documentation link.

The Python `jdll-unet` and `jdll-cross-goose` readers must include NumPy support;
publishing their changes and updating JDLL's pinned Git revisions is required
before newly installed environments can use this capability.

## Filesystem Inference

Opening NumPy files for inference remains the responsibility of the host
application (Icy, Fiji, etc.). Training does not require host NumPy support. If
the host cannot open your array, convert it to a supported TIFF or open it using
a suitable host importer first. Consumers can provide application-specific
guidance through `getImageFormatDocumentationUrl()`.
