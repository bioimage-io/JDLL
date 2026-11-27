# Label-to-contour benchmark

Run this directory's `benchmark.py` with the Pixi Appose Python, passing the path to a 2D label TIFF and `--jdk PATH_TO_JDK`. It compiles the two Java files in a temporary directory, then starts a fresh JVM and a Python worker. It uses the installed Maven jars or a classpath supplied with `--classpath`. The TIFF is read-only.

## Results

Windows x64, Intel Core i5-7200U, Java 21.0.11, Python 3.11.18, Java Appose 0.12.0, Python Appose 0.11.0, NumPy 2.0.2 and Numba 0.60.0. Times are per plane in milliseconds, with five warmups followed by the median of 15 interleaved measurements. The harness also prints the 90th percentile.

| Measurement | ProteoVir, 2048², 46 IDs | Synthetic disks, 2048², 4096 IDs | Fragmented, 512², 256 IDs |
|---|---:|---:|---:|
| Resident Java raster to runs | 3.52 | 4.40 | 2.34 |
| Java tracing with raster and runs already available | 0.86 | 9.78 | 5.13 |
| Resident Java raster to runs and rings | 3.88 | 14.90 | 7.10 |
| Resident Python raster to runs and rings, compiled with Numba | 8.68 | 21.49 | 10.69 |
| Send raster over Appose, then extract runs and rings in Java | 31.76 | 38.65 | 12.83 |
| Send Python-computed runs and rings over Appose | 14.25 | 31.98 | 17.70 |
| Send NumPy runs over Appose and trace in Java | 12.98 | 35.05 | 16.41 |
| Send Numba runs over Appose and trace in Java | 13.85 | 27.55 | 15.00 |
| Send NumPy runs over Appose, ending when Java has the runs | 11.61 | 21.61 | 10.25 |
| Packed runs and rings in bytes | 91,792 | 3,768,320 | 2,586,440 |
| Full float32 plane in bytes | 16,777,216 | 16,777,216 | 1,048,576 |

Both tracers follow oriented pixel edges with 4-connected foreground and emit only corners. Runs are packed int32 triples of label ID, row-major start and length. The benchmark checks coordinates and runs word for word across the two languages. An independent scan conversion reconstructs each ID's original runs from its rings, then reconstructs the source plane exactly. A separate fixture exercises an object with a hole and island, diagonal contacts, adjacent labels, single pixels, image borders and sparse IDs.

IPC measurements start with the plane in Python and finish with primitive buffers in Java. They include dispatch, allocation and copying via fresh named shared memory. The runs-to-Java variant materializes a temporary plane for tracing. Times exclude model execution, JIT startup, arbitrary ImgLib2 view materialization, Icy ROI construction, assignment of holes to polygon exteriors, GeoJSON topology normalization and file writing. The installed Appose worker occasionally reports a task dead while its thread starts. Retries of those read-only requests are included in wall times. The measured run reported zero retries for the real image and two each for the two synthetic datasets across setup, warmup and timed tasks.
