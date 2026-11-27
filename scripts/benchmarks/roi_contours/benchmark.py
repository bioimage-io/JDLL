"""Compare lossless extraction from float32 instance-label planes. No model is loaded."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def launch():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="a 2D TIFF of integer-valued instance labels")
    parser.add_argument("--jdk", type=Path, default=os.environ.get("JAVA_HOME"), help="JDK 17+")
    parser.add_argument("--classpath", help="Java classpath for JDLL's Appose and its runtime dependencies")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    if args.classpath:
        classpath = args.classpath
    else:
        repo = Path.home() / ".m2/repository"
        artifacts = [
            ("org/apposed", "appose", "0.12.0"),
            ("org/apache/groovy", "groovy", "4.0.28"),
            ("org/apache/groovy", "groovy-json", "4.0.28"),
            ("net/java/dev/jna", "jna", "5.14.0"),
            ("net/java/dev/jna", "jna-platform", "5.14.0"),
        ]
        jars = [repo / group / name / version / f"{name}-{version}.jar" for group, name, version in artifacts]
        if not all(path.is_file() for path in jars):
            parser.error("provide --classpath pointing to your installed Appose runtime")
        classpath = os.pathsep.join(map(str, jars))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    java_bin = args.jdk / "bin" if args.jdk else Path()
    with tempfile.TemporaryDirectory(prefix="jdll-roi-bench-") as tmp:
        subprocess.run([str(java_bin / "javac"), "-cp", classpath, "-d", tmp,
                        str(source / "LabelContours.java"), str(source / "RoiContourBenchmark.java")],
                       env=env, check=True)
        subprocess.run([str(java_bin / "java"), "-Xmx512m", "-cp", tmp + os.pathsep + classpath,
                        "RoiContourBenchmark", sys.executable, str(source), str(args.labels.resolve())],
                       env=env, check=True)


if __name__ == "__main__":
    launch()
    sys.exit(0)


# The Appose worker imports the remainder. The two implementations trace identical
# pixel edges, allowing the packed buffers to be compared word for word.
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter_ns

import numpy as np
import tifffile
from numba import njit


@njit
def runs(pixels):
    h, w = pixels.shape
    output = np.empty(384, np.int32)
    size = 0
    for y in range(h):
        x = 0
        while x < w:
            start = x
            label = pixels[y, x]
            x += 1
            while x < w and pixels[y, x] == label:
                x += 1
            if label == 0:
                continue
            if size == len(output):
                bigger = np.empty(size * 2, np.int32)
                bigger[:size] = output
                output = bigger
            output[size] = int(label)
            output[size+1] = y * w + start
            output[size+2] = x - start
            size += 3
    return output[:size]


@njit(inline="always")
def member(pixels, x, y, label):
    return 0 <= x < pixels.shape[1] and 0 <= y < pixels.shape[0] and pixels[y, x] == label


def numpy_runs(pixels):
    """An optional producer-side encoder requiring only the NumPy already used by the workflows."""
    cuts = np.empty(pixels.shape, dtype=np.bool_)
    np.not_equal(pixels[:, :-1], pixels[:, 1:], out=cuts[:, :-1])
    cuts[:, -1] = True
    ends = np.flatnonzero(cuts).astype(np.int32) + 1
    starts = np.empty_like(ends)
    starts[0], starts[1:] = 0, ends[:-1]
    labels = pixels.ravel()[starts]
    keep = labels != 0
    output = np.empty((np.count_nonzero(keep), 3), np.int32)
    output[:, 0] = labels[keep]
    output[:, 1] = starts[keep]
    output[:, 2] = (ends - starts)[keep]
    return output.ravel()


@njit
def rings(pixels, spans):
    h, w = pixels.shape
    seen = np.zeros((h, w), np.bool_)
    output = np.empty(len(spans) // 3 * 10, np.int32)
    size = 0
    dx, dy = (1, 0, -1, 0), (0, 1, 0, -1)
    right_x, right_y = (0, -1, -1, 0), (0, 0, -1, -1)
    left_x, left_y = (0, 0, -1, -1), (-1, 0, 0, -1)
    for i in range(0, len(spans), 3):
        label, start, length = spans[i], spans[i+1], spans[i+2]
        y0, east_pixel = divmod(start + length - 1, w)
        if seen[y0, east_pixel]:
            continue
        x0 = east_pixel + 1
        x, y, direction = x0, y0, 1
        count_index = size + 1
        output[size] = label
        size += 2
        while True:
            if direction == 1:
                seen[y, x-1] = True
            x, y = x + dx[direction], y + dy[direction]
            following = direction
            if not member(pixels, x + right_x[direction], y + right_y[direction], label):
                following = (direction + 1) & 3
            elif member(pixels, x + left_x[direction], y + left_y[direction], label):
                following = (direction + 3) & 3
            if direction != following:
                output[size], output[size+1] = x, y
                size += 2
            direction = following
            if x == x0 and y == y0 and direction == 1:
                break
        output[count_index] = (size - count_index - 1) // 2
    return output[:size]


def verify(pixels, spans, paths):
    """Independently scan-convert each ID's rings, including holes, to its original row runs."""
    expected = spans.reshape(-1, 3)
    crossings = np.empty((2 * len(expected), 3), np.int32)
    position = index = num_rings = holes = 0
    while index < len(paths):
        label, count = map(int, paths[index:index+2])
        vertices = paths[index+2:index+2+2*count].reshape(-1, 2)
        previous = np.roll(vertices, 1, axis=0)
        area2 = (previous[:, 0].astype(np.int64) * vertices[:, 1] -
                 previous[:, 1].astype(np.int64) * vertices[:, 0]).sum()
        assert count >= 4 and area2 != 0
        holes += int(area2 < 0)
        num_rings += 1
        for (x0, y0), (x1, y1) in zip(previous, vertices):
            assert (x0 == x1) != (y0 == y1)
            if y0 == y1:
                continue
            lo, hi = sorted((int(y0), int(y1)))
            n = hi - lo
            crossings[position:position+n, 0] = label
            crossings[position:position+n, 1] = np.arange(lo, hi)
            crossings[position:position+n, 2] = x0
            position += n
        index += 2 + count * 2
    assert position == len(crossings) and index == len(paths)
    crossings = crossings[np.lexsort(crossings.T[::-1])].reshape(-1, 2, 3)
    left, right = crossings[:, 0], crossings[:, 1]
    assert np.array_equal(left[:, :2], right[:, :2])
    restored = np.column_stack((left[:, 0], left[:, 1] * pixels.shape[1] + left[:, 2],
                                right[:, 2] - left[:, 2]))
    ordered = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    np.testing.assert_array_equal(restored, ordered)
    canvas = np.zeros(pixels.size, dtype=np.float32)
    for label, start, length in expected:
        canvas[start:start+length] = label
    np.testing.assert_array_equal(canvas.reshape(pixels.shape), pixels)
    return num_rings, holes


def fixture(name, path):
    if name == "real":
        raw = tifffile.imread(path)
    elif name == "cells":
        y, x = np.ogrid[:2048, :2048]
        row, col = y // 32, x // 32
        label = 1 + row * 64 + col
        inside = ((x % 32 - 15) ** 2 + (y % 32 - 15) ** 2) <= (12 + label % 4) ** 2
        raw = np.where(inside, label, 0)
    elif name == "fragmented":
        y, x = np.ogrid[:512, :512]
        raw = np.where(np.random.default_rng(42).random((512, 512)) < 0.5,
                       1 + (y // 32) * 16 + x // 32, 0)
    else:
        raw = np.zeros((18, 24), np.int32)
        raw[:11, :11] = 7
        raw[2:9, 2:9] = 0
        raw[4:7, 4:7] = 7
        raw[0:5, 11] = 19
        raw[12, 12] = raw[13, 13] = 7
        raw[-1, -1] = 100019
    pixels = np.ascontiguousarray(raw, dtype=np.float32)
    if (raw.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all() or
            np.any(pixels < 0) or np.any(pixels > 2**24) or not np.array_equal(raw, pixels) or
            not np.array_equal(pixels, np.trunc(pixels))):
        raise ValueError("expected a nonempty plane with exactly representable integer label IDs")
    return pixels


class Plane:
    def __init__(self, name, path):
        self.pixels = fixture(name, path)
        self.handles = {}
        spans = runs(self.pixels)
        np.testing.assert_array_equal(spans, numpy_runs(self.pixels))
        paths = rings(self.pixels, spans)
        ring_count, hole_count = verify(self.pixels, spans, paths)
        if name == "topology":
            assert (ring_count, hole_count) == (7, 1)
        self.info = dict(width=self.pixels.shape[1], height=self.pixels.shape[0],
                         ids=len(np.unique(spans[::3])), runs=len(spans)//3, rings=ring_count,
                         holes=hole_count, packedBytes=spans.nbytes + paths.nbytes)

    def release(self, name):
        handle = self.handles.pop(name, None)
        if handle is not None:
            handle.close()
            handle.unlink()

    def close(self):
        for name in list(self.handles):
            self.release(name)

    def send(self, kind, previous):
        self.release(previous)
        start = perf_counter_ns()
        if kind == "raster":
            blocks, run_words = [self.pixels], 0
        else:
            spans = numpy_runs(self.pixels) if kind == "numpy_runs" else runs(self.pixels)
            blocks, run_words = [spans], len(spans)
            if kind == "geometry":
                blocks.append(rings(self.pixels, spans))
        extract_ns = perf_counter_ns() - start
        size = sum(b.nbytes for b in blocks)
        handle = SharedMemory(create=True, size=max(4, size))
        self.handles[handle.name] = handle
        offset = 0
        for b in blocks:
            handle.buf[offset:offset+b.nbytes] = memoryview(b).cast("B")
            offset += b.nbytes
        return dict(name=handle.name, size=size, runWords=run_words, extractNs=extract_ns)
