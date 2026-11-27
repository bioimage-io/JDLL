import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import org.apposed.appose.Service;
import org.apposed.appose.SharedMemory;
import org.apposed.appose.TaskException;

/** Real Appose IPC. Each result uses a newly created shared-memory segment. */
public class RoiContourBenchmark {
    private record Payload(float[] pixels, int[] spans, int[] paths, double extractMs) {}
    private record Geometry(int[] spans, int[] paths) {}
    private static volatile Object sink;
    private static String previousSegment = "";
    private static int retries;
    private static final int WARMUP = 5, SAMPLES = 15;
    private static final String[] ROUTES = {
        "java_runs", "java_geometry", "dense_to_java_runs", "python_runs",
        "dense_to_java_geometry", "python_geometry", "python_runs_to_java_geometry",
        "java_contours_from_ready_runs", "numpy_runs", "numpy_runs_to_java_geometry"
    };

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        try (Service service = new Service(new File(args[1]), args[0], "-B", "-u", "-c",
                "import faulthandler; faulthandler.enable(); import benchmark; benchmark.Plane('topology', ''); "
                + "from appose.python_worker import main; main()")) {
            service.start();
            if (Boolean.getBoolean("roi.debug")) service.debug(System.err::println);
            service.task("import benchmark; task.export(bench=benchmark)", "main").waitFor();
            for (String name : new String[]{"topology", "real", "cells", "fragmented"}) {
                Map<String, Object> info = service.task(
                        "plane = bench.Plane(name, path)\ntask.export(plane=plane)\ntask.outputs.update(plane.info)",
                        Map.of("name", name, "path", args[2]), "main").waitFor().outputs;
                System.out.println(name + " " + info);
                previousSegment = "";
                retries = 0;
                examine(service, ((Number) info.get("width")).intValue(), name.equals("topology"));
                service.task("plane.close()", "main").waitFor();
                System.out.println("  Appose task retries included in timing: " + retries);
            }
        }
    }

    private static Payload receive(Service service, String kind) throws Exception {
        Map<String, Object> inputs = Map.of("kind", kind, "previous", previousSegment);
        Map<String, Object> result;
        for (int attempt = 0; ; attempt++) {
            try {
                result = service.task("task.outputs.update(plane.send(kind, previous))", inputs).waitFor().outputs;
                break;
            } catch (TaskException e) {
                // The installed worker can report a dead task between thread registration and start.
                // Each read-only attempt owns its output segment. Cleanup acknowledges only consumed segments.
                if (attempt == 2 || !e.getMessage().contains("thread death")) throw e;
                retries++;
            }
        }
        previousSegment = (String) result.get("name");
        int size = ((Number) result.get("size")).intValue();
        int runWords = ((Number) result.get("runWords")).intValue();
        double extract = ((Number) result.get("extractNs")).doubleValue() / 1e6;
        try (SharedMemory memory = SharedMemory.attach((String) result.get("name"), Math.max(4, size))) {
            memory.unlinkOnClose(false);
            ByteBuffer buffer = memory.buf().order(ByteOrder.nativeOrder());
            if (kind.equals("raster")) {
                float[] values = new float[size / 4];
                buffer.asFloatBuffer().get(values);
                return new Payload(values, null, null, extract);
            }
            IntBuffer words = buffer.asIntBuffer();
            int[] spans = new int[runWords];
            int[] paths = new int[size / 4 - runWords];
            words.get(spans).get(paths);
            return new Payload(null, spans, paths, extract);
        }
    }

    private static Geometry geometry(float[] pixels, int width) {
        int[] spans = LabelContours.runs(pixels, width);
        return new Geometry(spans, LabelContours.rings(pixels, width, spans));
    }

    private static Geometry fromRuns(Payload payload, int width, int size) {
        float[] pixels = new float[size];
        int[] spans = payload.spans;
        for (int i = 0; i < spans.length; i += 3) {
            int start = spans[i + 1];
            Arrays.fill(pixels, start, start + spans[i + 2], spans[i]);
        }
        return new Geometry(spans, LabelContours.rings(pixels, width, spans));
    }

    private static void examine(Service service, int width, boolean checkOnly) throws Exception {
        float[] pixels = receive(service, "raster").pixels;
        Geometry java = geometry(pixels, width);
        Payload python = receive(service, "geometry");
        Geometry hybrid = fromRuns(receive(service, "runs"), width, pixels.length);
        if (!Arrays.equals(java.spans, python.spans) || !Arrays.equals(java.paths, python.paths)
                || !Arrays.equals(hybrid.paths, java.paths)) throw new AssertionError("geometry mismatch");
        if (checkOnly) return;
        double[][] times = new double[ROUTES.length][SAMPLES];
        double[][] kernels = new double[2][SAMPLES];
        for (int trial = 0; trial < WARMUP + SAMPLES; trial++) {
            // Rotate the first route to avoid always paying GC/IPC first in one route.
            for (int order = 0; order < ROUTES.length; order++) {
                int route = (trial + order) % ROUTES.length;
                long start = System.nanoTime();
                double pythonMs = 0;
                switch (route) {
                    case 0 -> sink = LabelContours.runs(pixels, width);
                    case 1 -> sink = geometry(pixels, width);
                    case 2 -> sink = LabelContours.runs(receive(service, "raster").pixels, width);
                    case 3 -> {
                        Payload p = receive(service, "runs");
                        sink = p;
                        pythonMs = p.extractMs;
                    }
                    case 4 -> sink = geometry(receive(service, "raster").pixels, width);
                    case 5 -> {
                        Payload p = receive(service, "geometry");
                        sink = p;
                        pythonMs = p.extractMs;
                    }
                    case 6 -> sink = fromRuns(receive(service, "runs"), width, pixels.length);
                    case 7 -> sink = LabelContours.rings(pixels, width, java.spans);
                    case 8 -> sink = receive(service, "numpy_runs");
                    case 9 -> sink = fromRuns(receive(service, "numpy_runs"), width, pixels.length);
                    default -> throw new AssertionError();
                }
                double ms = (System.nanoTime() - start) / 1e6;
                if (trial >= WARMUP) {
                    int sample = trial - WARMUP;
                    times[route][sample] = ms;
                    if (route == 3 || route == 5) kernels[route == 3 ? 0 : 1][sample] = pythonMs;
                }
            }
        }
        for (int i = 0; i < ROUTES.length; i++) report(ROUTES[i], times[i]);
        report("python_runs_kernel", kernels[0]);
        report("python_geometry_kernel", kernels[1]);
        sink = null;
    }

    private static void report(String route, double[] samples) {
        Arrays.sort(samples);
        System.out.printf("  %-32s median %7.2f ms  p90 %7.2f ms%n", route, samples[SAMPLES / 2],
                samples[(int) Math.ceil(SAMPLES * .9) - 1]);
    }
}
