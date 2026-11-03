package io.bioimage.modelrunner.gui.custom.yolo;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentResolver;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.system.GpuCompatibility;
import io.bioimage.modelrunner.system.PlatformDetection;

/** Short-lived, coalesced checks of already-installed backend runtimes; never installs anything. */
final class AccelerationAvailability {
    private static final Map<String, Probe> PROBES = new ConcurrentHashMap<>();

    private AccelerationAvailability() {}

    static boolean check(PixiEnvironmentSpec environment, YoloAccelerationCheckBox.Backend backend) {
        boolean mac = PlatformDetection.isMacOS();
        String name = (environment.getSelectedEnvironment() + environment.getEnvironmentDirectory().getName())
                .toLowerCase(java.util.Locale.ROOT);
        if (!mac && (!name.contains("cuda") || name.contains("no-cuda"))) return false;
        File prefix = new File(environment.getEnvironmentDirectory(), ".pixi/envs/" + environment.getSelectedEnvironment());
        File python = new File(prefix, PlatformDetection.isWindows() ? "python.exe" : "bin/python");
        // Before first installation only the live driver can be tested. Workload startup checks the backend too.
        if (!python.isFile()) return mac ? GpuCompatibility.isMetalAvailable() : GpuCompatibility.isCudaAvailable();
        String key = python.getAbsolutePath() + ":" + python.lastModified() + ":" + backend;
        Probe probe = PROBES.compute(key, (k, previous) -> previous != null
                && System.nanoTime() - previous.started < TimeUnit.SECONDS.toNanos(30)
                ? previous : new Probe(prefix, python, backend, mac ? "mps" : "cuda"));
        probe.result.run();
        try {
            return probe.result.get();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return false;
        } catch (java.util.concurrent.ExecutionException error) {
            return false;
        }
    }

    private static final class Probe {
        final long started = System.nanoTime();
        final FutureTask<Boolean> result;

        Probe(File prefix, File python, YoloAccelerationCheckBox.Backend backend, String device) {
            result = new FutureTask<>(() -> run(prefix, python, backend, device));
        }
    }

    private static boolean run(File prefix, File python, YoloAccelerationCheckBox.Backend backend,
            String device) throws Exception {
        String source = PixiEnvironmentResolver.readClasspathResourceAsString("python/device_selection.py");
        String function = backend == YoloAccelerationCheckBox.Backend.TENSORFLOW ? "tensorflow_device" : "torch_device";
        String script = source + "\nresolved = " + function + "('" + device + "', lambda **event: None)\n"
                + "print('JDLL_ACCELERATION=' + ('no' if resolved in ('cpu', '/CPU:0') else 'yes'))\n";
        Path output = Files.createTempFile("jdll-acceleration-", ".log");
        Process process = null;
        try {
            ProcessBuilder builder = new ProcessBuilder(python.getAbsolutePath(), "-c", script)
                    .redirectErrorStream(true).redirectOutput(output.toFile());
            Map<String, String> env = builder.environment();
            env.put("CONDA_PREFIX", prefix.getAbsolutePath());
            env.put("TF_CPP_MIN_LOG_LEVEL", "3");
            env.put("PYTHONPATH", "");
            env.put("DYLD_LIBRARY_PATH", "");
            for (String key : new String[] {"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"})
                env.put(key, PlatformDetection.isWindows() ? new File(prefix, "Library").getAbsolutePath() : "");
            String pathKey = env.keySet().stream().filter(key -> key.equalsIgnoreCase("PATH")).findFirst().orElse("PATH");
            String path = env.getOrDefault(pathKey, "");
            env.put(pathKey, new File(prefix, "bin") + File.pathSeparator
                    + new File(prefix, "Library/bin") + File.pathSeparator + prefix + File.pathSeparator + path);
            if (PlatformDetection.isLinux()) {
                env.put("LD_LIBRARY_PATH", new File(prefix, "lib").getAbsolutePath());
            }
            process = builder.start();
            if (!process.waitFor(20, TimeUnit.SECONDS)) return false;
            return process.exitValue() == 0 && new String(Files.readAllBytes(output), StandardCharsets.UTF_8)
                    .contains("JDLL_ACCELERATION=yes");
        } finally {
            if (process != null && process.isAlive()) {
                process.destroyForcibly();
                process.waitFor();
            }
            Files.deleteIfExists(output);
        }
    }
}
