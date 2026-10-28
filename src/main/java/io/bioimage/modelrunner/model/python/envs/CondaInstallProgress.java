package io.bioimage.modelrunner.model.python.envs;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;

import org.apposed.appose.util.Platforms;
import org.yaml.snakeyaml.Yaml;

/** Appose 0.12's indentation-based lock parser misses Pixi's indentless YAML sequences. */
final class CondaInstallProgress implements AutoCloseable {
    private final File directory;
    private final String environment;
    private final BiConsumer<Long, Long> progress;
    private final AtomicBoolean active = new AtomicBoolean();
    private ScheduledExecutorService executor;
    private long total;

    CondaInstallProgress(File directory, String environment, BiConsumer<Long, Long> progress) {
        this.directory = directory;
        this.environment = environment;
        this.progress = progress;
    }

    void start() {
        if (directory == null || active.get()) return;
        total = packageCount();
        if (total <= 0) return;
        active.set(true);
        poll();
        executor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread thread = new Thread(r, "jdll-conda-install-progress");
            thread.setDaemon(true);
            return thread;
        });
        executor.scheduleWithFixedDelay(this::poll, 500, 500, TimeUnit.MILLISECONDS);
    }

    private void poll() {
        if (!active.get()) return;
        File metadata = new File(directory, ".pixi/envs/" + environment + "/conda-meta");
        String[] files = metadata.list((folder, name) -> name.endsWith(".json"));
        if (active.get()) progress.accept(Math.min(total, files == null ? 0L : files.length), total);
    }

    long packageCount() {
        try (Reader reader = Files.newBufferedReader(new File(directory, "pixi.lock").toPath(), StandardCharsets.UTF_8)) {
            Object packages = new Yaml().load(reader);
            for (String key : new String[] {"environments", environment, "packages", platform()}) {
                if (!(packages instanceof Map)) return 0;
                packages = ((Map<?, ?>) packages).get(key);
            }
            if (!(packages instanceof List)) return 0;
            return ((List<?>) packages).stream().filter(entry -> entry instanceof Map
                    && ((Map<?, ?>) entry).containsKey("conda")).count();
        } catch (IOException | RuntimeException e) {
            return 0; // Unknown totals remain indeterminate.
        }
    }

    static String platform() {
        switch (Platforms.PLATFORM) {
            case "LINUX|ARM64": return "linux-aarch64";
            case "LINUX|X64": return "linux-64";
            case "WINDOWS|ARM64": return "win-arm64";
            case "WINDOWS|X64": return "win-64";
            case "MACOS|ARM64": return "osx-arm64";
            case "MACOS|X64": return "osx-64";
            default: return "";
        }
    }

    boolean isActive() { return active.get(); }
    long total() { return total; }

    @Override public void close() {
        active.set(false);
        if (executor != null) executor.shutdownNow();
    }
}
