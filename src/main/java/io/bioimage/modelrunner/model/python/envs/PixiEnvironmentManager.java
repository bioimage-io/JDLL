package io.bioimage.modelrunner.model.python.envs;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Supplier;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.builder.PixiBuilder;
import org.apposed.appose.tool.Pixi;

/**
 * Centralized environment planning, installation and validation logic for the
 * protected PyTorch model runner.
 */
public final class PixiEnvironmentManager {
    private static final ThreadLocal<Operation> CURRENT_OPERATION = new ThreadLocal<>();

    private PixiEnvironmentManager() {
        // Utility class.
    }

    /** Reuse setup only within this thread's operation; closing never closes model workers. */
    public static Operation beginOperation() {
        Operation operation = new Operation(CURRENT_OPERATION.get());
        CURRENT_OPERATION.set(operation);
        return operation;
    }

    /** Resolve hardware/platform settings once per backend family within an operation. */
    public static PixiEnvironmentSpec resolveForOperation(String family, Supplier<PixiEnvironmentSpec> resolver) {
        Operation operation = CURRENT_OPERATION.get();
        return operation == null ? resolver.get()
                : operation.specifications.computeIfAbsent(family, key -> resolver.get());
    }

    public static final class Operation implements AutoCloseable {
        private final Operation previous;
        private final Map<String, PixiEnvironmentSpec> specifications = new HashMap<>();
        private final Map<String, PreparedEnvironment> environments = new HashMap<>();
        private boolean closed;

        private Operation(Operation previous) { this.previous = previous; }

        private Environment prepared(PixiEnvironmentSpec spec) {
            String key = environmentKey(spec);
            PreparedEnvironment prepared = environments.get(key);
            if (prepared == null) return null;
            if (prepared.matches(spec)) return prepared.environment;
            environments.remove(key);
            return null;
        }

        @Override public void close() {
            if (closed) return;
            if (CURRENT_OPERATION.get() != this) {
                throw new IllegalStateException("Environment operations must close on their owning thread in nesting order.");
            }
            specifications.clear();
            environments.clear();
            closed = true;
            if (previous == null) CURRENT_OPERATION.remove();
            else CURRENT_OPERATION.set(previous);
        }
    }

    private static String environmentKey(PixiEnvironmentSpec spec) {
        return spec.getEnvironmentDirectory().toPath().toAbsolutePath().normalize() + ":" + spec.getSelectedEnvironment();
    }

    private static final class PreparedEnvironment {
        final Environment environment;
        final String manifest;
        final List<String> manualNoDeps;
        final String builderState;
        final File pixiExecutable;

        PreparedEnvironment(PixiEnvironmentSpec spec, Environment environment) {
            this.environment = environment;
            manifest = spec.getPixiTomlContent();
            manualNoDeps = new ArrayList<>(spec.getManualNoDeps());
            builderState = readState(new File(spec.getEnvironmentDirectory(), "appose.json"));
            pixiExecutable = new File(environment.launchArgs().get(0));
        }

        boolean matches(PixiEnvironmentSpec spec) {
            if (!manifest.equals(spec.getPixiTomlContent()) || !manualNoDeps.equals(spec.getManualNoDeps())
                    || !hasMatchingManifest(spec) || !pixiExecutable.isFile() || builderState == null
                    || !builderState.equals(readState(new File(spec.getEnvironmentDirectory(), "appose.json")))) return false;
            File prefix = new File(spec.getEnvironmentDirectory(), ".pixi/envs/" + spec.getSelectedEnvironment());
            if (!new File(prefix, io.bioimage.modelrunner.system.PlatformDetection.isWindows()
                    ? "python.exe" : "bin/python").isFile()) return false;
            return manualNoDeps.isEmpty() || (manifest + "\n" + manualNoDeps).equals(
                    readState(new File(spec.getEnvironmentDirectory(), ".jdll-no-deps")));
        }
    }

    private static String readState(File file) {
        try {
            return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * Creates an unstarted worker with NumPy initialized before Appose reads stdin.
     * Importing NumPy for the first time from a task can deadlock on Windows.
     *
     * @param environment the installed Python environment.
     * @return an unstarted service configured for early NumPy initialization.
     */
    public static Service pythonService(Environment environment) {
        return pythonService(environment, "");
    }

    /** Adds backend-specific initialization without replacing the required NumPy import. */
    public static Service pythonService(Environment environment, String initialization) {
        Objects.requireNonNull(environment, "environment");
        EnvironmentSetupLog.startingPython();
        String script = "import numpy as np";
        if (initialization != null && !initialization.trim().isEmpty()) script += "\n" + initialization;
        return environment.python().init(script);
    }

    /**
     * Checks whether the environment described by the given spec appears to be
     * installed.
     *
     * <p>This method verifies only that:
     * <ul>
     * <li>the environment directory exists,</li>
     * <li>the selected Conda prefix exists,</li>
     * <li>one of the expected manifest files exists,</li>
     * <li>the first manifest found in the order {@code pixi.toml},
     * {@code pyproject.toml}, {@code environment.yml} matches the expected
     * content from the spec,</li>
     * <li>and Pixi is installed.</li>
     * </ul>
     *
     * <p>It does <strong>not</strong> validate that the actual environment contents
     * still match the manifest. If the environment was modified after creation,
     * this method may still return {@code true}.</p>
     *
     * @param spec the spec.
     * @return true if installed; false otherwise.
     */
    public static boolean isInstalled(final PixiEnvironmentSpec spec) {
        Objects.requireNonNull(spec, "spec");

        Operation operation = CURRENT_OPERATION.get();
        if (operation != null && operation.prepared(spec) != null) return true;
        return hasMatchingManifest(spec) && new Pixi().isInstalled();
    }

    private static boolean hasMatchingManifest(PixiEnvironmentSpec spec) {

        final File envDir = spec.getEnvironmentDirectory();
        if (!envDir.isDirectory() || !hasPrefix(spec)) {
            return false;
        }

        final File[] candidateFiles = new File[] {
            new File(envDir, "pixi.toml"),
            new File(envDir, "pyproject.toml"),
            new File(envDir, "environment.yml")
        };

        File manifestFile = null;
        for (File candidate : candidateFiles) {
            if (candidate.isFile()) {
                manifestFile = candidate;
                break;
            }
        }

        if (manifestFile == null) {
            return false;
        }

        final String fileContent;
        try {
            fileContent = new String(
                Files.readAllBytes(manifestFile.toPath()), StandardCharsets.UTF_8);
        } catch (IOException e) {
            return false;
        }

        if (!spec.getPixiTomlContent().equals(fileContent)) {
            return false;
        }

        return true;
    }

    /**
     * Ensures the environment described by the given specification is installed.
     *
     * @param spec the spec.
     * @param consumer the consumer callback.
     * @throws InterruptedException if the current thread is interrupted.
     * @throws BuildException if the Python environment or service cannot be built.
     */
    public static void installRequirements(
            final PixiEnvironmentSpec spec,
            final Consumer<String> consumer) throws InterruptedException, BuildException {
        Objects.requireNonNull(spec, "spec");
        openEnvironment(spec, consumer);
    }

    /**
     * Build or reuse an environment with the same reporting for installation and model loading.
     * Raw Appose/Pixi output is kept separately from user-facing progress summaries.
     */
    public static Environment openEnvironment(PixiEnvironmentSpec spec, Consumer<String> consumer) throws BuildException {
        return openEnvironment(spec, consumer, PixiEnvironmentManager::buildEnvironment);
    }

    @FunctionalInterface
    interface EnvironmentBuilder {
        Environment build(PixiEnvironmentSpec spec, Consumer<String> consumer) throws BuildException;
    }

    static Environment openEnvironment(PixiEnvironmentSpec spec, Consumer<String> consumer,
            EnvironmentBuilder builder) throws BuildException {
        Objects.requireNonNull(spec, "spec");
        Operation operation = CURRENT_OPERATION.get();
        Environment prepared = operation == null ? null : operation.prepared(spec);
        if (prepared != null) return prepared;
        Environment environment = builder.build(spec, consumer);
        if (operation != null) operation.environments.put(environmentKey(spec), new PreparedEnvironment(spec, environment));
        return environment;
    }

    private static Environment buildEnvironment(PixiEnvironmentSpec spec, Consumer<String> consumer) throws BuildException {
        Objects.requireNonNull(spec, "spec");
        File directory = spec.getEnvironmentDirectory();
        EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter(directory.getAbsolutePath(), consumer,
                directory, spec.getSelectedEnvironment());
        // Appose performs the authoritative Pixi check; the announcement only needs local state.
        reporter.announceSetup(hasMatchingManifest(spec));
        File diagnostics = new File(new File(directory, "setup-logs"),
                "setup-" + System.currentTimeMillis() + "-" + UUID.randomUUID().toString().substring(0, 8) + ".log");
        boolean success = false;
        try {
            Environment env;
            try (PrintWriter writer = diagnosticWriter(diagnostics, reporter)) {
                Consumer<String> output = line -> {
                    if (line == null) return;
                    writer.println(line);
                    reporter.output(line);
                };
                try {
                    boolean prefixPresent = hasPrefix(spec);
                    // Appose 0.12 checks builder state, but not whether its prefix was deleted.
                    if (!prefixPresent) Files.deleteIfExists(new File(directory, "appose.json").toPath());
                    File manualState = new File(directory, ".jdll-no-deps");
                    String manualSignature = spec.getPixiTomlContent() + "\n" + spec.getManualNoDeps();
                    boolean manualInstalled = prefixPresent && manualState.isFile() && manualSignature.equals(
                            new String(Files.readAllBytes(manualState.toPath()), StandardCharsets.UTF_8));
                    PixiBuilder builder = Appose.pixi().content(spec.getPixiTomlContent()).scheme("pixi.toml").base(directory);
                    builder.subscribeOutput(output);
                    builder.subscribeError(output);
                    builder.subscribeProgress(reporter::progress);
                    env = builder.build();
                    // JDLL's per-device manifests use default. Activating it again would run install twice.
                    if (!"default".equals(spec.getSelectedEnvironment())) env = env.activate(spec.getSelectedEnvironment());
                    if (!manualInstalled && !spec.getManualNoDeps().isEmpty()) {
                        for (String pckge : spec.getManualNoDeps()) {
                            reporter.phase("Installing " + pckge, 0, -1);
                            installNoDeps(env, pckge, output);
                        }
                        Files.write(manualState.toPath(), manualSignature.getBytes(StandardCharsets.UTF_8));
                    }
                } catch (BuildException | IOException | RuntimeException e) {
                    e.printStackTrace(writer);
                    for (Throwable cause = e; cause != null; cause = cause.getCause()) {
                        if (cause instanceof InterruptedException) Thread.currentThread().interrupt();
                    }
                    throw e;
                } finally {
                    if (writer.checkError()) reporter.message("Could not write complete environment diagnostics at: " + diagnostics);
                }
            }
            success = true;
            return env;
        } catch (IOException e) {
            throw new BuildException(e);
        } finally {
            reporter.finish(success, diagnostics.getAbsolutePath());
        }
    }

    private static boolean hasPrefix(PixiEnvironmentSpec spec) {
        return new File(spec.getEnvironmentDirectory(), ".pixi/envs/" + spec.getSelectedEnvironment()
                + "/conda-meta").isDirectory();
    }

    private static PrintWriter diagnosticWriter(File file, EnvironmentSetupLog.Reporter reporter) {
        try {
            Files.createDirectories(file.getParentFile().toPath());
            return new PrintWriter(Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8));
        } catch (IOException e) {
            reporter.message("Could not save environment diagnostics at " + file + ": " + e.getMessage());
            // Keep draining process output even if the diagnostic destination is unavailable.
            return new PrintWriter(new java.io.OutputStream() {
                @Override public void write(int value) {}
            });
        }
    }

    private static void installNoDeps(Environment env, String pckge, Consumer<String> diagnostics) throws BuildException {
        List<String> pythonExes = Arrays.asList("python", "python3", "python.exe");
        try (Service service = env.service(pythonExes, "-m", "pip", "install", pckge, "--no-deps")) {
            service.debug(diagnostics);
            service.start();
            int exit = service.waitFor();
            if (exit != 0) throw new BuildException("Package installation failed for " + pckge + " (exit " + exit + ")");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BuildException(e);
        } catch (IOException e) {
            throw new BuildException(e);
        }
    }
}
