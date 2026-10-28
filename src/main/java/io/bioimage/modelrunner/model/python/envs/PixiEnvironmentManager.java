package io.bioimage.modelrunner.model.python.envs;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.UUID;
import java.util.function.Consumer;

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

    private static final String OPENCV_COMPAT = "opencv-python-compat/pyproject.toml";

    private PixiEnvironmentManager() {
        // Utility class.
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
     * <li>any bundled compatibility resource matches its classpath source,</li>
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

        try {
            return resourcesMatch(spec) && new Pixi().isInstalled();
        } catch (IOException e) {
            return false;
        }
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
        Objects.requireNonNull(spec, "spec");
        File directory = spec.getEnvironmentDirectory();
        EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter(directory.getAbsolutePath(), consumer,
                directory, spec.getSelectedEnvironment());
        reporter.announceSetup(isInstalled(spec));
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
                    prepareResources(spec);
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

    /** Install the metadata-only OpenCV alias beside the manifest, not in site-packages. */
    static void prepareResources(PixiEnvironmentSpec spec) throws IOException {
        if (!needsOpenCvCompat(spec)) return;
        File target = new File(spec.getEnvironmentDirectory(), OPENCV_COMPAT);
        String content = openCvCompatContent();
        if (target.isFile() && content.equals(new String(Files.readAllBytes(target.toPath()), StandardCharsets.UTF_8))) return;
        Files.createDirectories(target.getParentFile().toPath());
        Files.write(target.toPath(), content.getBytes(StandardCharsets.UTF_8));
    }

    static boolean resourcesMatch(PixiEnvironmentSpec spec) throws IOException {
        if (!needsOpenCvCompat(spec)) return true;
        File target = new File(spec.getEnvironmentDirectory(), OPENCV_COMPAT);
        return target.isFile() && openCvCompatContent().equals(
                new String(Files.readAllBytes(target.toPath()), StandardCharsets.UTF_8));
    }

    private static boolean needsOpenCvCompat(PixiEnvironmentSpec spec) {
        return spec.getPixiTomlContent().contains("path = \"./opencv-python-compat\"");
    }

    private static String openCvCompatContent() {
        return PixiEnvironmentResolver.readClasspathResourceAsString("python/" + OPENCV_COMPAT);
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
