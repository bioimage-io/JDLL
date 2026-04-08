package io.bioimage.modelrunner.model.python.envs;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Consumer;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Builder.ProgressConsumer;
import org.apposed.appose.builder.PixiBuilder;
import org.apposed.appose.tool.Pixi;

/**
 * Centralized environment planning, installation and validation logic for the
 * protected PyTorch model runner.
 */
public final class PixiEnvironmentManager {

    private PixiEnvironmentManager() {
        // Utility class.
    }

    /**
     * Checks whether the environment described by the given spec appears to be
     * installed.
     *
     * <p>This method verifies only that:
     * <ul>
     *   <li>the environment directory exists,</li>
     *   <li>one of the expected manifest files exists,</li>
     *   <li>the first manifest found in the order {@code pixi.toml},
     *       {@code pyproject.toml}, {@code environment.yml} matches the expected
     *       content from the spec,</li>
     *   <li>and Pixi is installed.</li>
     * </ul>
     *
     * <p>It does <strong>not</strong> validate that the actual environment contents
     * still match the manifest. If the environment was modified after creation,
     * this method may still return {@code true}.</p>
     *
     * @param spec
     *     the resolved environment specification
     * @return {@code true} if the environment directory and manifest look valid for
     *     the given spec, {@code false} otherwise
     */
    public static boolean isInstalled(final PixiEnvironmentSpec spec) {
        Objects.requireNonNull(spec, "spec");

        final File envDir = spec.getEnvironmentDirectory();
        if (!envDir.isDirectory()) {
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

        return new Pixi().isInstalled();
    }

    /**
     * Ensures the environment described by the given specification is installed.
     *
     * @param spec
     *     the resolved environment specification
     * @param consumer
     *     optional consumer for stdout, stderr and progress updates
     * @throws InterruptedException
     *     if installation is interrupted
     * @throws BuildException
     *     if the Pixi environment cannot be built or a required follow-up step fails
     */
    public static void installRequirements(
            final PixiEnvironmentSpec spec,
            final Consumer<String> consumer) throws InterruptedException, BuildException {
        Objects.requireNonNull(spec, "spec");

        final PixiBuilder pixi = Appose.pixi().content(spec.getPixiTomlContent());
        subscribeLogs(pixi, consumer);

        final boolean installed = isInstalled(spec);
        if (!installed) {
            final Environment env = pixi.environment(spec.getSelectedEnvironment()).build();

            if (spec.getManualNoDeps().size() > 0) {
            	for (String pckge : spec.getManualNoDeps()) {
            		consumer.accept("Installing: " + pckge);
                	installNoDeps(env, pckge);
            	}
            }
        }
    }

    /**
     * Installs {@code biapy==3.5.10} in an already built environment without
     * installing its dependencies.
     *
     * @param env
     *     the environment where Biapy should be installed
     * @param pckge
     *     the package we want to be installed as a string someone would use with `pip install package`
     * @throws BuildException
     *     if the installation fails
     */
    private static void installNoDeps(final Environment env, final String pckge) throws BuildException {
		List<String> pythonExes = Arrays.asList("python", "python3", "python.exe");
        try {
			Service serv = env.service(pythonExes, "-m", "pip", "install", "biapy==3.5.10", "--no-deps").start();
			serv.waitFor();
		} catch (InterruptedException | IOException e) {
			throw new BuildException(e);
		}
    }

    private static void subscribeLogs(final PixiBuilder pixi, final Consumer<String> consumer) {
        if (consumer == null) {
            return;
        }

        pixi.subscribeOutput(consumer);
        pixi.subscribeError(consumer);

        final ProgressConsumer progress = (title, current, maximum) -> {
            final double pct = (maximum <= 0) ? 0.0 : (100.0 * current / maximum);
            final String label = (title == null || title.trim().isEmpty()) ? "Downloading" : title;
            consumer.accept(label + ": " + String.format(Locale.US, "%.1f", pct) + "%");
        };
        pixi.subscribeProgress(progress);
    }
}