package io.bioimage.modelrunner.model.python.envs;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskException;
import org.apposed.appose.Builder.ProgressConsumer;
import org.apposed.appose.builder.PixiBuilder;

/**
 * Centralized environment planning, installation and validation logic for the
 * protected PyTorch model runner.
 */
public final class PytorchEnvironmentManager {

    private PytorchEnvironmentManager() {
        // Utility class.
    }

    /**
     * Checks whether the environment described by the given spec appears to be
     * fully installed.
     *
     * @param spec
     *     the resolved environment specification
     * @return {@code true} if the expected dependencies are installed,
     *     {@code false} otherwise
     * @throws BuildException 
     *     if the Pixi environment cannot be built or a required follow-up step fails
     */
    public static boolean isInstalled(final PixiEnvironmentSpec spec) throws BuildException {
        Objects.requireNonNull(spec, "spec");

        Environment env = Appose.pixi().content(spec.getPixiTomlContent()).environment(spec.getSelectedEnvironment()).build();

        Service service = null;
        try {
            service = env.python();
            final Task task = service.task(buildDependencyValidationScript(spec));
            task.waitFor();

            if (task.status != TaskStatus.COMPLETE) {
                return false;
            }

            final Object installed = task.outputs.get("installed");
            return Boolean.TRUE.equals(installed);
        } catch (InterruptedException | TaskException e) {
            return false;
        } finally {
            if (service != null) {
                service.close();
            }
        }
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

        }
    }

    /**
     * Installs {@code biapy==3.5.10} in an already built environment without
     * installing its dependencies.
     *
     * @param env
     *     the environment where Biapy should be installed
     * @throws BuildException
     *     if the installation fails
     * @throws InterruptedException
     *     if the installation is interrupted
     */
    private static void installBiapyNoDeps(final Environment env) throws BuildException, InterruptedException {
        Objects.requireNonNull(env, "env");

        Service service = null;
        try {
            service = env.python();

            final String code =
                    "import subprocess" + System.lineSeparator() +
                    "import sys" + System.lineSeparator() +
                    "subprocess.check_call([" +
                    "sys.executable, '-m', 'pip', 'install', 'biapy==3.5.10', '--no-deps'])";

            final Task task = service.task(code);
            task.waitFor();
            ensureTaskSucceeded(task, "Failed to install biapy==3.5.10 --no-deps.");
        } catch (TaskException e) {
            throw new BuildException("Failed to install biapy==3.5.10 --no-deps.", e);
        } finally {
            if (service != null) {
                service.close();
            }
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

    private static void ensureTaskSucceeded(final Task task, final String defaultMessage) throws BuildException {
        if (task.status == TaskStatus.COMPLETE) {
            return;
        }
        if (task.status == TaskStatus.CANCELED) {
            throw new BuildException(defaultMessage + " Task canceled.");
        }
        if (task.status == TaskStatus.FAILED || task.status == TaskStatus.CRASHED) {
            throw new BuildException(defaultMessage + " " + task.error);
        }
        throw new BuildException(defaultMessage + " Unexpected task status: " + task.status);
    }

    private static String buildDependencyValidationScript(final PixiEnvironmentSpec spec) {
        final Map<String, String> exact = new LinkedHashMap<String, String>();
        final Map<String, String> minimum = new LinkedHashMap<String, String>();
        final Set<String> present = new LinkedHashSet<String>();

        // Python version
        // Checked separately in Python via sys.version_info.

        // Torch / torchvision versions
        addTorchRequirements(spec, exact);

        if (true) {
            // Legacy macOS branch
            exact.put("timm", "1.0.14");
            exact.put("pytorch-msssim", "1.0.0");
            exact.put("torchmetrics", "1.4.3");
            exact.put("cellpose", "3.1.1.1");
            exact.put("torch-fidelity", "0.3.0");
            exact.put("xarray", "2025.1.2");
            exact.put("scipy", "1.15.2");
            exact.put("bioimageio.core", "0.7.0");
            exact.put("biapy", "3.5.10");

            minimum.put("pooch", "1.8.1");
            minimum.put("imagecodecs", "2024.1.1");
            minimum.put("h5py", "3.9.0");
            minimum.put("torchinfo", "1.8.0");
            minimum.put("pandas", "1.5.3");
            minimum.put("fill-voids", "2.0.6");
            minimum.put("edt", "2.3.2");
            minimum.put("tqdm", "4.66.1");
            minimum.put("yacs", "0.1.8");
            minimum.put("zarr", "2.16.1");
            minimum.put("pydot", "1.4.2");
            minimum.put("matplotlib", "3.7.1");
            minimum.put("imgaug", "0.4.0");
            minimum.put("tensorboardX", "2.6.2.2");
            minimum.put("scikit-learn", "1.4.0");
            minimum.put("opencv-python", "4.8.0.76");
            minimum.put("scikit-image", "0.21.0");

            present.add("careamics");
            present.add("appose");
            present.add("numpy");
        } else {
            // Standard branch
            exact.put("timm", "1.0.14");
            exact.put("pytorch-msssim", "1.0.0");
            exact.put("torchmetrics", "1.4.3");
            exact.put("cellpose", "3.1.1.1");
            exact.put("scipy", "1.15.2");
            exact.put("torch-fidelity", "0.3.0");
            exact.put("biapy", "3.5.10");

            present.add("careamics");
            present.add("appose");
        }

        final StringBuilder code = new StringBuilder();

        code.append("import sys").append(System.lineSeparator());
        code.append("from importlib.metadata import version, PackageNotFoundError").append(System.lineSeparator());
        code.append("from packaging.version import Version").append(System.lineSeparator());
        code.append("exact = ").append(toPythonDict(exact)).append(System.lineSeparator());
        code.append("minimum = ").append(toPythonDict(minimum)).append(System.lineSeparator());
        code.append("present = ").append(toPythonList(present)).append(System.lineSeparator());
        code.append("problems = []").append(System.lineSeparator());
        code.append("ok = True").append(System.lineSeparator());

        code.append("if sys.version_info[:2] != (3, 10):").append(System.lineSeparator());
        code.append("    ok = False").append(System.lineSeparator());
        code.append("    problems.append('python==3.10 expected, found ' + sys.version.split()[0])")
                .append(System.lineSeparator());

        code.append("for name, expected in exact.items():").append(System.lineSeparator());
        code.append("    try:").append(System.lineSeparator());
        code.append("        found = version(name)").append(System.lineSeparator());
        code.append("    except PackageNotFoundError:").append(System.lineSeparator());
        code.append("        ok = False").append(System.lineSeparator());
        code.append("        problems.append(name + ' not installed')").append(System.lineSeparator());
        code.append("        continue").append(System.lineSeparator());
        code.append("    if found != expected:").append(System.lineSeparator());
        code.append("        ok = False").append(System.lineSeparator());
        code.append("        problems.append(name + '==' + expected + ' expected, found ' + found)")
                .append(System.lineSeparator());

        code.append("for name, expected in minimum.items():").append(System.lineSeparator());
        code.append("    try:").append(System.lineSeparator());
        code.append("        found = version(name)").append(System.lineSeparator());
        code.append("    except PackageNotFoundError:").append(System.lineSeparator());
        code.append("        ok = False").append(System.lineSeparator());
        code.append("        problems.append(name + ' not installed')").append(System.lineSeparator());
        code.append("        continue").append(System.lineSeparator());
        code.append("    if Version(found) < Version(expected):").append(System.lineSeparator());
        code.append("        ok = False").append(System.lineSeparator());
        code.append("        problems.append(name + '>=' + expected + ' expected, found ' + found)")
                .append(System.lineSeparator());

        code.append("for name in present:").append(System.lineSeparator());
        code.append("    try:").append(System.lineSeparator());
        code.append("        version(name)").append(System.lineSeparator());
        code.append("    except PackageNotFoundError:").append(System.lineSeparator());
        code.append("        ok = False").append(System.lineSeparator());
        code.append("        problems.append(name + ' not installed')").append(System.lineSeparator());

        // Legacy macOS also constrained numpy<2 in the original Java logic.
        if (true) {
            code.append("try:").append(System.lineSeparator());
            code.append("    numpy_found = version('numpy')").append(System.lineSeparator());
            code.append("    if Version(numpy_found) >= Version('2'):").append(System.lineSeparator());
            code.append("        ok = False").append(System.lineSeparator());
            code.append("        problems.append('numpy<2 expected, found ' + numpy_found)")
                    .append(System.lineSeparator());
            code.append("except PackageNotFoundError:").append(System.lineSeparator());
            code.append("    ok = False").append(System.lineSeparator());
            code.append("    problems.append('numpy not installed')").append(System.lineSeparator());
        }

        code.append("task.outputs['installed'] = ok").append(System.lineSeparator());
        code.append("task.outputs['problems'] = problems").append(System.lineSeparator());

        return code.toString();
    }

    private static void addTorchRequirements(
            final PixiEnvironmentSpec spec,
            final Map<String, String> exact) {
        final String env = spec.getSelectedEnvironment();

        if ("win-no-cuda".equals(env) || "win-cuda".equals(env)) {
            exact.put("torch", "2.4.1");
            exact.put("torchvision", "0.19.1");
        } else if ("linux-x86_64-no-cuda".equals(env) || "linux-x86_64-cuda".equals(env)) {
            exact.put("torch", "2.4.0");
            exact.put("torchvision", "0.19.0");
        } else if ("macos-x86_64".equals(env) || "macos-x86_64-legacy".equals(env)) {
            exact.put("torch", "2.2.2");
            exact.put("torchvision", "0.17.2");
        } else if ("macos-arm64".equals(env) || "macos-arm64-legacy".equals(env)) {
            exact.put("torch", "2.4.0");
            exact.put("torchvision", "0.19.0");
        } else {
            throw new IllegalArgumentException("Unknown environment: " + env);
        }
    }

    private static String toPythonDict(final Map<String, String> map) {
        final StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, String> entry : map.entrySet()) {
            if (!first) {
                sb.append(", ");
            }
            sb.append("'").append(escapePython(entry.getKey())).append("': ");
            sb.append("'").append(escapePython(entry.getValue())).append("'");
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }

    private static String toPythonList(final Set<String> values) {
        final StringBuilder sb = new StringBuilder();
        sb.append("[");
        boolean first = true;
        for (String value : values) {
            if (!first) {
                sb.append(", ");
            }
            sb.append("'").append(escapePython(value)).append("'");
            first = false;
        }
        sb.append("]");
        return sb.toString();
    }

    private static String escapePython(final String value) {
        return value.replace("\\", "\\\\").replace("'", "\\'");
    }
}