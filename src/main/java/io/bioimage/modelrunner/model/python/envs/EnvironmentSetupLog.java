package io.bioimage.modelrunner.model.python.envs;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.File;

/** Run-scoped setup reporting, including builds triggered inside model factories. */
public final class EnvironmentSetupLog {
    private static final ThreadLocal<Scope> CURRENT = new ThreadLocal<>();

    private EnvironmentSetupLog() {}

    /** Bind callbacks to the calling worker only; always close the returned scope. */
    public static Scope open(Consumer<String> messages, Consumer<Progress> progress) {
        Scope scope = new Scope(messages, progress, CURRENT.get());
        CURRENT.set(scope);
        return scope;
    }

    public static final class Scope implements AutoCloseable {
        private final Consumer<String> messages;
        private final Consumer<Progress> progress;
        private final Scope previous;
        private final Set<String> ready = new HashSet<>();

        private Scope(Consumer<String> messages, Consumer<Progress> progress, Scope previous) {
            this.messages = messages;
            this.progress = progress;
            this.previous = previous;
        }

        @Override public void close() {
            try {
                if (progress != null) progress.accept(null);
            } finally {
                if (previous == null) CURRENT.remove();
                else CURRENT.set(previous);
            }
        }
    }

    /** Immutable snapshot. A non-positive maximum means indeterminate, not zero percent. */
    public static final class Progress {
        public final String title;
        public final long current;
        public final long maximum;
        public final String detail;

        public Progress(String title, long current, long maximum) {
            this(title, current, maximum, "");
        }

        private Progress(String title, long current, long maximum, String detail) {
            this.title = title;
            this.current = Math.max(0, current);
            this.maximum = maximum;
            this.detail = detail;
        }

        public String format(int frame) {
            String spinner = new String[] {"/", "|", "\\", "|"}[Math.floorMod(frame, 4)];
            String label = title + (detail.isEmpty() ? "" : " - " + detail);
            if (maximum <= 0) return label + " " + spinner;
            double fraction = Math.min(1.0, (double) current / maximum);
            StringBuilder bar = new StringBuilder("[");
            for (int i = 0; i < 20; i++) bar.append(i < (int) (20 * fraction) ? '#' : '.');
            bar.append("] ").append(String.format(Locale.ROOT, "%.1f%%", 100 * fraction));
            if (!title.equals("Downloading Pixi")) bar.append(" (").append(current).append('/').append(maximum).append(" packages)");
            return label + " " + bar + " " + spinner;
        }
    }

    static final class Reporter {
        private static final Pattern PACKAGE = Pattern.compile(
                "^\\s*((?:Downloading|Building|Installing) [A-Za-z0-9][A-Za-z0-9_.-]*)\\s*(?:\\(|\\[|$)");
        private static final Pattern ANSI = Pattern.compile("\\x1B\\[[;\\d]*m");
        private static final Pattern PYPI_REQUIRED = Pattern.compile("pixi_install_pypi: (\\d+) of (\\d+) required packages");
        private static final Pattern PYPI_COUNT = Pattern.compile("pixi_install_pypi: (Prepared|Installed) (\\d+) packages?\\b");
        private static final Pattern INSTALLING = Pattern.compile("pixi_install_pypi: \\*installing\\* .*: ([A-Za-z0-9_.-]+)(?: [^\\s]+)?\\s*$");
        private final Scope scope = CURRENT.get();
        private final Consumer<String> messages;
        private final String environment;
        private final boolean previouslyReady;
        private final CondaInstallProgress conda;
        private boolean nativeCondaProgress;
        private long pypiToInstall;
        private Progress latest;
        private boolean finished;

        Reporter(String environment, Consumer<String> fallback) {
            this(environment, fallback, null, "default");
        }

        Reporter(String environment, Consumer<String> fallback, File directory, String selectedEnvironment) {
            this.environment = environment;
            messages = scope != null ? scope.messages : fallback;
            previouslyReady = scope != null && scope.ready.contains(environment);
            conda = new CondaInstallProgress(directory, selectedEnvironment, this::condaProgress);
            if (!previouslyReady) phase("Preparing required Python environment", 0, -1);
        }

        void announceSetup(boolean installed) {
            if (previouslyReady) return;
            message((installed
                    ? "Checking the required Python environment; installing or updating dependencies if needed: "
                    : "Installing the required Python environment (missing or out of date): ") + environment);
        }

        synchronized void progress(String title, long current, long maximum) {
            if (finished) return;
            if ("Downloading pixi".equalsIgnoreCase(title)) {
                phase("Downloading Pixi", current, maximum);
                if (maximum > 0 && current >= maximum) phase("Resolving dependencies", 0, -1);
            } else if ("Solving".equals(title)) {
                // Appose's solve denominator is inferred from log records, not work remaining.
                phase("Resolving dependencies", 0, -1);
            } else if ("Downloading PyPI packages".equals(title)) {
                phase("Preparing PyPI packages", current, maximum);
            } else if ("Done".equals(title)) {
                phase("Finalizing Python environment", 0, -1);
            } else {
                if ("Installing conda packages".equals(title) && maximum > 0) {
                    nativeCondaProgress = true;
                    conda.close();
                }
                phase(title, current, maximum);
            }
        }

        synchronized void output(String line) {
            if (finished) return;
            // Deliberately whitelist landmarks. Resolver/debug/stderr text stays on disk.
            Matcher required = PYPI_REQUIRED.matcher(line);
            Matcher count = PYPI_COUNT.matcher(line);
            if (line.contains("fetched") && line.contains("records") && line.contains("solve{")) {
                if (latest == null || latest.title.startsWith("Preparing required")) phase("Resolving dependencies", 0, -1);
            } else if (line.contains("resolved conda environment for")) {
                phase("Resolving Python dependencies", 0, -1);
            } else if (line.contains("conda_prefix: updating prefix for")) {
                if (latest == null || !latest.title.equals("Installing conda packages")) phase("Installing conda packages", 0, -1);
                if (!nativeCondaProgress) conda.start();
            } else if (line.contains("conda_metadata: Prefix file updated")) {
                conda.close();
                if (!nativeCondaProgress && conda.total() > 0) phase("Installing conda packages", conda.total(), conda.total());
            } else if (required.find()) {
                pypiToInstall = Math.max(0, Long.parseLong(required.group(2)) - Long.parseLong(required.group(1)));
                phase("Preparing PyPI packages", 0, pypiToInstall > 0 ? pypiToInstall : -1);
            } else if (count.find() && pypiToInstall > 0) {
                phase("Prepared".equals(count.group(1)) ? "Preparing PyPI packages" : "Installing PyPI packages",
                        Long.parseLong(count.group(2)), pypiToInstall);
            } else if (latest != null) {
                Matcher packageName = PACKAGE.matcher(ANSI.matcher(line).replaceAll(""));
                Matcher installing = INSTALLING.matcher(line);
                String detail = packageName.find() ? packageName.group(1)
                        : installing.find() ? "Installing " + installing.group(1) : null;
                if (detail != null) {
                    latest = new Progress(latest.title, latest.current, latest.maximum, detail);
                    if (scope != null && scope.progress != null) scope.progress.accept(latest);
                }
            }
        }

        synchronized void phase(String title, long current, long maximum) {
            if (finished || title == null || title.trim().isEmpty()) return;
            if (latest != null && !latest.title.equals(title)) archive();
            String detail = latest != null && latest.title.equals(title) ? latest.detail : "";
            latest = new Progress(title, current, maximum, detail);
            if (scope != null && scope.progress != null) scope.progress.accept(latest);
        }

        synchronized void finish(boolean success, String diagnostics) {
            if (finished) return;
            finished = true;
            conda.close();
            if (success) {
                archive();
                if (!previouslyReady || latest != null) message("Python environment ready: " + environment);
                if (scope != null) scope.ready.add(environment);
            } else {
                message((Thread.currentThread().isInterrupted() ? "Python environment setup interrupted." : "Python environment setup failed.")
                        + " Diagnostics: " + diagnostics);
            }
            latest = null;
            if (scope != null && scope.progress != null) scope.progress.accept(null);
        }

        private synchronized void condaProgress(long current, long maximum) {
            if (conda.isActive()) phase("Installing conda packages", current, maximum);
        }

        private void archive() {
            if (latest != null && !latest.title.startsWith("Preparing required")
                    && !latest.title.startsWith("Finalizing")) {
                if (latest.title.startsWith("Resolving")) {
                    message(latest.title.replaceFirst("Resolving", "Resolved") + ".");
                    return;
                }
                if (latest.title.equals("Preparing PyPI packages") && latest.current < latest.maximum) {
                    message("PyPI package preparation completed (cached packages can be reused).");
                    return;
                }
                String text = latest.format(0);
                message(text.substring(0, text.length() - 2));
            }
        }

        void message(String text) {
            if (messages != null) messages.accept(text);
        }
    }
}
