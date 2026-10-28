package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;

import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

public class EnvironmentSetupLogTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void setupIsAnnouncedBeforeProgressAndNotRepeatedForTheSameReadyEnvironment() {
        List<String> lines = new ArrayList<>();
        try (EnvironmentSetupLog.Scope scope = EnvironmentSetupLog.open(lines::add, progress -> {})) {
            EnvironmentSetupLog.Reporter missing = new EnvironmentSetupLog.Reporter("new-env", null);
            missing.announceSetup(false);
            assertEquals("Installing the required Python environment (missing or out of date): new-env", lines.get(0));
            missing.progress("Solving", 1, 1);
            missing.finish(true, "diagnostics.log");
            int count = lines.size();
            EnvironmentSetupLog.Reporter reused = new EnvironmentSetupLog.Reporter("new-env", null);
            reused.announceSetup(true);
            reused.finish(true, "diagnostics.log");
            assertEquals(count, lines.size());
            EnvironmentSetupLog.Reporter installed = new EnvironmentSetupLog.Reporter("existing-env", null);
            installed.announceSetup(true);
            assertEquals("Checking the required Python environment; installing or updating dependencies if needed: existing-env",
                    lines.get(lines.size() - 1));
            installed.finish(true, "diagnostics.log");
        }
    }

    @Test public void phasesUseHonestUnitsAndOnlyArchiveSummaries() {
        List<String> lines = new ArrayList<>();
        List<EnvironmentSetupLog.Progress> updates = new ArrayList<>();
        try (EnvironmentSetupLog.Scope scope = EnvironmentSetupLog.open(lines::add, updates::add)) {
            EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test-env", null);
            reporter.progress("Downloading pixi", 1, 4);
            assertTrue(updates.get(updates.size() - 1).format(0).contains("25.0%"));
            assertFalse(updates.get(updates.size() - 1).format(0).contains("packages"));
            for (int i = 0; i < 1000; i++) reporter.progress("Downloading pixi", 2, 4);
            assertEquals("Download ticks must not append UI/disk lines", 0, lines.size());
            reporter.progress("Downloading pixi", 4, 4);
            assertEquals("Resolving dependencies /", updates.get(updates.size() - 1).format(0));
            reporter.progress("Solving", 3, 2);
            assertFalse(updates.get(updates.size() - 1).format(0).contains("%"));
            reporter.output("DEBUG arbitrary stderr not intended for users");
            reporter.progress("Installing conda packages", 7, 10);
            String bar = updates.get(updates.size() - 1).format(2);
            assertTrue(bar, bar.contains("70.0% (7/10 packages)"));
            assertTrue(bar, bar.endsWith("\\"));
            reporter.progress("Downloading PyPI packages", 3, 6);
            assertEquals("Preparing PyPI packages", updates.get(updates.size() - 1).title);
            reporter.progress("Installing PyPI packages", 6, 6);
            reporter.progress("Done", 1, 1);
            reporter.finish(true, "diagnostics.log");
            assertNull(updates.get(updates.size() - 1));
            assertTrue(lines.get(lines.size() - 1).contains("environment ready"));
            assertFalse(lines.toString().contains("arbitrary stderr"));
            assertFalse(lines.toString().contains("Finalizing"));
            assertTrue(lines.toString().contains("100.0%"));
            int count = lines.size();
            new EnvironmentSetupLog.Reporter("test-env", null).finish(true, "diagnostics.log");
            assertEquals("A second no-op build in the same run should be silent", count, lines.size());
        }
        List<String> fallback = new ArrayList<>();
        new EnvironmentSetupLog.Reporter("other-env", fallback::add).finish(false, "diagnostics.log");
        assertEquals(1, fallback.size());
        assertTrue(fallback.get(0).contains("failed"));
        assertTrue(fallback.get(0).contains("diagnostics.log"));
    }

    @Test public void scopesAreNestedAndWorkerLocal() throws Exception {
        List<String> outer = new ArrayList<>(), inner = new ArrayList<>(), worker = new ArrayList<>();
        try (EnvironmentSetupLog.Scope a = EnvironmentSetupLog.open(outer::add, p -> {})) {
            try (EnvironmentSetupLog.Scope b = EnvironmentSetupLog.open(inner::add, p -> {})) {
                new EnvironmentSetupLog.Reporter("inner", null).finish(true, "");
            }
            Thread thread = new Thread(() -> new EnvironmentSetupLog.Reporter("worker", worker::add).finish(true, ""));
            thread.start();
            thread.join();
            new EnvironmentSetupLog.Reporter("outer", null).finish(true, "");
        }
        assertEquals(1, outer.size());
        assertEquals(1, inner.size());
        assertEquals(1, worker.size());
        assertTrue(outer.get(0).contains("outer"));
    }

    @Test public void currentPixiLockLayoutAndSingularPackageMessagesHaveCounts() throws Exception {
        File directory = temporary.newFolder("environment");
        File metadata = new File(directory, ".pixi/envs/default/conda-meta");
        Files.createDirectories(metadata.toPath());
        Files.write(new File(metadata, "python.json").toPath(), new byte[0]);
        String lock = "version: 6\nenvironments:\n  default:\n    packages:\n      " + CondaInstallProgress.platform()
                + ":\n      - conda: python.conda\n      - conda: pip.conda\n      - pypi: colorama.whl\n";
        File lockFile = new File(directory, "pixi.lock");
        Files.write(lockFile.toPath(), lock.getBytes(StandardCharsets.UTF_8));
        List<String> lines = new ArrayList<>();
        List<EnvironmentSetupLog.Progress> updates = new ArrayList<>();
        try (EnvironmentSetupLog.Scope scope = EnvironmentSetupLog.open(lines::add, updates::add)) {
            EnvironmentSetupLog.Reporter reporter = new EnvironmentSetupLog.Reporter("test", null, directory, "default");
            try {
                reporter.output("INFO update: resolved conda environment for environment 'default' 'linux-64'");
                assertEquals("Resolving Python dependencies", last(updates).title);
                reporter.output("DEBUG pixi_core::environment::conda_prefix: updating prefix for 'default'");
                assertEquals(2, last(updates).maximum);
                assertEquals(1, last(updates).current);
                reporter.output("DEBUG pixi_core::environment::conda_metadata: Prefix file updated with: 'prefix'");
                assertEquals(2, last(updates).current);
                reporter.output("DEBUG pixi_install_pypi: 0 of 1 required packages are considered installed and up-to-date");
                reporter.output("DEBUG pixi_install_pypi: *installing* from remote because no version is cached: colorama 0.4.6");
                assertEquals("Installing colorama", last(updates).detail);
                reporter.output("DEBUG pixi_install_pypi: *installing* cached version because cache is up-to-date: numpy");
                assertEquals("Installing numpy", last(updates).detail);
                reporter.output("INFO pixi_install_pypi: Prepared 1 package in 367ms");
                assertEquals(1, last(updates).current);
                assertEquals(1, last(updates).maximum);
                reporter.output("INFO pixi_install_pypi: Installed 1 package in 2ms");
                assertEquals("Installing PyPI packages", last(updates).title);
                assertEquals(1, last(updates).current);
                assertFalse(lines.toString().contains("DEBUG"));
            } finally {
                reporter.finish(true, "diagnostics");
            }
        }
        // YAML parsers must also accept the older, indented sequence representation.
        Files.write(lockFile.toPath(), lock.replace("      - ", "        - ").getBytes(StandardCharsets.UTF_8));
        try (CondaInstallProgress progress = new CondaInstallProgress(directory, "default", (c, m) -> {})) {
            assertEquals(2, progress.packageCount());
        }
    }

    private static EnvironmentSetupLog.Progress last(List<EnvironmentSetupLog.Progress> updates) {
        return updates.get(updates.size() - 1);
    }
}
