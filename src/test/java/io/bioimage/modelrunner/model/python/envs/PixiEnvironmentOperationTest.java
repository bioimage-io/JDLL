package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.system.PlatformDetection;

public class PixiEnvironmentOperationTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void resolutionIsSharedWithinAnOperationButNotBetweenThreadsOrRuns() throws Exception {
        PixiEnvironmentSpec spec = spec("shared", "manifest");
        AtomicInteger resolutions = new AtomicInteger();
        java.util.function.Supplier<PixiEnvironmentSpec> resolver = () -> {
            resolutions.incrementAndGet();
            return spec;
        };
        try (EnvironmentSetupLog.Scope scope = EnvironmentSetupLog.open(null, null)) {
            assertSame(spec, PixiEnvironmentManager.resolveForOperation("pytorch", resolver));
            assertSame(spec, PixiEnvironmentManager.resolveForOperation("pytorch", resolver));
            assertEquals(1, resolutions.get());
            Thread other = new Thread(() -> PixiEnvironmentManager.resolveForOperation("pytorch", resolver));
            other.start();
            other.join();
            assertEquals(2, resolutions.get());
            try (PixiEnvironmentManager.Operation nested = PixiEnvironmentManager.beginOperation()) {
                PixiEnvironmentManager.resolveForOperation("pytorch", resolver);
                assertEquals(3, resolutions.get());
            }
            PixiEnvironmentManager.resolveForOperation("pytorch", resolver);
            assertEquals("Closing nested operations restores the outer scope", 3, resolutions.get());
        }
        try (PixiEnvironmentManager.Operation next = PixiEnvironmentManager.beginOperation()) {
            PixiEnvironmentManager.resolveForOperation("pytorch", resolver);
            assertEquals(4, resolutions.get());
        }
    }

    @Test public void installerAndWorkerReuseOneSuccessfulBuildUntilTheOperationCloses() throws Exception {
        PixiEnvironmentSpec spec = spec("reuse", "manifest");
        AtomicInteger builds = new AtomicInteger();
        try (EnvironmentSetupLog.Scope scope = EnvironmentSetupLog.open(null, null)) {
            Environment first = open(spec, builds);
            assertSame(first, open(new PixiEnvironmentSpec("default", "manifest", spec.getEnvironmentDirectory()), builds));
            assertSame(first, PixiEnvironmentManager.openEnvironment(spec, null));
            assertTrue(PixiEnvironmentManager.isInstalled(spec));
            assertEquals(1, builds.get());
        }
        try (PixiEnvironmentManager.Operation next = PixiEnvironmentManager.beginOperation()) {
            open(spec, builds);
            assertEquals("The next operation must validate again", 2, builds.get());
        }
        open(spec, builds);
        open(spec, builds);
        assertEquals("Unscoped library calls must not use a persistent cache", 4, builds.get());
    }

    @Test public void changedOrDeletedEnvironmentFilesInvalidateReuse() throws Exception {
        PixiEnvironmentSpec spec = spec("invalidation", "manifest");
        AtomicInteger builds = new AtomicInteger();
        try (PixiEnvironmentManager.Operation operation = PixiEnvironmentManager.beginOperation()) {
            open(spec, builds);
            for (String path : Arrays.asList("pixi.toml", "appose.json")) {
                write(new File(spec.getEnvironmentDirectory(), path), "modified");
                open(spec, builds);
            }
            File prefix = prefix(spec);
            Files.delete(new File(prefix, python()).toPath());
            open(spec, builds);
            Files.delete(new File(prefix, "conda-meta").toPath());
            open(spec, builds);
            Files.delete(new File(spec.getEnvironmentDirectory(), "appose.json").toPath());
            open(spec, builds);
            Files.delete(new File(spec.getEnvironmentDirectory(), "pixi-fixture").toPath());
            open(spec, builds);
            assertEquals(7, builds.get());
            PixiEnvironmentSpec changed = new PixiEnvironmentSpec("default", "new manifest", spec.getEnvironmentDirectory());
            open(changed, builds);
            assertEquals(8, builds.get());
            assertSame(open(changed, builds), open(changed, builds));
            assertEquals(8, builds.get());
        }
    }

    @Test public void manualDependenciesAndSelectedEnvironmentArePartOfTheIdentity() throws Exception {
        PixiEnvironmentSpec base = spec("identity", "manifest");
        PixiEnvironmentSpec manual = new PixiEnvironmentSpec("default", "manifest", base.getEnvironmentDirectory(),
                Collections.singletonList("package==1"));
        PixiEnvironmentSpec other = new PixiEnvironmentSpec("other", "manifest", base.getEnvironmentDirectory());
        AtomicInteger builds = new AtomicInteger();
        try (PixiEnvironmentManager.Operation operation = PixiEnvironmentManager.beginOperation()) {
            Environment first = open(base, builds);
            Environment withManual = open(manual, builds);
            assertNotSame(first, withManual);
            assertSame(withManual, open(manual, builds));
            Files.delete(new File(base.getEnvironmentDirectory(), ".jdll-no-deps").toPath());
            assertNotSame(withManual, open(manual, builds));
            assertNotSame(first, open(other, builds));
            assertEquals(4, builds.get());
        }
    }

    @Test public void failuresAreNotCachedAndConcurrentOperationsAreIndependent() throws Exception {
        PixiEnvironmentSpec spec = spec("failure", "manifest");
        AtomicInteger builds = new AtomicInteger();
        try (PixiEnvironmentManager.Operation operation = PixiEnvironmentManager.beginOperation()) {
            assertThrows(BuildException.class, () -> PixiEnvironmentManager.openEnvironment(spec, null, (s, log) -> {
                builds.incrementAndGet();
                throw new BuildException("injected failure");
            }));
            Environment successful = open(spec, builds);
            assertEquals(2, builds.get());
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread other = new Thread(() -> {
                try (PixiEnvironmentManager.Operation separate = PixiEnvironmentManager.beginOperation()) {
                    assertNotSame(successful, open(spec, builds));
                } catch (Throwable error) {
                    failure.set(error);
                }
            });
            other.start();
            other.join();
            if (failure.get() != null) throw new AssertionError(failure.get());
            assertSame(successful, open(spec, builds));
            assertEquals(3, builds.get());
        }
    }

    private PixiEnvironmentSpec spec(String directory, String manifest) throws IOException {
        return new PixiEnvironmentSpec("default", manifest, temporary.newFolder(directory));
    }

    private static Environment open(PixiEnvironmentSpec spec, AtomicInteger builds) throws BuildException {
        return PixiEnvironmentManager.openEnvironment(spec, null, (requested, log) -> {
            builds.incrementAndGet();
            try {
                Files.createDirectories(new File(prefix(requested), "conda-meta").toPath());
                write(new File(prefix(requested), python()), "python fixture");
                write(new File(requested.getEnvironmentDirectory(), "pixi.toml"), requested.getPixiTomlContent());
                write(new File(requested.getEnvironmentDirectory(), "appose.json"), "builder state");
                write(new File(requested.getEnvironmentDirectory(), "pixi-fixture"), "pixi fixture");
                if (!requested.getManualNoDeps().isEmpty()) {
                    write(new File(requested.getEnvironmentDirectory(), ".jdll-no-deps"),
                            requested.getPixiTomlContent() + "\n" + requested.getManualNoDeps());
                }
            } catch (IOException e) {
                throw new BuildException(e);
            }
            return (Environment) Proxy.newProxyInstance(Environment.class.getClassLoader(), new Class<?>[] {Environment.class},
                    (proxy, method, args) -> {
                        if (method.getName().equals("launchArgs")) return Collections.singletonList(
                                new File(requested.getEnvironmentDirectory(), "pixi-fixture").getAbsolutePath());
                        throw new AssertionError("Reuse must not start a worker: " + method);
                    });
        });
    }

    private static File prefix(PixiEnvironmentSpec spec) {
        return new File(spec.getEnvironmentDirectory(), ".pixi/envs/" + spec.getSelectedEnvironment());
    }

    private static String python() { return PlatformDetection.isWindows() ? "python.exe" : "bin/python"; }

    private static void write(File file, String contents) throws IOException {
        Files.createDirectories(file.getParentFile().toPath());
        Files.write(file.toPath(), contents.getBytes(StandardCharsets.UTF_8));
    }
}
