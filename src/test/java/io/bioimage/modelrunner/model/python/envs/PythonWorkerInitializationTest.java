package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;
import java.io.File;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.junit.Test;

public class PythonWorkerInitializationTest {
    @Test public void registersNumpyInitializationBeforeReturningAnUnstartedWorker() {
        assertInitialization(null, "import numpy as np");
    }

    @Test public void backendInitializationKeepsNumpyFirstInOneInitCall() {
        assertInitialization("import numba", "import numpy as np\nimport numba");
    }

    @Test public void reportsStartupBeforeCreatingTheWorker() {
        List<String> messages = new ArrayList<>();
        Service worker = new Service(new File("."), "unused");
        Environment environment = (Environment) Proxy.newProxyInstance(Environment.class.getClassLoader(),
                new Class<?>[] {Environment.class}, (proxy, method, args) -> {
                    assertEquals("python", method.getName());
                    assertEquals(1, messages.size());
                    assertTrue(messages.get(0).startsWith("Starting Python worker"));
                    return worker;
                });
        try (EnvironmentSetupLog.Scope scope = EnvironmentSetupLog.open(messages::add, null)) {
            assertSame(worker, PixiEnvironmentManager.pythonService(environment));
            assertFalse(worker.isAlive());
        }
    }

    private void assertInitialization(String extra, String expected) {
        AtomicReference<String> initialization = new AtomicReference<>();
        Service worker = new Service(new File("."), "unused") {
            @Override public synchronized Service init(String script) {
                assertFalse(isAlive());
                assertNull("Appose init replaces the previous script", initialization.get());
                initialization.set(script);
                return this;
            }
        };
        Environment environment = (Environment) Proxy.newProxyInstance(Environment.class.getClassLoader(),
                new Class<?>[] {Environment.class}, (proxy, method, args) -> {
                    assertEquals("python", method.getName());
                    return worker;
                });
        assertSame(worker, extra == null ? PixiEnvironmentManager.pythonService(environment)
                : PixiEnvironmentManager.pythonService(environment, extra));
        assertEquals(expected, initialization.get());
        assertFalse(worker.isAlive());
    }
}
