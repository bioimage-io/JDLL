package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;
import java.io.File;
import java.lang.reflect.Proxy;
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
