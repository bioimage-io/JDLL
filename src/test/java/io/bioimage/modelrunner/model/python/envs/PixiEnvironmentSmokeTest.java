package io.bioimage.modelrunner.model.python.envs;

import static org.junit.Assert.*;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.junit.Assume;
import org.junit.Test;

/** Opt-in integration test against the exact prefix installed by the model CI job. */
public class PixiEnvironmentSmokeTest {
    @Test(timeout = 300000) public void apposeWorkerExecutesNumpyAndSharedMemory() throws Exception {
        String path = System.getenv("JDLL_PIXI_SMOKE_MANIFEST");
        Assume.assumeNotNull(path);
        File manifest = new File(path);
        String selected = System.getenv("JDLL_PIXI_SMOKE_ENVIRONMENT");
        PixiEnvironmentSpec spec = new PixiEnvironmentSpec(selected == null ? "default" : selected,
                new String(Files.readAllBytes(manifest.toPath()), StandardCharsets.UTF_8), manifest.getParentFile());
        Environment environment = PixiEnvironmentManager.openEnvironment(spec, System.out::println);
        try (Service service = environment.python()) {
            Task task = service.task("import numpy as np\n"
                    + "from multiprocessing import shared_memory\n"
                    + "shm = shared_memory.SharedMemory(create=True, size=16)\n"
                    + "try:\n"
                    + "    values = np.ndarray((4,), dtype=np.float32, buffer=shm.buf)\n"
                    + "    values[:] = [1, 2, 3, 4]\n"
                    + "    task.outputs['sum'] = float(values.sum())\n"
                    + "finally:\n"
                    + "    shm.close()\n"
                    + "    shm.unlink()\n");
            task.waitFor();
            assertEquals(task.error, TaskStatus.COMPLETE, task.status);
            assertEquals(10, ((Number) task.outputs.get("sum")).doubleValue(), 0);
        }
    }
}
