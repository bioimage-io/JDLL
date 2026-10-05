/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.model.special.stardist;

import static org.junit.Assert.*;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.apposed.appose.Service;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class StarDistShutdownTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test(timeout = 5000)
    public void removesWorkspaceAfterGracefulExitWithoutKilling() throws Exception {
        Worker worker = new Worker(true);
        Path workspace = workspace();
        StarDist.closeTrainingService(worker, workspace, null);
        assertTrue(worker.closed);
        assertFalse(worker.killed);
        awaitCleanup(workspace);
    }

    @Test(timeout = 5000)
    public void alreadyStoppedWorkerNeedsNoCloseOrKill() throws Exception {
        Worker worker = new Worker(false);
        worker.exited.countDown();
        Path workspace = workspace();
        StarDist.closeTrainingService(worker, workspace, null);
        assertFalse(worker.closed);
        assertFalse(worker.killed);
        awaitCleanup(workspace);
    }

    @Test(timeout = 5000)
    public void timeoutKillsButDefersDeletionUntilExit() throws Exception {
        Worker worker = new Worker(false);
        Path workspace = workspace();
        try {
            long start = System.nanoTime();
            StarDist.closeTrainingService(worker, workspace, null);
            assertTrue(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start) < 2000);
            assertTrue(worker.closed);
            assertTrue(worker.killed);
            assertTrue(worker.waiting.await(1, TimeUnit.SECONDS));
            assertTrue(Files.exists(workspace.resolve("mapped-data")));
        } finally {
            worker.exited.countDown();
            awaitCleanup(workspace);
        }
    }

    @Test(timeout = 5000)
    public void interruptedCallerKeepsInterruptAndStillStopsWorker() throws Exception {
        Worker worker = new Worker(false);
        Path workspace = workspace();
        try {
            Thread.currentThread().interrupt();
            StarDist.closeTrainingService(worker, workspace, null);
            assertTrue(Thread.currentThread().isInterrupted());
            assertTrue(worker.killed);
            assertTrue(Files.exists(workspace.resolve("mapped-data")));
        } finally {
            Thread.interrupted();
            worker.exited.countDown();
            awaitCleanup(workspace);
        }
    }

    private Path workspace() throws Exception {
        Path path = temporaryFolder.newFolder().toPath();
        Files.createFile(path.resolve("mapped-data"));
        return path;
    }

    private static void awaitCleanup(Path path) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (Files.exists(path) && System.nanoTime() < deadline) Thread.sleep(10);
        assertFalse(Files.exists(path));
    }

    /** A worker whose OS exit can lag behind kill(), without starting a Python environment. */
    private static final class Worker extends Service {
        final CountDownLatch exited = new CountDownLatch(1);
        final CountDownLatch waiting = new CountDownLatch(1);
        final boolean graceful;
        volatile boolean closed;
        volatile boolean killed;

        Worker(boolean graceful) {
            super(new File("."), "unused");
            this.graceful = graceful;
        }

        @Override public boolean isAlive() { return exited.getCount() != 0; }
        @Override public void close() {
            closed = true;
            if (graceful) exited.countDown();
        }
        @Override public void kill() { killed = true; }
        @Override public int waitFor() throws InterruptedException {
            waiting.countDown();
            exited.await();
            return 0;
        }
    }
}
