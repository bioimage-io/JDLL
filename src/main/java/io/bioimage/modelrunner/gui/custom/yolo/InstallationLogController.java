package io.bioimage.modelrunner.gui.custom.yolo;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import javax.swing.SwingUtilities;
import javax.swing.Timer;

import io.bioimage.modelrunner.model.python.envs.EnvironmentSetupLog;
import io.bioimage.modelrunner.model.python.envs.EnvironmentSetupLog.Progress;

/** Coalesces arbitrarily frequent download callbacks into at most ten UI updates/second. */
final class InstallationLogController {
    private final AtomicReference<Progress> latest = new AtomicReference<>();
    private final Consumer<String> messages;
    private final Consumer<String> liveLine;
    private final Timer timer;
    private int frame;
    private long generation;
    private volatile boolean running;

    InstallationLogController(Consumer<String> messages, Consumer<String> liveLine) {
        this.messages = messages;
        this.liveLine = liveLine;
        timer = new Timer(100, e -> render());
    }

    EnvironmentSetupLog.Scope openScope() {
        final long run;
        synchronized (this) { run = ++generation; }
        return EnvironmentSetupLog.open(message -> {
            synchronized (this) { if (run != generation) return; }
            messages.accept(message);
        }, progress -> {
            synchronized (this) {
                if (run != generation) return;
                latest.set(progress);
            }
            // Starting is needed only once, not for every byte-progress callback.
            if (progress == null) SwingUtilities.invokeLater(() -> {
                synchronized (this) { if (run != generation) return; }
                if (latest.get() == null) { timer.stop(); running = false; liveLine.accept(null); }
            });
            else startTimer();
        });
    }

    private boolean startPending;

    private synchronized void startTimer() {
        if (startPending || running) return;
        startPending = true;
        SwingUtilities.invokeLater(() -> {
            synchronized (this) { startPending = false; }
            if (latest.get() != null && !timer.isRunning()) { running = true; render(); timer.start(); }
        });
    }

    private void render() {
        Progress progress = latest.get();
        if (progress != null) liveLine.accept(progress.format(frame++));
    }

    void stop() {
        synchronized (this) { generation++; latest.set(null); }
        timer.stop();
        running = false;
        liveLine.accept(null);
    }
}
