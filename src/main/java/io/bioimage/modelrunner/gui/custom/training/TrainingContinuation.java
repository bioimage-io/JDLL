package io.bioimage.modelrunner.gui.custom.training;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/** Lightweight view of the trainer's atomic checkpoint index for the model chooser. */
public final class TrainingContinuation {
    public final Path directory;
    public final int epoch;
    public final int budget;
    public final String checkpoint;
    private JsonObject legacy;
    private String dataset;

    public String dataset() { return dataset; }

    public static Object legacyMetadata(String selection) {
        TrainingContinuation run = find(selection);
        return run == null || run.legacy == null ? null : new com.google.gson.Gson().fromJson(run.legacy, java.util.Map.class);
    }

    private TrainingContinuation(Path directory, int epoch, int budget, String checkpoint) {
        this.directory = directory;
        this.epoch = epoch;
        this.budget = budget;
        this.checkpoint = checkpoint;
    }

    public static TrainingContinuation find(String selection) {
        if (selection == null || selection.trim().isEmpty()) return null;
        File selected = new File(selection);
        if (!selected.exists()) return null;
        Path folder = selected.toPath().toAbsolutePath();
        if (!selected.isDirectory()) folder = folder.getParent();
        for (int i = 0; i < 2 && folder != null; i++, folder = folder.getParent()) {
            Path index = folder.resolve("continuation.json");
            try {
                boolean legacy = !Files.isRegularFile(index);
                JsonObject run = legacy ? OlderTrainingRun.inspect(folder) : JsonParser.parseString(Files.readString(index)).getAsJsonObject();
                if (run == null) continue;
                if ("complete".equals(run.get("status").getAsString())) return null;
                Path lock = folder.resolve(".training.lock");
                if (Files.isRegularFile(lock)) {
                    try (java.nio.channels.FileChannel channel = java.nio.channels.FileChannel.open(lock,
                            java.nio.file.StandardOpenOption.WRITE);
                         java.nio.channels.FileLock lease = channel.tryLock(0, 1, false)) {
                        if (lease == null) return null;
                    }
                }
                JsonObject slots = run.getAsJsonObject("checkpoints");
                for (String field : new String[]{"state", "weights"}) {
                    for (String kind : new String[]{"best", "last"}) {
                        JsonObject slot = slots.getAsJsonObject(kind);
                        if (slot == null || !slot.has("weights") || !Files.isRegularFile(folder.resolve(slot.get("weights").getAsString()))) continue;
                        if (slot.has(field) && Files.isRegularFile(folder.resolve(slot.get(field).getAsString()))) {
                            Path file = folder.resolve(slot.get(field).getAsString());
                            if ("state".equals(field) && slot.has("snapshot") && !Files.isRegularFile(folder.resolve(slot.get("snapshot").getAsString()))) continue;
                            if ("state".equals(field) && slot.has("state_bytes") && Files.size(file) != slot.get("state_bytes").getAsLong()) continue;
                            if (slot.has("companions") && "state".equals(field)) {
                                boolean missing = false;
                                for (com.google.gson.JsonElement item : slot.getAsJsonArray("companions"))
                                    missing |= !Files.isRegularFile(folder.resolve(item.getAsString()));
                                if (missing) continue;
                            }
                            TrainingContinuation result = new TrainingContinuation(folder, slot.get("epoch").getAsInt(), run.get("epochs").getAsInt(), kind);
                            if (run.has("dataset")) result.dataset = run.get("dataset").getAsString();
                            if (legacy) result.legacy = run;
                            return result;
                        }
                    }
                }
            } catch (IOException | RuntimeException ignored) { /* Imported or incomplete metadata. */ }
            return null;
        }
        return null;
    }
}
