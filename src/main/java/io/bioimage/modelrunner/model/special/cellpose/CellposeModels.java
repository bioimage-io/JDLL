package io.bioimage.modelrunner.model.special.cellpose;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/** Registry and format dispatch shared by training, inference and model browsing. */
public final class CellposeModels {
    public static final String NATIVE = "cellpose3-cpnet";
    public static final List<String> OFFICIAL = java.util.Collections.unmodifiableList(
            Arrays.asList("cyto3", "cyto2", "cyto", "nuclei"));

    private CellposeModels() {}

    /** Resolve a native checkpoint or download an official pair into the host's shared models directory. */
    public static String resolve(String model, String modelsDir, Consumer<String> logConsumer)
            throws IOException, InterruptedException, ExecutionException {
        if (model == null || model.trim().isEmpty()) throw new IOException("Select a Cellpose base model.");
        String modelsDirectory = new File(modelsDir == null ? "models" : modelsDir).getAbsolutePath();
        File custom = new File(model);
        if (custom.exists()) {
            return weights(custom).getAbsolutePath();
        }
        if (!OFFICIAL.contains(model)) throw new IOException("Cellpose weights do not exist: " + model);
        String installed = Cellpose.findPretrainedModelInstalled(model, modelsDirectory);
        if (installed != null) {
            return installed;
        }
        accept(logConsumer, "Downloading Cellpose " + model + " weights.");
        ThrottledProgress progress = new ThrottledProgress(logConsumer);
        String resolved = Cellpose.donwloadPretrained(model, modelsDirectory,
                value -> progress.accept("Downloading Cellpose " + model + " weights: "
                        + Math.round(value * 1000.0) / 10.0 + "%"));
        progress.acceptNow("Downloading Cellpose " + model + " weights: 100.0%");
        return resolved;
    }

    private static void accept(Consumer<String> consumer, String message) {
        if (consumer != null) {
            consumer.accept(message);
        }
    }

    private static final class ThrottledProgress {
        private final Consumer<String> consumer;
        private long lastUpdate;

        private ThrottledProgress(Consumer<String> consumer) {
            this.consumer = consumer;
        }

        private void accept(String message) {
            long now = System.currentTimeMillis();
            if (consumer != null && now - lastUpdate >= 100L) {
                lastUpdate = now;
                consumer.accept(message);
            }
        }

        private void acceptNow(String message) {
            if (consumer != null) {
                lastUpdate = System.currentTimeMillis();
                consumer.accept(message);
            }
        }
    }

    public static File familyDirectory(String modelsDir) {
        return new File(modelsDir == null ? "models" : modelsDir, "cellpose").getAbsoluteFile();
    }

    private static JsonObject manifest(File dir) throws IOException {
        try (Reader reader = Files.newBufferedReader(new File(dir, "model.json").toPath(), StandardCharsets.UTF_8)) {
            JsonObject json = new Gson().fromJson(reader, JsonObject.class);
            if (json == null || json.get("format_version").getAsInt() != 1
                    || !NATIVE.equals(json.get("backend").getAsString()))
                throw new IOException("Unsupported Cellpose model format in " + dir);
            if (!json.has("weights") || !json.get("weights").isJsonPrimitive())
                throw new IOException("Missing weight filename in " + dir);
            com.google.gson.JsonArray channels = json.getAsJsonArray("channels");
            if (channels == null || channels.size() != 2 || channels.get(0).getAsInt() < 0
                    || channels.get(1).getAsInt() < 0) throw new IOException("Invalid channels in " + dir);
            return json;
        } catch (RuntimeException e) {
            throw new IOException("Invalid Cellpose model metadata in " + dir, e);
        }
    }

    public static boolean isSelectable(String value) {
        if (value == null || value.trim().isEmpty()) return false;
        if (OFFICIAL.contains(value)) return true;
        File file = new File(value);
        if (!file.exists()) return false;
        try { weights(file); return true; } catch (IOException e) { return false; }
    }

    public static File weights(File path) throws IOException {
        if (path.isFile()) return path.getCanonicalFile();
        JsonObject info = manifest(path);
        File weights = new File(path, info.get("weights").getAsString()).getCanonicalFile();
        if (path.getCanonicalFile().equals(weights.getParentFile()) && !weights.isFile()
                && io.bioimage.modelrunner.gui.custom.training.TrainingContinuation.find(path.getPath()) != null) {
            File last = new File(path, "last.pt");
            if (last.isFile()) return last.getCanonicalFile();
        }
        if (!path.getCanonicalFile().equals(weights.getParentFile()) || !weights.isFile())
            throw new IOException("Missing or invalid Cellpose weights in " + path);
        return weights;
    }

    public static int[] channels(File path) throws IOException {
        File dir = path.isDirectory() ? path : path.getAbsoluteFile().getParentFile();
        if (!new File(dir, "model.json").isFile()) return new int[] {0, 0};
        com.google.gson.JsonArray values = manifest(dir).getAsJsonArray("channels");
        return new int[] {values.get(0).getAsInt(), values.get(1).getAsInt()};
    }

    public static LinkedHashMap<String,String> entries(String modelsDir) {
        LinkedHashMap<String,String> entries = new LinkedHashMap<>();
        for (String name : OFFICIAL) entries.put(name, name);
        File[] dirs = familyDirectory(modelsDir).listFiles(File::isDirectory);
        if (dirs != null) {
            Arrays.sort(dirs);
            for (File dir : dirs) {
                if (!new File(dir, "model.json").isFile()) continue;
                try {
                    weights(dir);
                    entries.put("[Custom] " + dir.getName(), dir.getAbsolutePath());
                } catch (IOException ignored) { /* Incomplete or unsupported run. */ }
            }
        }
        return entries;
    }
}
