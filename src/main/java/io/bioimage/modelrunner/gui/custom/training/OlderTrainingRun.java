package io.bioimage.modelrunner.gui.custom.training;

import com.google.gson.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.Locale;

/** Recover evidence of an interrupted run from the files written before continuation.json existed. */
final class OlderTrainingRun {
    private static JsonObject json(Path folder, String name) throws IOException {
        Path p = folder.resolve(name);
        return Files.isRegularFile(p) ? JsonParser.parseString(Files.readString(p)).getAsJsonObject() : new JsonObject();
    }

    static JsonObject inspect(Path folder) throws IOException {
        JsonObject cfg = json(folder, "config.json"), slots = new JsonObject();
        int budget, last = 0, best = 0;
        if (Files.isRegularFile(folder.resolve("run.json")) && cfg.has("backend") && cfg.has("epochs")) {
            JsonObject status = json(folder, "run.json");
            if ("complete".equals(status.get("status").getAsString())) return null;
            budget = cfg.get("epochs").getAsInt();
            last = status.get("completed_epochs").getAsInt();
            JsonObject model = json(folder, "model.json");
            best = model.has("best_epoch") ? model.get("best_epoch").getAsInt() : 0;
            slot(folder, slots, "best", best, "best.pt", null);
            slot(folder, slots, "last", last, "last.pt", null);
        } else if (cfg.has("training") && Files.isRegularFile(folder.resolve("metrics.json"))) {
            budget = cfg.getAsJsonObject("training").get("epochs").getAsInt();
            JsonArray history = json(folder, "metrics.json").getAsJsonArray("history");
            if (history == null || history.size() == 0) return null;
            double top = Double.NEGATIVE_INFINITY, lowest = Double.POSITIVE_INFINITY;
            for (JsonElement value : history) {
                JsonObject row = value.getAsJsonObject();
                last = row.get("epoch").getAsInt();
                JsonObject selection = row.getAsJsonObject("validation_selection");
                if (selection != null && !selection.get("evaluated").getAsBoolean()) continue;
                JsonObject scores = row.has("full_validation") ? row.getAsJsonObject("full_validation") : row.getAsJsonObject("val_metrics");
                double score = selection != null ? selection.get("value").getAsDouble() : Double.NEGATIVE_INFINITY;
                if (selection == null) for (String key : new String[]{"panoptic_quality", "mean_dice", "dice"}) {
                    if (scores.has(key)) { score = scores.get(key).getAsDouble(); break; }
                }
                double loss = row.getAsJsonObject("val_losses").get("total_loss").getAsDouble();
                if (score > top || score == top && (selection == null || loss < lowest)) {
                    best = last; top = score; lowest = loss;
                }
            }
            if (last >= budget || finishedUnet(folder)) return null;
            slot(folder, slots, "best", best, "weights_best.pt", "weights_best.pt");
            // Older 2D runs could overwrite last with mid-epoch weights. Only best is known to be validated.
            if (!Files.exists(folder.resolve("weights_best.pt")) || Files.exists(folder.resolve("validation_state.json")))
                slot(folder, slots, "last", last, "weights_last.pt", "weights_last.pt");
        } else if (cfg.has("train_epochs") && Files.isRegularFile(folder.resolve("validation_metrics.json"))) {
            budget = cfg.get("train_epochs").getAsInt();
            double lowest = Double.POSITIVE_INFINITY;
            for (JsonElement value : json(folder, "validation_metrics.json").getAsJsonArray("history")) {
                JsonObject row = value.getAsJsonObject();
                last = row.get("epoch").getAsInt();
                double score = row.getAsJsonObject("losses").get("loss").getAsDouble();
                if (score < lowest) { lowest = score; best = last; }
            }
            if (last >= budget) return null;
            slot(folder, slots, "best", best, "weights_best.h5", String.format(Locale.ROOT,"training_state-%06d.index",best));
            slot(folder, slots, "last", last, "weights_last.h5", String.format(Locale.ROOT,"training_state-%06d.index",last));
        } else if (Files.isRegularFile(folder.resolve("config.yaml")) && Files.isRegularFile(folder.resolve("history.json"))) {
            if (Files.exists(folder.resolve("result.json"))) return null;
            JsonObject history = json(folder, "history.json");
            JsonArray epochs = history.getAsJsonArray("epochs"), scores = history.getAsJsonObject("metrics").getAsJsonArray("val/ap_0.75");
            if (epochs.size() == 0 || epochs.size() != scores.size()) return null;
            budget = 0;
            for (String line : Files.readAllLines(folder.resolve("config.yaml"))) {
                if (line.trim().startsWith("max_epochs:")) { budget = Integer.parseInt(line.split(":",2)[1].trim()); break; }
            }
            double high = Double.NEGATIVE_INFINITY;
            for (int i=0; i<epochs.size(); i++) {
                last = epochs.get(i).getAsInt();
                if (scores.get(i).getAsDouble() > high) { high = scores.get(i).getAsDouble(); best = last; }
            }
            if (last >= budget) return null;
            slot(folder, slots, "best",best,"checkpoints/best.ckpt","checkpoints/best.ckpt");
            slot(folder, slots, "last",last,"checkpoints/last.ckpt","checkpoints/last.ckpt");
        } else if (cfg.has("framework") && "yolo".equals(cfg.get("framework").getAsString())) {
            JsonObject status = json(folder, "training_results.json");
            if (status.has("status") && "completed".equals(status.get("status").getAsString())) return null;
            budget = cfg.getAsJsonObject("training").get("epochs").getAsInt();
            Path csv = folder.resolve("results.csv");
            if (!Files.isRegularFile(csv)) return null;
            java.util.List<String> rows = Files.readAllLines(csv);
            String[] columns = rows.get(0).trim().split(",");
            int fitness = -1;
            for (int i=0; i<columns.length; i++) if (columns[i].trim().equals("metrics/mAP50-95(B)")) fitness = i;
            double high = Double.NEGATIVE_INFINITY;
            for (int i=1; i<rows.size(); i++) {
                String[] values = rows.get(i).trim().split(",");
                if (values.length != columns.length) continue;
                last = Integer.parseInt(values[0].trim());
                if (fitness >= 0 && Double.parseDouble(values[fitness]) >= high) {
                    best = last; high = Double.parseDouble(values[fitness]);
                }
            }
            if (last >= budget) return null;
            slot(folder,slots,"best",best,"weights/best.pt",null);
            slot(folder,slots,"last",last,"weights/last.pt",null);
        } else return null;
        if (slots.size() == 0 || budget <= 0) return null;
        JsonObject run = new JsonObject();
        run.addProperty("epochs",budget);
        run.addProperty("status","interrupted");
        run.addProperty("directory",folder.toString());
        run.add("checkpoints",slots);
        return run;
    }

    private static boolean finishedUnet(Path dir) throws IOException {
        Path log = dir.resolve("training.log");
        if (!Files.isRegularFile(log)) return false;
        try (java.util.stream.Stream<String> lines = Files.lines(log)) {
            return lines.anyMatch(line -> line.contains("Training complete:"));
        }
    }
    private static void slot(Path dir, JsonObject slots, String kind, int epoch, String weights, String state) {
        if (epoch <= 0 || !Files.isRegularFile(dir.resolve(weights))) return;
        JsonObject slot = new JsonObject();
        slot.addProperty("epoch",epoch);
        slot.addProperty("weights",weights);
        if (state != null && Files.isRegularFile(dir.resolve(state))) slot.addProperty("state",state);
        slots.add(kind,slot);
    }
}
