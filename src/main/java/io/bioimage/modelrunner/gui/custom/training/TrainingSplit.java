package io.bioimage.modelrunner.gui.custom.training;

import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import com.google.gson.*;

/** Assign original annotated sources before materializing links, planes or crops. */
public final class TrainingSplit {
    private final Path root, output;
    private final TrainingContinuation previous;
    private final Consumer<String> log;
    private JsonArray spatial = new JsonArray();
    private File spatialSource;
    private boolean reuseSpatial;
    public File spatialSource() { return spatialSource; }
    public TrainingSplit(File root, File output, String base, Consumer<String> log) {
        this.root = root.toPath().toAbsolutePath().normalize();
        this.output = output.toPath().toAbsolutePath();
        this.previous = TrainingContinuation.find(base);
        this.log = log;
    }

    public <T> boolean assign(List<T> train, List<T> val, Function<T, File> image, Function<T, File> label,
            Predicate<T> positive, double fraction, long seed) throws IOException {
        return assign(train,val,image,label,positive,fraction,seed,false,null);
    }

    @FunctionalInterface
    public interface AnnotatedRegion<T> {
        boolean containsObject(T sample, int[][] bounds) throws IOException;
    }

    public <T> boolean assignDetections(List<T> train, List<T> val, Function<T, File> image, Function<T, File> label,
            Predicate<T> positive, double fraction, long seed, AnnotatedRegion<T> region) throws IOException {
        return assign(train,val,image,label,positive,fraction,seed,false,region);
    }

    <T> boolean assign(List<T> train, List<T> val, Function<T, File> image, Function<T, File> label,
            Predicate<T> positive, double fraction, long seed, boolean backendRecordsDomains) throws IOException {
        return assign(train,val,image,label,positive,fraction,seed,backendRecordsDomains,null);
    }

    private <T> boolean assign(List<T> train, List<T> val, Function<T, File> image, Function<T, File> label,
            Predicate<T> positive, double fraction, long seed, boolean backendRecordsDomains, AnnotatedRegion<T> region) throws IOException {
        boolean automatic = val.isEmpty();
        JsonObject old = new JsonObject();
        if (previous != null && Files.isRegularFile(previous.directory.resolve("split.json"))) {
            old = JsonParser.parseString(Files.readString(previous.directory.resolve("split.json"))).getAsJsonObject();
        }
        if (previous != null && automatic && old.size() == 0)
            message("No previous automatic split is available. Creating a validation holdout from the current dataset.");
        if (automatic && old.has("spatial")) spatial = old.getAsJsonArray("spatial");
        Map<List<String>, String> known = new HashMap<>();
        for (String group : Arrays.asList("train", "val")) if (old.has(group)) {
            for (JsonElement source : old.getAsJsonArray(group)) {
                JsonArray pair = source.getAsJsonArray();
                known.put(Arrays.asList(pair.get(0).getAsString(), pair.get(1).getAsString()), group);
            }
        }
        Function<T, List<String>> key = item -> Arrays.asList(name(image.apply(item)), name(label.apply(item)));
        if (spatial.size() > 0 && train.stream().noneMatch(p -> previouslySpatial(image.apply(p)) && positive.test(p)))
            spatial = new JsonArray();
        boolean changed = false;
        if (automatic && train.size() > 1) {
            List<T> fresh = new ArrayList<>(), retained = new ArrayList<>();
            for (T item : train) {
                String prior = known.get(key.apply(item));
                if ("val".equals(prior)) val.add(item);
                else if ("train".equals(prior)) retained.add(item);
                else fresh.add(item);
            }
            fresh.sort(Comparator.comparing(item -> key.apply(item).get(0)));
            Collections.shuffle(fresh, new Random(seed));
            if (previous != null) message("Recovered " + retained.size() + " training and " + val.size()
                    + " validation sources. Assigning " + fresh.size() + " new sources.");
            int count = (int)Math.round(fraction * fresh.size());
            if (val.isEmpty()) {
                int target = Math.max(1, (int)Math.round(fraction * train.size()));
                count = Math.min(fresh.size(), target);
                if (count < target && !retained.isEmpty()) {
                    Collections.shuffle(retained, new Random(seed));
                    while (count + val.size() < target && retained.size() + fresh.size() - count > 1)
                        val.add(retained.remove(retained.size()-1));
                    if (!val.isEmpty()) message("Previous validation unavailable. Rebuilt the holdout using previously trained images.");
                }
            }
            if (retained.isEmpty()) count = Math.min(count, Math.max(0, fresh.size()-1));
            val.addAll(fresh.subList(0, count));
            retained.addAll(fresh.subList(count, fresh.size()));
            if (retained.isEmpty() && val.size() > 1) retained.add(val.remove(val.size()-1));
            train.clear();
            train.addAll(retained);
            changed = true;
        }
        if (spatial.size() == 0 && !val.isEmpty() && val.stream().noneMatch(positive)) {
            T candidate = train.stream().filter(positive)
                    .min(Comparator.comparing(item -> known.containsKey(key.apply(item)))).orElse(null);
            if (candidate == null)
                throw new IOException("Validation has no annotated objects. Provide an annotated holdout and at least one annotated training source.");
            if (train.stream().filter(positive).limit(2).count() == 1) {
                spatialSource = image.apply(candidate);
                message("Only one source contains annotations. Reserving a disjoint region of " + spatialSource.getName() + " for validation.");
            } else {
                train.remove(candidate);
                val.add(candidate);
                message("Validation has no annotated objects. Moved " + key.apply(candidate).get(0) + " from training to validation.");
                changed = true;
            }
        }
        if (spatial.size() == 0 && train.stream().noneMatch(positive) && val.stream().filter(positive).limit(2).count() == 2) {
            T annotated = val.stream().filter(positive).findFirst().get();
            val.remove(annotated);
            train.add(annotated);
            changed = true;
            message("Moved " + key.apply(annotated).get(0) + " into training so both splits contain annotations.");
        } else if (spatial.size() == 0 && train.stream().noneMatch(positive) && val.stream().anyMatch(positive)) {
            T annotated = val.stream().filter(positive).findFirst().get();
            spatialSource = image.apply(annotated);
            val.remove(annotated);
            train.add(annotated);
            changed = true;
            message("Only the validation source contains annotations. Reserving disjoint training and validation regions within it.");
        }
        if (region != null && spatial.size() > 0) {
            List<T> samples = new ArrayList<>(train);
            samples.addAll(val);
            for (T shared : samples) {
                File file = image.apply(shared);
                int[][] trainBounds = previousBounds(file, "train"), valBounds = previousBounds(file, "val");
                if (trainBounds == null || valBounds == null) continue;
                List<T> trainObjects = new ArrayList<>(), valObjects = new ArrayList<>();
                for (T item : train) if (item != shared && positive.test(item)) trainObjects.add(item);
                for (T item : val) if (item != shared && positive.test(item)) valObjects.add(item);
                boolean trainRegion = region.containsObject(shared, trainBounds);
                boolean valRegion = region.containsObject(shared, valBounds);
                if (!valRegion && valObjects.isEmpty() && !trainObjects.isEmpty() && (trainRegion || trainObjects.size() > 1)) {
                    T moved = trainObjects.remove(0);
                    train.remove(moved);
                    val.add(moved);
                    valObjects.add(moved);
                    changed = true;
                    message("The saved validation region contains no objects. Moved " + key.apply(moved).get(0) + " into validation.");
                } else if (!trainRegion && trainObjects.isEmpty() && !valObjects.isEmpty() && (valRegion || valObjects.size() > 1)) {
                    T moved = valObjects.remove(0);
                    val.remove(moved);
                    train.add(moved);
                    trainObjects.add(moved);
                    changed = true;
                    message("The saved training region contains no objects. Moved " + key.apply(moved).get(0) + " into training.");
                }
                reuseSpatial = (valRegion || !valObjects.isEmpty()) && (trainRegion || !trainObjects.isEmpty());
                break;
            }
        }
        Files.createDirectories(output);
        if (automatic) {
            JsonObject split = new JsonObject();
            split.add("train", entries(train, key));
            split.add("val", entries(val, key));
            write("split.json", split.toString());
            // Geometry-aware trainers fingerprint the actual eligible images and held-out regions.
            if (backendRecordsDomains) return changed;
        }
        // A one-source spatial split also depends on the contents of that source.
        List<T> held = new ArrayList<>(val.isEmpty() ? train : val);
        held.sort(Comparator.comparing(item -> key.apply(item).get(0)));
        try {
            MessageDigest hash = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[1024*1024];
            for (T item : held) {
                hash.update(new Gson().toJson(key.apply(item)).getBytes(java.nio.charset.StandardCharsets.UTF_8));
                for (File f : Arrays.asList(image.apply(item), label.apply(item))) {
                    hash.update(fileDigest(f, buffer));
                }
            }
            write("validation.sha256", Base64.getEncoder().encodeToString(hash.digest()));
        } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
        return changed;
    }

    public boolean previouslySpatial(File image) {
        if (image.equals(spatialSource)) return true;
        for (JsonElement entry : spatial)
            if (entry.getAsJsonObject().getAsJsonArray("source").get(0).getAsString().equals(name(image))) return true;
        return false;
    }

    public int[][] retainedBounds(File image, String group) {
        return reuseSpatial ? previousBounds(image, group) : null;
    }

    private int[][] previousBounds(File image, String group) {
        for (JsonElement entry : spatial) {
            JsonObject part = entry.getAsJsonObject();
            if (part.getAsJsonArray("source").get(0).getAsString().equals(name(image)) && group.equals(part.get("split").getAsString()))
                return new Gson().fromJson(part.get("region"), int[][].class);
        }
        return null;
    }

    public void recordSpatial(File image, File label, int[][] training, int[][] validation) throws IOException {
        try {
            MessageDigest signature = MessageDigest.getInstance("SHA-256");
            signature.update(Files.readAllBytes(output.resolve("validation.sha256")));
            byte[] buffer = new byte[1024*1024];
            signature.update(fileDigest(image,buffer));
            signature.update(fileDigest(label,buffer));
            signature.update(new Gson().toJson(validation).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            write("validation.sha256",Base64.getEncoder().encodeToString(signature.digest()));
        } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
        Path path = output.resolve("split.json");
        if (!Files.isRegularFile(path)) return;
        JsonObject split = JsonParser.parseString(Files.readString(path)).getAsJsonObject();
        JsonElement source = new Gson().toJsonTree(Arrays.asList(name(image),name(label)));
        JsonArray regions = new JsonArray();
        for (String group : Arrays.asList("train","val")) {
            JsonArray entries = split.getAsJsonArray(group);
            if (!entries.contains(source)) entries.add(source);
            JsonObject region = new JsonObject();
            region.add("source",source);
            region.addProperty("split",group);
            region.add("region",new Gson().toJsonTree("train".equals(group) ? training : validation));
            regions.add(region);
        }
        split.add("spatial",regions);
        write("split.json",split.toString());
    }

    private static byte[] fileDigest(File file, byte[] buffer) throws NoSuchAlgorithmException, IOException {
        MessageDigest content = MessageDigest.getInstance("SHA-256");
        if (file != null && file.isFile()) try (InputStream in = Files.newInputStream(file.toPath())) {
            int n;
            while ((n = in.read(buffer)) != -1) content.update(buffer, 0, n);
        }
        return content.digest();
    }

    private void write(String name, String data) throws IOException {
        Path tmp = Files.createTempFile(output, ".split-", ".tmp");
        try {
            Files.writeString(tmp, data);
            try {
                Files.move(tmp, output.resolve(name), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(tmp, output.resolve(name), StandardCopyOption.REPLACE_EXISTING);
            }
        } finally { Files.deleteIfExists(tmp); }
    }
    private String name(File f) {
        if (f == null) return "";
        Path path = f.toPath().toAbsolutePath().normalize();
        if (!root.getRoot().equals(path.getRoot())) return path.toString().replace(File.separatorChar, '/');
        return root.relativize(path).toString().replace(File.separatorChar, '/');
    }
    private static <T> JsonArray entries(List<T> items, Function<T, List<String>> key) {
        JsonArray result = new JsonArray();
        Gson gson = new Gson();
        for (T item : items) result.add(gson.toJsonTree(key.apply(item)));
        return result;
    }
    private void message(String text) { if (log != null) log.accept(text); }
}
