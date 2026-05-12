/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Stream;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

final class YoloDatasetPreparer {

    private static final long SPLIT_SEED = 5489L;
    private static final double TRAIN_FRACTION = 0.8;
    private static final double SMALL_OBJECT_IMAGE_AREA_RATIO = 0.001;
    private static final double TARGET_OBJECT_TILE_AREA_RATIO = 0.01;
    private static final int MIN_OBJECT_VISIBILITY_PX = 64;
    private static final int MAX_REPETITIONS_PER_OBJECT = 3;
    private static final String DEFAULT_CLASS_NAME = "object";
    private static final String GENERATED_YAML_NAME = "data.yaml";

    private static final Set<String> IMAGE_EXTENSIONS = new HashSet<String>();
    private static final Set<String> MASK_SUFFIXES = new HashSet<String>();
    static {
        Collections.addAll(IMAGE_EXTENSIONS, ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff");
        Collections.addAll(MASK_SUFFIXES, "_mask", "_masks", "_label", "_labels");
    }

    private YoloDatasetPreparer() {}

    static File prepare(String datasetPath, String modelName, String modelsDir, Consumer<String> logConsumer)
            throws IOException {
        if (datasetPath == null || datasetPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a YOLO training dataset path.");
        }
        File input = new File(datasetPath.trim()).getAbsoluteFile();
        if (!input.exists()) {
            throw new IllegalArgumentException("The training dataset path does not exist: " + input);
        }

        File yaml = input.isFile() && isYaml(input) ? input : findDatasetYaml(input);
        if (yaml != null) {
            YamlDataset dataset = readYamlDataset(yaml);
            if (dataset.hasTrainAndVal()) {
                log(logConsumer, "Using YOLO dataset YAML: " + yaml.getAbsolutePath());
                return yaml.getAbsoluteFile();
            }
            if (dataset.hasTrainOnly()) {
                File generatedYaml = createGeneratedYoloDataset(dataset.sourceName(), modelName, modelsDir,
                        dataset.namesOrInferred(), dataset.train.samples, Collections.<YoloSample>emptyList(),
                        true, logConsumer);
                log(logConsumer, "Generated train/validation YOLO dataset: " + generatedYaml.getAbsolutePath());
                return generatedYaml;
            }
            throw new IllegalArgumentException("The YOLO YAML must define at least a valid train split: " + yaml);
        }

        if (!input.isDirectory()) {
            throw new IllegalArgumentException("The training dataset must be a YAML file or a dataset folder: " + input);
        }

        SplitData trainSplit = findYoloSplit(input, "train");
        SplitData valSplit = firstNonNull(findYoloSplit(input, "val"),
                findYoloSplit(input, "validation"),
                findYoloSplit(input, "valid"));
        if (trainSplit != null && !trainSplit.samples.isEmpty()
                && valSplit != null && !valSplit.samples.isEmpty()) {
            File generatedYaml = writeFolderYaml(input, trainSplit, valSplit);
            log(logConsumer, "Created YOLO dataset YAML: " + generatedYaml.getAbsolutePath());
            return generatedYaml;
        }
        if (trainSplit != null && !trainSplit.samples.isEmpty()) {
            File generatedYaml = createGeneratedYoloDataset(input.getName(), modelName, modelsDir,
                    inferNames(trainSplit.samples), trainSplit.samples, Collections.<YoloSample>emptyList(),
                    true, logConsumer);
            log(logConsumer, "Generated train/validation YOLO dataset: " + generatedYaml.getAbsolutePath());
            return generatedYaml;
        }

        List<MaskSample> trainMasks = findMaskSamples(input, "train");
        List<MaskSample> valMasks = firstNonEmpty(findMaskSamples(input, "val"),
                findMaskSamples(input, "validation"),
                findMaskSamples(input, "valid"));
        if (!trainMasks.isEmpty()) {
            boolean splitTrain = valMasks.isEmpty();
            File generatedYaml = createGeneratedMaskDataset(input.getName(), modelName, modelsDir,
                    trainMasks, valMasks, splitTrain, logConsumer);
            log(logConsumer, "Generated YOLO dataset from instance masks: " + generatedYaml.getAbsolutePath());
            return generatedYaml;
        }

        throw new IllegalArgumentException("Could not recognize a YOLO dataset or an instance-mask dataset in: "
                + input.getAbsolutePath());
    }

    private static File findDatasetYaml(File input) throws IOException {
        if (input == null || !input.isDirectory()) {
            return null;
        }
        for (String name : new String[] {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}) {
            File candidate = new File(input, name);
            if (candidate.isFile() && looksLikeYoloYaml(candidate)) {
                return candidate;
            }
        }
        try (Stream<Path> stream = Files.walk(input.toPath(), 3)) {
            return stream
                    .filter(Files::isRegularFile)
                    .map(Path::toFile)
                    .filter(YoloDatasetPreparer::isYaml)
                    .filter(YoloDatasetPreparer::looksLikeYoloYaml)
                    .findFirst()
                    .orElse(null);
        }
    }

    private static boolean looksLikeYoloYaml(File file) {
        try {
            return readYamlKeys(file).containsKey("train");
        } catch (IOException e) {
            return false;
        }
    }

    private static YamlDataset readYamlDataset(File yaml) throws IOException {
        LinkedHashMap<String, String> keys = readYamlKeys(yaml);
        File yamlDir = yaml.getParentFile() == null ? new File(".") : yaml.getParentFile();
        File root = yamlDir;
        String pathValue = keys.get("path");
        if (pathValue != null && !pathValue.trim().isEmpty()) {
            root = resolvePath(yamlDir, pathValue);
        }
        SplitData train = collectYoloSplit("train", resolvePath(root, keys.get("train")));
        String valValue = keys.containsKey("val") ? keys.get("val") : keys.get("validation");
        SplitData val = collectYoloSplit("val", resolvePath(root, valValue));
        return new YamlDataset(yaml, keys, train, val, parseNames(keys.get("names")));
    }

    private static LinkedHashMap<String, String> readYamlKeys(File yaml) throws IOException {
        LinkedHashMap<String, String> keys = new LinkedHashMap<String, String>();
        try (BufferedReader reader = Files.newBufferedReader(yaml.toPath(), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty() || line.trim().startsWith("#")) {
                    continue;
                }
                if (Character.isWhitespace(line.charAt(0))) {
                    continue;
                }
                int colon = line.indexOf(':');
                if (colon <= 0) {
                    continue;
                }
                String key = line.substring(0, colon).trim().toLowerCase(Locale.ROOT);
                String value = stripYamlComment(line.substring(colon + 1).trim());
                keys.put(key, stripQuotes(value));
            }
        }
        return keys;
    }

    private static String stripYamlComment(String value) {
        boolean single = false;
        boolean dbl = false;
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (c == '\'' && !dbl) {
                single = !single;
            } else if (c == '"' && !single) {
                dbl = !dbl;
            } else if (c == '#' && !single && !dbl) {
                return value.substring(0, i).trim();
            }
        }
        return value.trim();
    }

    private static String stripQuotes(String value) {
        if (value == null) {
            return null;
        }
        String clean = value.trim();
        if (clean.length() >= 2
                && ((clean.startsWith("\"") && clean.endsWith("\""))
                        || (clean.startsWith("'") && clean.endsWith("'")))) {
            return clean.substring(1, clean.length() - 1);
        }
        return clean;
    }

    private static List<String> parseNames(String namesValue) {
        if (namesValue == null || namesValue.trim().isEmpty()) {
            return Collections.emptyList();
        }
        String value = namesValue.trim();
        if (value.startsWith("[") && value.endsWith("]")) {
            value = value.substring(1, value.length() - 1);
            List<String> names = new ArrayList<String>();
            for (String part : value.split(",")) {
                String name = stripQuotes(part.trim());
                if (!name.isEmpty()) {
                    names.add(name);
                }
            }
            return names;
        }
        return Collections.emptyList();
    }

    private static File resolvePath(File base, String value) {
        if (value == null || value.trim().isEmpty()) {
            return null;
        }
        String clean = stripQuotes(value.trim());
        if (clean.startsWith("[") || clean.startsWith("{")) {
            return null;
        }
        File file = new File(clean);
        return file.isAbsolute() ? file : new File(base, clean);
    }

    private static SplitData collectYoloSplit(String name, File splitPath) throws IOException {
        if (splitPath == null || !splitPath.exists()) {
            return null;
        }
        if (splitPath.isFile()) {
            return collectYoloSplitFromList(name, splitPath);
        }
        SplitLayout layout = inferSplitLayout(splitPath);
        if (layout == null) {
            return null;
        }
        return collectYoloSplit(name, layout.imageRoot, layout.labelRoot);
    }

    private static SplitData collectYoloSplitFromList(String name, File listFile) throws IOException {
        List<YoloSample> samples = new ArrayList<YoloSample>();
        File base = listFile.getParentFile() == null ? new File(".") : listFile.getParentFile();
        try (BufferedReader reader = Files.newBufferedReader(listFile.toPath(), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String clean = line.trim();
                if (clean.isEmpty() || clean.startsWith("#")) {
                    continue;
                }
                File image = resolvePath(base, clean);
                File label = labelForImagePath(image);
                if (image.isFile() && label != null && label.isFile()) {
                    YoloSample sample = readYoloSample(image, label);
                    if (sample != null) {
                        samples.add(sample);
                    }
                }
            }
        }
        return samples.isEmpty() ? null : new SplitData(name, null, null, samples);
    }

    private static SplitLayout inferSplitLayout(File splitPath) {
        File imagesChild = new File(splitPath, "images");
        File labelsChild = new File(splitPath, "labels");
        if (imagesChild.isDirectory() && labelsChild.isDirectory()) {
            return new SplitLayout(imagesChild, labelsChild);
        }
        File labelsBySegment = replacePathSegment(splitPath, "images", "labels");
        if (labelsBySegment != null && labelsBySegment.isDirectory()) {
            return new SplitLayout(splitPath, labelsBySegment);
        }
        File parent = splitPath.getParentFile();
        if (parent != null) {
            File labelsSibling = new File(new File(parent, "labels"), splitPath.getName());
            if (labelsSibling.isDirectory()) {
                return new SplitLayout(splitPath, labelsSibling);
            }
        }
        if (labelsChild.isDirectory()) {
            return new SplitLayout(splitPath, labelsChild);
        }
        return null;
    }

    private static File replacePathSegment(File path, String from, String to) {
        Path absolute = path.toPath().toAbsolutePath();
        Path rebuilt = absolute.getRoot();
        boolean replaced = false;
        for (Path segment : absolute) {
            String name = segment.toString();
            if (!replaced && from.equals(name)) {
                name = to;
                replaced = true;
            }
            rebuilt = rebuilt == null ? new File(name).toPath() : rebuilt.resolve(name);
        }
        return replaced ? rebuilt.toFile() : null;
    }

    private static File labelForImagePath(File image) {
        Path path = image.toPath().toAbsolutePath();
        Path rebuilt = path.getRoot();
        boolean replaced = false;
        for (Path segment : path) {
            String name = segment.toString();
            if (!replaced && "images".equals(name)) {
                name = "labels";
                replaced = true;
            }
            rebuilt = rebuilt == null ? new File(name).toPath() : rebuilt.resolve(name);
        }
        if (!replaced) {
            return null;
        }
        File candidate = rebuilt.toFile();
        return new File(candidate.getParentFile(), removeExtension(candidate.getName()) + ".txt");
    }

    private static SplitData findYoloSplit(File root, String splitName) throws IOException {
        for (SplitLayout layout : candidateSplitLayouts(root, splitName)) {
            SplitData split = collectYoloSplit(splitName, layout.imageRoot, layout.labelRoot);
            if (split != null && !split.samples.isEmpty()) {
                return split;
            }
        }
        return null;
    }

    private static List<SplitLayout> candidateSplitLayouts(File root, String splitName) {
        List<SplitLayout> layouts = new ArrayList<SplitLayout>();
        layouts.add(new SplitLayout(new File(root, "images/" + splitName), new File(root, "labels/" + splitName)));
        layouts.add(new SplitLayout(new File(root, splitName + "/images"), new File(root, splitName + "/labels")));
        layouts.add(new SplitLayout(new File(root, splitName), new File(root, "labels/" + splitName)));
        layouts.add(new SplitLayout(new File(root, splitName), new File(root, splitName + "/labels")));
        return layouts;
    }

    private static SplitData collectYoloSplit(String name, File imageRoot, File labelRoot) throws IOException {
        if (imageRoot == null || labelRoot == null || !imageRoot.isDirectory() || !labelRoot.isDirectory()) {
            return null;
        }
        List<File> imageFiles = collectImages(imageRoot);
        List<YoloSample> samples = new ArrayList<YoloSample>();
        for (File image : imageFiles) {
            Path rel = imageRoot.toPath().toAbsolutePath().relativize(image.toPath().toAbsolutePath());
            File label = new File(labelRoot, replaceExtension(rel.toString(), ".txt"));
            if (!label.isFile()) {
                continue;
            }
            YoloSample sample = readYoloSample(image, label);
            if (sample != null) {
                samples.add(sample);
            }
        }
        return samples.isEmpty() ? null : new SplitData(name, imageRoot, labelRoot, samples);
    }

    private static YoloSample readYoloSample(File image, File label) throws IOException {
        ImageSize size = readImageSize(image);
        if (size == null) {
            return null;
        }
        List<Box> boxes = readYoloBoxes(label, size.width, size.height);
        return boxes.isEmpty() ? null : new YoloSample(image, label, size.width, size.height, boxes);
    }

    private static List<Box> readYoloBoxes(File label, int imageWidth, int imageHeight) throws IOException {
        List<Box> boxes = new ArrayList<Box>();
        try (BufferedReader reader = Files.newBufferedReader(label.toPath(), StandardCharsets.UTF_8)) {
            String line;
            int objectIndex = 0;
            while ((line = reader.readLine()) != null) {
                String clean = line.trim();
                if (clean.isEmpty() || clean.startsWith("#")) {
                    continue;
                }
                String[] parts = clean.split("\\s+");
                if (parts.length < 5) {
                    continue;
                }
                try {
                    int cls = Integer.parseInt(parts[0]);
                    double cx = Double.parseDouble(parts[1]) * imageWidth;
                    double cy = Double.parseDouble(parts[2]) * imageHeight;
                    double bw = Double.parseDouble(parts[3]) * imageWidth;
                    double bh = Double.parseDouble(parts[4]) * imageHeight;
                    Box box = new Box(cls, cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0,
                            objectIndex++);
                    if (box.isValid()) {
                        boxes.add(box.clamped(imageWidth, imageHeight));
                    }
                } catch (NumberFormatException e) {
                    // Ignore malformed label rows; they should not block usable samples.
                }
            }
        }
        return boxes;
    }

    private static File writeFolderYaml(File root, SplitData train, SplitData val) throws IOException {
        File yaml = uniqueYamlFile(root);
        List<String> names = inferNames(train.samples, val.samples);
        writeYaml(yaml, root, pathRelativeTo(root, train.imageRoot), pathRelativeTo(root, val.imageRoot), names);
        return yaml;
    }

    private static File uniqueYamlFile(File root) {
        File yaml = new File(root, GENERATED_YAML_NAME);
        if (!yaml.exists()) {
            return yaml;
        }
        for (int i = 1; ; i++) {
            File candidate = new File(root, "data-" + i + ".yaml");
            if (!candidate.exists()) {
                return candidate;
            }
        }
    }

    private static File createGeneratedYoloDataset(String sourceName, String modelName, String modelsDir,
            List<String> names, List<YoloSample> trainSamples, List<YoloSample> valSamples,
            boolean splitTrainSamples, Consumer<String> logConsumer) throws IOException {
        File root = createUniqueGeneratedRoot(sourceName, modelName, modelsDir);
        List<GeneratedSample> trainGenerated = toGeneratedSamples(trainSamples);
        List<GeneratedSample> valGenerated = toGeneratedSamples(valSamples);
        writeGeneratedDataset(root, names.isEmpty() ? inferNames(trainSamples, valSamples) : names,
                trainGenerated, valGenerated, splitTrainSamples, logConsumer);
        return new File(root, GENERATED_YAML_NAME);
    }

    private static File createGeneratedMaskDataset(String sourceName, String modelName, String modelsDir,
            List<MaskSample> trainMasks, List<MaskSample> valMasks, boolean splitTrainSamples,
            Consumer<String> logConsumer) throws IOException {
        File root = createUniqueGeneratedRoot(sourceName, modelName, modelsDir);
        List<GeneratedSample> trainGenerated = toGeneratedSamplesFromMasks(trainMasks);
        List<GeneratedSample> valGenerated = toGeneratedSamplesFromMasks(valMasks);
        writeGeneratedDataset(root, Collections.singletonList(DEFAULT_CLASS_NAME), trainGenerated, valGenerated,
                splitTrainSamples, logConsumer);
        return new File(root, GENERATED_YAML_NAME);
    }

    private static void writeGeneratedDataset(File root, List<String> names,
            List<GeneratedSample> trainGenerated, List<GeneratedSample> valGenerated,
            boolean splitTrainSamples, Consumer<String> logConsumer) throws IOException {
        if (splitTrainSamples) {
            List<GeneratedSample> all = new ArrayList<GeneratedSample>(trainGenerated);
            Collections.shuffle(all, new Random(SPLIT_SEED));
            int trainCount = splitIndex(all.size());
            trainGenerated = new ArrayList<GeneratedSample>(all.subList(0, trainCount));
            valGenerated = new ArrayList<GeneratedSample>(all.subList(trainCount, all.size()));
        }
        if (trainGenerated.isEmpty() || valGenerated.isEmpty()) {
            throw new IllegalArgumentException("Could not create non-empty train and validation splits.");
        }

        File trainImages = new File(root, "images/train");
        File valImages = new File(root, "images/val");
        File trainLabels = new File(root, "labels/train");
        File valLabels = new File(root, "labels/val");
        Files.createDirectories(trainImages.toPath());
        Files.createDirectories(valImages.toPath());
        Files.createDirectories(trainLabels.toPath());
        Files.createDirectories(valLabels.toPath());

        sortBySource(trainGenerated);
        sortBySource(valGenerated);
        writeSamples(trainGenerated, trainImages, trainLabels);
        writeSamples(valGenerated, valImages, valLabels);
        writeYaml(new File(root, GENERATED_YAML_NAME), root, "images/train", "images/val", names);
        log(logConsumer, "Prepared " + trainGenerated.size() + " training and " + valGenerated.size()
                + " validation YOLO samples.");
    }

    private static void sortBySource(List<GeneratedSample> samples) {
        Collections.sort(samples, Comparator
                .comparing((GeneratedSample sample) -> sample.sourceImage.getAbsolutePath())
                .thenComparing(sample -> sample.crop == null ? "" : sample.crop.toString()));
    }

    private static int splitIndex(int size) {
        if (size < 2) {
            return size;
        }
        int trainCount = (int) Math.round(size * TRAIN_FRACTION);
        trainCount = Math.max(1, trainCount);
        trainCount = Math.min(size - 1, trainCount);
        return trainCount;
    }

    private static List<GeneratedSample> toGeneratedSamples(List<YoloSample> samples) {
        List<GeneratedSample> generated = new ArrayList<GeneratedSample>();
        for (YoloSample sample : samples) {
            generated.addAll(buildGeneratedSamples(sample.image, sample.width, sample.height, sample.boxes));
        }
        return generated;
    }

    private static List<GeneratedSample> toGeneratedSamplesFromMasks(List<MaskSample> samples) {
        List<GeneratedSample> generated = new ArrayList<GeneratedSample>();
        for (MaskSample sample : samples) {
            generated.addAll(buildGeneratedSamples(sample.image, sample.width, sample.height, sample.boxes));
        }
        return generated;
    }

    private static List<GeneratedSample> buildGeneratedSamples(File image, int width, int height, List<Box> boxes) {
        List<GeneratedSample> generated = new ArrayList<GeneratedSample>();
        if (boxes == null || boxes.isEmpty()) {
            return generated;
        }
        double imageArea = Math.max(1.0, width * (double) height);
        List<Box> largeBoxes = new ArrayList<Box>();
        List<Box> smallBoxes = new ArrayList<Box>();
        for (Box box : boxes) {
            if (box.area() < SMALL_OBJECT_IMAGE_AREA_RATIO * imageArea) {
                smallBoxes.add(box);
            } else {
                largeBoxes.add(box);
            }
        }
        if (smallBoxes.isEmpty()) {
            generated.add(GeneratedSample.full(image, width, height, boxes));
            return generated;
        }
        if (!largeBoxes.isEmpty()) {
            generated.add(GeneratedSample.full(image, width, height, largeBoxes));
        }
        Map<Integer, Integer> repetitions = new HashMap<Integer, Integer>();
        for (Box target : smallBoxes) {
            if (count(repetitions, target.objectIndex) >= MAX_REPETITIONS_PER_OBJECT) {
                continue;
            }
            Rectangle crop = cropAround(target, width, height);
            List<Box> cropBoxes = new ArrayList<Box>();
            for (Box candidate : smallBoxes) {
                if (count(repetitions, candidate.objectIndex) >= MAX_REPETITIONS_PER_OBJECT) {
                    continue;
                }
                if (candidate.fullyInside(crop)) {
                    cropBoxes.add(candidate.relativeTo(crop));
                }
            }
            if (cropBoxes.isEmpty()) {
                cropBoxes.add(target.relativeTo(crop));
            }
            for (Box cropBox : cropBoxes) {
                repetitions.put(cropBox.objectIndex, count(repetitions, cropBox.objectIndex) + 1);
            }
            generated.add(GeneratedSample.crop(image, crop, cropBoxes));
        }
        return generated;
    }

    private static int count(Map<Integer, Integer> map, int key) {
        Integer value = map.get(key);
        return value == null ? 0 : value.intValue();
    }

    private static Rectangle cropAround(Box box, int imageWidth, int imageHeight) {
        double targetArea = Math.max(box.area() / TARGET_OBJECT_TILE_AREA_RATIO,
                MIN_OBJECT_VISIBILITY_PX * (double) MIN_OBJECT_VISIBILITY_PX);
        int side = (int) Math.ceil(Math.sqrt(targetArea));
        side = Math.max(side, (int) Math.ceil(Math.max(box.width(), box.height())));
        side = Math.max(1, side);
        int cropW = Math.min(imageWidth, side);
        int cropH = Math.min(imageHeight, side);
        int x = (int) Math.round((box.x1 + box.x2) / 2.0 - cropW / 2.0);
        int y = (int) Math.round((box.y1 + box.y2) / 2.0 - cropH / 2.0);
        x = clamp(x, 0, Math.max(0, imageWidth - cropW));
        y = clamp(y, 0, Math.max(0, imageHeight - cropH));
        if (box.x1 < x) {
            x = Math.max(0, (int) Math.floor(box.x1));
        }
        if (box.y1 < y) {
            y = Math.max(0, (int) Math.floor(box.y1));
        }
        if (box.x2 > x + cropW) {
            x = Math.min(Math.max(0, imageWidth - cropW), (int) Math.ceil(box.x2 - cropW));
        }
        if (box.y2 > y + cropH) {
            y = Math.min(Math.max(0, imageHeight - cropH), (int) Math.ceil(box.y2 - cropH));
        }
        return new Rectangle(x, y, cropW, cropH);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void writeSamples(List<GeneratedSample> samples, File imageDir, File labelDir)
            throws IOException {
        int index = 0;
        File cachedSource = null;
        BufferedImage cachedImage = null;
        for (GeneratedSample sample : samples) {
            String stem = safeFileName(removeExtension(sample.sourceImage.getName())) + "_" + String.format("%05d", index++);
            File imageOut;
            if (sample.crop == null) {
                imageOut = new File(imageDir, stem + extensionOrDefault(sample.sourceImage, ".png"));
                linkOrCopy(sample.sourceImage.toPath(), imageOut.toPath());
            } else {
                imageOut = new File(imageDir, stem + ".png");
                if (!sample.sourceImage.equals(cachedSource)) {
                    cachedImage = ImageIO.read(sample.sourceImage);
                    cachedSource = sample.sourceImage;
                }
                writeCrop(sample.sourceImage, cachedImage, sample.crop, imageOut);
            }
            writeLabel(new File(labelDir, stem + ".txt"), sample.boxes, sample.width, sample.height);
        }
    }

    private static void writeCrop(File source, BufferedImage image, Rectangle crop, File output) throws IOException {
        if (image == null) {
            throw new IOException("Could not read image for crop: " + source);
        }
        BufferedImage cropped = image.getSubimage(crop.x, crop.y, crop.width, crop.height);
        Files.createDirectories(output.getParentFile().toPath());
        if (!ImageIO.write(cropped, "png", output)) {
            throw new IOException("Could not write cropped image: " + output);
        }
    }

    private static void writeLabel(File label, List<Box> boxes, int imageWidth, int imageHeight) throws IOException {
        Files.createDirectories(label.getParentFile().toPath());
        DecimalFormat df = new DecimalFormat("0.########", DecimalFormatSymbols.getInstance(Locale.US));
        try (BufferedWriter writer = Files.newBufferedWriter(label.toPath(), StandardCharsets.UTF_8)) {
            for (Box box : boxes) {
                Box clean = box.clamped(imageWidth, imageHeight);
                if (!clean.isValid()) {
                    continue;
                }
                double cx = ((clean.x1 + clean.x2) / 2.0) / imageWidth;
                double cy = ((clean.y1 + clean.y2) / 2.0) / imageHeight;
                double w = clean.width() / imageWidth;
                double h = clean.height() / imageHeight;
                writer.write(clean.classId + " " + df.format(cx) + " " + df.format(cy)
                        + " " + df.format(w) + " " + df.format(h));
                writer.newLine();
            }
        }
    }

    private static void linkOrCopy(Path source, Path target) throws IOException {
        Files.createDirectories(target.getParent());
        try {
            Files.createSymbolicLink(target, source.toAbsolutePath());
            return;
        } catch (UnsupportedOperationException | SecurityException | IOException e) {
            // Fall through to hard link/copy. Symlinks are not always available, especially on Windows.
        }
        try {
            Files.createLink(target, source.toAbsolutePath());
            return;
        } catch (UnsupportedOperationException | SecurityException | IOException e) {
            // Fall through to copy.
        }
        try {
            Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
        } catch (FileAlreadyExistsException e) {
            Files.delete(target);
            Files.copy(source, target);
        }
    }

    private static List<MaskSample> findMaskSamples(File root, String splitName) throws IOException {
        File splitDir = new File(root, splitName);
        if (!splitDir.isDirectory()) {
            return Collections.emptyList();
        }
        List<MaskPair> pairs = findMaskPairs(splitDir);
        List<MaskSample> samples = new ArrayList<MaskSample>();
        for (MaskPair pair : pairs) {
            MaskSample sample = readMaskSample(pair.image, pair.mask);
            if (sample != null && !sample.boxes.isEmpty()) {
                samples.add(sample);
            }
        }
        return samples;
    }

    private static List<MaskPair> findMaskPairs(File splitDir) throws IOException {
        List<File> files = collectImages(splitDir);
        Map<String, File> images = new HashMap<String, File>();
        Map<String, File> masks = new HashMap<String, File>();
        for (File file : files) {
            String stem = removeExtension(file.getName());
            String core = maskCore(stem);
            if (core == null) {
                images.put(normalizeImageCore(stem), file);
                images.put(stem, file);
            } else {
                masks.put(core, file);
            }
        }
        List<MaskPair> pairs = new ArrayList<MaskPair>();
        for (Map.Entry<String, File> entry : masks.entrySet()) {
            File image = images.get(entry.getKey());
            if (image != null) {
                pairs.add(new MaskPair(image, entry.getValue()));
            }
        }
        Collections.sort(pairs, Comparator.comparing(pair -> pair.image.getAbsolutePath()));
        return pairs;
    }

    private static String normalizeImageCore(String stem) {
        return stem.endsWith("_image") ? stem.substring(0, stem.length() - "_image".length()) : stem;
    }

    private static String maskCore(String stem) {
        for (String suffix : MASK_SUFFIXES) {
            if (stem.endsWith(suffix)) {
                return stem.substring(0, stem.length() - suffix.length());
            }
        }
        return null;
    }

    private static MaskSample readMaskSample(File imageFile, File maskFile) throws IOException {
        ImageSize size = readImageSize(imageFile);
        BufferedImage mask = ImageIO.read(maskFile);
        if (size == null || mask == null) {
            return null;
        }
        Raster raster = mask.getRaster();
        Map<Integer, int[]> bounds = new HashMap<Integer, int[]>();
        int width = Math.min(size.width, mask.getWidth());
        int height = Math.min(size.height, mask.getHeight());
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = raster.getSample(x, y, 0);
                if (value <= 0) {
                    continue;
                }
                int[] b = bounds.get(value);
                if (b == null) {
                    b = new int[] {x, y, x, y};
                    bounds.put(value, b);
                } else {
                    b[0] = Math.min(b[0], x);
                    b[1] = Math.min(b[1], y);
                    b[2] = Math.max(b[2], x);
                    b[3] = Math.max(b[3], y);
                }
            }
        }
        List<Box> boxes = new ArrayList<Box>();
        int objectIndex = 0;
        for (int[] b : bounds.values()) {
            Box box = new Box(0, b[0], b[1], b[2] + 1.0, b[3] + 1.0, objectIndex++);
            if (box.isValid()) {
                boxes.add(box);
            }
        }
        return new MaskSample(imageFile, size.width, size.height, boxes);
    }

    private static List<File> collectImages(File root) throws IOException {
        if (root == null || !root.isDirectory()) {
            return Collections.emptyList();
        }
        List<File> images = new ArrayList<File>();
        try (Stream<Path> stream = Files.walk(root.toPath())) {
            stream.filter(Files::isRegularFile)
                    .map(Path::toFile)
                    .filter(YoloDatasetPreparer::isImage)
                    .forEach(images::add);
        }
        Collections.sort(images, Comparator.comparing(File::getAbsolutePath));
        return images;
    }

    private static ImageSize readImageSize(File imageFile) throws IOException {
        try (ImageInputStream input = ImageIO.createImageInputStream(imageFile)) {
            if (input == null) {
                return null;
            }
            Iterator<ImageReader> readers = ImageIO.getImageReaders(input);
            if (!readers.hasNext()) {
                return null;
            }
            ImageReader reader = readers.next();
            try {
                reader.setInput(input);
                return new ImageSize(reader.getWidth(0), reader.getHeight(0));
            } finally {
                reader.dispose();
            }
        }
    }

    private static File createUniqueGeneratedRoot(String sourceName, String modelName, String modelsDir)
            throws IOException {
        File models = modelsDir == null ? new File("models") : new File(modelsDir);
        File softwareRoot = models.getParentFile() == null ? new File(".") : models.getParentFile();
        File datasetsRoot = new File(softwareRoot, "datasets");
        Files.createDirectories(datasetsRoot.toPath());
        String baseName = safeFileName(sourceName) + "-" + safeFileName(modelName);
        File root = new File(datasetsRoot, baseName);
        int suffix = 1;
        while (root.exists()) {
            root = new File(datasetsRoot, baseName + "-" + suffix++);
        }
        Files.createDirectories(root.toPath());
        return root;
    }

    private static void writeYaml(File yaml, File root, String trainPath, String valPath, List<String> names)
            throws IOException {
        Files.createDirectories(yaml.getParentFile().toPath());
        List<String> safeNames = names == null || names.isEmpty() ? Collections.singletonList(DEFAULT_CLASS_NAME) : names;
        try (BufferedWriter writer = Files.newBufferedWriter(yaml.toPath(), StandardCharsets.UTF_8)) {
            writer.write("path: " + root.getAbsolutePath());
            writer.newLine();
            writer.write("train: " + trainPath.replace(File.separatorChar, '/'));
            writer.newLine();
            writer.write("val: " + valPath.replace(File.separatorChar, '/'));
            writer.newLine();
            writer.write("names:");
            writer.newLine();
            for (int i = 0; i < safeNames.size(); i++) {
                writer.write("  " + i + ": " + safeYamlValue(safeNames.get(i)));
                writer.newLine();
            }
        }
    }

    @SafeVarargs
    private static <T> T firstNonNull(T... values) {
        for (T value : values) {
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    @SafeVarargs
    private static <T> List<T> firstNonEmpty(List<T>... lists) {
        for (List<T> list : lists) {
            if (list != null && !list.isEmpty()) {
                return list;
            }
        }
        return Collections.emptyList();
    }

    @SafeVarargs
    private static List<String> inferNames(List<YoloSample>... sampleLists) {
        int maxClass = -1;
        for (List<YoloSample> samples : sampleLists) {
            if (samples == null) {
                continue;
            }
            for (YoloSample sample : samples) {
                for (Box box : sample.boxes) {
                    maxClass = Math.max(maxClass, box.classId);
                }
            }
        }
        if (maxClass < 0) {
            return Collections.singletonList(DEFAULT_CLASS_NAME);
        }
        List<String> names = new ArrayList<String>();
        for (int i = 0; i <= maxClass; i++) {
            names.add("class_" + i);
        }
        return names;
    }

    private static String pathRelativeTo(File root, File child) {
        try {
            return root.toPath().toAbsolutePath().relativize(child.toPath().toAbsolutePath()).toString();
        } catch (Exception e) {
            return child.getAbsolutePath();
        }
    }

    private static String safeYamlValue(String value) {
        if (value == null || value.trim().isEmpty()) {
            return DEFAULT_CLASS_NAME;
        }
        String clean = value.trim().replace("'", "''");
        return "'" + clean + "'";
    }

    private static String safeFileName(String name) {
        String clean = name == null ? "" : name.trim();
        if (clean.toLowerCase(Locale.ROOT).endsWith(YoloModelRegistry.YOLO_WEIGHTS_EXTENSION)) {
            clean = clean.substring(0, clean.length() - YoloModelRegistry.YOLO_WEIGHTS_EXTENSION.length());
        }
        clean = clean.replaceAll("[^A-Za-z0-9._-]+", "_");
        clean = clean.replaceAll("_+", "_");
        if (clean.isEmpty() || ".".equals(clean) || "..".equals(clean)) {
            return "dataset";
        }
        return clean;
    }

    private static boolean isYaml(File file) {
        String name = file.getName().toLowerCase(Locale.ROOT);
        return file.isFile() && (name.endsWith(".yaml") || name.endsWith(".yml"));
    }

    private static boolean isImage(File file) {
        String name = file.getName().toLowerCase(Locale.ROOT);
        for (String ext : IMAGE_EXTENSIONS) {
            if (name.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    private static String extensionOrDefault(File file, String defaultExtension) {
        String name = file.getName();
        int dot = name.lastIndexOf('.');
        return dot < 0 ? defaultExtension : name.substring(dot);
    }

    private static String removeExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? name : name.substring(0, dot);
    }

    private static String replaceExtension(String path, String extension) {
        int dot = path.lastIndexOf('.');
        return (dot < 0 ? path : path.substring(0, dot)) + extension;
    }

    private static void log(Consumer<String> logConsumer, String message) {
        if (logConsumer != null) {
            logConsumer.accept(message);
        }
    }

    private static final class YamlDataset {
        private final File yaml;
        private final LinkedHashMap<String, String> keys;
        private final SplitData train;
        private final SplitData val;
        private final List<String> names;

        private YamlDataset(File yaml, LinkedHashMap<String, String> keys,
                SplitData train, SplitData val, List<String> names) {
            this.yaml = yaml;
            this.keys = keys;
            this.train = train;
            this.val = val;
            this.names = names;
        }

        private boolean hasTrainAndVal() {
            return train != null && !train.samples.isEmpty() && val != null && !val.samples.isEmpty();
        }

        private boolean hasTrainOnly() {
            return train != null && !train.samples.isEmpty() && (val == null || val.samples.isEmpty());
        }

        private List<String> namesOrInferred() {
            return names == null || names.isEmpty() ? inferNames(train.samples) : names;
        }

        private String sourceName() {
            File parent = yaml.getParentFile();
            if (parent != null && parent.getName() != null && !parent.getName().trim().isEmpty()) {
                return parent.getName();
            }
            String value = keys.get("path");
            return value == null || value.trim().isEmpty() ? removeExtension(yaml.getName()) : new File(value).getName();
        }
    }

    private static final class SplitData {
        private final String name;
        private final File imageRoot;
        private final File labelRoot;
        private final List<YoloSample> samples;

        private SplitData(String name, File imageRoot, File labelRoot, List<YoloSample> samples) {
            this.name = name;
            this.imageRoot = imageRoot;
            this.labelRoot = labelRoot;
            this.samples = samples;
        }
    }

    private static final class SplitLayout {
        private final File imageRoot;
        private final File labelRoot;

        private SplitLayout(File imageRoot, File labelRoot) {
            this.imageRoot = imageRoot;
            this.labelRoot = labelRoot;
        }
    }

    private static final class YoloSample {
        private final File image;
        private final File label;
        private final int width;
        private final int height;
        private final List<Box> boxes;

        private YoloSample(File image, File label, int width, int height, List<Box> boxes) {
            this.image = image;
            this.label = label;
            this.width = width;
            this.height = height;
            this.boxes = boxes;
        }
    }

    private static final class MaskPair {
        private final File image;
        private final File mask;

        private MaskPair(File image, File mask) {
            this.image = image;
            this.mask = mask;
        }
    }

    private static final class MaskSample {
        private final File image;
        private final int width;
        private final int height;
        private final List<Box> boxes;

        private MaskSample(File image, int width, int height, List<Box> boxes) {
            this.image = image;
            this.width = width;
            this.height = height;
            this.boxes = boxes;
        }
    }

    private static final class GeneratedSample {
        private final File sourceImage;
        private final Rectangle crop;
        private final int width;
        private final int height;
        private final List<Box> boxes;

        private GeneratedSample(File sourceImage, Rectangle crop, int width, int height, List<Box> boxes) {
            this.sourceImage = sourceImage;
            this.crop = crop;
            this.width = width;
            this.height = height;
            this.boxes = boxes;
        }

        private static GeneratedSample full(File sourceImage, int width, int height, List<Box> boxes) {
            return new GeneratedSample(sourceImage, null, width, height, new ArrayList<Box>(boxes));
        }

        private static GeneratedSample crop(File sourceImage, Rectangle crop, List<Box> boxes) {
            return new GeneratedSample(sourceImage, crop, crop.width, crop.height, new ArrayList<Box>(boxes));
        }
    }

    private static final class Box {
        private final int classId;
        private final double x1;
        private final double y1;
        private final double x2;
        private final double y2;
        private final int objectIndex;

        private Box(int classId, double x1, double y1, double x2, double y2, int objectIndex) {
            this.classId = classId;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.objectIndex = objectIndex;
        }

        private double width() {
            return Math.max(0.0, x2 - x1);
        }

        private double height() {
            return Math.max(0.0, y2 - y1);
        }

        private double area() {
            return width() * height();
        }

        private boolean isValid() {
            return width() > 0.0 && height() > 0.0;
        }

        private boolean fullyInside(Rectangle rectangle) {
            return x1 >= rectangle.x && y1 >= rectangle.y
                    && x2 <= rectangle.x + rectangle.width
                    && y2 <= rectangle.y + rectangle.height;
        }

        private Box relativeTo(Rectangle rectangle) {
            return new Box(classId, x1 - rectangle.x, y1 - rectangle.y,
                    x2 - rectangle.x, y2 - rectangle.y, objectIndex)
                    .clamped(rectangle.width, rectangle.height);
        }

        private Box clamped(int width, int height) {
            return new Box(classId,
                    Math.max(0.0, Math.min(width, x1)),
                    Math.max(0.0, Math.min(height, y1)),
                    Math.max(0.0, Math.min(width, x2)),
                    Math.max(0.0, Math.min(height, y2)),
                    objectIndex);
        }
    }

    private static final class ImageSize {
        private final int width;
        private final int height;

        private ImageSize(int width, int height) {
            this.width = width;
            this.height = height;
        }
    }
}
