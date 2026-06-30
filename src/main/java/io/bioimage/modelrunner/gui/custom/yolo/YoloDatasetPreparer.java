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
    private static final double TINY_OBJECT_IMAGE_AREA_RATIO = 0.00005;
    private static final double TINY_OBJECT_REFERENCE_FACTOR = 3.0;
    private static final double MIN_LABEL_RESIZED_SIDE_PX = 4.0;
    private static final double CRITICAL_OBJECT_RESIZED_SIDE_PX = 8.0;
    private static final double PREFERRED_CROP_RESIZED_SIDE_PX = 12.0;
    private static final double MIN_OBJECT_CROP_OVERLAP_RATIO = 0.3;
    private static final double MAX_OBJECT_CROP_AREA_RATIO = 0.9;
    private static final double BORDER_OBJECT_MARGIN_PX = 1.0;
    private static final int MIN_RESOLUTION_AWARE_CROP_SIDE_PX = 64;
    private static final String DEFAULT_CLASS_NAME = "object";
    private static final String GENERATED_YAML_NAME = "data.yaml";

    private static final Set<String> IMAGE_EXTENSIONS = new HashSet<String>();
    private static final Set<String> IMAGE_SUFFIXES = new HashSet<String>();
    private static final Set<String> MASK_SUFFIXES = new HashSet<String>();
    static {
        Collections.addAll(IMAGE_EXTENSIONS, ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff");
        Collections.addAll(IMAGE_SUFFIXES, "_img", "_image", "_sample", "_images", "_imgs", "_samples");
        Collections.addAll(MASK_SUFFIXES, "_mask", "_masks", "_label", "_labels");
    }

    private YoloDatasetPreparer() {}

    static File prepare(String datasetPath, String modelName, String modelsDir, Consumer<String> logConsumer)
            throws IOException {
        return prepare(datasetPath, modelName, modelsDir, YoloTrainingConfig.DEFAULT_IMAGE_SIZE, logConsumer);
    }

    static File prepare(String datasetPath, String modelName, String modelsDir, int imageSize,
            Consumer<String> logConsumer)
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
                    inferNamesFromLabelFiles(trainSplit.samples), trainSplit.samples, Collections.<YoloSample>emptyList(),
                    true, logConsumer);
            log(logConsumer, "Generated train/validation YOLO dataset: " + generatedYaml.getAbsolutePath());
            return generatedYaml;
        }

        List<MaskSample> trainMasks = firstNonEmpty(findMaskSamples(input, ""), findMaskSamples(input, "train"));
        List<MaskSample> valMasks = firstNonEmpty(findMaskSamples(input, "val"),
                findMaskSamples(input, "validation"),
                findMaskSamples(input, "valid"));
        if (!trainMasks.isEmpty()) {
            boolean splitTrain = valMasks.isEmpty();
            File generatedYaml = createGeneratedMaskDataset(input.getName(), modelName, modelsDir,
                    trainMasks, valMasks, splitTrain, imageSize, logConsumer);
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
        if (pathValue != null && !pathValue.trim().isEmpty() && resolvePath(yamlDir, pathValue).exists()) {
            root = resolvePath(yamlDir, pathValue);
        }
        SplitData train = collectYoloSplit("train", resolvePath(root, keys.get("train")), false);
        String valValue = keys.containsKey("val") ? keys.get("val") : keys.get("validation");
        SplitData val = collectYoloSplit("val", resolvePath(root, valValue), false);
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

    private static SplitData collectYoloSplit(String name, File splitPath, boolean parseBoxes) throws IOException {
        if (splitPath == null || !splitPath.exists()) {
            return null;
        }
        if (splitPath.isFile()) {
            return collectYoloSplitFromList(name, splitPath, parseBoxes);
        }
        SplitLayout layout = inferSplitLayout(splitPath);
        if (layout == null) {
            return null;
        }
        return collectYoloSplit(name, layout.imageRoot, layout.labelRoot, parseBoxes);
    }

    private static SplitData collectYoloSplitFromList(String name, File listFile, boolean parseBoxes) throws IOException {
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
                    YoloSample sample = readYoloSample(image, label, parseBoxes);
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
            SplitData split = collectYoloSplit(splitName, layout.imageRoot, layout.labelRoot, false);
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
        if ("train".equals(splitName)) {
            layouts.add(new SplitLayout(new File(root, "images"), new File(root, "labels")));
        }
        return layouts;
    }

    private static SplitData collectYoloSplit(String name, File imageRoot, File labelRoot, boolean parseBoxes)
            throws IOException {
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
            YoloSample sample = readYoloSample(image, label, parseBoxes);
            if (sample != null) {
                samples.add(sample);
            }
        }
        return samples.isEmpty() ? null : new SplitData(name, imageRoot, labelRoot, samples);
    }

    private static YoloSample readYoloSample(File image, File label, boolean parseBoxes) throws IOException {
        if (!parseBoxes) {
            return new YoloSample(image, label, -1, -1, Collections.<Box>emptyList());
        }
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
        List<String> names = inferNamesFromLabelFiles(train.samples, val.samples);
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
        writeLinkedYoloDataset(root,
                names == null || names.isEmpty() ? inferNamesFromLabelFiles(trainSamples, valSamples) : names,
                trainSamples, valSamples, splitTrainSamples, logConsumer);
        return new File(root, GENERATED_YAML_NAME);
    }

    private static void writeLinkedYoloDataset(File root, List<String> names,
            List<YoloSample> trainSamples, List<YoloSample> valSamples,
            boolean splitTrainSamples, Consumer<String> logConsumer) throws IOException {
        if (splitTrainSamples) {
            List<YoloSample> all = new ArrayList<YoloSample>(trainSamples);
            Collections.shuffle(all, new Random(SPLIT_SEED));
            int trainCount = splitIndex(all.size());
            trainSamples = new ArrayList<YoloSample>(all.subList(0, trainCount));
            valSamples = new ArrayList<YoloSample>(all.subList(trainCount, all.size()));
        }
        if (trainSamples.isEmpty() || valSamples.isEmpty()) {
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

        sortYoloSamples(trainSamples);
        sortYoloSamples(valSamples);
        writeLinkedSamples(trainSamples, trainImages, trainLabels);
        writeLinkedSamples(valSamples, valImages, valLabels);
        writeYaml(new File(root, GENERATED_YAML_NAME), root, "images/train", "images/val", names);
        log(logConsumer, "Prepared " + trainSamples.size() + " training and " + valSamples.size()
                + " validation YOLO samples.");
    }

    private static void sortYoloSamples(List<YoloSample> samples) {
        Collections.sort(samples, Comparator.comparing(sample -> sample.image.getAbsolutePath()));
    }

    private static void writeLinkedSamples(List<YoloSample> samples, File imageDir, File labelDir)
            throws IOException {
        int index = 0;
        for (YoloSample sample : samples) {
            String stem = safeFileName(removeExtension(sample.image.getName())) + "_" + String.format("%05d", index++);
            File imageOut = new File(imageDir, stem + extensionOrDefault(sample.image, ".png"));
            File labelOut = new File(labelDir, stem + ".txt");
            linkOrCopy(sample.image.toPath(), imageOut.toPath());
            linkOrCopy(sample.label.toPath(), labelOut.toPath());
        }
    }

    private static File createGeneratedMaskDataset(String sourceName, String modelName, String modelsDir,
            List<MaskSample> trainMasks, List<MaskSample> valMasks, boolean splitTrainSamples,
            int imageSize, Consumer<String> logConsumer) throws IOException {
        File root = createUniqueGeneratedRoot(sourceName, modelName, modelsDir);
        List<GeneratedSample> trainGenerated = toGeneratedSamplesFromMasks(trainMasks, imageSize);
        List<GeneratedSample> valGenerated = toGeneratedSamplesFromMasks(valMasks, imageSize);
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

    private static List<GeneratedSample> toGeneratedSamplesFromMasks(List<MaskSample> samples, int imageSize) {
        List<GeneratedSample> generated = new ArrayList<GeneratedSample>();
        for (MaskSample sample : samples) {
            generated.addAll(buildGeneratedSamples(sample.image, sample.width, sample.height, sample.boxes, imageSize));
        }
        return generated;
    }

    private static List<GeneratedSample> buildGeneratedSamples(
            File image, int width, int height, List<Box> boxes, int imageSize) {
        List<GeneratedSample> generated = new ArrayList<GeneratedSample>();
        if (boxes == null || boxes.isEmpty()) {
            return generated;
        }
        int yoloImageSize = Math.max(1, imageSize);
        int fullSampleSide = Math.max(width, height);
        double imageArea = Math.max(1.0, width * (double) height);
        double tinyObjectAreaReference = tinyObjectAreaReference(boxes);
        List<Box> validBoxes = new ArrayList<Box>();
        List<Box> fullBoxes = new ArrayList<Box>();
        List<Box> criticalBoxes = new ArrayList<Box>();
        for (Box box : boxes) {
            if (!box.isValid()) {
                continue;
            }
            if (shouldIgnoreTinyObject(box, imageArea, tinyObjectAreaReference)) {
                continue;
            }
            validBoxes.add(box);
            if (resizedMinSide(box, fullSampleSide, yoloImageSize) >= MIN_LABEL_RESIZED_SIDE_PX) {
                fullBoxes.add(box);
            }
            if (!touchesImageBorder(box, width, height)
                    && resizedMinSide(box, fullSampleSide, yoloImageSize) < CRITICAL_OBJECT_RESIZED_SIDE_PX) {
                criticalBoxes.add(box);
            }
        }
        if (hasBoxWithResizedMinSide(validBoxes, fullSampleSide, yoloImageSize, CRITICAL_OBJECT_RESIZED_SIDE_PX)
                && !fullBoxes.isEmpty()) {
            generated.add(GeneratedSample.full(image, width, height, fullBoxes));
        }

        List<Box> uncovered = new ArrayList<Box>(criticalBoxes);
        while (!uncovered.isEmpty()) {
            CropCandidate best = null;
            for (Box target : uncovered) {
                Rectangle crop = cropAround(target, width, height, yoloImageSize);
                CropCandidate candidate = cropCandidate(crop, validBoxes, uncovered, yoloImageSize);
                if (candidate.coveredCriticalBoxes.isEmpty()) {
                    continue;
                }
                if (best == null || candidate.compareTo(best) > 0) {
                    best = candidate;
                }
            }
            if (best == null) {
                break;
            }
            generated.add(GeneratedSample.crop(image, best.crop, best.labelBoxes));
            removeCovered(uncovered, best.coveredCriticalBoxes);
        }
        return generated;
    }

    private static boolean hasBoxWithResizedMinSide(
            List<Box> boxes, int sampleSide, int imageSize, double threshold) {
        for (Box box : boxes) {
            if (box.isValid() && resizedMinSide(box, sampleSide, imageSize) >= threshold) {
                return true;
            }
        }
        return false;
    }

    private static boolean touchesImageBorder(Box box, int width, int height) {
        return box.x1 <= BORDER_OBJECT_MARGIN_PX
                || box.y1 <= BORDER_OBJECT_MARGIN_PX
                || box.x2 >= width - BORDER_OBJECT_MARGIN_PX
                || box.y2 >= height - BORDER_OBJECT_MARGIN_PX;
    }

    private static boolean shouldIgnoreTinyObject(Box box, double imageArea, double tinyObjectAreaReference) {
        return tinyObjectAreaReference >= 0.0
                && box.objectArea() / imageArea < TINY_OBJECT_IMAGE_AREA_RATIO
                && box.objectArea() < tinyObjectAreaReference;
    }

    private static double tinyObjectAreaReference(List<Box> boxes) {
        if (boxes == null || boxes.size() < 2) {
            return -1.0;
        }
        List<Double> areas = new ArrayList<Double>();
        for (Box box : boxes) {
            areas.add(box.objectArea());
        }
        Collections.sort(areas);
        if (areas.size() == 2) {
            return areas.get(1).doubleValue() / TINY_OBJECT_REFERENCE_FACTOR;
        }
        return percentile(areas, 3.0) / TINY_OBJECT_REFERENCE_FACTOR;
    }

    private static double percentile(List<Double> sortedValues, double percentile) {
        if (sortedValues == null || sortedValues.isEmpty()) {
            return Double.NaN;
        }
        if (sortedValues.size() == 1) {
            return sortedValues.get(0).doubleValue();
        }
        double position = Math.max(0.0, Math.min(100.0, percentile)) / 100.0
                * (sortedValues.size() - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) {
            return sortedValues.get(lower).doubleValue();
        }
        double fraction = position - lower;
        return sortedValues.get(lower).doubleValue() * (1.0 - fraction)
                + sortedValues.get(upper).doubleValue() * fraction;
    }

    private static CropCandidate cropCandidate(
            Rectangle crop, List<Box> boxes, List<Box> uncoveredCriticalBoxes, int imageSize) {
        List<Box> labelBoxes = new ArrayList<Box>();
        for (Box box : boxes) {
            Box relative = box.relativeTo(crop);
            if (!relative.isValid()
                    || resizedMinSide(relative, Math.max(crop.width, crop.height), imageSize)
                            < MIN_LABEL_RESIZED_SIDE_PX) {
                continue;
            }
            if (uncoveredCriticalBoxes.contains(box)) {
                labelBoxes.add(relative);
            } else if (box.areaInside(crop) >= MIN_OBJECT_CROP_OVERLAP_RATIO * box.area()
                    && box.area() < MAX_OBJECT_CROP_AREA_RATIO * crop.width * (double) crop.height) {
                labelBoxes.add(relative);
            }
        }

        List<Box> coveredCriticalBoxes = new ArrayList<Box>();
        int sampleSide = Math.max(crop.width, crop.height);
        for (Box box : uncoveredCriticalBoxes) {
            if (box.fullyInside(crop)
                    && resizedMinSide(box, sampleSide, imageSize) >= CRITICAL_OBJECT_RESIZED_SIDE_PX) {
                coveredCriticalBoxes.add(box);
            }
        }
        return new CropCandidate(crop, labelBoxes, coveredCriticalBoxes);
    }

    private static void removeCovered(List<Box> uncovered, List<Box> covered) {
        Set<Integer> coveredIndices = new HashSet<Integer>();
        for (Box box : covered) {
            coveredIndices.add(box.objectIndex);
        }
        for (Iterator<Box> iterator = uncovered.iterator(); iterator.hasNext();) {
            if (coveredIndices.contains(iterator.next().objectIndex)) {
                iterator.remove();
            }
        }
    }

    private static double resizedMinSide(Box box, int sampleSide, int imageSize) {
        if (sampleSide <= 0) {
            return 0.0;
        }
        return Math.min(box.width(), box.height()) * imageSize / sampleSide;
    }

    private static Rectangle cropAround(
            Box box,
            int imageWidth,
            int imageHeight,
            int imageSize
    ) {
        double minSide = Math.max(1.0, Math.min(box.width(), box.height()));
        int side = (int) Math.ceil(minSide * imageSize / PREFERRED_CROP_RESIZED_SIDE_PX);
        side = Math.max(side, MIN_RESOLUTION_AWARE_CROP_SIDE_PX);
        side = Math.max(side, (int) Math.ceil(Math.max(box.width(), box.height())));
        side = Math.max(1, Math.min(side, Math.min(imageWidth, imageHeight)));
        int cropW = side;
        int cropH = side;
        int x = centeredCropStart((box.x1 + box.x2) / 2.0, cropW, imageWidth);
        int y = centeredCropStart((box.y1 + box.y2) / 2.0, cropH, imageHeight);
        return new Rectangle(x, y, cropW, cropH);
    }

    private static int centeredCropStart(double center, int cropSize, int imageSize) {
        int max = Math.max(0, imageSize - cropSize);
        int start = (int) Math.round(center - cropSize / 2.0);
        return Math.max(0, Math.min(max, start));
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
        for (SplitLayout layout : candidateMaskLayouts(root, splitName)) {
            List<MaskPair> pairs = findMaskPairs(layout.imageRoot, layout.labelRoot);
            if (!pairs.isEmpty()) {
                return readMaskSamples(pairs);
            }
        }
        if (!splitDir.isDirectory()) {
            return Collections.emptyList();
        }
        List<MaskPair> pairs = findMaskPairs(splitDir);
        return readMaskSamples(pairs);
    }

    private static List<SplitLayout> candidateMaskLayouts(File root, String splitName) {
        List<SplitLayout> layouts = new ArrayList<SplitLayout>();
        if (splitName == null || splitName.isEmpty()) {
            layouts.add(new SplitLayout(new File(root, "images"), new File(root, "labels")));
            return layouts;
        }
        layouts.add(new SplitLayout(new File(root, "images/" + splitName), new File(root, "labels/" + splitName)));
        layouts.add(new SplitLayout(new File(root, splitName + "/images"), new File(root, splitName + "/labels")));
        layouts.add(new SplitLayout(new File(root, splitName), new File(root, "labels/" + splitName)));
        layouts.add(new SplitLayout(new File(root, splitName), new File(root, splitName + "/labels")));
        return layouts;
    }

    private static List<MaskSample> readMaskSamples(List<MaskPair> pairs) throws IOException {
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

    private static List<MaskPair> findMaskPairs(File imageRoot, File maskRoot) throws IOException {
        if (imageRoot == null || maskRoot == null || !imageRoot.isDirectory() || !maskRoot.isDirectory()) {
            return Collections.emptyList();
        }
        Map<String, File> images = new HashMap<String, File>();
        for (File file : collectImages(imageRoot)) {
            images.put(normalizeImageCore(relativeStem(imageRoot, file)), file);
        }
        Map<String, File> masks = new HashMap<String, File>();
        for (File file : collectImages(maskRoot)) {
            String stem = relativeStem(maskRoot, file);
            String core = maskCore(stem);
            masks.put(core == null ? stem : core, file);
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

    private static String relativeStem(File root, File file) {
        Path rel = root.toPath().toAbsolutePath().relativize(file.toPath().toAbsolutePath());
        return removeExtension(rel.toString()).replace(File.separatorChar, '/');
    }

    private static String normalizeImageCore(String stem) {
        for (String suffix : IMAGE_SUFFIXES) {
            if (stem.endsWith(suffix)) {
                return stem.substring(0, stem.length() - suffix.length());
            }
        }
        return stem;
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
        Map<Integer, Integer> pixelAreas = new HashMap<Integer, Integer>();
        int width = Math.min(size.width, mask.getWidth());
        int height = Math.min(size.height, mask.getHeight());
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = raster.getSample(x, y, 0);
                if (value <= 0) {
                    continue;
                }
                pixelAreas.put(value, pixelAreas.getOrDefault(value, 0) + 1);
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
        for (Map.Entry<Integer, int[]> entry : bounds.entrySet()) {
            int[] b = entry.getValue();
            int pixelArea = pixelAreas.getOrDefault(entry.getKey(), 0);
            Box box = new Box(0, b[0], b[1], b[2] + 1.0, b[3] + 1.0, objectIndex++, pixelArea);
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
    private static List<String> inferNamesFromLabelFiles(List<YoloSample>... sampleLists) {
        int maxClass = -1;
        for (List<YoloSample> samples : sampleLists) {
            if (samples == null) {
                continue;
            }
            for (YoloSample sample : samples) {
                maxClass = Math.max(maxClass, maxClassId(sample.label));
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

    private static int maxClassId(File labelFile) {
        if (labelFile == null || !labelFile.isFile()) {
            return -1;
        }
        int maxClass = -1;
        try (BufferedReader reader = Files.newBufferedReader(labelFile.toPath(), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String clean = line.trim();
                if (clean.isEmpty() || clean.startsWith("#")) {
                    continue;
                }
                String[] parts = clean.split("\\s+");
                if (parts.length == 0) {
                    continue;
                }
                try {
                    maxClass = Math.max(maxClass, Integer.parseInt(parts[0]));
                } catch (NumberFormatException e) {
                    // Ignore malformed label rows.
                }
            }
        } catch (IOException e) {
            return maxClass;
        }
        return maxClass;
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
            return names == null || names.isEmpty() ? inferNamesFromLabelFiles(train.samples) : names;
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
        private final File imageRoot;
        private final List<YoloSample> samples;

        private SplitData(String name, File imageRoot, File labelRoot, List<YoloSample> samples) {
            this.imageRoot = imageRoot;
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

        private YoloSample(File image, File label, int width, int height, List<Box> boxes) {
            this.image = image;
            this.label = label;
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

    private static final class CropCandidate implements Comparable<CropCandidate> {
        private final Rectangle crop;
        private final List<Box> labelBoxes;
        private final List<Box> coveredCriticalBoxes;

        private CropCandidate(Rectangle crop, List<Box> labelBoxes, List<Box> coveredCriticalBoxes) {
            this.crop = crop;
            this.labelBoxes = labelBoxes;
            this.coveredCriticalBoxes = coveredCriticalBoxes;
        }

        @Override
        public int compareTo(CropCandidate other) {
            int covered = Integer.compare(coveredCriticalBoxes.size(), other.coveredCriticalBoxes.size());
            if (covered != 0) {
                return covered;
            }
            int labels = Integer.compare(labelBoxes.size(), other.labelBoxes.size());
            if (labels != 0) {
                return labels;
            }
            return Integer.compare(Math.max(other.crop.width, other.crop.height), Math.max(crop.width, crop.height));
        }
    }

    private static final class Box {
        private final int classId;
        private final double x1;
        private final double y1;
        private final double x2;
        private final double y2;
        private final int objectIndex;
        private final int objectArea;

        private Box(int classId, double x1, double y1, double x2, double y2, int objectIndex) {
            this(classId, x1, y1, x2, y2, objectIndex, -1);
        }

        private Box(int classId, double x1, double y1, double x2, double y2, int objectIndex, int objectArea) {
            this.classId = classId;
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.objectIndex = objectIndex;
            this.objectArea = objectArea;
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

        private double objectArea() {
            return objectArea > 0 ? objectArea : area();
        }

        private boolean isValid() {
            return width() > 0.0 && height() > 0.0;
        }

        private boolean fullyInside(Rectangle rectangle) {
            return x1 >= rectangle.x && y1 >= rectangle.y
                    && x2 <= rectangle.x + rectangle.width
                    && y2 <= rectangle.y + rectangle.height;
        }

        private double areaInside(Rectangle rectangle) {
            double overlapX1 = Math.max(x1, rectangle.x);
            double overlapY1 = Math.max(y1, rectangle.y);
            double overlapX2 = Math.min(x2, rectangle.x + rectangle.width);
            double overlapY2 = Math.min(y2, rectangle.y + rectangle.height);
            double overlapWidth = Math.max(0.0, overlapX2 - overlapX1);
            double overlapHeight = Math.max(0.0, overlapY2 - overlapY1);
            return overlapWidth * overlapHeight;
        }

        private Box relativeTo(Rectangle rectangle) {
            return new Box(classId, x1 - rectangle.x, y1 - rectangle.y,
                    x2 - rectangle.x, y2 - rectangle.y, objectIndex, objectArea)
                    .clamped(rectangle.width, rectangle.height);
        }

        private Box clamped(int width, int height) {
            return new Box(classId,
                    Math.max(0.0, Math.min(width, x1)),
                    Math.max(0.0, Math.min(height, y1)),
                    Math.max(0.0, Math.min(width, x2)),
                    Math.max(0.0, Math.min(height, y2)),
                    objectIndex,
                    objectArea);
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
