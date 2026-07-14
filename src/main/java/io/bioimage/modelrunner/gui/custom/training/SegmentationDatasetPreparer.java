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
package io.bioimage.modelrunner.gui.custom.training;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

/**
 * Thin Java-side normalizer for image/mask segmentation datasets.
 * <p>
 * It keeps already-supported layouts untouched and creates linked canonical
 * train/val folders only when the backend would otherwise misread the dataset.
 * </p>
 */
public final class SegmentationDatasetPreparer {

    public enum Framework {
        STARDIST,
        UNET
    }

    private static final long SPLIT_SEED = 5489L;
    private static final double DEFAULT_VALID_FRACTION = 0.15d;
    private static final int MAX_REPORTED_ITEMS = 5;
    private static final int SUSPICIOUS_UNIQUE_LABEL_COUNT = 128;
    private static final double SUSPICIOUS_DENSE_LABEL_FRACTION = 0.85d;
    private static final String GENERATED_INFO_NAME = "dataset-links.txt";

    private static final Set<String> IMAGE_EXTENSIONS = new HashSet<String>();
    private static final Set<String> MASK_EXTENSIONS = new HashSet<String>();
    private static final List<String> IMAGE_ALIASES = Arrays.asList(
            "images", "image", "imgs", "img", "data");
    private static final List<String> MASK_ALIASES = Arrays.asList(
            "masks", "mask", "labels", "label", "gt");
    private static final List<String> VAL_ALIASES = Arrays.asList("val", "validation", "valid");
    private static final List<String> IMAGE_SUFFIXES = Arrays.asList(
            "_image", "-image", "_images", "-images",
            "_img", "-img", "_imgs", "-imgs",
            "_sample", "-sample", "_samples", "-samples",
            "_raw", "-raw");
    private static final List<String> MASK_SUFFIXES = Arrays.asList(
            "_mask", "-mask", "_masks", "-masks",
            "_label", "-label", "_labels", "-labels",
            "_gt", "-gt", "_seg", "-seg", "_segmentation", "-segmentation");
    private static final List<String> UNET_IMAGE_SUFFIXES = Arrays.asList(
            "_image", "-image", "_img", "-img", "_raw", "-raw");
    private static final List<String> UNET_MASK_SUFFIXES = Arrays.asList(
            "_mask", "-mask", "_label", "-label", "_labels", "-labels", "_gt", "-gt");
    private static final List<String> STARDIST_IMAGE_SUFFIXES = Collections.singletonList("_image");
    private static final List<String> STARDIST_MASK_SUFFIXES = Arrays.asList(
            "_mask", "_masks", "_label", "_labels");

    static {
        Collections.addAll(IMAGE_EXTENSIONS, ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp");
        Collections.addAll(MASK_EXTENSIONS, ".tif", ".tiff", ".png", ".bmp");
    }

    private SegmentationDatasetPreparer() {}

    /**
     * Prepares a segmentation dataset for a Python backend.
     *
     * @param datasetPath the user-provided dataset path.
     * @param modelName the model name, used when a linked dataset is needed.
     * @param modelsDir the software models directory.
     * @param validFraction the validation fraction used for generated train/val splits.
     * @param framework the target framework.
     * @param logConsumer optional log consumer.
     * @return the prepared dataset.
     * @throws IOException if dataset preparation fails.
     */
    public static PreparedDataset prepare(String datasetPath, String modelName, String modelsDir,
            double validFraction, Framework framework, Consumer<String> logConsumer) throws IOException {
        if (datasetPath == null || datasetPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Please provide a training dataset path.");
        }
        Framework target = framework == null ? Framework.UNET : framework;
        File input = new File(datasetPath.trim()).getAbsoluteFile();
        if (!input.isDirectory()) {
            throw new IllegalArgumentException("The training dataset path must be a readable folder: " + input);
        }

        log(logConsumer, "Analyzing segmentation dataset at: " + input.getAbsolutePath());
        Discovery discovery = discover(input, target);
        if (discovery.train.rawPairs.isEmpty()) {
            throw new IllegalArgumentException("Could not find image/mask pairs in: " + input);
        }

        Summary summary = new Summary();
        summary.sourceDescription = discovery.description;
        summary.explicitValidationSplit = !discovery.val.rawPairs.isEmpty();
        Split train = validateSplit("train", discovery.train, summary, target);
        Split val = validateSplit("val", discovery.val, summary, target);
        if (target == Framework.UNET && summary.missingMasks > 0) {
            throw new IllegalArgumentException("UNet training requires one mask per image. Missing masks: "
                    + summary.missingMasks + firstItems(summary.missingMaskExamples));
        }
        if (train.pairs.isEmpty()) {
            throw new IllegalArgumentException("No valid training image/mask pairs remain after dataset checks.");
        }

        boolean backendCanPairDirectly = backendCanPairDirectly(train, target)
                && backendCanPairDirectly(val, target);
        boolean needsGeneratedDataset = !discovery.directlyDigestible || !backendCanPairDirectly
                || summary.requiresFilteredDataset();
        if (needsGeneratedDataset && val.pairs.isEmpty() && train.pairs.size() > 1) {
            SplitPair split = splitTrainVal(train.pairs, validFraction);
            train = new Split("train", split.train);
            val = new Split("val", split.val);
            summary.generatedValidationSplit = !val.pairs.isEmpty();
        }

        File datasetRoot = input;
        boolean generated = false;
        if (needsGeneratedDataset) {
            datasetRoot = writeLinkedDataset(input.getName(), modelName, modelsDir, train, val, logConsumer);
            generated = true;
        }
        logSummary(logConsumer, datasetRoot, generated, summary, train, val);
        return new PreparedDataset(datasetRoot, generated, summary);
    }

    private static Discovery discover(File root, Framework framework) throws IOException {
        LayoutResult train = findSplit(root, "train", framework);
        if (!train.rawPairs.isEmpty()) {
            LayoutResult val = new LayoutResult("val", true, Collections.<RawPair>emptyList());
            boolean valDirect = true;
            for (String alias : VAL_ALIASES) {
                LayoutResult candidate = findSplit(root, alias, framework);
                if (!candidate.rawPairs.isEmpty()) {
                    val = candidate;
                    valDirect = candidate.directlyDigestible;
                    break;
                }
            }
            boolean direct = train.directlyDigestible && valDirect;
            return new Discovery(train, val, direct, "split folders");
        }

        LayoutResult rootSeparated = findSeparated(root, framework);
        if (!rootSeparated.rawPairs.isEmpty()) {
            return new Discovery(rootSeparated, new LayoutResult("val", true, Collections.<RawPair>emptyList()),
                    rootSeparated.directlyDigestible, "images/masks folders");
        }

        LayoutResult rootMixed = findMixed(root, "root mixed folder");
        if (!rootMixed.rawPairs.isEmpty()) {
            return new Discovery(rootMixed, new LayoutResult("val", true, Collections.<RawPair>emptyList()),
                    false, "mixed image/mask folder");
        }
        return new Discovery(new LayoutResult("train", false, Collections.<RawPair>emptyList()),
                new LayoutResult("val", false, Collections.<RawPair>emptyList()), false, "unsupported layout");
    }

    private static LayoutResult findSplit(File root, String splitName, Framework framework) throws IOException {
        File splitRoot = new File(root, splitName);
        for (SplitLayout layout : candidateLayouts(root, splitName, framework)) {
            List<RawPair> pairs = pair(layout.imageRoot, layout.maskRoot, false);
            if (!pairs.isEmpty()) {
                return new LayoutResult(splitName, layout.directlyDigestible, pairs);
            }
        }
        if (splitRoot.isDirectory()) {
            LayoutResult mixed = findMixed(splitRoot, splitName + " mixed folder");
            if (!mixed.rawPairs.isEmpty()) {
                return mixed;
            }
        }
        return new LayoutResult(splitName, false, Collections.<RawPair>emptyList());
    }

    private static LayoutResult findSeparated(File root, Framework framework) throws IOException {
        File imageRoot = firstAliasFolder(root, IMAGE_ALIASES);
        File maskRoot = firstAliasFolder(root, MASK_ALIASES);
        List<RawPair> pairs = pair(imageRoot, maskRoot, false);
        return new LayoutResult("root", !pairs.isEmpty(), pairs);
    }

    private static LayoutResult findMixed(File root, String description) throws IOException {
        return new LayoutResult(description, false, pair(root, root, true));
    }

    private static List<SplitLayout> candidateLayouts(File root, String splitName, Framework framework) {
        List<SplitLayout> layouts = new ArrayList<SplitLayout>();
        File splitRoot = new File(root, splitName);
        for (String imageAlias : IMAGE_ALIASES) {
            for (String maskAlias : MASK_ALIASES) {
                boolean direct = "train".equals(splitName) || "val".equals(splitName)
                        || (framework == Framework.STARDIST && "validation".equals(splitName));
                layouts.add(new SplitLayout(new File(splitRoot, imageAlias), new File(splitRoot, maskAlias), direct));
                layouts.add(new SplitLayout(new File(new File(root, imageAlias), splitName),
                        new File(new File(root, maskAlias), splitName), false));
            }
        }
        for (String maskAlias : MASK_ALIASES) {
            layouts.add(new SplitLayout(splitRoot, new File(new File(root, maskAlias), splitName), false));
            layouts.add(new SplitLayout(splitRoot, new File(splitRoot, maskAlias), false));
        }
        return layouts;
    }

    private static List<RawPair> pair(File imageRoot, File maskRoot, boolean mixed) throws IOException {
        if (imageRoot == null || maskRoot == null || !imageRoot.isDirectory() || !maskRoot.isDirectory()) {
            return Collections.emptyList();
        }
        List<File> imageFiles = firstLevelFiles(imageRoot, IMAGE_EXTENSIONS);
        List<File> maskFiles = firstLevelFiles(maskRoot, MASK_EXTENSIONS);
        if (mixed) {
            List<File> classifiedImages = new ArrayList<File>();
            List<File> classifiedMasks = new ArrayList<File>();
            for (File file : imageFiles) {
                if (maskSuffix(file) == null) {
                    classifiedImages.add(file);
                }
            }
            for (File file : maskFiles) {
                if (maskSuffix(file) != null) {
                    classifiedMasks.add(file);
                }
            }
            imageFiles = classifiedImages;
            maskFiles = classifiedMasks;
        }

        Map<String, Integer> extensionCounts = new HashMap<String, Integer>();
        Map<String, Integer> suffixCounts = new HashMap<String, Integer>();
        Map<String, List<MaskCandidate>> masksByKey = new HashMap<String, List<MaskCandidate>>();
        for (File image : imageFiles) {
            increment(extensionCounts, extension(image));
        }
        for (File mask : maskFiles) {
            String extension = extension(mask);
            String suffix = maskSuffix(mask);
            increment(extensionCounts, extension);
            increment(suffixCounts, suffix == null ? "" : suffix);
            String key = canonicalMaskKey(mask);
            MaskCandidate candidate = new MaskCandidate(mask, extension, suffix == null ? "" : suffix);
            masksByKey.computeIfAbsent(key, k -> new ArrayList<MaskCandidate>()).add(candidate);
        }

        List<RawPair> pairs = new ArrayList<RawPair>();
        for (File image : imageFiles) {
            LinkedHashSet<MaskCandidate> candidates = new LinkedHashSet<MaskCandidate>();
            for (String key : imageKeys(image)) {
                List<MaskCandidate> found = masksByKey.get(key);
                if (found != null) {
                    candidates.addAll(found);
                }
            }
            if (candidates.isEmpty()) {
                pairs.add(RawPair.missing(image));
                continue;
            }
            AmbiguityChoice choice = chooseMask(new ArrayList<MaskCandidate>(candidates),
                    extensionCounts, suffixCounts);
            if (choice.mask == null) {
                pairs.add(RawPair.ambiguous(image, files(candidates)));
            } else {
                pairs.add(RawPair.paired(image, choice.mask.file, canonicalImageKey(image), choice.resolvedAmbiguity));
            }
        }
        Collections.sort(pairs, Comparator.comparing(pair -> pair.image.getAbsolutePath()));
        return pairs;
    }

    private static AmbiguityChoice chooseMask(List<MaskCandidate> candidates,
            Map<String, Integer> extensionCounts, Map<String, Integer> suffixCounts) {
        if (candidates.size() == 1) {
            return new AmbiguityChoice(candidates.get(0), false);
        }
        MaskCandidate best = null;
        int[] bestScore = null;
        boolean tie = false;
        for (MaskCandidate candidate : candidates) {
            int[] score = new int[] {
                    extensionCounts.getOrDefault(candidate.extension, 0),
                    suffixCounts.getOrDefault(candidate.suffix, 0),
                    candidate.suffix.isEmpty() ? 0 : 1
            };
            if (best == null || compareScore(score, bestScore) > 0) {
                best = candidate;
                bestScore = score;
                tie = false;
            } else if (compareScore(score, bestScore) == 0) {
                tie = true;
            }
        }
        return tie ? new AmbiguityChoice(null, false) : new AmbiguityChoice(best, true);
    }

    private static int compareScore(int[] a, int[] b) {
        if (b == null) {
            return 1;
        }
        for (int i = 0; i < a.length; i++) {
            int cmp = Integer.compare(a[i], b[i]);
            if (cmp != 0) {
                return cmp;
            }
        }
        return 0;
    }

    private static Split validateSplit(String name, LayoutResult layout, Summary summary, Framework framework)
            throws IOException {
        List<Pair> valid = new ArrayList<Pair>();
        for (RawPair raw : layout.rawPairs) {
            summary.sourceImages++;
            if (raw.missingMask) {
                summary.missingMasks++;
                addExample(summary.missingMaskExamples, raw.image.getName());
                continue;
            }
            if (raw.ambiguousMasks != null) {
                summary.ambiguousDropped++;
                addExample(summary.ambiguousExamples, raw.image.getName());
                continue;
            }
            if (raw.resolvedAmbiguity) {
                summary.ambiguousResolved++;
            }
            PairStats stats = inspectPair(raw, framework);
            if (stats.unreadable) {
                summary.unreadablePairs++;
                addExample(summary.unreadableExamples, raw.image.getName());
                continue;
            }
            if (stats.shapeMismatch) {
                summary.shapeMismatches++;
                addExample(summary.shapeMismatchExamples, raw.image.getName());
                continue;
            }
            if (stats.unsupportedMask) {
                summary.unsupportedMasks++;
                addExample(summary.unsupportedMaskExamples, raw.mask.getName());
                continue;
            }
            if (stats.emptyMask) {
                summary.emptyMasks++;
            }
            if (stats.multichannelMask) {
                summary.multichannelMasks++;
                addExample(summary.multichannelExamples, raw.mask.getName());
            }
            if (stats.suspiciousContinuousMask) {
                summary.suspiciousMasks++;
                addExample(summary.suspiciousMaskExamples, raw.mask.getName());
            }
            summary.sourceMasks++;
            summary.objects += stats.objectCount;
            valid.add(new Pair(raw.image, raw.mask, raw.key, stats.objectCount));
        }
        return new Split(name, valid);
    }

    private static PairStats inspectPair(RawPair raw, Framework framework) throws IOException {
        ImageSize imageSize = readImageSize(raw.image);
        MaskStats maskStats = readMaskStats(raw.mask);
        PairStats stats = new PairStats();
        if (imageSize == null || maskStats == null) {
            stats.unreadable = true;
            return stats;
        }
        stats.shapeMismatch = imageSize.width != maskStats.width || imageSize.height != maskStats.height;
        stats.unsupportedMask = maskStats.floatData;
        stats.emptyMask = maskStats.objectCount == 0;
        stats.objectCount = maskStats.objectCount;
        stats.multichannelMask = maskStats.numBands > 1;
        stats.suspiciousContinuousMask = maskStats.suspiciousContinuousLike;
        return stats;
    }

    private static MaskStats readMaskStats(File maskFile) throws IOException {
        BufferedImage image = ImageIO.read(maskFile);
        if (image == null) {
            return null;
        }
        Raster raster = image.getRaster();
        int dataType = raster.getTransferType();
        MaskStats stats = new MaskStats();
        stats.width = image.getWidth();
        stats.height = image.getHeight();
        stats.numBands = raster.getNumBands();
        stats.floatData = dataType == DataBuffer.TYPE_FLOAT || dataType == DataBuffer.TYPE_DOUBLE;
        if (stats.floatData) {
            return stats;
        }
        Map<Integer, Integer> histogram = new HashMap<Integer, Integer>();
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int value = raster.getSample(x, y, 0);
                if (value <= 0) {
                    continue;
                }
                histogram.put(value, histogram.getOrDefault(value, 0) + 1);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }
        stats.objectCount = histogram.size();
        if (!histogram.isEmpty()) {
            int valueRange = Math.max(1, max - min + 1);
            double denseFraction = histogram.size() / (double) valueRange;
            stats.suspiciousContinuousLike = histogram.size() >= SUSPICIOUS_UNIQUE_LABEL_COUNT
                    && denseFraction >= SUSPICIOUS_DENSE_LABEL_FRACTION;
        }
        return stats;
    }

    private static File writeLinkedDataset(String sourceName, String modelName, String modelsDir,
            Split train, Split val, Consumer<String> logConsumer) throws IOException {
        File root = createUniqueGeneratedRoot(sourceName, modelName, modelsDir);
        log(logConsumer, "Creating linked segmentation dataset at: " + root.getAbsolutePath());
        writeSplit(root, train);
        writeSplit(root, val);
        Files.write(new File(root, GENERATED_INFO_NAME).toPath(), Arrays.asList(
                "This dataset contains links to the original images and masks.",
                "Generated by JDLL to normalize a segmentation dataset layout."));
        return root;
    }

    private static boolean backendCanPairDirectly(Split split, Framework framework) {
        List<String> imageSuffixes = framework == Framework.STARDIST
                ? STARDIST_IMAGE_SUFFIXES : UNET_IMAGE_SUFFIXES;
        List<String> maskSuffixes = framework == Framework.STARDIST
                ? STARDIST_MASK_SUFFIXES : UNET_MASK_SUFFIXES;
        for (Pair pair : split.pairs) {
            String imageStem = removeExtension(pair.image.getName()).toLowerCase(Locale.ROOT);
            String maskStem = removeExtension(pair.mask.getName()).toLowerCase(Locale.ROOT);
            String imageKey = stripSuffix(imageStem, imageSuffixes);
            String maskKey = stripSuffix(maskStem, maskSuffixes);
            if (!imageStem.equals(maskStem) && !imageStem.equals(maskKey)
                    && !imageKey.equals(maskStem) && !imageKey.equals(maskKey)) {
                return false;
            }
        }
        return true;
    }

    private static void writeSplit(File root, Split split) throws IOException {
        if (split == null || split.pairs.isEmpty()) {
            return;
        }
        Set<String> usedNames = new HashSet<String>();
        File imageRoot = new File(new File(root, split.name), "images");
        File maskRoot = new File(new File(root, split.name), "masks");
        Files.createDirectories(imageRoot.toPath());
        Files.createDirectories(maskRoot.toPath());
        for (Pair pair : split.pairs) {
            String base = uniqueBaseName(safeFileName(pair.key), usedNames);
            File imageTarget = new File(imageRoot, base + extensionOrDefault(pair.image, ".tif"));
            File maskTarget = new File(maskRoot, base + "_mask" + extensionOrDefault(pair.mask, ".tif"));
            linkOnly(pair.image.toPath(), imageTarget.toPath());
            linkOnly(pair.mask.toPath(), maskTarget.toPath());
        }
    }

    private static SplitPair splitTrainVal(List<Pair> pairs, double validFraction) {
        double fraction = validFraction > 0 && validFraction < 1 ? validFraction : DEFAULT_VALID_FRACTION;
        List<Pair> shuffled = new ArrayList<Pair>(pairs);
        Collections.shuffle(shuffled, new Random(SPLIT_SEED));
        int valCount = Math.max(1, (int) Math.round(shuffled.size() * fraction));
        valCount = Math.min(valCount, shuffled.size() - 1);
        List<Pair> val = new ArrayList<Pair>(shuffled.subList(0, valCount));
        List<Pair> train = new ArrayList<Pair>(shuffled.subList(valCount, shuffled.size()));
        return new SplitPair(train, val);
    }

    private static void logSummary(Consumer<String> logConsumer, File datasetRoot, boolean generated,
            Summary summary, Split train, Split val) {
        log(logConsumer, "Segmentation dataset source: " + summary.sourceDescription
                + (generated ? "; generated linked dataset." : "; reused original dataset."));
        log(logConsumer, "Final dataset path: " + datasetRoot.getAbsolutePath());
        log(logConsumer, "Training split: images=" + train.pairs.size()
                + ", masks=" + train.pairs.size()
                + ", objects=" + objectCount(train) + ".");
        log(logConsumer, "Validation split: images=" + val.pairs.size()
                + ", masks=" + val.pairs.size()
                + ", objects=" + objectCount(val)
                + validationSplitNote(summary, val));
        log(logConsumer, "Dataset checks: source_images=" + summary.sourceImages
                + ", source_masks=" + summary.sourceMasks
                + ", missing_masks=" + summary.missingMasks
                + ", shape_mismatches=" + summary.shapeMismatches
                + ", ambiguous_resolved=" + summary.ambiguousResolved
                + ", ambiguous_dropped=" + summary.ambiguousDropped
                + ", empty_masks=" + summary.emptyMasks
                + ", unsupported_masks=" + summary.unsupportedMasks
                + ", unreadable_pairs=" + summary.unreadablePairs
                + ", multichannel_masks=" + summary.multichannelMasks
                + ", suspicious_integer_masks=" + summary.suspiciousMasks + ".");
        logExamples(logConsumer, "Missing mask examples", summary.missingMaskExamples);
        logExamples(logConsumer, "Shape mismatch examples", summary.shapeMismatchExamples);
        logExamples(logConsumer, "Ambiguous mask examples", summary.ambiguousExamples);
        logExamples(logConsumer, "Unsupported mask examples", summary.unsupportedMaskExamples);
        logExamples(logConsumer, "Suspicious integer mask examples", summary.suspiciousMaskExamples);
        if (summary.multichannelMasks > 0) {
            logExamples(logConsumer, "Multichannel masks using first channel", summary.multichannelExamples);
        }
    }

    private static long objectCount(Split split) {
        long count = 0;
        for (Pair pair : split.pairs) {
            count += pair.objectCount;
        }
        return count;
    }

    private static String validationSplitNote(Summary summary, Split val) {
        if (summary.generatedValidationSplit) {
            return " (created automatically).";
        }
        if (!summary.explicitValidationSplit && val.pairs.isEmpty()) {
            return " (backend will create validation split from training data).";
        }
        return ".";
    }

    private static void logExamples(Consumer<String> logConsumer, String label, List<String> examples) {
        if (examples.isEmpty()) {
            return;
        }
        log(logConsumer, label + ": " + String.join(", ", examples)
                + (examples.size() == MAX_REPORTED_ITEMS ? ", ..." : ""));
    }

    private static List<File> files(Set<MaskCandidate> candidates) {
        List<File> files = new ArrayList<File>();
        for (MaskCandidate candidate : candidates) {
            files.add(candidate.file);
        }
        return files;
    }

    private static List<File> firstLevelFiles(File root, Set<String> extensions) throws IOException {
        if (root == null || !root.isDirectory()) {
            return Collections.emptyList();
        }
        List<File> files = new ArrayList<File>();
        File[] listed = root.listFiles();
        if (listed == null) {
            return files;
        }
        for (File file : listed) {
            if (file.isFile() && hasExtension(file, extensions) && !file.getName().startsWith(".")) {
                files.add(file.getAbsoluteFile());
            }
        }
        Collections.sort(files, Comparator.comparing(File::getAbsolutePath));
        return files;
    }

    private static File firstAliasFolder(File root, List<String> aliases) {
        if (root == null || !root.isDirectory()) {
            return null;
        }
        File[] children = root.listFiles(File::isDirectory);
        if (children == null) {
            return null;
        }
        Map<String, File> byName = new HashMap<String, File>();
        for (File child : children) {
            byName.put(child.getName().toLowerCase(Locale.ROOT), child);
        }
        for (String alias : aliases) {
            File folder = byName.get(alias);
            if (folder != null) {
                return folder;
            }
        }
        return null;
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

    private static void linkOnly(Path source, Path target) throws IOException {
        Files.createDirectories(target.getParent());
        try {
            Files.createSymbolicLink(target, source.toAbsolutePath());
            return;
        } catch (UnsupportedOperationException | SecurityException | IOException e) {
            // Fall through to hard links. Copying is intentionally avoided for training datasets.
        }
        Files.createLink(target, source.toAbsolutePath());
    }

    private static String uniqueBaseName(String base, Set<String> used) {
        String safe = base == null || base.isEmpty() ? "sample" : base;
        String candidate = safe;
        int suffix = 1;
        while (!used.add(candidate)) {
            candidate = safe + "-" + suffix++;
        }
        return candidate;
    }

    private static Set<String> imageKeys(File image) {
        LinkedHashSet<String> keys = new LinkedHashSet<String>();
        String stem = removeExtension(image.getName());
        keys.add(stem.toLowerCase(Locale.ROOT));
        keys.add(canonicalImageKey(image));
        return keys;
    }

    private static String canonicalImageKey(File image) {
        return stripSuffix(removeExtension(image.getName()), IMAGE_SUFFIXES).toLowerCase(Locale.ROOT);
    }

    private static String canonicalMaskKey(File mask) {
        return stripSuffix(removeExtension(mask.getName()), MASK_SUFFIXES).toLowerCase(Locale.ROOT);
    }

    private static String stripSuffix(String stem, List<String> suffixes) {
        String lower = stem.toLowerCase(Locale.ROOT);
        for (String suffix : suffixes) {
            if (lower.endsWith(suffix)) {
                return stem.substring(0, stem.length() - suffix.length());
            }
        }
        return stem;
    }

    private static String maskSuffix(File mask) {
        String stem = removeExtension(mask.getName()).toLowerCase(Locale.ROOT);
        for (String suffix : MASK_SUFFIXES) {
            if (stem.endsWith(suffix)) {
                return suffix;
            }
        }
        return null;
    }

    private static boolean hasExtension(File file, Set<String> extensions) {
        String name = file.getName().toLowerCase(Locale.ROOT);
        for (String extension : extensions) {
            if (name.endsWith(extension)) {
                return true;
            }
        }
        return false;
    }

    private static String extension(File file) {
        String name = file.getName();
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot).toLowerCase(Locale.ROOT);
    }

    private static String extensionOrDefault(File file, String defaultExtension) {
        String ext = extension(file);
        return ext.isEmpty() ? defaultExtension : ext;
    }

    private static String removeExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? name : name.substring(0, dot);
    }

    private static String safeFileName(String name) {
        String clean = name == null ? "" : name.trim();
        clean = clean.replaceAll("[^A-Za-z0-9._-]+", "_");
        clean = clean.replaceAll("_+", "_");
        if (clean.isEmpty() || ".".equals(clean) || "..".equals(clean)) {
            return "dataset";
        }
        return clean;
    }

    private static void increment(Map<String, Integer> counts, String key) {
        counts.put(key, counts.getOrDefault(key, 0) + 1);
    }

    private static void addExample(List<String> examples, String value) {
        if (examples.size() < MAX_REPORTED_ITEMS) {
            examples.add(value);
        }
    }

    private static String firstItems(List<String> examples) {
        return examples.isEmpty() ? "" : " (" + String.join(", ", examples) + ")";
    }

    private static void log(Consumer<String> logConsumer, String message) {
        if (logConsumer != null) {
            logConsumer.accept(message);
        }
    }

    public static final class PreparedDataset {
        private final File datasetRoot;
        private final boolean generated;
        private final Summary summary;

        private PreparedDataset(File datasetRoot, boolean generated, Summary summary) {
            this.datasetRoot = datasetRoot;
            this.generated = generated;
            this.summary = summary;
        }

        public File getDatasetRoot() {
            return datasetRoot;
        }

        public boolean isGenerated() {
            return generated;
        }

        public Summary getSummary() {
            return summary;
        }
    }

    public static final class Summary {
        private String sourceDescription = "";
        private int sourceImages;
        private int sourceMasks;
        private int missingMasks;
        private int shapeMismatches;
        private int ambiguousResolved;
        private int ambiguousDropped;
        private int emptyMasks;
        private int unsupportedMasks;
        private int unreadablePairs;
        private int multichannelMasks;
        private int suspiciousMasks;
        private long objects;
        private boolean generatedValidationSplit;
        private boolean explicitValidationSplit;
        private final List<String> missingMaskExamples = new ArrayList<String>();
        private final List<String> shapeMismatchExamples = new ArrayList<String>();
        private final List<String> ambiguousExamples = new ArrayList<String>();
        private final List<String> unsupportedMaskExamples = new ArrayList<String>();
        private final List<String> unreadableExamples = new ArrayList<String>();
        private final List<String> multichannelExamples = new ArrayList<String>();
        private final List<String> suspiciousMaskExamples = new ArrayList<String>();

        private boolean requiresFilteredDataset() {
            return shapeMismatches > 0 || ambiguousResolved > 0 || ambiguousDropped > 0
                    || unsupportedMasks > 0 || unreadablePairs > 0 || missingMasks > 0;
        }
    }

    private static final class Discovery {
        private final LayoutResult train;
        private final LayoutResult val;
        private final boolean directlyDigestible;
        private final String description;

        private Discovery(LayoutResult train, LayoutResult val, boolean directlyDigestible, String description) {
            this.train = train;
            this.val = val;
            this.directlyDigestible = directlyDigestible;
            this.description = description;
        }
    }

    private static final class LayoutResult {
        private final String name;
        private final boolean directlyDigestible;
        private final List<RawPair> rawPairs;

        private LayoutResult(String name, boolean directlyDigestible, List<RawPair> rawPairs) {
            this.name = name;
            this.directlyDigestible = directlyDigestible;
            this.rawPairs = rawPairs;
        }
    }

    private static final class SplitLayout {
        private final File imageRoot;
        private final File maskRoot;
        private final boolean directlyDigestible;

        private SplitLayout(File imageRoot, File maskRoot, boolean directlyDigestible) {
            this.imageRoot = imageRoot;
            this.maskRoot = maskRoot;
            this.directlyDigestible = directlyDigestible;
        }
    }

    private static final class Split {
        private final String name;
        private final List<Pair> pairs;

        private Split(String name, List<Pair> pairs) {
            this.name = name;
            this.pairs = pairs == null ? Collections.<Pair>emptyList() : pairs;
        }
    }

    private static final class SplitPair {
        private final List<Pair> train;
        private final List<Pair> val;

        private SplitPair(List<Pair> train, List<Pair> val) {
            this.train = train;
            this.val = val;
        }
    }

    private static final class RawPair {
        private final File image;
        private final File mask;
        private final String key;
        private final boolean missingMask;
        private final boolean resolvedAmbiguity;
        private final List<File> ambiguousMasks;

        private RawPair(File image, File mask, String key, boolean missingMask,
                boolean resolvedAmbiguity, List<File> ambiguousMasks) {
            this.image = image;
            this.mask = mask;
            this.key = key;
            this.missingMask = missingMask;
            this.resolvedAmbiguity = resolvedAmbiguity;
            this.ambiguousMasks = ambiguousMasks;
        }

        private static RawPair paired(File image, File mask, String key, boolean resolvedAmbiguity) {
            return new RawPair(image, mask, key, false, resolvedAmbiguity, null);
        }

        private static RawPair missing(File image) {
            return new RawPair(image, null, canonicalImageKey(image), true, false, null);
        }

        private static RawPair ambiguous(File image, List<File> masks) {
            return new RawPair(image, null, canonicalImageKey(image), false, false, masks);
        }
    }

    private static final class Pair {
        private final File image;
        private final File mask;
        private final String key;
        private final int objectCount;

        private Pair(File image, File mask, String key, int objectCount) {
            this.image = image;
            this.mask = mask;
            this.key = key;
            this.objectCount = objectCount;
        }
    }

    private static final class MaskCandidate {
        private final File file;
        private final String extension;
        private final String suffix;

        private MaskCandidate(File file, String extension, String suffix) {
            this.file = file;
            this.extension = extension;
            this.suffix = suffix;
        }
    }

    private static final class AmbiguityChoice {
        private final MaskCandidate mask;
        private final boolean resolvedAmbiguity;

        private AmbiguityChoice(MaskCandidate mask, boolean resolvedAmbiguity) {
            this.mask = mask;
            this.resolvedAmbiguity = resolvedAmbiguity;
        }
    }

    private static final class PairStats {
        private boolean unreadable;
        private boolean shapeMismatch;
        private boolean unsupportedMask;
        private boolean emptyMask;
        private boolean multichannelMask;
        private boolean suspiciousContinuousMask;
        private int objectCount;
    }

    private static final class MaskStats {
        private int width;
        private int height;
        private int numBands;
        private boolean floatData;
        private boolean suspiciousContinuousLike;
        private int objectCount;
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
