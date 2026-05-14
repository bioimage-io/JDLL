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

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import io.bioimage.modelrunner.model.detection.Detection;

public final class YoloDetectionGeoJsonWriter {

    private static final String EXTENSION = ".geojson";

    private YoloDetectionGeoJsonWriter() {}

    public static List<File> write(Map<String, List<Detection>> detectionsByImagePath) throws IOException {
        List<File> files = new ArrayList<File>();
        for (Map.Entry<String, List<Detection>> entry : detectionsByImagePath.entrySet()) {
            File outputFile = outputFileFor(new File(entry.getKey()));
            write(outputFile, entry.getKey(), entry.getValue());
            files.add(outputFile);
        }
        return files;
    }

    public static File outputFileFor(File imageFile) {
        File parent = imageFile == null ? null : imageFile.getParentFile();
        if (parent == null) {
            parent = new File(".");
        }
        String name = imageFile == null ? "image" : stripExtension(imageFile.getName());
        if (name == null || name.trim().isEmpty()) {
            name = "image";
        }
        return new File(parent, name + EXTENSION);
    }

    public static void write(File outputFile, String imagePath, List<Detection> detections) throws IOException {
        File parent = outputFile.getParentFile();
        if (parent != null) {
            Files.createDirectories(parent.toPath());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile.toPath(), StandardCharsets.UTF_8)) {
            writer.write("{\"type\":\"FeatureCollection\",\"properties\":{\"image_path\":");
            writer.write(json(imagePath));
            writer.write("},\"features\":[");
            if (detections != null) {
                boolean first = true;
                for (Detection detection : detections) {
                    if (!validBox(detection)) {
                        continue;
                    }
                    if (!first) {
                        writer.write(',');
                    }
                    first = false;
                    writeFeature(writer, detection);
                }
            }
            writer.write("]}");
        }
    }

    private static void writeFeature(BufferedWriter writer, Detection detection) throws IOException {
        writer.write("{\"type\":\"Feature\",\"properties\":{");
        writer.write("\"class_id\":");
        writer.write(Integer.toString(detection.getClassId()));
        writer.write(",\"confidence\":");
        writer.write(number(detection.getConfidence()));
        writer.write("},\"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[");
        writeCoordinate(writer, detection.getX1(), detection.getY1());
        writer.write("],[");
        writeCoordinate(writer, detection.getX2(), detection.getY1());
        writer.write("],[");
        writeCoordinate(writer, detection.getX2(), detection.getY2());
        writer.write("],[");
        writeCoordinate(writer, detection.getX1(), detection.getY2());
        writer.write("],[");
        writeCoordinate(writer, detection.getX1(), detection.getY1());
        writer.write("]]]}}");
    }

    private static void writeCoordinate(BufferedWriter writer, double x, double y) throws IOException {
        writer.write(number(x));
        writer.write(',');
        writer.write(number(y));
    }

    private static boolean validBox(Detection detection) {
        return detection != null
                && Double.isFinite(detection.getX1())
                && Double.isFinite(detection.getY1())
                && Double.isFinite(detection.getX2())
                && Double.isFinite(detection.getY2())
                && Double.isFinite(detection.getConfidence())
                && detection.getX2() > detection.getX1()
                && detection.getY2() > detection.getY1();
    }

    private static String stripExtension(String name) {
        if (name == null) {
            return null;
        }
        int dot = name.lastIndexOf('.');
        return dot <= 0 ? name : name.substring(0, dot);
    }

    private static String number(double value) {
        if (!Double.isFinite(value)) {
            return "0.00000000";
        }
        return String.format(Locale.US, "%.8f", value);
    }

    private static String json(String value) {
        String safe = value == null ? "" : value;
        StringBuilder builder = new StringBuilder(safe.length() + 2);
        builder.append('"');
        for (int i = 0; i < safe.length(); i++) {
            char c = safe.charAt(i);
            switch (c) {
                case '"':
                    builder.append("\\\"");
                    break;
                case '\\':
                    builder.append("\\\\");
                    break;
                case '\b':
                    builder.append("\\b");
                    break;
                case '\f':
                    builder.append("\\f");
                    break;
                case '\n':
                    builder.append("\\n");
                    break;
                case '\r':
                    builder.append("\\r");
                    break;
                case '\t':
                    builder.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        builder.append(String.format(Locale.US, "\\u%04x", (int) c));
                    } else {
                        builder.append(c);
                    }
                    break;
            }
        }
        builder.append('"');
        return builder.toString();
    }
}
