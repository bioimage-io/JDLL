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
import java.util.List;
import java.util.Locale;
import java.util.Map;

import io.bioimage.modelrunner.model.detection.Detection;

public final class YoloDetectionCsvWriter {

    private static final String SUFFIX = "_deepicy_yolo_prediction.csv";

    private YoloDetectionCsvWriter() {}

    public static File outputFileFor(File source) {
        File base = source == null ? new File(".") : source;
        File parent = base.getParentFile();
        if (parent == null) {
            parent = base.isDirectory() ? base : new File(".");
        }
        String name = base.isDirectory() ? base.getName() : stripExtension(base.getName());
        if (name == null || name.trim().isEmpty()) {
            name = "image";
        }
        return new File(parent, name + SUFFIX);
    }

    public static void write(File outputFile, Map<String, List<Detection>> detectionsByImagePath) throws IOException {
        File parent = outputFile.getParentFile();
        if (parent != null) {
            Files.createDirectories(parent.toPath());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile.toPath(), StandardCharsets.UTF_8)) {
            writer.write("image_path,x1,y1,x2,y2,class_id,confidence");
            writer.newLine();
            for (Map.Entry<String, List<Detection>> entry : detectionsByImagePath.entrySet()) {
                List<Detection> detections = entry.getValue();
                if (detections == null || detections.isEmpty()) {
                    writer.write(csv(entry.getKey()));
                    writer.write(",,,,,,");
                    writer.newLine();
                    continue;
                }
                for (Detection detection : detections) {
                    writer.write(csv(entry.getKey()));
                    writer.write(',');
                    writer.write(number(detection.getX1()));
                    writer.write(',');
                    writer.write(number(detection.getY1()));
                    writer.write(',');
                    writer.write(number(detection.getX2()));
                    writer.write(',');
                    writer.write(number(detection.getY2()));
                    writer.write(',');
                    writer.write(Integer.toString(detection.getClassId()));
                    writer.write(',');
                    writer.write(number(detection.getConfidence()));
                    writer.newLine();
                }
            }
        }
    }

    private static String stripExtension(String name) {
        if (name == null) {
            return null;
        }
        int dot = name.lastIndexOf('.');
        return dot <= 0 ? name : name.substring(0, dot);
    }

    private static String number(double value) {
        return String.format(Locale.US, "%.8f", value);
    }

    private static String csv(String value) {
        String safe = value == null ? "" : value;
        if (safe.indexOf('"') >= 0 || safe.indexOf(',') >= 0 || safe.indexOf('\n') >= 0 || safe.indexOf('\r') >= 0) {
            return "\"" + safe.replace("\"", "\"\"") + "\"";
        }
        return safe;
    }
}
