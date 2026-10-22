/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.cellpose;

import javax.swing.JComboBox;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

/** Source channels are one-based; zero means mean intensity for primary, none for nuclei. */
public final class CellposeChannels {
    private CellposeChannels() { }

    public static void configure(JComboBox<String> primary, JComboBox<String> nuclei, int count) {
        if (count < 1) throw new IllegalArgumentException("The image has no channels.");
        String first = (String) primary.getSelectedItem();
        String second = (String) nuclei.getSelectedItem();
        primary.removeAllItems();
        nuclei.removeAllItems();
        primary.addItem("gray");
        nuclei.addItem("none");
        for (int c = 1; c <= count; c++) {
            String name = count == 3 ? new String[] {"red", "green", "blue"}[c - 1] : "channel " + c;
            primary.addItem(name);
            nuclei.addItem(name);
        }
        selectExisting(primary, first);
        selectExisting(nuclei, second);
    }

    private static void selectExisting(JComboBox<String> combo, String selected) {
        for (int i = 0; i < combo.getItemCount(); i++) {
            if (combo.getItemAt(i).equals(selected)) {
                combo.setSelectedIndex(i);
                return;
            }
        }
        combo.setSelectedIndex(0);
    }

    public static int parse(String channel) {
        if ("gray".equals(channel) || "none".equals(channel)) return 0;
        if ("red".equals(channel)) return 1;
        if ("green".equals(channel)) return 2;
        if ("blue".equals(channel)) return 3;
        if (channel != null && channel.startsWith("channel ")) return Integer.parseInt(channel.substring(8));
        throw new IllegalArgumentException("Please select a valid Cellpose channel.");
    }

    public static void validate(int[] channels, long count) {
        if (channels == null || channels.length != 2 || channels[0] < 0 || channels[1] < 0
                || channels[0] > count || channels[1] > count) {
            throw new IllegalArgumentException("Selected Cellpose channels are not available in this "
                    + count + "-channel image.");
        }
    }

    public static <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<FloatType> pack(
            RandomAccessibleInterval<T> plane, int[] channels) {
        validate(channels, plane.dimension(2));
        RandomAccessibleInterval<FloatType> packed = ArrayImgs.floats(plane.dimension(0), plane.dimension(1), 3);
        for (int dest = 0; dest < 2; dest++) {
            if (dest == 1 && channels[1] == 0) continue;
            int first = channels[dest] == 0 ? 0 : channels[dest] - 1;
            int end = channels[dest] == 0 ? Math.toIntExact(plane.dimension(2)) : first + 1;
            for (int c = first; c < end; c++) {
                Cursor<T> source = Views.flatIterable(Views.hyperSlice(plane, 2, c)).cursor();
                Cursor<FloatType> target = Views.flatIterable(Views.hyperSlice(packed, 2, dest)).cursor();
                while (source.hasNext()) {
                    float value = source.next().getRealFloat() / (end - first);
                    FloatType pixel = target.next();
                    pixel.set(pixel.get() + value);
                }
            }
        }
        return packed;
    }
}
