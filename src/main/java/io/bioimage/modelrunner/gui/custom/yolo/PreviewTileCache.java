/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom.yolo;

import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BooleanSupplier;

import io.bioimage.modelrunner.gui.custom.PreviewReader;

/** Bounded RGB pyramid tiles. No full-resolution plane/volume is retained. */
final class PreviewTileCache implements AutoCloseable {
    static final int TILE_SIZE = 256;
    static final long MAX_BYTES = 64L * 1024 * 1024;
    static final int MAX_POSITIONS = 10;
    private final PreviewReader reader;
    private final LinkedHashMap<Key, BufferedImage> tiles = new LinkedHashMap<>(32, 0.75f, true);
    private long bytes;
    private boolean closed;

    PreviewTileCache(PreviewReader reader) { this.reader = reader; }
    PreviewReader.Info info() { return reader.info(); }

    BufferedImage render(int z, int t, Rectangle2D.Double region, int width, int height,
            BooleanSupplier abort) throws IOException {
        int level = level(region, width, height);
        long scale = 1L << level;
        int firstX = (int) (region.x / scale / TILE_SIZE), firstY = (int) (region.y / scale / TILE_SIZE);
        int lastX = (int) Math.ceil((region.x + region.width) / scale / TILE_SIZE);
        int lastY = (int) Math.ceil((region.y + region.height) / scale / TILE_SIZE);
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = output.createGraphics();
        try {
            graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            graphics.scale(width / region.width, height / region.height);
            graphics.translate(-region.x, -region.y);
            graphics.scale(scale, scale);
            for (int y = firstY; y < lastY; y++) for (int x = firstX; x < lastX; x++) {
                if (abort.getAsBoolean()) return null;
                BufferedImage tile = tile(new Key(z, t, level, x, y), abort);
                if (tile == null) return null;
                graphics.drawImage(tile, AffineTransform.getTranslateInstance(x * (double) TILE_SIZE,
                        y * (double) TILE_SIZE), null);
            }
        } finally { graphics.dispose(); }
        return output;
    }

    void prefetch(int z, int t, Rectangle2D.Double region, int width, int height, BooleanSupplier abort)
            throws IOException {
        int count = 1;
        for (int distance = 1; distance < MAX_POSITIONS && count < MAX_POSITIONS; distance++) {
            for (int dz = -distance; dz <= distance && count < MAX_POSITIONS; dz++) {
                int dt = distance - Math.abs(dz);
                for (int sign : dt == 0 ? new int[] {1} : new int[] {-1, 1}) {
                    int nextZ = z + dz, nextT = t + sign * dt;
                    if (nextZ < 0 || nextZ >= info().depth || nextT < 0 || nextT >= info().frames) continue;
                    if (abort.getAsBoolean()) return;
                    render(nextZ, nextT, region, width, height, abort);
                    if (++count >= MAX_POSITIONS) return;
                }
            }
        }
    }

    static int level(Rectangle2D.Double region, int width, int height) {
        double ratio = Math.min(region.width / width, region.height / height);
        return Math.max(0, Math.min(30, (int) Math.floor(Math.log(Math.max(1, ratio)) / Math.log(2))));
    }

    private synchronized BufferedImage tile(Key key, BooleanSupplier abort) throws IOException {
        if (abort.getAsBoolean()) return null;
        if (closed) throw new IOException("Preview reader is closed.");
        BufferedImage cached = tiles.get(key);
        if (cached != null) return cached;
        long scale = 1L << key.level;
        int x = Math.toIntExact(key.x * TILE_SIZE * scale), y = Math.toIntExact(key.y * TILE_SIZE * scale);
        int w = (int) Math.min(TILE_SIZE * scale, info().width - (long) x);
        int h = (int) Math.min(TILE_SIZE * scale, info().height - (long) y);
        BufferedImage decoded = reader.read(key.z, key.t, key.level, new Rectangle(x, y, w, h), abort);
        if (abort.getAsBoolean()) return null;
        if (decoded == null) throw new IOException("Preview reader returned no pixels.");
        int targetW = (int) ((w + scale - 1) / scale), targetH = (int) ((h + scale - 1) / scale);
        BufferedImage result = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = result.createGraphics();
        graphics.drawImage(decoded, 0, 0, targetW, targetH, null);
        graphics.dispose();
        tiles.put(key, result);
        bytes += targetW * (long) targetH * 4;
        Set<Long> positions = new HashSet<>();
        for (Key entry : tiles.keySet()) positions.add(((long) entry.t << 32) | entry.z);
        Iterator<Map.Entry<Key, BufferedImage>> entries = tiles.entrySet().iterator();
        while ((bytes > MAX_BYTES || positions.size() > MAX_POSITIONS) && entries.hasNext()) {
            Map.Entry<Key, BufferedImage> entry = entries.next();
            bytes -= entry.getValue().getWidth() * (long) entry.getValue().getHeight() * 4;
            entries.remove();
            positions.clear();
            for (Key retained : tiles.keySet()) positions.add(((long) retained.t << 32) | retained.z);
        }
        return result;
    }

    @Override public synchronized void close() throws IOException {
        if (closed) return;
        closed = true;
        tiles.clear();
        bytes = 0;
        reader.close();
    }

    private static final class Key {
        final int z, t, level, x, y;
        Key(int z, int t, int level, int x, int y) { this.z = z; this.t = t; this.level = level; this.x = x; this.y = y; }
        @Override public boolean equals(Object other) {
            if (!(other instanceof Key)) return false;
            Key key = (Key) other;
            return z == key.z && t == key.t && level == key.level && x == key.x && y == key.y;
        }
        @Override public int hashCode() { return Objects.hash(z, t, level, x, y); }
    }
}
