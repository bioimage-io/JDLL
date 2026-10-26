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

import static org.junit.Assert.*;

import java.awt.Rectangle;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import io.bioimage.modelrunner.gui.custom.PreviewReader;

public class PreviewTileCacheTest {
    @Test public void readsOnlyVisiblePyramidTilesAndReusesThem() throws Exception {
        AtomicInteger reads = new AtomicInteger();
        AtomicInteger closes = new AtomicInteger();
        PreviewReader reader = new PreviewReader() {
            public Info info() { return new Info(16384, 8192, 3, 40, 20, 0, 2, 4); }
            public BufferedImage read(int z, int t, int level, Rectangle region) {
                assertEquals(7, z);
                assertEquals(3, t);
                assertEquals(2, level);
                assertEquals(new Rectangle(1024, 2048, 1024, 1024), region);
                reads.incrementAndGet();
                BufferedImage image = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
                image.setRGB(0, 0, 0x123456);
                return image;
            }
            public void close() { closes.incrementAndGet(); }
        };
        PreviewTileCache cache = new PreviewTileCache(reader);
        Rectangle2D.Double region = new Rectangle2D.Double(1024, 2048, 1024, 1024);
        assertEquals(0x123456, cache.render(7, 3, region, 256, 256, () -> false).getRGB(0, 0) & 0xffffff);
        cache.render(7, 3, region, 256, 256, () -> false);
        assertEquals(1, reads.get());
        assertNull(cache.render(7, 3, region, 256, 256, () -> true));
        assertEquals(1, reads.get());
        cache.close();
        cache.close();
        assertEquals(1, closes.get());
    }

    @Test public void cacheBoundsBothPositionsAndRgbBytes() throws Exception {
        PreviewReader reader = new PreviewReader() {
            public Info info() { return new Info(8192, 8192, 1, 20, 1); }
            public BufferedImage read(int z, int t, int level, Rectangle region) {
                return new BufferedImage(region.width, region.height, BufferedImage.TYPE_INT_RGB);
            }
            public void close() {}
        };
        try (PreviewTileCache cache = new PreviewTileCache(reader)) {
            Rectangle2D.Double tile = new Rectangle2D.Double(0, 0, 256, 256);
            for (int z = 0; z < 20; z++) cache.render(z, 0, tile, 256, 256, () -> false);
            assertTrue(((Map<?, ?>) field(cache, "tiles")).size() <= PreviewTileCache.MAX_POSITIONS);
            cache.render(0, 0, new Rectangle2D.Double(0, 0, 4352, 4352), 4352, 4352, () -> false);
            assertTrue((Long) field(cache, "bytes") <= PreviewTileCache.MAX_BYTES);
        }
    }

    private static Object field(Object owner, String name) throws Exception {
        Field field = owner.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(owner);
    }
}
