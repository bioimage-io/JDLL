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
package io.bioimage.modelrunner.gui.custom;

import java.awt.geom.Rectangle2D;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntFunction;

/** One shared reference, with optional file/time-point exceptions. Z is never part of a key. */
public final class SizeHints {
    private Rectangle2D.Double shared;
    private final Map<Key, Rectangle2D.Double> individual = new HashMap<>();

    public void reset() { shared = null; individual.clear(); }
    public boolean hasAny() { return shared != null || !individual.isEmpty(); }
    public boolean usesShared(File file, int time) { return !individual.containsKey(new Key(file, time)); }

    public Rectangle2D.Double get(File file, int time) {
        Key key = new Key(file, time);
        Rectangle2D.Double rectangle = individual.get(key);
        return copy(rectangle == null ? shared : rectangle);
    }

    public void set(File file, int time, Rectangle2D.Double rectangle) {
        Key key = new Key(file, time);
        if (individual.containsKey(key)) individual.put(key, copy(rectangle));
        else shared = copy(rectangle);
    }

    public void useShared(File file, int time, boolean useShared) {
        Key key = new Key(file, time);
        if (useShared) individual.remove(key);
        else if (!individual.containsKey(key)) individual.put(key, copy(shared));
    }

    /** Clearing an individual exception restores inheritance; clearing shared leaves exceptions intact. */
    public void clear(File file, int time) {
        Key key = new Key(file, time);
        if (individual.containsKey(key)) individual.remove(key);
        else shared = null;
    }

    public Snapshot snapshot() { return new Snapshot(shared, individual); }

    /** Immutable run settings, independent of subsequent preview edits. */
    public static final class Snapshot {
        private final Rectangle2D.Double shared;
        private final Map<Key, Rectangle2D.Double> individual = new HashMap<>();

        private Snapshot(Rectangle2D.Double shared, Map<Key, Rectangle2D.Double> individual) {
            this.shared = copy(shared);
            individual.forEach((key, rectangle) -> this.individual.put(key, copy(rectangle)));
        }

        public IntFunction<Rectangle2D.Double> forImage(File file) {
            return time -> {
                Key key = new Key(file, time);
                Rectangle2D.Double rectangle = individual.get(key);
                return copy(rectangle == null ? shared : rectangle);
            };
        }
    }

    private static Rectangle2D.Double copy(Rectangle2D.Double rectangle) {
        return rectangle == null ? null : (Rectangle2D.Double) rectangle.clone();
    }

    private static final class Key {
        final String file;
        final int time;
        Key(File file, int time) {
            this.file = file == null ? "" : file.toPath().toAbsolutePath().normalize().toString();
            this.time = time;
        }
        @Override public boolean equals(Object other) {
            if (!(other instanceof Key)) return false;
            Key key = (Key) other;
            return time == key.time && file.equals(key.file);
        }
        @Override public int hashCode() { return Objects.hash(file, time); }
    }
}
