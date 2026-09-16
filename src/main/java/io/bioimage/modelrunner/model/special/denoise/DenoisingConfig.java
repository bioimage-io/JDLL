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
package io.bioimage.modelrunner.model.special.denoise;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/** Immutable configuration passed to {@code jdll_denoise}. */
public final class DenoisingConfig {

    private final String method;
    private final String effort;
    private final String device;
    private final String dimensions;
    private final String axes;
    private final String noiseStructure;
    private final long seed;

    private DenoisingConfig(Builder builder) {
        method = normalize(builder.method, "noise2fast");
        effort = normalize(builder.effort, "balanced").replace('-', '_');
        if ("balanced_high".equals(effort) && !"zs_n2n".equals(method)) {
            throw new IllegalArgumentException("Balanced-high effort is only supported for ZS-N2N.");
        }
        device = normalizeDevice(builder.device);
        dimensions = normalize(builder.dimensions, "auto");
        axes = normalize(builder.axes, "yx");
        noiseStructure = normalize(builder.noiseStructure, "auto");
        seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getMethod() {
        return method;
    }

    public String getEffort() {
        return effort;
    }

    public String getDevice() {
        return device;
    }

    public String getDimensions() {
        return dimensions;
    }

    public String getAxes() {
        return axes;
    }

    public String getNoiseStructure() {
        return noiseStructure;
    }

    public long getSeed() {
        return seed;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> config = new LinkedHashMap<String, Object>();
        config.put("method", method);
        config.put("effort", effort);
        config.put("device", device);
        config.put("dimensions", dimensions);
        config.put("axes", axes);
        config.put("seed", seed);
        Map<String, Object> normalization = new LinkedHashMap<String, Object>();
        normalization.put("mode", "percentile");
        boolean zsN2N = "zs_n2n".equals(method);
        normalization.put("low", zsN2N ? 0d : 0.1d);
        normalization.put("high", zsN2N ? 100d : 99.9d);
        normalization.put("per_channel", true);
        normalization.put("clip", true);
        config.put("normalization", normalization);
        Map<String, Object> tiling = new LinkedHashMap<String, Object>();
        tiling.put("enabled", "auto");
        tiling.put("tile_size", "auto");
        tiling.put("overlap", "auto");
        tiling.put("blend", "gaussian");
        config.put("tiling", tiling);
        Map<String, Object> methodConfig = new LinkedHashMap<String, Object>();
        if ("structn2v".equals(method)) methodConfig.put("noise_structure", noiseStructure);
        config.put("method_config", methodConfig);
        config.put("allow_cpu_fallback", false);
        return config;
    }

    private static String normalize(String value, String fallback) {
        return value == null || value.trim().isEmpty()
                ? fallback : value.trim().toLowerCase(Locale.ROOT);
    }

    private static String normalizeDevice(String value) {
        String normalized = normalize(value, "cpu");
        return "cuda".equals(normalized) || "mps".equals(normalized) ? normalized : "cpu";
    }

    public static final class Builder {
        private String method = "noise2fast";
        private String effort = "balanced";
        private String device = "cpu";
        private String dimensions = "auto";
        private String axes = "yx";
        private String noiseStructure = "auto";
        private long seed = 5489L;

        private Builder() {}

        public Builder method(String value) { method = value; return this; }
        public Builder effort(String value) { effort = value; return this; }
        public Builder device(String value) { device = value; return this; }
        public Builder dimensions(String value) { dimensions = value; return this; }
        public Builder axes(String value) { axes = value; return this; }
        public Builder noiseStructure(String value) { noiseStructure = value; return this; }
        public Builder seed(long value) { seed = value; return this; }
        public DenoisingConfig build() { return new DenoisingConfig(this); }
    }
}
