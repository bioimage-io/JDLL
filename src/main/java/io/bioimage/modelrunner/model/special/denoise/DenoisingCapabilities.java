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
package io.bioimage.modelrunner.model.special.denoise;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Read-only view of {@code jdll_denoise.capabilities()}. */
public final class DenoisingCapabilities {
    private final Map<String, MethodCapability> methods;

    private DenoisingCapabilities(Map<String, MethodCapability> methods) {
        this.methods = Collections.unmodifiableMap(methods);
    }

    public static DenoisingCapabilities from(Map<String, Object> values) {
        Map<String, MethodCapability> parsed = new LinkedHashMap<String, MethodCapability>();
        Object methodsValue = values == null ? null : values.get("methods");
        if (methodsValue instanceof Map) {
            Map<?, ?> methodMap = (Map<?, ?>) methodsValue;
            for (Map.Entry<?, ?> entry : methodMap.entrySet()) {
                if (entry.getValue() instanceof Map) {
                    Map<?, ?> capability = (Map<?, ?>) entry.getValue();
                    boolean available = !Boolean.FALSE.equals(capability.get("available"));
                    java.util.Set<String> devices = new java.util.LinkedHashSet<String>();
                    Object deviceValue = capability.get("devices");
                    if (deviceValue instanceof List) {
                        for (Object device : (List<?>) deviceValue) devices.add(String.valueOf(device));
                    }
                    parsed.put(String.valueOf(entry.getKey()), new MethodCapability(available, devices));
                }
            }
        }
        return new DenoisingCapabilities(parsed);
    }

    public boolean isMethodAvailable(String method) {
        MethodCapability capability = methods.get(method);
        return capability != null && capability.available;
    }

    public boolean supportsDevice(String method, String device) {
        MethodCapability capability = methods.get(method);
        return capability != null && capability.available && capability.devices.contains(device);
    }

    private static final class MethodCapability {
        private final boolean available;
        private final java.util.Set<String> devices;
        private MethodCapability(boolean available, java.util.Set<String> devices) {
            this.available = available;
            this.devices = Collections.unmodifiableSet(devices);
        }
    }
}
