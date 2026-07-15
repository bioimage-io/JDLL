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
package io.bioimage.modelrunner.gui.custom.denoise;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.Service;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.special.denoise.Denoising;
import io.bioimage.modelrunner.model.special.denoise.DenoisingConfig;
import io.bioimage.modelrunner.model.special.denoise.DenoisingCapabilities;
import io.bioimage.modelrunner.model.special.denoise.DenoisingProgress;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

/** Owns the persistent Appose process and current denoising task. */
public final class DenoisingService {

    private final DenoisingInstaller installer;
    private volatile Denoising model;
    private Map<String, Object> loadedConfig;
    private volatile Service capabilityService;

    public DenoisingService(DenoisingInstaller installer) {
        this.installer = installer;
    }

    public <T extends RealType<T> & NativeType<T>>
    RandomAccessibleInterval<FloatType> run(DenoisingConfig config, RandomAccessibleInterval<T> input,
            boolean preview, Consumer<DenoisingProgress> progressConsumer, Consumer<String> installConsumer)
            throws BuildException, InterruptedException, LoadModelException, RunModelException {
        ensureLoaded(config, progressConsumer, installConsumer);
        Denoising active = model;
        active.setPreview(preview);
        List<Tensor<FloatType>> outputs = active.inference(Tensor.build("input", config.getAxes(), input));
        if (outputs.size() != 1) {
            throw new RunModelException("The denoising backend returned " + outputs.size() + " outputs; expected one.");
        }
        return outputs.get(0).getData();
    }

    public Map<String, Object> getMetadata() {
        return model == null ? null : model.getMetadata();
    }

    public DenoisingCapabilities capabilities(Consumer<String> installConsumer) throws Exception {
        installer.installIfNeeded(installConsumer);
        return Denoising.capabilities(service -> capabilityService = service);
    }

    public void cancel() {
        Denoising active = model;
        if (active != null) active.cancelCurrentInference();
    }

    public void close() {
        Denoising active = model;
        if (active != null) active.close();
        Service probe = capabilityService;
        if (probe != null && probe.isAlive()) probe.close();
        capabilityService = null;
        model = null;
        loadedConfig = null;
    }

    private void ensureLoaded(DenoisingConfig config, Consumer<DenoisingProgress> progressConsumer,
            Consumer<String> installConsumer) throws BuildException, InterruptedException, LoadModelException {
        Map<String, Object> requested = config.toMap();
        if (model != null && !requested.equals(loadedConfig)) close();
        if (model == null) {
            installer.installIfNeeded(installConsumer);
            model = new Denoising(config, progressConsumer);
            model.setMaxSharedMemoryPixelCount(Long.MAX_VALUE);
            model.loadModel();
            loadedConfig = requested;
        }
    }
}
