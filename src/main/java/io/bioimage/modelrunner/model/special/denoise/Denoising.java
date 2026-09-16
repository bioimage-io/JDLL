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
 * #L%
 */
package io.bioimage.modelrunner.model.special.denoise;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.TaskException;

import com.google.gson.Gson;

import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Appose/shared-memory bridge for the {@code jdll_denoise} package. */
public final class Denoising extends DLModelPytorchProtected {

    private final DenoisingConfig config;
    private final Consumer<DenoisingProgress> progressConsumer;
    private volatile boolean preview;
    private volatile Map<String, Object> metadata;

    public Denoising(DenoisingConfig config, Consumer<DenoisingProgress> progressConsumer) {
        super(config.toMap(), config.getDevice());
        this.config = config;
        this.progressConsumer = progressConsumer;
    }

    public void setPreview(boolean preview) {
        this.preview = preview;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    /**
     * Queries capabilities in a short-lived service without creating a denoiser.
     *
     * @return backend capabilities.
     */
    public static DenoisingCapabilities capabilities() throws Exception {
        return capabilities(null);
    }

    /**
     * Queries capabilities and reports ownership of the temporary service.
     *
     * @param serviceConsumer receives the active service and then {@code null}.
     * @return backend capabilities.
     */
    public static DenoisingCapabilities capabilities(Consumer<Service> serviceConsumer) throws Exception {
        PixiEnvironmentSpec spec = resolvePytorchEnv();
        Environment environment = Appose.pixi().environment(spec.getSelectedEnvironment())
                .wrap(spec.getEnvironmentDirectory());
        Service service = environment.python();
        if (serviceConsumer != null) serviceConsumer.accept(service);
        try {
            String nl = System.lineSeparator();
            String code = "import os, sys" + nl + addSourcePathCode()
                    + "from jdll_denoise import capabilities as _jdll_denoise_capabilities" + nl
                    + "task.outputs['capabilities'] = _jdll_denoise_capabilities()" + nl;
            Task task = service.task(code);
            task.waitFor();
            if (task.status == TaskStatus.FAILED || task.status == TaskStatus.CRASHED) {
                throw new TaskException("Could not query jdll-denoise capabilities.", task);
            }
            Object result = task.outputs.get("capabilities");
            if (!(result instanceof Map)) {
                throw new IllegalStateException("jdll-denoise returned invalid capabilities.");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> values = (Map<String, Object>) result;
            return DenoisingCapabilities.from(values);
        } finally {
            if (service.isAlive()) service.close();
            if (serviceConsumer != null) serviceConsumer.accept(null);
        }
    }

    @Override
    protected String buildModelCode() {
        String nl = System.lineSeparator();
        return "if 'sys' not in globals().keys():" + nl
                + "  import sys" + nl
                + "  task.export(sys=sys)" + nl
                + "if 'os' not in globals().keys():" + nl
                + "  import os" + nl
                + "  task.export(os=os)" + nl
                + "if 'np' not in globals().keys():" + nl
                + "  import numpy as np" + nl
                + "  task.export(np=np)" + nl
                + "if 'torch' not in globals().keys():" + nl
                + "  import torch" + nl
                + "  task.export(torch=torch)" + nl
                + "if 'shared_memory' not in globals().keys():" + nl
                + "  from multiprocessing import shared_memory" + nl
                + "  task.export(shared_memory=shared_memory)" + nl
                + addSourcePathCode()
                + "from jdll_denoise import denoise as jdll_denoise_run" + nl
                + "from jdll_denoise import preview as jdll_denoise_preview" + nl
                + backendCompatibilityCode(config)
                + "task.export(jdll_denoise_run=jdll_denoise_run)" + nl
                + "task.export(jdll_denoise_preview=jdll_denoise_preview)" + nl;
    }

    @Override
    protected <T extends RealType<T> & NativeType<T>> String createInputsCode(
            List<Tensor<T>> tensors, List<String> names) {
        metadata = null;
        String nl = System.lineSeparator();
        String input = names.get(0);
        StringBuilder code = new StringBuilder();
        code.append("created_shms.clear()").append(nl);
        code.append("task.outputs.clear()").append(nl);
        code.append(SHM_NAMES_KEY).append(" = []").append(nl);
        code.append(DTYPES_KEY).append(" = []").append(nl);
        code.append(DIMS_KEY).append(" = []").append(nl);
        List<SharedMemoryArray> shmas = createSharedMemoryArraysForInputs(tensors);
        for (int i = 0; i < tensors.size(); i++) {
            SharedMemoryArray shma = shmas.get(i);
            code.append(codeToConvertShmaToPython(shma, names.get(i)));
            inShmaList.add(shma);
        }
        code.append(denoiseCallCode(input, config, preview));
        code.append("_jdll_denoise_metadata = dict(_jdll_denoise_result.get('metadata', {}))").append(nl);
        code.append("task.update(message='Denoising result ready', info={'type': 'result_metadata', ")
                .append("'metadata': _jdll_denoise_metadata})").append(nl);
        code.append("handle_output(np.asarray(_jdll_denoise_result['denoised'], dtype=np.float32), ")
                .append(SHMS_KEY).append(", ").append(SHM_NAMES_KEY).append(", ")
                .append(DTYPES_KEY).append(", ").append(DIMS_KEY).append(")").append(nl);
        code.append(taskOutputsCode());
        return code.toString();
    }

    static String backendCompatibilityCode(DenoisingConfig config) {
        if (!"zs_n2n".equals(config.getMethod())) return "";
        String nl = System.lineSeparator();
        // Older packages accept Quick/Balanced but resolve a different architecture.
        return "from jdll_denoise import capabilities as _jdll_denoise_capabilities" + nl
                + "_jdll_zs_capability = _jdll_denoise_capabilities().get('methods', {}).get('zs_n2n', {})" + nl
                + "_jdll_zs_version = _jdll_zs_capability.get('effort_presets_version', 0)" + nl
                + "if (not isinstance(_jdll_zs_version, int) or _jdll_zs_version < 1 or '"
                + TrainingCodeUtils.py(config.getEffort()) + "' not in _jdll_zs_capability.get('efforts', [])):" + nl
                + "  raise RuntimeError('Update jdll-denoise: ZS-N2N requires effort presets version 1 "
                + "with Quick, Balanced, Balanced-high and Thorough support.')" + nl;
    }

    static String denoiseCallCode(String input, DenoisingConfig config, boolean preview) {
        String nl = System.lineSeparator();
        String configJson = new Gson().toJson(config.toMap());
        return "_jdll_denoise_config = __import__('json').loads(r''' "
                + TrainingCodeUtils.py(configJson) + " ''')" + nl
                + "def _jdll_denoise_callback(event):" + nl
                + "  task.update(message=str(event.get('message', '')), current=event.get('current'), "
                + "maximum=event.get('maximum'), info=event)" + nl
                + "  return True" + nl
                + "_jdll_denoise_result = " + (preview ? "jdll_denoise_preview" : "jdll_denoise_run")
                + "(" + input + ", _jdll_denoise_config, callbacks=_jdll_denoise_callback)" + nl;
    }

    @Override
    protected String getOutputTensorAxes(int outputCount) {
        if (outputCount != 0) {
            throw new IllegalArgumentException("Denoising returns one image.");
        }
        Map<String, Object> resultMetadata = metadata;
        if (resultMetadata != null && resultMetadata.get("output_axes") instanceof String) {
            return (String) resultMetadata.get("output_axes");
        }
        return config.getAxes();
    }

    @Override
    protected void configureTask(Task task) {
        task.listen(this::handleEvent);
    }

    private void handleEvent(TaskEvent event) {
        if (!ResponseType.UPDATE.equals(event.responseType) || event.info == null) {
            return;
        }
        if ("result_metadata".equals(String.valueOf(event.info.get("type")))) {
            Object value = event.info.get("metadata");
            if (value instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> result = (Map<String, Object>) value;
                metadata = result;
            }
            return;
        }
        if (progressConsumer != null) {
            progressConsumer.accept(new DenoisingProgress(
                    String.valueOf(event.info.get("phase")), event.message,
                    (long) event.current, (long) event.maximum, event.info));
        }
    }

    private static String addSourcePathCode() {
        String nl = System.lineSeparator();
        String source = sourceDirectory();
        return "_jdll_denoise_source = r'" + TrainingCodeUtils.py(source) + "'" + nl
                + "if _jdll_denoise_source and os.path.isdir(_jdll_denoise_source) "
                + "and _jdll_denoise_source not in sys.path:" + nl
                + "  sys.path.insert(0, _jdll_denoise_source)" + nl;
    }

    private static String sourceDirectory() {
        String configured = System.getProperty("jdll.denoise.source");
        if (configured == null || configured.trim().isEmpty()) {
            configured = System.getenv("JDLL_DENOISE_SOURCE");
        }
        if (configured != null && new File(configured).isDirectory()) {
            return new File(configured).getAbsolutePath();
        }
        File development = new File("/home/carlos/hack_git/jdll-denoise");
        return development.isDirectory() ? development.getAbsolutePath() : "";
    }
}
