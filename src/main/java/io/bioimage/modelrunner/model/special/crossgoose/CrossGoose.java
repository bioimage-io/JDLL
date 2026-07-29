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
package io.bioimage.modelrunner.model.special.crossgoose;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.gui.custom.crossgoose.CrossGooseModelRegistry;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.python.methods.ConvertDims;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/** Cross-GOOSE model backed by the future {@code jdll-cross-goose} adapter. */
public final class CrossGoose extends DLModelPytorchProtected {

    private static final String DEFAULT_SOURCE_DIR = "/home/carlos/hack_git/jdll-cross-goose";
    private final String modelPath;
    private volatile Double objectSize;

    private CrossGoose(String modelPath, Consumer<InferenceProgress> progressConsumer, String device) {
        super(modelPath, modelPath, modelPath, modelPath, new LinkedHashMap<String, Object>(), true, device);
        this.modelPath = modelPath;
        setMaxSharedMemoryPixelCount(Long.MAX_VALUE);
        environmentSpec = resolvePytorchEnv();
        setInferenceProgressConsumer(progressConsumer);
    }

    public static CrossGoose fromFile(String modelPath, Consumer<InferenceProgress> progressConsumer,
            String device) throws IOException, BuildException, LoadModelException {
        File directory = CrossGooseModelRegistry.modelDirectory(new File(modelPath));
        if (directory == null) {
            throw new IOException("Path does not point to a Cross-GOOSE model directory: " + modelPath);
        }
        CrossGoose model = new CrossGoose(directory.getAbsolutePath(), progressConsumer, device);
        model.loadModel();
        return model;
    }

    public static boolean isInstalled() {
        try {
            return PixiEnvironmentManager.isInstalled(resolvePytorchEnv());
        } catch (Exception error) {
            return false;
        }
    }

    public static PixiEnvironmentSpec resolvePytorchEnv() {
        return DLModelPytorchProtected.resolvePytorchEnv();
    }

    public void setObjectSize(Double size) {
        objectSize = size != null && size.doubleValue() > 0.0 ? size : null;
    }

    public static void train(Map<String, Object> config,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer,
            Consumer<String> logConsumer, Consumer<Service> serviceConsumer)
            throws IOException, BuildException, InterruptedException, TaskException {
        PixiEnvironmentSpec spec = resolvePytorchEnv();
        Environment environment = Appose.pixi().environment(spec.getSelectedEnvironment())
                .wrap(spec.getEnvironmentDirectory());
        Service python = environment.python();
        if (serviceConsumer != null) serviceConsumer.accept(python);
        try {
            Task task = python.task(buildTrainingCode(config));
            task.listen(event -> handleTrainingEvent(event, progressConsumer, previewConsumer, logConsumer));
            task.waitFor();
        } finally {
            if (python.isAlive()) python.close();
            if (serviceConsumer != null) serviceConsumer.accept(null);
        }
    }

    @Override protected boolean reportsOwnInferenceProgress() { return true; }

    @Override protected void configureTask(Task task) { task.listen(this::handleInferenceEvent); }

    @Override
    protected String getOutputTensorAxes(int outputCount) {
        if (outputCount != 0) throw new IllegalArgumentException(
                "Cross-GOOSE inference returns one label image.");
        return "yx";
    }

    @Override
    protected String buildModelCode() {
        String nl = System.lineSeparator();
        return ""
                + "if 'sys' not in globals().keys():" + nl
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
                + sourcePathCode()
                + "from jdll_cross_goose.appose_api import infer as jdll_cross_goose_infer" + nl
                + "from jdll_cross_goose.appose_api import load_model as jdll_cross_goose_load_model" + nl
                + "_jdll_cross_goose_device = '" + TrainingCodeUtils.py(device) + "'" + nl
                + MODEL_VAR_NAME + ", _jdll_cross_goose_model_config = jdll_cross_goose_load_model(r'"
                + TrainingCodeUtils.py(modelPath) + "', _jdll_cross_goose_device)" + nl
                + "task.export(jdll_cross_goose_infer=jdll_cross_goose_infer)" + nl
                + "task.export(_jdll_cross_goose_device=_jdll_cross_goose_device)" + nl
                + "task.export(_jdll_cross_goose_model_config=_jdll_cross_goose_model_config)" + nl
                + "task.export(" + MODEL_VAR_NAME + "=" + MODEL_VAR_NAME + ")" + nl;
    }

    @Override
    protected <T extends RealType<T> & NativeType<T>>
    String createInputsCode(List<Tensor<T>> inputs, List<String> names) {
        String nl = System.lineSeparator();
        String input = names.get(0);
        StringBuilder code = new StringBuilder();
        code.append(ConvertDims.getMethodDeclaration()).append(nl)
                .append("created_shms.clear()").append(nl)
                .append("task.outputs.clear()").append(nl)
                .append(SHM_NAMES_KEY).append(" = []").append(nl)
                .append(DTYPES_KEY).append(" = []").append(nl)
                .append(DIMS_KEY).append(" = []").append(nl);
        List<SharedMemoryArray> arrays = createSharedMemoryArraysForInputs(inputs);
        for (int i = 0; i < inputs.size(); i++) {
            code.append(codeToConvertShmaToPython(arrays.get(i), names.get(i)));
            inShmaList.add(arrays.get(i));
        }
        code.append(input).append(" = ").append(ConvertDims.getMethodName()).append('(')
                .append(input).append(", '")
                .append(inputs.get(0).getAxesOrderString().toLowerCase(Locale.ROOT))
                .append("', out_order='cyx', output_type='numpy', contiguous=True, n_channels=2)")
                .append(nl)
                .append("def _jdll_cross_goose_callback(event):").append(nl)
                .append("  task.update(message=str(event.get('message', '')), current=event.get('current'), ")
                .append("maximum=event.get('maximum'), info=event)").append(nl)
                .append("  return True").append(nl)
                .append("_jdll_cross_goose_infer_config = {'model_path': r'")
                .append(TrainingCodeUtils.py(modelPath))
                .append("', 'device': _jdll_cross_goose_device}").append(nl);
        if (objectSize != null) {
            code.append("_jdll_cross_goose_infer_config['object_size'] = ")
                    .append(objectSize).append(nl);
        }
        code
                .append("_jdll_cross_goose_result = jdll_cross_goose_infer(")
                .append("_jdll_cross_goose_infer_config, {'image': ").append(input)
                .append("}, model=").append(MODEL_VAR_NAME)
                .append(", callback=_jdll_cross_goose_callback)").append(nl)
                .append("_jdll_cross_goose_outputs = _jdll_cross_goose_result.get('outputs', ")
                .append("_jdll_cross_goose_result)").append(nl)
                .append("_jdll_cross_goose_labels = _jdll_cross_goose_outputs['labels']").append(nl)
                .append(String.format("handle_output(np.asarray(_jdll_cross_goose_labels), %s, %s, %s, %s)",
                        SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY)).append(nl)
                .append(taskOutputsCode());
        return code.toString();
    }

    private void handleInferenceEvent(TaskEvent event) {
        if (!ResponseType.UPDATE.equals(event.responseType) || event.info == null
                || !"inference_progress".equals(event.info.get("type"))) return;
        String phase = String.valueOf(event.info.get("phase"));
        int total = TrainingCodeUtils.asInt(event.info.get("total_patches"), (int) event.maximum);
        int patch = TrainingCodeUtils.asInt(event.info.get("patch_index"), (int) event.current);
        if ("inference_start".equals(phase)) {
            if (tileCounter != null) tileCounter.acceptTotal((long) total);
            emitProgress(InferenceProgress.inferenceStart(total));
        }
        else if ("patch_start".equals(phase)) emitProgress(InferenceProgress.patchStart(patch, total));
        else if ("patch_end".equals(phase)) {
            if (tileCounter != null) tileCounter.acceptProgress((long) patch);
            emitProgress(InferenceProgress.patchEnd(patch, total));
        } else if ("merge_start".equals(phase)) emitProgress(InferenceProgress.mergeStart());
        else if ("inference_end".equals(phase)) emitProgress(InferenceProgress.inferenceEnd());
    }

    private static String sourcePathCode() {
        String nl = System.lineSeparator();
        return "_jdll_cross_goose_source = r'" + TrainingCodeUtils.py(sourceDirectory()) + "'" + nl
                + "if _jdll_cross_goose_source and os.path.isdir(_jdll_cross_goose_source) "
                + "and _jdll_cross_goose_source not in sys.path:" + nl
                + "  sys.path.insert(0, _jdll_cross_goose_source)" + nl;
    }

    private static String sourceDirectory() {
        String path = System.getProperty("jdll.crossgoose.path");
        if (path == null || path.trim().isEmpty()) path = System.getenv("JDLL_CROSS_GOOSE_PATH");
        return path == null || path.trim().isEmpty() ? DEFAULT_SOURCE_DIR : path.trim();
    }

    private static String buildTrainingCode(Map<String, Object> config) {
        String nl = System.lineSeparator();
        return "import json, os, sys" + nl
                + TrainingCodeUtils.apposeStdoutCapture()
                + "import torch" + nl
                + sourcePathCode()
                + "from jdll_cross_goose.appose_api import train as jdll_cross_goose_train" + nl
                + TrainingCodeUtils.pytorchMemoryCleanupFunction("_jdll_cleanup_pytorch_memory")
                + "_jdll_cross_goose_config = json.loads(r'''"
                + TrainingCodeUtils.toJson(config) + "''')" + nl
                + "try:" + nl
                + "  _jdll_cross_goose_result = jdll_cross_goose_train(_jdll_cross_goose_config, task=task)" + nl
                + "  for _jdll_kind, _jdll_key in (('last', 'last_checkpoint'), ('best', 'best_checkpoint')):" + nl
                + "    _jdll_path = _jdll_cross_goose_result.get(_jdll_key)" + nl
                + "    if _jdll_path:" + nl
                + "      task.update(message='Cross-GOOSE %s checkpoint: %s' % (_jdll_kind, _jdll_path), "
                + "info={'type': 'checkpoint', 'kind': _jdll_kind, 'path': _jdll_path})" + nl
                + "  task.outputs['result'] = _jdll_cross_goose_result.get('model_dir')" + nl
                + "finally:" + nl
                + "  try:" + nl
                + "    del _jdll_cross_goose_result" + nl
                + "  except Exception:" + nl
                + "    pass" + nl
                + "  _jdll_cleanup_pytorch_memory()" + nl;
    }

    private static void handleTrainingEvent(TaskEvent event,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer, Consumer<String> logConsumer) {
        if (!ResponseType.UPDATE.equals(event.responseType) || event.info == null) return;
        if (event.message != null && logConsumer != null) logConsumer.accept(event.message);
        Object type = event.info.get("type");
        if ("progress".equals(type) && progressConsumer != null) {
            progressConsumer.accept(new UnetTrainingProgress(
                    TrainingCodeUtils.asInt(event.info.get("epoch"), (int) event.current),
                    TrainingCodeUtils.asInt(event.info.get("step"), (int) event.current),
                    TrainingCodeUtils.asInt(event.info.get("total_epochs"), 0),
                    TrainingCodeUtils.asInt(event.info.get("total_steps"), (int) event.maximum),
                    TrainingCodeUtils.asDoubleMap(event.info.get("losses")),
                    TrainingCodeUtils.asDoubleMap(event.info.get("metrics"))));
        } else if ("preview".equals(type) && previewConsumer != null) {
            previewConsumer.accept(new UnetValidationPreview(
                    TrainingCodeUtils.asInt(event.info.get("epoch"), (int) event.current),
                    string(event.info.get("preview_path")),
                    string(event.info.get("latest_preview_path"))));
        }
    }

    private static String string(Object value) { return value == null ? null : value.toString(); }
}
