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
package io.bioimage.modelrunner.model.special.unet;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Locale;
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
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.gui.custom.unet.UnetModelRegistry;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.python.methods.ConvertDims;
import io.bioimage.modelrunner.model.special.common.TrainingCodeUtils;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * JDLL UNet model backed by the local {@code jdll-unet} Python package.
 */
public final class Unet extends DLModelPytorchProtected {

    private static final String DEFAULT_UNET_SOURCE_DIR = "/home/carlos/hack_git/jdll-unet";

    private final String modelPath;
    private final Map<String, Object> config;
    private final String task;
    private final String dimensions;
    private final int inputChannels;
    private volatile Double objectSize;

    private Unet(String modelPath, Map<String, Object> config,
            Consumer<InferenceProgress> inferenceProgressConsumer, String device) {
        super(modelPath, modelPath, modelPath, modelPath, config, true, device);
        this.modelPath = new File(modelPath).getAbsolutePath();
        this.config = normalizedConfig(config);
        this.task = normalizedTask(this.config.get("task"));
        this.dimensions = nestedConfigString(this.config, "architecture_config", "dimensions", "2d");
        int modelInputChannels = configInt(this.config, "input_channels", 1);
        int contextSlices = nestedConfigInt(this.config, "architecture_config", "context_slices", 3);
        this.inputChannels = "2.5d".equals(this.dimensions)
                ? Math.max(1, modelInputChannels / Math.max(1, contextSlices))
                : modelInputChannels;
        setMaxSharedMemoryPixelCount(Long.MAX_VALUE);
        this.environmentSpec = resolvePytorchEnv();
        super.setInferenceProgressConsumer(inferenceProgressConsumer);
    }

    /**
     * Creates and loads a UNet model from a checkpoint or model directory.
     *
     * @param modelPath the model path.
     * @param inferenceProgressConsumer the inference progress consumer.
     * @param device the requested device.
     * @return the loaded UNet model.
     * @throws IOException if the config cannot be read.
     * @throws BuildException if the Appose environment cannot be built.
     * @throws LoadModelException if the model cannot be loaded.
     */
    public static Unet fromFile(String modelPath, Consumer<InferenceProgress> inferenceProgressConsumer, String device)
            throws IOException, BuildException, LoadModelException {
        File modelFile = resolveModelFile(modelPath);
        Map<String, Object> config = loadModelConfig(modelFile);
        Unet model = new Unet(modelFile.getAbsolutePath(), config, inferenceProgressConsumer, device);
        model.loadModel();
        return model;
    }

    /**
     * Creates and loads a UNet model from a checkpoint or model directory.
     *
     * @param modelPath the model path.
     * @param inferenceProgressConsumer the inference progress consumer.
     * @return the loaded UNet model.
     * @throws IOException if the config cannot be read.
     * @throws BuildException if the Appose environment cannot be built.
     * @throws LoadModelException if the model cannot be loaded.
     */
    public static Unet fromFile(String modelPath, Consumer<InferenceProgress> inferenceProgressConsumer)
            throws IOException, BuildException, LoadModelException {
        return fromFile(modelPath, inferenceProgressConsumer, "cpu");
    }

    /**
     * Returns the number of model input channels.
     *
     * @return the input channels.
     */
    public int getInputChannels() {
        return inputChannels;
    }

    /**
     * Returns the model task.
     *
     * @return the model task.
     */
    public String getTask() {
        return task;
    }

    /**
     * Returns the model dimensionality.
     *
     * @return {@code 2d}, {@code 2.5d}, or {@code 3d}.
     */
    public String getDimensions() {
        return dimensions;
    }

    /**
     * Returns whether this model consumes a volume.
     *
     * @return true for 2.5D and 3D models.
     */
    public boolean isVolumeModel() {
        return !"2d".equals(dimensions);
    }

    /**
     * Sets the approximate object or semantic-region size in input pixels.
     *
     * @param size the size, or null to disable the inference override.
     */
    public void setObjectSize(Double size) {
        this.objectSize = size != null && size > 0.0 ? size : null;
    }

    /**
     * Returns whether the shared PyTorch environment is installed.
     *
     * @return true if installed.
     */
    public static boolean isInstalled() {
        try {
            return PixiEnvironmentManager.isInstalled(resolvePytorchEnv());
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Resolves the UNet environment.
     *
     * @return the environment spec.
     */
    public static PixiEnvironmentSpec resolvePytorchEnv() {
        return DLModelPytorchProtected.resolvePytorchEnv();
    }

    /**
     * Runs UNet training through the Python package.
     *
     * @param config the training config.
     * @param progressConsumer the progress consumer.
     * @param logConsumer the log consumer.
     * @param serviceConsumer the Appose service consumer.
     * @throws IOException if an I/O error occurs.
     * @throws BuildException if the environment cannot be built.
     * @throws InterruptedException if interrupted.
     * @throws TaskException if the task fails.
     */
    public static void train(Map<String, Object> config,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<String> logConsumer,
            Consumer<Service> serviceConsumer)
            throws IOException, BuildException, InterruptedException, TaskException {
        train(config, progressConsumer, null, logConsumer, serviceConsumer);
    }

    /**
     * Runs UNet training through the Python package.
     *
     * @param config the training config.
     * @param progressConsumer the progress consumer.
     * @param previewConsumer the validation preview consumer.
     * @param logConsumer the log consumer.
     * @param serviceConsumer the Appose service consumer.
     * @throws IOException if an I/O error occurs.
     * @throws BuildException if the environment cannot be built.
     * @throws InterruptedException if interrupted.
     * @throws TaskException if the task fails.
     */
    public static void train(Map<String, Object> config,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer,
            Consumer<String> logConsumer,
            Consumer<Service> serviceConsumer)
            throws IOException, BuildException, InterruptedException, TaskException {
        validateTrainingConfig(config);
        PixiEnvironmentSpec envSpec = resolvePytorchEnv();
        Environment env = Appose.pixi()
                .environment(envSpec.getSelectedEnvironment())
                .wrap(envSpec.getEnvironmentDirectory());
        Service python = env.python();
        if (serviceConsumer != null) {
            serviceConsumer.accept(python);
        }
        try {
            Task task = python.task(buildTrainingCode(config));
            task.listen(event -> handleTrainingEvent(event, progressConsumer, previewConsumer, logConsumer));
            task.waitFor();
        } finally {
            if (python.isAlive()) {
                python.close();
            }
            if (serviceConsumer != null) {
                serviceConsumer.accept(null);
            }
        }
    }

    @Override
    protected boolean reportsOwnInferenceProgress() {
        return true;
    }

    @Override
    protected void configureTask(final Task task) {
        task.listen(this::handleInferenceEvent);
    }

    /**
     * Returns the output tensor axes.
     *
     * @param outputCount the output count.
     * @return the axes.
     */
    @Override
    protected String getOutputTensorAxes(int outputCount) {
        if (outputCount != 0) {
            throw new IllegalArgumentException("UNet inference returns one final label image.");
        }
        return isVolumeModel() ? "zyx" : "yx";
    }

    /**
     * Builds the Python model-loading code.
     *
     * @return the code.
     */
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
                + addUnetSourcePathCode()
                + "from jdll_unet.appose_api import infer as jdll_unet_infer" + nl
                + "from jdll_unet.infer import load_model as jdll_unet_load_model" + nl
                + "_jdll_unet_device = '" + TrainingCodeUtils.py(device) + "'" + nl
                + MODEL_VAR_NAME + ", _jdll_unet_model_config = jdll_unet_load_model(r'"
                + TrainingCodeUtils.py(modelPath) + "', _jdll_unet_device)" + nl
                + "_jdll_unet_task = str(_jdll_unet_model_config.get('task', 'binary_semantic'))" + nl
                + "task.export(jdll_unet_infer=jdll_unet_infer)" + nl
                + "task.export(_jdll_unet_model_config=_jdll_unet_model_config)" + nl
                + "task.export(_jdll_unet_task=_jdll_unet_task)" + nl
                + "task.export(_jdll_unet_device=_jdll_unet_device)" + nl
                + "task.export(" + MODEL_VAR_NAME + "=" + MODEL_VAR_NAME + ")" + nl;
    }

    /**
     * Creates the Python whole-image inference code.
     *
     * @param <T> the T type parameter.
     * @param inRais the input RAIs.
     * @param names the Python variable names.
     * @return the code.
     */
    @Override
    protected <T extends RealType<T> & NativeType<T>>
    String createInputsCode(List<Tensor<T>> inRais, List<String> names) {
        String nl = System.lineSeparator();
        String inputName = names.get(0);
        String code = "";
        code += ConvertDims.getMethodDeclaration() + nl;
        code += "created_shms.clear()" + nl;
        code += "task.outputs.clear()" + nl;
        code += SHM_NAMES_KEY + " = []" + nl;
        code += DTYPES_KEY + " = []" + nl;
        code += DIMS_KEY + " = []" + nl;
        List<SharedMemoryArray> shmas = createSharedMemoryArraysForInputs(inRais);
        for (int i = 0; i < inRais.size(); i ++) {
            SharedMemoryArray shma = shmas.get(i);
            code += codeToConvertShmaToPython(shma, names.get(i));
            inShmaList.add(shma);
        }
        String outOrder = isVolumeModel() ? "czyx" : "cyx";
        code += inputName + " = " + ConvertDims.getMethodName() + "(" + inputName
                + ", '" + inRais.get(0).getAxesOrderString().toLowerCase(Locale.ROOT)
                + "', out_order='" + outOrder + "', output_type='numpy', contiguous=True, n_channels="
                + inputChannels + ")" + nl;
        code += "_jdll_unet_infer_config = {'model_path': r'" + TrainingCodeUtils.py(modelPath)
                + "', 'device': _jdll_unet_device}" + nl;
        if (objectSize != null) {
            code += "_jdll_unet_infer_config['object_size'] = " + objectSize + nl;
        }
        code += "def _jdll_unet_callback(event):" + nl;
        code += "  task.update(message=str(event.get('message', '')), current=event.get('current'), "
                + "maximum=event.get('maximum'), info=event)" + nl;
        code += "  return True" + nl;
        code += "_jdll_unet_result = jdll_unet_infer(_jdll_unet_infer_config, {'image': " + inputName
                + "}, callback=_jdll_unet_callback)" + nl;
        code += "_jdll_unet_outputs = _jdll_unet_result['outputs']" + nl;
        code += "if _jdll_unet_task == 'instance_friendly':" + nl;
        code += "  _jdll_unet_labels = _jdll_unet_outputs['labels']" + nl;
        code += "elif _jdll_unet_task == 'multiclass_semantic':" + nl;
        code += "  _jdll_unet_labels = _jdll_unet_outputs['mask']" + nl;
        code += "else:" + nl;
        code += "  _jdll_unet_labels = _jdll_unet_outputs.get('labels', _jdll_unet_outputs['mask'])" + nl;
        code += String.format("handle_output(np.asarray(_jdll_unet_labels), %s, %s, %s, %s)",
                SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY) + nl;
        code += taskOutputsCode();
        return code;
    }

    private void handleInferenceEvent(final TaskEvent event) {
        if (!ResponseType.UPDATE.equals(event.responseType) || event.info == null
                || !"inference_progress".equals(event.info.get("type"))) {
            return;
        }
        String phase = String.valueOf(event.info.get("phase"));
        int total = TrainingCodeUtils.asInt(event.info.get("total_patches"), (int) event.maximum);
        int patch = TrainingCodeUtils.asInt(event.info.get("patch_index"), (int) event.current);
        if ("inference_start".equals(phase)) {
            if (tileCounter != null) {
                tileCounter.acceptTotal((long) total);
            }
            emitProgress(InferenceProgress.inferenceStart(total));
        } else if ("patch_start".equals(phase)) {
            emitProgress(InferenceProgress.patchStart(patch, total));
        } else if ("patch_end".equals(phase)) {
            if (tileCounter != null) {
                tileCounter.acceptProgress((long) patch);
            }
            emitProgress(InferenceProgress.patchEnd(patch, total));
        } else if ("merge_start".equals(phase)) {
            emitProgress(InferenceProgress.mergeStart());
        } else if ("inference_end".equals(phase)) {
            emitProgress(InferenceProgress.inferenceEnd());
        }
    }

    private static File resolveModelFile(String modelPath) {
        if (modelPath == null || modelPath.trim().isEmpty()) {
            throw new IllegalArgumentException("UNet model path cannot be empty.");
        }
        File path = new File(modelPath);
        if (path.isFile()) {
            return path.getAbsoluteFile();
        }
        File modelFile = UnetModelRegistry.findModelFile(path);
        if (modelFile != null) {
            return modelFile.getAbsoluteFile();
        }
        throw new IllegalArgumentException("Path provided does not point to a UNet model: " + modelPath);
    }

    private static Map<String, Object> loadModelConfig(File modelFile) throws IOException {
        File dir = modelFile.getParentFile();
        File configFile = new File(dir, "config.json");
        if (!configFile.isFile()) {
            Map<String, Object> fallback = new LinkedHashMap<String, Object>();
            fallback.put("task", "binary_semantic");
            fallback.put("input_channels", 1);
            fallback.put("num_classes", 1);
            fallback.put("postprocessing", new LinkedHashMap<String, Object>());
            return fallback;
        }
        return JSONUtils.load(configFile.getAbsolutePath());
    }

    private static Map<String, Object> normalizedConfig(Map<String, Object> source) {
        Map<String, Object> normalized = new LinkedHashMap<String, Object>();
        if (source != null) {
            normalized.putAll(source);
        }
        normalized.putIfAbsent("task", "binary_semantic");
        normalized.putIfAbsent("input_channels", 1);
        normalized.putIfAbsent("num_classes", 1);
        normalized.putIfAbsent("postprocessing", new LinkedHashMap<String, Object>());
        return normalized;
    }

    private static String normalizedTask(Object value) {
        String task = value == null ? "binary_semantic" : value.toString().trim().toLowerCase(Locale.ROOT);
        if ("classes".equals(task)) {
            return "multiclass_semantic";
        }
        if ("objects".equals(task)) {
            return "instance_friendly";
        }
        return task.isEmpty() || "auto".equals(task) ? "binary_semantic" : task;
    }

    private static int configInt(Map<String, Object> config, String key, int fallback) {
        Object value = config.get(key);
        return value instanceof Number ? ((Number) value).intValue() : fallback;
    }

    private static int nestedConfigInt(Map<String, Object> config, String section, String key, int fallback) {
        Object value = nestedConfigValue(config, section, key);
        return value instanceof Number ? ((Number) value).intValue() : fallback;
    }

    private static String nestedConfigString(Map<String, Object> config, String section, String key,
            String fallback) {
        Object value = nestedConfigValue(config, section, key);
        return value == null ? fallback : value.toString().trim().toLowerCase(Locale.ROOT);
    }

    private static Object nestedConfigValue(Map<String, Object> config, String section, String key) {
        Object nested = config.get(section);
        return nested instanceof Map ? ((Map<?, ?>) nested).get(key) : null;
    }

    private static String addUnetSourcePathCode() {
        String nl = System.lineSeparator();
        return "_jdll_unet_source = r'" + TrainingCodeUtils.py(jdllUnetSourceDir()) + "'" + nl
                + "if _jdll_unet_source and os.path.isdir(_jdll_unet_source) and _jdll_unet_source not in sys.path:" + nl
                + "  sys.path.insert(0, _jdll_unet_source)" + nl;
    }

    private static String jdllUnetSourceDir() {
        String fromProperty = System.getProperty("jdll.unet.path");
        if (fromProperty != null && !fromProperty.trim().isEmpty()) {
            return fromProperty.trim();
        }
        String fromEnv = System.getenv("JDLL_UNET_PATH");
        if (fromEnv != null && !fromEnv.trim().isEmpty()) {
            return fromEnv.trim();
        }
        return DEFAULT_UNET_SOURCE_DIR;
    }

    private static void validateTrainingConfig(Map<String, Object> config) {
        if (config == null) {
            throw new IllegalArgumentException("UNet training config cannot be null.");
        }
        Object outputDir = config.get("output_dir");
        if (outputDir == null || outputDir.toString().trim().isEmpty()) {
            throw new IllegalArgumentException("UNet output_dir cannot be empty.");
        }
        Object datasetPath = config.get("dataset_path");
        if (datasetPath == null || !new File(datasetPath.toString()).exists()) {
            throw new IllegalArgumentException("UNet dataset path does not exist: " + datasetPath);
        }
    }

    private static String buildTrainingCode(Map<String, Object> config) {
        String nl = System.lineSeparator();
        return ""
                + "import json, os, sys" + nl
                + TrainingCodeUtils.apposeStdoutCapture()
                + "import torch" + nl
                + addUnetSourcePathCode()
                + "from jdll_unet.appose_api import train as jdll_unet_train" + nl
                + TrainingCodeUtils.pytorchMemoryCleanupFunction("_jdll_cleanup_pytorch_memory")
                + "_jdll_unet_config = json.loads(r'''" + TrainingCodeUtils.toJson(config) + "''')" + nl
                + "try:" + nl
                + "  _jdll_unet_result = jdll_unet_train(_jdll_unet_config, task=task)" + nl
                + "  for _jdll_kind, _jdll_key in (('last', 'last_checkpoint'), ('best', 'best_checkpoint')):" + nl
                + "    _jdll_path = _jdll_unet_result.get(_jdll_key)" + nl
                + "    if _jdll_path:" + nl
                + "      task.update(message='UNet %s checkpoint: %s' % (_jdll_kind, _jdll_path), info={'type': 'checkpoint', 'kind': _jdll_kind, 'path': _jdll_path})" + nl
                + "  if _jdll_unet_result.get('model_path'):" + nl
                + "    task.update(message='Exported/final UNet model file: ' + str(_jdll_unet_result.get('model_path')), info={'type': 'checkpoint', 'kind': 'final', 'path': _jdll_unet_result.get('model_path')})" + nl
                + "  if _jdll_unet_result.get('model_dir'):" + nl
                + "    task.update(message='Exported/final UNet model directory: ' + str(_jdll_unet_result.get('model_dir')), info={'type': 'checkpoint', 'kind': 'final', 'path': _jdll_unet_result.get('model_dir')})" + nl
                + "  task.outputs['result'] = _jdll_unet_result.get('model_path')" + nl
                + "  task.outputs['model_dir'] = _jdll_unet_result.get('model_dir')" + nl
                + "finally:" + nl
                + "  try:" + nl
                + "    del _jdll_unet_result" + nl
                + "  except Exception:" + nl
                + "    pass" + nl
                + "  _jdll_cleanup_pytorch_memory()" + nl;
    }

    private static void handleTrainingEvent(TaskEvent event,
            Consumer<UnetTrainingProgress> progressConsumer,
            Consumer<UnetValidationPreview> previewConsumer,
            Consumer<String> logConsumer) {
        if (!event.responseType.equals(ResponseType.UPDATE) || event.info == null) {
            return;
        }
        if (event.message != null && logConsumer != null) {
            logConsumer.accept(event.message);
        }
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
                    stringOrNull(event.info.get("preview_path")),
                    stringOrNull(event.info.get("latest_preview_path"))));
        } else if ("complete".equals(type) && logConsumer != null) {
            Object modelDir = event.info.get("model_dir");
            logConsumer.accept("UNet training complete: " + (modelDir == null ? "" : modelDir.toString()));
        } else if ("warning".equals(type) && logConsumer != null) {
            Object message = event.info.get("message");
            if (message != null) {
                logConsumer.accept(message.toString());
            }
        } else if ("training_plan".equals(type) && logConsumer != null) {
            logTrainingPlan(event.info, logConsumer);
        }
    }

    private static void logTrainingPlan(Map<String, Object> plan, Consumer<String> logConsumer) {
        StringBuilder model = new StringBuilder("Resolved UNet model: architecture=")
                .append(plan.get("architecture"))
                .append(", dimensions=").append(plan.get("dimensions"))
                .append(", patch_size=").append(plan.get("patch_size"));
        if (plan.get("context_slices") != null) {
            model.append(", context_slices=").append(plan.get("context_slices"))
                    .append(", context_stride=").append(plan.get("context_stride_policy"));
        }
        model.append(", deep_supervision=").append(plan.get("deep_supervision"))
                .append(", augmentation=").append(plan.get("augmentation_profile"));
        logConsumer.accept(model.toString());
        logConsumer.accept("Resolved UNet runtime: microbatch=" + plan.get("microbatch_size")
                + ", accumulation_steps=" + plan.get("accumulation_steps")
                + ", effective_batch=" + plan.get("effective_batch_size")
                + ", steps_per_epoch=" + plan.get("steps_per_epoch"));

        Object memoryValue = plan.get("memory_plan");
        if (memoryValue instanceof Map) {
            Map<?, ?> memory = (Map<?, ?>) memoryValue;
            logConsumer.accept("Resolved UNet memory plan: preferred_patch=" + memory.get("preferred_patch")
                    + ", resolved_patch=" + memory.get("resolved_patch")
                    + ", budget_gb=" + memory.get("planning_budget_gb")
                    + ", reductions=" + memory.get("reductions"));
        }
    }

    private static String stringOrNull(Object value) {
        return value == null ? null : value.toString();
    }
}
