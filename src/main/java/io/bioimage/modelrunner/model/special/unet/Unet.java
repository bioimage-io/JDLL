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
import java.util.ArrayList;
import java.util.Arrays;
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
import io.bioimage.modelrunner.model.tiling.TileInfo;
import io.bioimage.modelrunner.model.tiling.TileMaker;
import io.bioimage.modelrunner.model.tiling.merger.DenseMerger;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
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

    private static final long DEFAULT_DENSE_TILE_XY = 512L;
    private static final long DEFAULT_DENSE_OUTPUT_HALO_XY = 64L;
    private static final String DEFAULT_UNET_SOURCE_DIR = "/home/carlos/hack_git/jdll-unet";

    private final String modelPath;
    private final Map<String, Object> config;
    private final String task;
    private final int inputChannels;
    private final int outputClasses;

    private Unet(String modelPath, Map<String, Object> config,
            Consumer<InferenceProgress> inferenceProgressConsumer, String device) {
        super(modelPath, modelPath, modelPath, modelPath, config, true, device);
        this.modelPath = new File(modelPath).getAbsolutePath();
        this.config = normalizedConfig(config);
        this.task = normalizedTask(this.config.get("task"));
        this.inputChannels = configInt(this.config, "input_channels", 1);
        this.outputClasses = Math.max(1, configInt(this.config, "num_classes", 1));
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
        if (logConsumer != null) {
            python.debug(line -> TrainingCodeUtils.logTrainingDebug(line, logConsumer));
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

    /**
     * Returns the tile maker.
     *
     * @param <T> the T type parameter.
     * @param <R> the R type parameter.
     * @param inputs the inputs to process.
     * @return the resulting merger.
     */
    @Override
    protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    Merger<Tensor<T>, Tensor<R>> getTileMaker(final List<Tensor<T>> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("UNet tiling needs one input tensor.");
        }
        Tensor<T> reference = inputs.get(0);
        TileMaker tileMaker = TileMaker.build(
                Arrays.asList(createInputTileInfo(reference)),
                createDenseOutputTileInfo(reference));
        DenseMerger<T, R> merger = new DenseMerger<T, R>(tileMaker);
        merger.addCallback(this::runUnetPostprocess);
        merger.configure(inputs);
        return merger;
    }

    /**
     * Returns the output tensor axes.
     *
     * @param outputCount the output count.
     * @return the axes.
     */
    @Override
    protected String getOutputTensorAxes(int outputCount) {
        if ("multiclass_semantic".equals(task)) {
            if (outputCount != 0) {
                throw new IllegalArgumentException("Multiclass UNet has one probability output.");
            }
            return "byxc";
        }
        if ("instance_friendly".equals(task)) {
            if (outputCount > 1) {
                throw new IllegalArgumentException("Instance UNet has foreground and boundary outputs.");
            }
            return "byx";
        }
        if (outputCount != 0) {
            throw new IllegalArgumentException("Binary UNet has one foreground output.");
        }
        return "byx";
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
                + "from jdll_unet.infer import infer as jdll_unet_infer, load_model as jdll_unet_load_model" + nl
                + "from jdll_unet.postprocess import postprocess_binary, postprocess_instance, postprocess_multiclass" + nl
                + "_jdll_unet_device = '" + TrainingCodeUtils.py(device) + "'" + nl
                + MODEL_VAR_NAME + ", _jdll_unet_model_config = jdll_unet_load_model(r'"
                + TrainingCodeUtils.py(modelPath) + "', _jdll_unet_device)" + nl
                + "_jdll_unet_task = str(_jdll_unet_model_config.get('task', 'binary_semantic'))" + nl
                + "task.export(jdll_unet_infer=jdll_unet_infer)" + nl
                + "task.export(postprocess_binary=postprocess_binary)" + nl
                + "task.export(postprocess_instance=postprocess_instance)" + nl
                + "task.export(postprocess_multiclass=postprocess_multiclass)" + nl
                + "task.export(_jdll_unet_model_config=_jdll_unet_model_config)" + nl
                + "task.export(_jdll_unet_task=_jdll_unet_task)" + nl
                + "task.export(_jdll_unet_device=_jdll_unet_device)" + nl
                + "task.export(" + MODEL_VAR_NAME + "=" + MODEL_VAR_NAME + ")" + nl;
    }

    /**
     * Creates the Python tile inference code.
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
        code += inputName + " = " + ConvertDims.getMethodName() + "(" + inputName
                + ", '" + inRais.get(0).getAxesOrderString().toLowerCase(Locale.ROOT)
                + "', out_order='yxc', output_type='numpy', contiguous=True, n_channels="
                + inputChannels + ")" + nl;
        code += "_jdll_unet_tile_config = {'model_path': r'" + TrainingCodeUtils.py(modelPath)
                + "', 'device': _jdll_unet_device, 'tile_size': [int(" + inputName
                + ".shape[0]), int(" + inputName + ".shape[1])], 'tile_overlap': 0.0}" + nl;
        code += "_jdll_unet_result = jdll_unet_infer(_jdll_unet_tile_config, {'image': " + inputName + "})" + nl;
        code += "_jdll_unet_outputs = _jdll_unet_result['outputs']" + nl;
        code += "if _jdll_unet_task == 'multiclass_semantic':" + nl;
        code += "  " + OUTPUT_LIST_KEY + " = [np.expand_dims(np.moveaxis(_jdll_unet_outputs['probabilities'], 0, -1).astype(np.float32, copy=False), 0)]" + nl;
        code += "elif _jdll_unet_task == 'instance_friendly':" + nl;
        code += "  " + OUTPUT_LIST_KEY + " = [np.expand_dims(_jdll_unet_outputs['foreground_probability'].astype(np.float32, copy=False), 0), "
                + "np.expand_dims(_jdll_unet_outputs['boundary_probability'].astype(np.float32, copy=False), 0)]" + nl;
        code += "else:" + nl;
        code += "  " + OUTPUT_LIST_KEY + " = [np.expand_dims(_jdll_unet_outputs['foreground_probability'].astype(np.float32, copy=False), 0)]" + nl;
        code += String.format("handle_output_list(%s, %s, %s, %s, %s)",
                OUTPUT_LIST_KEY, SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY) + nl;
        code += taskOutputsCode();
        return code;
    }

    private <R extends RealType<R> & NativeType<R>> List<Tensor<R>> runUnetPostprocess(
            final List<Tensor<R>> reconstructed) {
        if (reconstructed == null || reconstructed.isEmpty()) {
            return reconstructed;
        }
        try {
            String nl = System.lineSeparator();
            String code = ConvertDims.getMethodDeclaration() + nl;
            code += "created_shms.clear()" + nl;
            code += "task.outputs.clear()" + nl;
            code += SHM_NAMES_KEY + " = []" + nl;
            code += DTYPES_KEY + " = []" + nl;
            code += DIMS_KEY + " = []" + nl;
            List<String> names = new ArrayList<String>();
            for (int i = 0; i < reconstructed.size(); i ++) {
                String name = "unet_reconstructed_" + i + "_" + java.util.UUID.randomUUID().toString().replace("-", "_");
                names.add(name);
                SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(reconstructed.get(i).getData(), false, false);
                code += codeToConvertShmaToPython(shma, name);
                inShmaList.add(shma);
            }
            code += "_jdll_unet_post = dict(_jdll_unet_model_config.get('postprocessing', {}))" + nl;
            if ("multiclass_semantic".equals(task)) {
                code += names.get(0) + " = " + ConvertDims.getMethodName() + "(" + names.get(0)
                        + ", 'byxc', out_order='cyx', n_channels=" + outputClasses
                        + ", output_type='numpy', contiguous=False)" + nl;
                code += "_jdll_unet_labels = postprocess_multiclass(" + names.get(0)
                        + ", min_object_size=int(_jdll_unet_post.get('min_object_size', 0)))['mask']" + nl;
            } else if ("instance_friendly".equals(task)) {
                code += names.get(0) + " = " + ConvertDims.getMethodName() + "(" + names.get(0)
                        + ", 'byx', out_order='yx', n_channels=1, output_type='numpy', contiguous=False)" + nl;
                code += names.get(1) + " = " + ConvertDims.getMethodName() + "(" + names.get(1)
                        + ", 'byx', out_order='yx', n_channels=1, output_type='numpy', contiguous=False)" + nl;
                code += "_jdll_unet_labels = postprocess_instance(" + names.get(0) + ", " + names.get(1)
                        + ", threshold=float(_jdll_unet_post.get('threshold', 0.5)), "
                        + "min_object_size=int(_jdll_unet_post.get('min_object_size', 0)))['labels']" + nl;
            } else {
                code += names.get(0) + " = " + ConvertDims.getMethodName() + "(" + names.get(0)
                        + ", 'byx', out_order='yx', n_channels=1, output_type='numpy', contiguous=False)" + nl;
                code += "_jdll_unet_binary = postprocess_binary(" + names.get(0)
                        + ", threshold=float(_jdll_unet_post.get('threshold', 0.5)), "
                        + "min_object_size=int(_jdll_unet_post.get('min_object_size', 0)), "
                        + "fill_holes=bool(_jdll_unet_post.get('fill_holes', False)), "
                        + "connected_components=bool(_jdll_unet_post.get('connected_components', True)))" + nl;
                code += "_jdll_unet_labels = _jdll_unet_binary.get('labels', _jdll_unet_binary['mask'])" + nl;
            }
            code += String.format("handle_output(_jdll_unet_labels.astype(np.float32, copy=False), %s, %s, %s, %s)",
                    SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY) + nl;
            code += taskOutputsCode();
            Map<String, RandomAccessibleInterval<R>> outputs = executeCode(code);
            if (outputs.isEmpty()) {
                return reconstructed;
            }
            return Arrays.asList(Tensor.build("labels", "yx", outputs.values().iterator().next()));
        } catch (RunModelException e) {
            throw new IllegalStateException("UNet postprocessing failed after dense tile reconstruction.", e);
        }
    }

    private static <T extends RealType<T> & NativeType<T>> TileInfo createInputTileInfo(final Tensor<T> input) {
        String axes = input.getAxesOrderString().toLowerCase(Locale.ROOT);
        long[] imageDims = input.getData().dimensionsAsLongArray();
        long[] tileDims = imageDims.clone();
        int xAxis = axisIndex(axes, 'x');
        int yAxis = axisIndex(axes, 'y');
        tileDims[xAxis] = Math.min(DEFAULT_DENSE_TILE_XY, imageDims[xAxis] * 3L);
        tileDims[yAxis] = Math.min(DEFAULT_DENSE_TILE_XY, imageDims[yAxis] * 3L);
        return TileInfo.build(input.getName(), imageDims, axes, tileDims, axes);
    }

    private <T extends RealType<T> & NativeType<T>> List<TileInfo> createDenseOutputTileInfo(final Tensor<T> reference) {
        String axes = reference.getAxesOrderString().toLowerCase(Locale.ROOT);
        long[] inputDims = reference.getData().dimensionsAsLongArray();
        long batch = axisSizeOrDefault(inputDims, axes, 'b', 1L);
        long y = axisSize(inputDims, axes, 'y');
        long x = axisSize(inputDims, axes, 'x');
        long tileY = Math.min(DEFAULT_DENSE_TILE_XY, y * 3L);
        long tileX = Math.min(DEFAULT_DENSE_TILE_XY, x * 3L);
        long haloY = safeOutputHalo(tileY);
        long haloX = safeOutputHalo(tileX);
        List<TileInfo> outputInfo = new ArrayList<TileInfo>();
        if ("multiclass_semantic".equals(task)) {
            TileInfo probabilities = TileInfo.build("output_0",
                    new long[] {batch, y, x, outputClasses}, "byxc",
                    new long[] {1L, tileY, tileX, outputClasses}, "byxc");
            probabilities.setHalo(new long[] {0L, haloY, haloX, 0L}, "byxc");
            outputInfo.add(probabilities);
        } else {
            TileInfo foreground = TileInfo.build("output_0",
                    new long[] {batch, y, x}, "byx",
                    new long[] {1L, tileY, tileX}, "byx");
            foreground.setHalo(new long[] {0L, haloY, haloX}, "byx");
            outputInfo.add(foreground);
            if ("instance_friendly".equals(task)) {
                TileInfo boundary = TileInfo.build("output_1",
                        new long[] {batch, y, x}, "byx",
                        new long[] {1L, tileY, tileX}, "byx");
                boundary.setHalo(new long[] {0L, haloY, haloX}, "byx");
                outputInfo.add(boundary);
            }
        }
        TileInfo.adaptHalos(outputInfo);
        return outputInfo;
    }

    private static long safeOutputHalo(final long outputTileSize) {
        return Math.min(DEFAULT_DENSE_OUTPUT_HALO_XY, Math.max(0L, (outputTileSize - 1L) / 2L));
    }

    private static int axisIndex(final String axes, final char axis) {
        int index = axes.indexOf(axis);
        if (index < 0) {
            throw new IllegalArgumentException("Axes '" + axes + "' do not contain axis '" + axis + "'.");
        }
        return index;
    }

    private static long axisSize(final long[] dims, final String axes, final char axis) {
        return dims[axisIndex(axes, axis)];
    }

    private static long axisSizeOrDefault(final long[] dims, final String axes, final char axis,
            final long defaultValue) {
        int index = axes.indexOf(axis);
        return index < 0 ? defaultValue : dims[index];
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
        if (event.message != null && logConsumer != null) {
            logConsumer.accept(event.message);
        }
        if (!event.responseType.equals(ResponseType.UPDATE) || event.info == null) {
            return;
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
        }
    }

    private static String stringOrNull(Object value) {
        return value == null ? null : value.toString();
    }
}
