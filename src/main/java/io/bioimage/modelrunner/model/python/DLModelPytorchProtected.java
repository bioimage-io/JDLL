package io.bioimage.modelrunner.model.python;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskException;
import org.apposed.appose.util.Environments;
import org.apposed.appose.util.Messages;

import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.bioimageio.tiling.TileMaker;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.java.DLModelJava.TilingConsumer;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.system.GpuCompatibility;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

/**
 * Class that defines the methods required to load a Python PyTorch model and
 * run inference through an isolated Python environment.
 *
 * @author Carlos Garcia
 */
public class DLModelPytorchProtected extends BaseModel {

    protected final String modelFile;

    protected final String callable;

    protected final String importModule;

    protected final String weightsPath;

    protected final Map<String, Object> kwargs;

    protected String envPath;

    private Service python;

    private final PixiEnvironmentSpec environmentSpec;

    protected List<SharedMemoryArray> inShmaList = new ArrayList<SharedMemoryArray>();

    private List<String> outShmNames;

    private List<String> outShmDTypes;

    private List<long[]> outShmDims;

    /**
     * List containing the desired tiling strategy for each of the input tensors.
     */
    protected List<TileInfo> inputTiles;

    /**
     * List containing the desired tiling strategy for each of the output tensors.
     */
    protected List<TileInfo> outputTiles;

    /**
     * Whether to do tiling or not when doing inference.
     */
    protected boolean tiling = false;

    /**
     * Consumer used to inform the current tile being processed and in how many
     * tiles the input images are going to be separated.
     */
    protected TilingConsumer tileCounter;

    /**
     * Default environment directory name.
     */
    public static final String COMMON_PYTORCH_ENV_NAME = "biapy";

    private static final String PIXI_TEMPLATE_RESOURCE = "/biapy-pixi.toml";
    
    private static final List<String> CUDA_COMPAT_VERSIONS = new ArrayList<>(Arrays.asList("12.4", "12.1", "11.8"));

    protected static final boolean IS_ARM = PlatformDetection.isMacOS()
            && (PlatformDetection.getArch().equals(PlatformDetection.ARCH_ARM64)
            || PlatformDetection.isUsingRosseta());

    protected static String INSTALLATION_DIR = Environments.apposeEnvsDir();

    protected static final String MODEL_VAR_NAME = "model_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
            + "if 'sys' not in globals().keys():" + System.lineSeparator()
            + "  import sys" + System.lineSeparator()
            + "  globals()['sys'] = sys" + System.lineSeparator()
            + "if 'np' not in globals().keys():" + System.lineSeparator()
            + "  import numpy as np" + System.lineSeparator()
            + "  globals()['np'] = np" + System.lineSeparator()
            + "if 'os' not in globals().keys():" + System.lineSeparator()
            + "  import os" + System.lineSeparator()
            + "  globals()['os'] = os" + System.lineSeparator()
            + "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
            + "  from multiprocessing import shared_memory" + System.lineSeparator()
            + "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
            + "%s" + System.lineSeparator()
            + "%s" + System.lineSeparator()
            + "if '%s' not in globals().keys():" + System.lineSeparator()
            + "  globals()['%s'] = %s" + System.lineSeparator();

    protected static final String OUTPUT_LIST_KEY = "out_list" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String SHMS_KEY = "shms_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String SHM_NAMES_KEY = "shm_names_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String DTYPES_KEY = "dtypes_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String DIMS_KEY = "dims_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String RECOVER_OUTPUTS_CODE = ""
            + "def handle_output(outs_i):" + System.lineSeparator()
            + "    if type(outs_i) == np.ndarray:" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=outs_i.nbytes)" + System.lineSeparator()
            + "      sh_np_array = np.ndarray(outs_i.shape, dtype=outs_i.dtype, buffer=shm.buf)" + System.lineSeparator()
            + "      np.copyto(sh_np_array, outs_i)" + System.lineSeparator()
            + "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
            + "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
            + "      " + DTYPES_KEY + ".append(str(outs_i.dtype))" + System.lineSeparator()
            + "      " + DIMS_KEY + ".append(outs_i.shape)" + System.lineSeparator()
            + "    elif str(type(outs_i)) == \"<class 'torch.Tensor'>\":" + System.lineSeparator()
            + "      if 'torch' not in globals().keys():" + System.lineSeparator()
            + "        import torch" + System.lineSeparator()
            + "        globals()['torch'] = torch" + System.lineSeparator()
            + "      else:" + System.lineSeparator()
            + "        torch = globals()['torch']" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=outs_i.numel() * outs_i.element_size())" + System.lineSeparator()
            + "      np_arr = np.ndarray(outs_i.shape, dtype=str(outs_i.dtype).split('.')[-1], buffer=shm.buf)" + System.lineSeparator()
            + "      tensor_np_view = torch.from_numpy(np_arr)" + System.lineSeparator()
            + "      tensor_np_view.copy_(outs_i)" + System.lineSeparator()
            + "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
            + "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
            + "      " + DTYPES_KEY + ".append(str(outs_i.dtype).split('.')[-1])" + System.lineSeparator()
            + "      " + DIMS_KEY + ".append(outs_i.shape)" + System.lineSeparator()
            + "    elif type(outs_i) == int:" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=8)" + System.lineSeparator()
            + "      shm.buf[:8] = outs_i.to_bytes(8, byteorder='little', signed=True)" + System.lineSeparator()
            + "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
            + "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
            + "      " + DTYPES_KEY + ".append('int64')" + System.lineSeparator()
            + "      " + DIMS_KEY + ".append((1))" + System.lineSeparator()
            + "    elif type(outs_i) == float:" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=8)" + System.lineSeparator()
            + "      shm.buf[:8] = outs_i.to_bytes(8, byteorder='little', signed=True)" + System.lineSeparator()
            + "      " + SHMS_KEY + ".append(shm)" + System.lineSeparator()
            + "      " + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator()
            + "      " + DTYPES_KEY + ".append('float64')" + System.lineSeparator()
            + "      " + DIMS_KEY + ".append((1))" + System.lineSeparator()
            + "    elif type(outs_i) == tuple or type(outs_i) == list:" + System.lineSeparator()
            + "      handle_output_list(outs_i)" + System.lineSeparator()
            + "    else:" + System.lineSeparator()
            + "      task.update('output type : ' + str(type(outs_i)) + ' not supported. "
            + "Only supported output types are: np.ndarray, torch.tensor, int and float, "
            + "or a list or tuple of any of those.')" + System.lineSeparator()
            + System.lineSeparator()
            + System.lineSeparator()
            + "def handle_output_list(out_list):" + System.lineSeparator()
            + "  if type(out_list) == tuple or type(out_list) == list:" + System.lineSeparator()
            + "    for outs_i in out_list:" + System.lineSeparator()
            + "      handle_output(outs_i)" + System.lineSeparator()
            + "  else:" + System.lineSeparator()
            + "    handle_output(out_list)" + System.lineSeparator()
            + System.lineSeparator()
            + System.lineSeparator()
            + "globals()['handle_output_list'] = handle_output_list" + System.lineSeparator()
            + "globals()['handle_output'] = handle_output" + System.lineSeparator()
            + System.lineSeparator()
            + System.lineSeparator();

    private static final String CLEAN_SHM_CODE = ""
            + "if '" + SHMS_KEY + "' in globals().keys():" + System.lineSeparator()
            + "  for s in " + SHMS_KEY + ":" + System.lineSeparator()
            + "    s.close()" + System.lineSeparator()
            + "    s.unlink()" + System.lineSeparator()
            + "    del s" + System.lineSeparator();

    private static final String JDLL_UUID = UUID.randomUUID().toString().replaceAll("-", "_");

    /**
     * Creates a new DLModelPytorchProtected.
     *
     * @param modelFile the modelFile parameter
     * @param callable the callable parameter
     * @param importModule the importModule parameter
     * @param weightsPath the weightsPath parameter
     * @param kwargs the kwargs parameter
     * @throws BuildException if there is an error building or opening the environment
     */
    protected DLModelPytorchProtected(
            final String modelFile,
            final String callable,
            final String importModule,
            final String weightsPath,
            final Map<String, Object> kwargs) throws BuildException {
        this(modelFile, callable, importModule, weightsPath, kwargs, false);
    }

    /**
     * Creates a new DLModelPytorchProtected.
     *
     * @param modelFile the modelFile parameter
     * @param callable the callable parameter
     * @param importModule the importModule parameter
     * @param weightsPath the weightsPath parameter
     * @param kwargs the kwargs parameter
     * @param customJDLL the customJDLL parameter
     * @throws BuildException if there is an error building or opening the environment
     */
    protected DLModelPytorchProtected(
            final String modelFile,
            final String callable,
            final String importModule,
            final String weightsPath,
            final Map<String, Object> kwargs,
            final boolean customJDLL) throws BuildException {

        if (!customJDLL && (new File(modelFile).isFile() == false || !modelFile.endsWith(".py")) && importModule == null) {
            throw new IllegalArgumentException("The model file does not correspond to an existing .py file.");
        }
        if (new File(weightsPath).isFile() == false
                || (!customJDLL && !(weightsPath.endsWith(".pt") || weightsPath.endsWith(".pth")))) {
            throw new IllegalArgumentException("The weights file does not correspond to an existing .pt/.pth file.");
        }

        this.callable = callable;
        if (!customJDLL && (modelFile != null && new File(modelFile).isFile())) {
            this.modelFile = new File(modelFile).getAbsolutePath();
        } else {
            this.modelFile = null;
        }

        if (!customJDLL && importModule != null) {
            this.importModule = importModule;
        } else {
            this.importModule = null;
        }

        if (new File(weightsPath).isFile()) {
            this.modelFolder = new File(weightsPath).getParentFile().getAbsolutePath();
        } else if (new File(modelFile).isFile()) {
            this.modelFolder = new File(modelFile).getParentFile().getAbsolutePath();
        }

        this.weightsPath = new File(weightsPath).getAbsolutePath();
        this.kwargs = kwargs;

        this.environmentSpec = resolvePytorchEnv();
        this.envPath = environmentSpec.getEnvironmentDirectory().getAbsolutePath();

        createPythonService();
    }

    /**
     * Creates the Python service used to run the model.
     *
     * @throws BuildException if the environment cannot be opened
     */
    protected void createPythonService() throws BuildException {
        final Environment env;

        env = Appose.pixi()
                .environment(environmentSpec.getSelectedEnvironment())
                .wrap(environmentSpec.getEnvironmentDirectory());

        python = env.python();
        python.debug(System.err::println);
    }

    /**
     * @return the Python service used to run the model
     */
    public Service getPythonSerice() {
        return this.python;
    }

    /**
     * @return the path to the Python environment used by the model
     */
    public String getEnvPath() {
        return this.envPath;
    }

    /**
     * @return the resolved Pixi environment name for the current machine
     */
    public String getSelectedEnvironment() {
        return environmentSpec.getSelectedEnvironment();
    }

    /**
     * Sets the path to the environment that should be used to run the model.
     *
     * @param envPath
     *     path to the environment that should be used to run the model
     * @throws BuildException if there is any error connecting to the Python environment
     */
    public void setCustomEnvPath(final String envPath) throws BuildException {
        this.envPath = envPath;
        if (this.python != null) {
            this.python.close();
        }
        createPythonService();
    }

    /**
     * @return whether the image is going to be processed in tiles or not
     */
    public boolean isTiling() {
        return this.tiling;
    }

    /**
     * Sets whether images should be processed in tiles or in one pass.
     *
     * @param doTiling
     *     whether to do tiling on inference
     */
    public void setTiling(final boolean doTiling) {
        this.tiling = doTiling;
    }

    /**
     * Sets the wanted tile specifications for each input and output tensor.
     *
     * @param inputTiles
     *     the specifications of how each input image can be tiled
     * @param outputTiles
     *     the specifications of how each output image will be tiled
     */
    public void setTileInfo(final List<TileInfo> inputTiles, final List<TileInfo> outputTiles) {
        this.inputTiles = inputTiles;
        this.outputTiles = outputTiles;
        this.tiling = true;
    }

    /**
     * Sets a consumer used to track tile execution progress.
     *
     * @param tileCounter
     *     consumer used to track tile inference
     */
    public void setTilingCounter(final TilingConsumer tileCounter) {
        this.tileCounter = tileCounter;
    }

    /**
     * Loads the model into the Python service.
     *
     * @throws LoadModelException if the model cannot be loaded
     */
    @Override
    public void loadModel() throws LoadModelException {
        if (loaded) {
            return;
        }
        if (closed) {
            throw new RuntimeException("Cannot load model after it has been closed");
        }

        try {
            String code = buildModelCode();
            code += RECOVER_OUTPUTS_CODE;
            final Task task = python.task(code);
            task.waitFor();
            ensureTaskSucceeded(task);
        } catch (IOException | InterruptedException | TaskException e) {
            throw new LoadModelException(Messages.stackTrace(e));
        }

        loaded = true;
    }

    /**
     * Copies a Python source file to another path.
     *
     * @param inputPath source file path
     * @param outputPath destination file path
     * @throws IOException if copying fails
     */
    private static void copyAndReplace(final String inputPath, final String outputPath) throws IOException {
        if (new File(outputPath).isFile()) {
            return;
        }
        Files.write(Paths.get(outputPath), Files.readAllBytes(Paths.get(inputPath)));
    }

    /**
     * Builds the Python code required to instantiate the model.
     *
     * @return the resulting Python code
     * @throws IOException if an I/O error occurs
     */
    protected String buildModelCode() throws IOException {
        String addPath = "";
        String importStr = "";
        String code = ""
                + "device = 'cpu'" + System.lineSeparator()
                + "if 'torch' not in globals().keys():" + System.lineSeparator()
                + "  import torch" + System.lineSeparator()
                + "  globals()['torch'] = torch" + System.lineSeparator()
                + (!IS_ARM ? ""
                : "  if torch.backends.mps.is_built() and torch.backends.mps.is_available():" + System.lineSeparator()
                + "    device = 'mps'" + System.lineSeparator());

        if (modelFile != null) {
            String moduleName = new File(modelFile).getName();
            moduleName = moduleName.substring(0, moduleName.length() - 3);
            if (moduleName.contains("+")) {
                final String newModelFile = modelFile.replaceAll("\\+", JDLL_UUID);
                copyAndReplace(modelFile, newModelFile);
                moduleName = new File(newModelFile).getName();
                moduleName = moduleName.substring(0, moduleName.length() - 3);
                addPath = String.format(
                        "sys.path.append(os.path.abspath(r'%s'))",
                        new File(newModelFile).getParentFile().getAbsolutePath());
                importStr = String.format("from %s import %s", moduleName, callable);
            } else {
                addPath = String.format(
                        "sys.path.append(os.path.abspath(r'%s'))",
                        new File(modelFile).getParentFile().getAbsolutePath());
                importStr = String.format("from %s import %s", moduleName, callable);
            }
        } else {
            importStr = String.format("from %s import %s", this.importModule, callable);
        }

        code += String.format(LOAD_MODEL_CODE_ABSTRACT, addPath, importStr, callable, callable, callable);

        code += MODEL_VAR_NAME + "=" + callable + "(" + codeForKwargs() + ")" + System.lineSeparator();
        code += "if any(isinstance(m, torch.nn.ConvTranspose3d) for m in "
                + MODEL_VAR_NAME + ".modules()):" + System.lineSeparator();
        code += "  device = 'cpu'" + System.lineSeparator();
        code += "globals()['device'] = device" + System.lineSeparator();
        code += MODEL_VAR_NAME + ".to(device)" + System.lineSeparator();
        code += "try:" + System.lineSeparator()
                + "  " + MODEL_VAR_NAME + ".load_state_dict("
                + "torch.load(r'" + this.weightsPath + "', map_location=" + MODEL_VAR_NAME + ".device))"
                + System.lineSeparator()
                + "except:" + System.lineSeparator()
                + "  " + MODEL_VAR_NAME + ".load_state_dict("
                + "torch.load(r'" + this.weightsPath + "', map_location=torch.device(device)))"
                + System.lineSeparator();
        code += "globals()['" + MODEL_VAR_NAME + "'] = " + MODEL_VAR_NAME + System.lineSeparator();
        return code;
    }

    private String codeForKwargsList(final List<Object> list) {
        String code = "[";
        for (Object codeVal : list) {
            if (codeVal == null) {
                code += "None";
            } else if ((codeVal instanceof Boolean && (Boolean) codeVal) || codeVal.equals("true")) {
                code += "True";
            } else if ((codeVal instanceof Boolean && !((Boolean) codeVal)) || codeVal.equals("false")) {
                code += "False";
            } else if (codeVal instanceof String) {
                code += "\"" + codeVal + "\"";
            } else if (codeVal instanceof List) {
                code += codeForKwargsList((List<Object>) codeVal);
            } else if (codeVal instanceof Map) {
                code += codeForKwargsMap((Map<String, Object>) codeVal);
            } else {
                code += codeVal;
            }
            code += ",";
        }
        code += "]";
        return code;
    }

    private String codeForKwargsMap(final Map<String, Object> map) {
        String code = "{";
        for (Entry<String, Object> entry : map.entrySet()) {
            Object codeVal = entry.getValue();
            code += "'" + entry.getKey() + "':";
            if (codeVal == null) {
                code += "None";
            } else if ((codeVal instanceof Boolean && (Boolean) codeVal) || codeVal.equals("true")) {
                code += "True";
            } else if ((codeVal instanceof Boolean && !((Boolean) codeVal)) || codeVal.equals("false")) {
                code += "False";
            } else if (codeVal instanceof String) {
                code += "\"" + codeVal + "\"";
            } else if (codeVal instanceof List) {
                code += codeForKwargsList((List<Object>) codeVal);
            } else if (codeVal instanceof Map) {
                code += codeForKwargsMap((Map<String, Object>) codeVal);
            } else {
                code += codeVal;
            }
            code += ",";
        }
        code += "}";
        return code;
    }

    private String codeForKwargs() {
        String code = "";
        for (Entry<String, Object> ee : kwargs.entrySet()) {
            Object codeVal = ee.getValue();
            if (codeVal == null) {
                codeVal = "None";
            } else if ((codeVal instanceof Boolean && (Boolean) codeVal) || codeVal.equals("true")) {
                codeVal = "True";
            } else if ((codeVal instanceof Boolean && !((Boolean) codeVal)) || codeVal.equals("false")) {
                codeVal = "False";
            } else if (codeVal instanceof String) {
                codeVal = "\"" + codeVal + "\"";
            } else if (codeVal instanceof List) {
                codeVal = codeForKwargsList((List<Object>) codeVal);
            } else if (codeVal instanceof Map) {
                codeVal = codeForKwargsMap((Map<String, Object>) codeVal);
            }
            code += ee.getKey() + "=" + codeVal + ",";
        }
        return code;
    }

    /**
     * Closes the Python service.
     */
    @Override
    public void close() {
        if (!loaded) {
            return;
        }
        python.close();
        loaded = false;
        closed = true;
    }

    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    Map<String, RandomAccessibleInterval<R>> predictForInputTensors(final List<Tensor<T>> inTensors)
            throws RunModelException {
        if (!loaded) {
            throw new RuntimeException("Please load the model first.");
        }
        final List<String> names = inTensors.stream()
                .map(tt -> tt.getName() + "_np")
                .collect(Collectors.toList());
        final List<RandomAccessibleInterval<T>> rais = inTensors.stream()
                .map(Tensor::getData)
                .collect(Collectors.toList());
        return executeCode(createInputsCode(rais, names));
    }

    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    Map<String, RandomAccessibleInterval<R>> executeCode(final String code) throws RunModelException {
        Map<String, RandomAccessibleInterval<R>> outMap;
        try {
            final Task task = python.task(code);
            task.waitFor();
            ensureTaskSucceeded(task);
            loaded = true;
            outMap = reconstructOutputs(task);
            cleanShm();
        } catch (InterruptedException | TaskException | IOException e) {
            try {
                cleanShm();
            } catch (InterruptedException | TaskException e1) {
                throw new RunModelException(Messages.stackTrace(e1));
            }
            throw new RunModelException(Messages.stackTrace(e));
        }
        return outMap;
    }

    /**
     * Runs inference directly on a list of input images.
     *
     * @param <T> input data type
     * @param <R> output data type
     * @param inputs input images
     * @return output images
     * @throws RunModelException if model execution fails
     */
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<RandomAccessibleInterval<R>> inference(final List<RandomAccessibleInterval<T>> inputs)
            throws RunModelException {

        if (!loaded) {
            throw new RuntimeException("Please load the model first.");
        }

        final List<String> names = IntStream.range(0, inputs.size())
                .mapToObj(i -> "var_" + UUID.randomUUID().toString().replace("-", "_"))
                .collect(Collectors.toList());

        final String code = createInputsCode(inputs, names);
        final Map<String, RandomAccessibleInterval<R>> map = executeCode(code);

        final List<RandomAccessibleInterval<R>> outRais = new ArrayList<RandomAccessibleInterval<R>>();
        for (Entry<String, RandomAccessibleInterval<R>> ee : map.entrySet()) {
            outRais.add(ee.getValue());
        }
        return outRais;
    }

    /**
     * Creates the Python code required to transfer inputs through shared memory.
     *
     * @param rais the input images
     * @param names the Python variable names
     * @param <T> input data type
     * @return the resulting Python code
     */
    protected <T extends RealType<T> & NativeType<T>> String createInputsCode(
            final List<RandomAccessibleInterval<T>> rais,
            final List<String> names) {
        String code = "created_shms = []" + System.lineSeparator();
        code += "try:" + System.lineSeparator();
        for (int i = 0; i < rais.size(); i++) {
            final SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(rais.get(i), false, false);
            code += codeToConvertShmaToPython(shma, names.get(i));
            inShmaList.add(shma);
        }
        code += "  " + MODEL_VAR_NAME + ".eval()" + System.lineSeparator();
        code += "  with torch.no_grad():" + System.lineSeparator();
        code += "    " + OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + "(";
        for (int i = 0; i < rais.size(); i++) {
            code += "torch.from_numpy(" + names.get(i) + ").to(device), ";
        }
        code = code.substring(0, code.length() - 2);
        code += ")" + System.lineSeparator();
        code += ""
                + "  " + SHMS_KEY + " = []" + System.lineSeparator()
                + "  " + SHM_NAMES_KEY + " = []" + System.lineSeparator()
                + "  " + DTYPES_KEY + " = []" + System.lineSeparator()
                + "  " + DIMS_KEY + " = []" + System.lineSeparator()
                + "  " + "globals()['" + SHMS_KEY + "'] = " + SHMS_KEY + System.lineSeparator()
                + "  " + "globals()['" + SHM_NAMES_KEY + "'] = " + SHM_NAMES_KEY + System.lineSeparator()
                + "  " + "globals()['" + DTYPES_KEY + "'] = " + DTYPES_KEY + System.lineSeparator()
                + "  " + "globals()['" + DIMS_KEY + "'] = " + DIMS_KEY + System.lineSeparator();
        code += "  " + "handle_output_list(" + OUTPUT_LIST_KEY + ")" + System.lineSeparator();

        final String closeEverythingWin = closeSHMWin();
        code += "  " + closeEverythingWin + System.lineSeparator();
        code += "except Exception as e:" + System.lineSeparator();
        code += "  " + closeEverythingWin + System.lineSeparator();
        code += "  raise e" + System.lineSeparator();
        code += taskOutputsCode();
        return code;
    }

    /**
     * Returns the Windows-only cleanup snippet for shared memory.
     *
     * @return the resulting Python code
     */
    protected static String closeSHMWin() {
        if (!PlatformDetection.isWindows()) {
            return "";
        }
        return "[(shm_i.close(), shm_i.unlink()) for shm_i in created_shms]";
    }

    /**
     * Returns the code that copies the Python task outputs into Appose task outputs.
     *
     * @return the resulting Python code
     */
    protected String taskOutputsCode() {
        return ""
                + "task.outputs['" + SHM_NAMES_KEY + "'] = " + SHM_NAMES_KEY + System.lineSeparator()
                + "task.outputs['" + DTYPES_KEY + "'] = " + DTYPES_KEY + System.lineSeparator()
                + "task.outputs['" + DIMS_KEY + "'] = " + DIMS_KEY + System.lineSeparator();
    }

    @Override
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<T>> run(final List<Tensor<R>> inputTensors) throws RunModelException {
        if (!this.isLoaded()) {
            throw new RunModelException("Please first load the model.");
        }
        if (!this.tiling) {
            throw new UnsupportedOperationException("Cannot run a DLModel if no information about the outputs is provided."
                    + " Either try with 'run( List< Tensor < T > > inTensors, List< Tensor < R > > outTensors )'"
                    + " or set the tiling information with 'setTileInfo(List<TileInfo> inputTiles, List<TileInfo> outputTiles)'."
                    + " Another option is to run simple inference over an ImgLib2 RandomAccessibleInterval with"
                    + " 'inference(List<RandomAccessibleInteral<T>> input)'");
        }
        if (this.isTiling() && (inputTiles != null || this.inputTiles.size() == 0)) {
            throw new UnsupportedOperationException("Tiling is set to 'true' but the input tiles are not well defined");
        } else if (this.isTiling() && (this.outputTiles == null || this.outputTiles.size() == 0)) {
            throw new UnsupportedOperationException("Tiling is set to 'true' but the output tiles are not well defined");
        }

        final TileMaker maker = TileMaker.build(inputTiles, outputTiles);
        final List<Tensor<T>> outTensors = createOutputTensors();
        runTiling(inputTensors, outTensors, maker);
        return outTensors;
    }

    private <T extends RealType<T> & NativeType<T>> List<Tensor<T>> createOutputTensors() {
        final List<Tensor<T>> outputTensors = new ArrayList<Tensor<T>>();
        for (TileInfo tt : this.outputTiles) {
            outputTensors.add((Tensor<T>) Tensor.buildBlankTensor(
                    tt.getName(),
                    tt.getImageAxesOrder(),
                    tt.getImageDims(),
                    (T) new FloatType()));
        }
        return outputTensors;
    }

    @Override
    public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void run(final List<Tensor<T>> inTensors, final List<Tensor<R>> outTensors) throws RunModelException {
        if (!this.isLoaded()) {
            throw new RunModelException("Please first load the model.");
        }
        if (!this.tiling) {
            this.runNoTiles(inTensors, outTensors);
            return;
        }
        if (this.isTiling() && (inputTiles != null || this.inputTiles.size() == 0)) {
            throw new UnsupportedOperationException("Tiling is set to 'true' but the input tiles are not well defined");
        } else if (this.isTiling() && (this.outputTiles == null || this.outputTiles.size() == 0)) {
            throw new UnsupportedOperationException("Tiling is set to 'true' but the output tiles are not well defined");
        }

        final TileMaker tiles = TileMaker.build(inputTiles, outputTiles);
        for (int i = 0; i < tiles.getNumberOfTiles(); i++) {
            final Tensor<R> tt = outTensors.get(i);
            final long[] expectedSize = tiles.getOutputImageSize(tt.getName());
            if (expectedSize == null) {
                throw new IllegalArgumentException("Tensor '" + tt.getName() + "' is missing in the outputs.");
            } else if (!tt.isEmpty() && Arrays.equals(expectedSize, tt.getData().dimensionsAsLongArray())) {
                throw new IllegalArgumentException("Tensor '" + tt.getName() + "' size is different than the expected size"
                        + " defined for the output image: " + Arrays.toString(tt.getData().dimensionsAsLongArray())
                        + " vs " + Arrays.toString(expectedSize) + ".");
            }
        }
        runTiling(inTensors, outTensors, tiles);
    }

    protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runTiling(final List<Tensor<R>> inputTensors, final List<Tensor<T>> outputTensors, final TileMaker tiles)
            throws RunModelException {
        for (int i = 0; i < tiles.getNumberOfTiles(); i++) {
            final int nTile = 0 + i;
            final List<Tensor<R>> inputTiles = inputTensors.stream()
                    .map(tt -> tiles.getNthTileInput(tt, nTile))
                    .collect(Collectors.toList());
            final List<Tensor<T>> outputTiles = outputTensors.stream()
                    .map(tt -> tiles.getNthTileOutput(tt, nTile))
                    .collect(Collectors.toList());
            runNoTiles(inputTiles, outputTiles);
        }
    }

    protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    void runNoTiles(final List<Tensor<T>> inTensors, final List<Tensor<R>> outTensors) throws RunModelException {
        final Map<String, RandomAccessibleInterval<R>> outMap = predictForInputTensors(inTensors);
        int c = 0;
        for (Entry<String, RandomAccessibleInterval<R>> ee : outMap.entrySet()) {
            final RandomAccessibleInterval<R> rai = ee.getValue();
            try {
                outTensors.get(c).setData(rai);
                c++;
            } catch (Exception ex) {
                // Preserve original behavior.
            }
        }
    }

    private void closeShm() {
        for (SharedMemoryArray shm : inShmaList) {
            shm.close();
        }
        inShmaList.clear();
    }

    /**
     * Cleans the shared memory resources used during execution.
     *
     * @throws InterruptedException if interrupted while waiting for cleanup
     * @throws TaskException if the cleanup task fails
     */
    protected void cleanShm() throws InterruptedException, TaskException {
        closeShm();
        if (PlatformDetection.isWindows()) {
            final Task closeSHMTask = python.task(CLEAN_SHM_CODE);
            closeSHMTask.waitFor();
            if (closeSHMTask.status == TaskStatus.FAILED || closeSHMTask.status == TaskStatus.CRASHED) {
                throw new TaskException("Unable to clean/close the opened shared memory arrays", closeSHMTask);
            }
        }
    }

    protected <T extends RealType<T> & NativeType<T>>
    Map<String, RandomAccessibleInterval<T>> reconstructOutputs(final Task task) throws IOException {
        buildOutShmList(task);
        buildOutDTypesList(task);
        buildOutDimsList(task);

        final LinkedHashMap<String, RandomAccessibleInterval<T>> outs =
                new LinkedHashMap<String, RandomAccessibleInterval<T>>();
        for (int i = 0; i < this.outShmNames.size(); i++) {
            final String shmName = outShmNames.get(i);
            final String dtype = outShmDTypes.get(i);
            final long[] dims = outShmDims.get(i);
            final RandomAccessibleInterval<T> rai = reconstruct(shmName, dtype, dims);
            outs.put("output_" + i, rai);
        }
        return outs;
    }

    private void buildOutShmList(final Task task) {
        this.outShmNames = new ArrayList<String>();
        if (!(task.outputs.get(SHM_NAMES_KEY) instanceof List)) {
            throw new RuntimeException("Unexpected type for '" + SHM_NAMES_KEY + "'.");
        }
        final List<?> list = (List<?>) task.outputs.get(SHM_NAMES_KEY);
        for (Object elem : list) {
            if (!(elem instanceof String)) {
                throw new RuntimeException("Unexpected type for element of '" + SHM_NAMES_KEY + "' list.");
            }
            outShmNames.add((String) elem);
        }
    }

    private void buildOutDTypesList(final Task task) {
        this.outShmDTypes = new ArrayList<String>();
        if (!(task.outputs.get(DTYPES_KEY) instanceof List)) {
            throw new RuntimeException("Unexpected type for '" + DTYPES_KEY + "'.");
        }
        final List<?> list = (List<?>) task.outputs.get(DTYPES_KEY);
        for (Object elem : list) {
            if (!(elem instanceof String)) {
                throw new RuntimeException("Unexpected type for element of '" + DTYPES_KEY + "' list.");
            }
            outShmDTypes.add((String) elem);
        }
    }

    private void buildOutDimsList(final Task task) {
        this.outShmDims = new ArrayList<long[]>();
        if (!(task.outputs.get(DIMS_KEY) instanceof List)) {
            throw new RuntimeException("Unexpected type for '" + DIMS_KEY + "'.");
        }
        final List<?> list = (List<?>) task.outputs.get(DIMS_KEY);
        for (Object elem : list) {
            if (!(elem instanceof Object[]) && !(elem instanceof List)) {
                throw new RuntimeException("Unexpected type for element of '" + DIMS_KEY + "' list.");
            }
            if (elem instanceof Object[]) {
                final Object[] arr = (Object[]) elem;
                final long[] longArr = new long[arr.length];
                for (int i = 0; i < arr.length; i++) {
                    if (!(arr[i] instanceof Number)) {
                        throw new RuntimeException("Unexpected type for array element of '" + DIMS_KEY + "' list.");
                    }
                    longArr[i] = ((Number) arr[i]).longValue();
                }
                outShmDims.add(longArr);
            } else if (elem instanceof List) {
                @SuppressWarnings("unchecked")
                final List<Object> arr = (List<Object>) elem;
                final long[] longArr = new long[arr.size()];
                for (int i = 0; i < arr.size(); i++) {
                    if (!(arr.get(i) instanceof Number)) {
                        throw new RuntimeException("Unexpected type for array element of '" + DIMS_KEY + "' list.");
                    }
                    longArr[i] = ((Number) arr.get(i)).longValue();
                }
                outShmDims.add(longArr);
            } else {
                throw new RuntimeException("Unexpected type for element of '" + DIMS_KEY + "' list.");
            }
        }
    }

    private <T extends RealType<T> & NativeType<T>>
    RandomAccessibleInterval<T> reconstruct(final String key, final String dtype, final long[] dims) throws IOException {
        final SharedMemoryArray shm = SharedMemoryArray.readOrCreate(
                key,
                dims,
                Cast.unchecked(CommonUtils.getImgLib2DataType(dtype)),
                false,
                false
        );

        // TODO I do not understand why it complains when the types align perfectly.
        final RandomAccessibleInterval<T> rai = shm.getSharedRAI();
        final RandomAccessibleInterval<T> raiCopy = Tensor.createCopyOfRaiInWantedDataType(
                Cast.unchecked(rai),
                Util.getTypeFromInterval(Cast.unchecked(rai))
        );

        shm.close();
        return raiCopy;
    }

    /**
     * Returns the Python code that recreates the shared-memory-backed NumPy view.
     *
     * @param shma the shared memory array
     * @param varName the Python variable name
     * @return the resulting Python code
     */
    protected static String codeToConvertShmaToPython(final SharedMemoryArray shma, final String varName) {
        String code = "";
        code += "  " + varName + "_shm = shared_memory.SharedMemory(name='"
                + shma.getNameForPython() + "', size=" + shma.getSize() + ")"
                + System.lineSeparator();
        code += "  " + "created_shms.append(" + varName + "_shm)" + System.lineSeparator();

        long nElems = 1;
        for (long elem : shma.getOriginalShape()) {
            nElems *= elem;
        }

        code += "  " + varName + " = np.ndarray(" + nElems + ", dtype='"
                + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
                + "', buffer=" + varName + "_shm.buf).reshape([";
        for (int i = 0; i < shma.getOriginalShape().length; i++) {
            code += shma.getOriginalShape()[i] + ", ";
        }
        code += "])" + System.lineSeparator();
        return code;
    }

    /**
     * Checks whether the default protected PyTorch environment is installed.
     *
     * @return {@code true} if the environment appears to be fully installed
     */
    public static boolean isInstalled() {
        try {
            final PixiEnvironmentSpec spec = resolvePytorchEnv();
            return PixiEnvironmentManager.isInstalled(spec);
        } catch (BuildException e) {
            return false;
        }
    }

    /**
     * Installs the requirements for the current model instance.
     *
     * @throws InterruptedException if installation is interrupted
     * @throws BuildException if installation fails
     */
    public void installRequirements() throws InterruptedException, BuildException {
        installRequirements(null);
    }

    /**
     * Installs the requirements for the current model instance.
     *
     * @param consumer
     *     optional consumer receiving installation logs and progress
     * @throws InterruptedException if installation is interrupted
     * @throws BuildException if installation fails
     */
    public void installRequirements(final Consumer<String> consumer)
            throws InterruptedException, BuildException {
        PixiEnvironmentManager.installRequirements(environmentSpec, consumer);

        if (!isInstalled()) {
            throw new RuntimeException("Not all the required packages were installed correctly. Please try again."
                    + " If the error persists, please post an issue at: https://github.com/bioimage-io/JDLL/issues");
        }
    }

    /**
     * Installs the default requirements for the current machine without creating
     * a model instance.
     *
     * @throws InterruptedException if installation is interrupted
     * @throws BuildException if installation fails
     */
    public static void installDefaultRequirements() throws InterruptedException, BuildException {
        installDefaultRequirements(null);
    }

    /**
     * Installs the default requirements for the current machine without creating
     * a model instance.
     *
     * @param consumer
     *     optional consumer receiving installation logs and progress
     * @throws InterruptedException if installation is interrupted
     * @throws BuildException if installation fails
     */
    public static void installDefaultRequirements(final Consumer<String> consumer)
            throws InterruptedException, BuildException {
        final PixiEnvironmentSpec spec = resolvePytorchEnv();
        PixiEnvironmentManager.installRequirements(spec, consumer);

        if (!isInstalled()) {
            throw new RuntimeException("Not all the required packages were installed correctly. Please try again."
                    + " If the error persists, please post an issue at: https://github.com/bioimage-io/JDLL/issues");
        }
    }

    /**
     * Sets the directory where the Python environment will be installed.
     *
     * @param installationDir
     *     directory where the Python environment will be created
     */
    public static void setInstallationDir(final String installationDir) {
        INSTALLATION_DIR = installationDir;
    }

    /**
     * @return the directory where the Python environment will be created
     */
    public static String getInstallationDir() {
        return INSTALLATION_DIR;
    }

    /**
     * Resolves the environment specification for the current machine.
     *
     * @param installationDir
     *     the base directory where the environment should live
     * @param environmentDirectoryName
     *     the directory name to use for the environment
     * @return the resolved environment specification
     * @throws BuildException
     *     if the pixi.toml template cannot be loaded
     */
    public static PixiEnvironmentSpec resolvePytorchEnv() throws BuildException {

        final String pixiTemplate = readClasspathResourceAsString(PIXI_TEMPLATE_RESOURCE);
        final String cudaVersion = GpuCompatibility.pickCudaVersion(CUDA_COMPAT_VERSIONS);

        final String pixiTomlContent;
        final String selectedEnvironment;
        final boolean installBiapyNoDeps;

        if (cudaVersion == null) {
            pixiTomlContent = String.format(Locale.ROOT, pixiTemplate, COMMON_PYTORCH_ENV_NAME, "", "", "", "");

            if (PlatformDetection.isLinux()) {
                selectedEnvironment = "linux-x86_64-no-cuda";
                installBiapyNoDeps = false;
            } else if (PlatformDetection.isWindows()) {
                selectedEnvironment = "win-x86_64-no-cuda";
                installBiapyNoDeps = false;
            } else if (isMacArmOrRosetta() && isLegacyMacOs()) {
                selectedEnvironment = "macos-arm64-legacy";
                installBiapyNoDeps = true;
            } else if (isMacArmOrRosetta()) {
                selectedEnvironment = "macos-arm64";
                installBiapyNoDeps = false;
            } else if (PlatformDetection.isMacOS() && !isMacArmOrRosetta() && isLegacyMacOs()) {
                selectedEnvironment = "macos-x86_64-legacy";
                installBiapyNoDeps = true;
            } else {
                selectedEnvironment = "macos-x86_64";
                installBiapyNoDeps = false;
            }
        } else {
            final String compactCuda = cudaVersion.replace(".", "");
            pixiTomlContent = String.format(
                    Locale.ROOT,
                    pixiTemplate, COMMON_PYTORCH_ENV_NAME,
                    compactCuda, compactCuda, compactCuda, compactCuda
            );
            if (PlatformDetection.isLinux()) {
                selectedEnvironment = "linux-x86_64-cuda";
            } else {
                selectedEnvironment = "win-x86_64-cuda";
            }
            installBiapyNoDeps = false;
        }

        final File environmentDirectory = new File(Environments.apposeEnvsDir(), COMMON_PYTORCH_ENV_NAME);
        return new PixiEnvironmentSpec(
                selectedEnvironment,
                pixiTomlContent,
                environmentDirectory,
                installBiapyNoDeps ? Arrays.asList("biapy==3.5.10") : null
        );
    }

    private static boolean isLegacyMacOs() {
        return PlatformDetection.isMacOS() && PlatformDetection.getOSVersion().getMajor() < 14;
    }

    private static boolean isMacArmOrRosetta() {
        return PlatformDetection.isMacOS()
                && (PlatformDetection.ARCH_ARM64.equals(PlatformDetection.getArch())
                || PlatformDetection.ARCH_AARCH64.equals(PlatformDetection.getArch())
                || PlatformDetection.isUsingRosseta());
    }

    /**
     * Reads a classpath resource fully as UTF-8 text.
     *
     * @param absoluteResourcePath
     *     absolute classpath resource path
     * @return the resource contents as UTF-8 text
     * @throws BuildException
     *     if the resource cannot be found or read
     */
    private static String readClasspathResourceAsString(final String absoluteResourcePath) throws BuildException {
        Objects.requireNonNull(absoluteResourcePath, "absoluteResourcePath");

        try (InputStream is = PixiEnvironmentManager.class.getResourceAsStream(absoluteResourcePath)) {
            if (is == null) {
                throw new BuildException("Required resource not found on classpath: " + absoluteResourcePath);
            }
            return new String(readAllBytesJava8(is), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new BuildException("Failed to read resource: " + absoluteResourcePath, e);
        }
    }

    /**
     * Java 8-compatible {@link InputStream} to byte array helper.
     *
     * @param is
     *     input stream to read
     * @return all bytes from the input stream
     * @throws IOException
     *     if reading fails
     */
    private static byte[] readAllBytesJava8(final InputStream is) throws IOException {
        final ByteArrayOutputStream baos = new ByteArrayOutputStream();
        final byte[] buffer = new byte[8192];
        int len;
        while ((len = is.read(buffer)) != -1) {
            baos.write(buffer, 0, len);
        }
        return baos.toByteArray();
    }

    private static void ensureTaskSucceeded(final Task task) {
        if (task.status == TaskStatus.COMPLETE) {
            return;
        }
        if (task.status == TaskStatus.CANCELED) {
            throw new RuntimeException("Task canceled");
        }
        if (task.status == TaskStatus.FAILED || task.status == TaskStatus.CRASHED) {
            throw new RuntimeException(task.error);
        }
        throw new RuntimeException("Unexpected task status: " + task.status);
    }
}