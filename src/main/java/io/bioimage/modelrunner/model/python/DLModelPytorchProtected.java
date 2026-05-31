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
import java.util.HashMap;
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
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.InferenceProgress;
import io.bioimage.modelrunner.model.java.DLModelJava.TilingConsumer;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.model.tiling.merger.Merger;
import io.bioimage.modelrunner.model.tiling.merger.NoTileMerger;
import io.bioimage.modelrunner.system.GpuCompatibility;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

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

    protected Service python;

    private volatile Task currentTask;

    private volatile boolean inferenceCancellationRequested;

    protected PixiEnvironmentSpec environmentSpec;

    protected List<SharedMemoryArray> inShmaList = new ArrayList<SharedMemoryArray>();

    private List<String> outShmNames;

    private List<String> outShmDTypes;

    private List<String> outAxes;

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
     * Consumer used to inform the current tile being processed and in how many
     * tiles the input images are going to be separated.
     */
    protected TilingConsumer tileCounter;

    /**
     * Consumer used to report structured inference progress events.
     */
    protected Consumer<InferenceProgress> inferenceProgressConsumer;
    
    /**
     * Device where the model will run
     */
	protected final String device;

    /**
     * Maximum number of pixels copied to shared memory for one image tensor.
     */
    protected long maxSharedMemoryPixelCount = 512L * 512L * 32L;

    /**
     * Scale metadata from the most recent input transfer.
     */
    private List<InputTransferScale> inputTransferScales = new ArrayList<InputTransferScale>();

    /**
     * Default environment directory name.
     */
    public static final String COMMON_PYTORCH_ENV_NAME = "biapy";

    private static final int MAX_TRANSIENT_TASK_RETRIES = 3;

    private static final String PIXI_TEMPLATE_RESOURCE = "tomls/biapy-pixi-%s.toml";
    
    private static final HashMap<String, String> TOML_SUFFIX = new HashMap<String, String>();
    static {
    	TOML_SUFFIX.put(PlatformDetection.OS_LINUX + PlatformDetection.ARCH_X86_64 + "false", "lin-x86");
    	TOML_SUFFIX.put(PlatformDetection.OS_WINDOWS + PlatformDetection.ARCH_X86_64 + "false", "win-x86");
    	TOML_SUFFIX.put(PlatformDetection.OS_OSX + PlatformDetection.ARCH_X86_64 + "false", "mac-x86");
    	TOML_SUFFIX.put(PlatformDetection.OS_OSX + PlatformDetection.ARCH_AARCH64 + "true", "mac-x86");
    	TOML_SUFFIX.put(PlatformDetection.OS_LINUX + PlatformDetection.ARCH_AARCH64 + "false", "mac-arm");
    }

    private static final List<String> CUDA_COMPAT_VERSIONS = new ArrayList<>(Arrays.asList("12.4", "12.1", "11.8"));

    protected static final boolean IS_ARM = PlatformDetection.isMacOS()
            && (PlatformDetection.getArch().equals(PlatformDetection.ARCH_ARM64)
            || PlatformDetection.isUsingRosseta());

    protected static String INSTALLATION_DIR = Environments.apposeEnvsDir();

    protected static final String MODEL_VAR_NAME = "model_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
            + "if 'sys' not in globals().keys():" + System.lineSeparator()
            + "  import sys" + System.lineSeparator()
            + "  task.export(sys=sys)" + System.lineSeparator()
            + "if 'np' not in globals().keys():" + System.lineSeparator()
            + "  import numpy as np" + System.lineSeparator()
            + "  task.export(np=np)" + System.lineSeparator()
            + "if 'os' not in globals().keys():" + System.lineSeparator()
            + "  import os" + System.lineSeparator()
            + "  task.export(os=os)" + System.lineSeparator()
            + "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
            + "  from multiprocessing import shared_memory" + System.lineSeparator()
            + "  task.export(shared_memory=shared_memory)" + System.lineSeparator()
            + "%s" + System.lineSeparator()
            + "%s" + System.lineSeparator()
            + "if '%s' not in globals().keys():" + System.lineSeparator()
            + "  task.export(%s=%s)" + System.lineSeparator();

    protected static final String OUTPUT_LIST_KEY = "out_list" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String SHMS_KEY = "shms_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String SHM_NAMES_KEY = "shm_names_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String DTYPES_KEY = "dtypes_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String DIMS_KEY = "dims_" + UUID.randomUUID().toString().replace("-", "_");

    protected static final String RECOVER_OUTPUTS_CODE = ""
			+ SHMS_KEY + " = []" + System.lineSeparator()
			+ "created_shms = []" + System.lineSeparator()
			+ "task.export(" + SHMS_KEY + "=" + SHMS_KEY + ")" + System.lineSeparator()
			+ "task.export(created_shms=created_shms)" + System.lineSeparator()
            + "def handle_output(outs_i, shms_key, shms_names, dtypes, dims):" + System.lineSeparator()
            + "    if type(outs_i) == np.ndarray:" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=outs_i.nbytes)" + System.lineSeparator()
            + "      sh_np_array = np.ndarray(outs_i.shape, dtype=outs_i.dtype, buffer=shm.buf)" + System.lineSeparator()
            + "      np.copyto(sh_np_array, outs_i)" + System.lineSeparator()
            + "      shms_key.append(shm)" + System.lineSeparator()
            + "      shms_names.append(shm.name)" + System.lineSeparator()
            + "      dtypes.append(str(outs_i.dtype))" + System.lineSeparator()
            + "      dims.append(outs_i.shape)" + System.lineSeparator()
            + "    elif str(type(outs_i)) == \"<class 'torch.Tensor'>\":" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=outs_i.numel() * outs_i.element_size())" + System.lineSeparator()
            + "      np_arr = np.ndarray(outs_i.shape, dtype=str(outs_i.dtype).split('.')[-1], buffer=shm.buf)" + System.lineSeparator()
            + "      tensor_np_view = torch.from_numpy(np_arr)" + System.lineSeparator()
            + "      tensor_np_view.copy_(outs_i)" + System.lineSeparator()
            + "      shms_key.append(shm)" + System.lineSeparator()
            + "      shms_names.append(shm.name)" + System.lineSeparator()
            + "      dtypes.append(str(outs_i.dtype).split('.')[-1])" + System.lineSeparator()
            + "      dims.append(outs_i.shape)" + System.lineSeparator()
            + "    elif type(outs_i) == int:" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=8)" + System.lineSeparator()
            + "      shm.buf[:8] = outs_i.to_bytes(8, byteorder='little', signed=True)" + System.lineSeparator()
            + "      shms_key.append(shm)" + System.lineSeparator()
            + "      shms_names.append(shm.name)" + System.lineSeparator()
            + "      dtypes.append('int64')" + System.lineSeparator()
            + "      dims.append((1))" + System.lineSeparator()
            + "    elif type(outs_i) == float:" + System.lineSeparator()
            + "      shm = shared_memory.SharedMemory(create=True, size=8)" + System.lineSeparator()
            + "      shm.buf[:8] = outs_i.to_bytes(8, byteorder='little', signed=True)" + System.lineSeparator()
            + "      shms_key.append(shm)" + System.lineSeparator()
            + "      shms_names.append(shm.name)" + System.lineSeparator()
            + "      dtypes.append('float64')" + System.lineSeparator()
            + "      dims.append((1))" + System.lineSeparator()
            + "    elif type(outs_i) == tuple or type(outs_i) == list:" + System.lineSeparator()
            + "      handle_output_list(outs_i, shms_key, shms_names, dtypes, dims)" + System.lineSeparator()
            + "    else:" + System.lineSeparator()
            + "      task.update('output type : ' + str(type(outs_i)) + ' not supported. "
            + "Only supported output types are: np.ndarray, torch.tensor, int and float, "
            + "or a list or tuple of any of those.')" + System.lineSeparator()
            + System.lineSeparator()
            + System.lineSeparator()
            + "def handle_output_list(out_list, shms_key, shms_names, dtypes, dims):" + System.lineSeparator()
            + "  if type(out_list) == tuple or type(out_list) == list:" + System.lineSeparator()
            + "    for outs_i in out_list:" + System.lineSeparator()
            + "      handle_output(outs_i, shms_key, shms_names, dtypes, dims)" + System.lineSeparator()
            + "  else:" + System.lineSeparator()
            + "    handle_output(out_list, shms_key, shms_names, dtypes, dims)" + System.lineSeparator()
            + System.lineSeparator()
            + System.lineSeparator()
            + "task.export(handle_output_list=handle_output_list)" + System.lineSeparator()
            + "task.export(handle_output=handle_output)" + System.lineSeparator()
            + System.lineSeparator()
            + System.lineSeparator();

    private static final String CLEAN_SHM_CODE_THREAD_DEATH = ""
            + "for s in " + SHMS_KEY + ":" + System.lineSeparator()
            + "    s.close()" + System.lineSeparator()
            + "    try:" + System.lineSeparator()
            + "        s.unlink()" + System.lineSeparator()
            + "    except FileNotFoundError:" + System.lineSeparator()
            + "        pass" + System.lineSeparator()
            + SHMS_KEY + ".clear()" + System.lineSeparator();

    private static final String CLEAN_SHM_CODE_WINDOWS = ""
            + "for s in " + SHMS_KEY + ":" + System.lineSeparator()
            + "    s.close()" + System.lineSeparator()
            + "    try:" + System.lineSeparator()
            + "        s.unlink()" + System.lineSeparator()
            + "    except FileNotFoundError:" + System.lineSeparator()
            + "        pass" + System.lineSeparator()
            + SHMS_KEY + ".clear()" + System.lineSeparator()
            + "for s in created_shms:" + System.lineSeparator()
            + "    s.close()" + System.lineSeparator()
            + "    try:" + System.lineSeparator()
            + "        s.unlink()" + System.lineSeparator()
            + "    except FileNotFoundError:" + System.lineSeparator()
            + "        pass" + System.lineSeparator()
            + "created_shms.clear()" + System.lineSeparator();

    private static final String CLEAN_SHM_CODE_POSIX = ""
            + "for s in list(" + SHMS_KEY + "):" + System.lineSeparator()
            + "    try:" + System.lineSeparator()
            + "        s.close()" + System.lineSeparator()
            + "    except Exception:" + System.lineSeparator()
            + "        pass" + System.lineSeparator()
            + SHMS_KEY + ".clear()" + System.lineSeparator()
            + "for s in list(created_shms):" + System.lineSeparator()
            + "    try:" + System.lineSeparator()
            + "        s.close()" + System.lineSeparator()
            + "    except Exception:" + System.lineSeparator()
            + "        pass" + System.lineSeparator()
            + "created_shms.clear()" + System.lineSeparator();

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
            final Map<String, Object> kwargs) {
        this(modelFile, callable, importModule, weightsPath, kwargs, false, null);
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
            final boolean customJDLL, 
            String device) {

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
        
        this.tileCounter = new TilingConsumer();
		String normalizedDevice = device == null ? "cpu" : device.trim().toLowerCase(Locale.ROOT);
		if (!"cuda".equals(normalizedDevice) && !"mps".equals(normalizedDevice)) {
			normalizedDevice = "cpu";
		}
		this.device = normalizedDevice;
    }

    /**
     * Creates the Python service used to run the model.
     *
     * @throws BuildException if the environment cannot be opened
     */
    protected void createPythonService() throws LoadModelException {
        final Environment env;

        try {
	        env = Appose.pixi()
	                .environment(environmentSpec.getSelectedEnvironment())
	                .wrap(environmentSpec.getEnvironmentDirectory());
	
	        python = env.python();
	        python.debug(System.err::println);
        } catch (Exception ex) {
        	throw new LoadModelException(Messages.stackTrace(ex));
        }
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
     * Sets a consumer used to track tile execution progress.
     *
     * @param tileCounter
     *     consumer used to track tile inference
     */
    public void setTilingCounter(final TilingConsumer tileCounter) {
        this.tileCounter = tileCounter;
    }
    
    /**
     * 
     * @return an object that can be used to track the progress processing tiles
     */
    public TilingConsumer getTilingCounter() {
    	return this.tileCounter;
    }

    /**
     * Sets a consumer used to receive structured inference progress events.
     *
     * @param consumer
     *     consumer called as inference advances
     */
    public void setInferenceProgressConsumer(final Consumer<InferenceProgress> consumer) {
        this.inferenceProgressConsumer = consumer;
    }

    /**
     * @return the configured structured inference progress consumer, or null
     */
    public Consumer<InferenceProgress> getInferenceProgressConsumer() {
        return inferenceProgressConsumer;
    }

    /**
     * @return maximum pixel count copied to shared memory for one image tensor
     */
    public long getMaxSharedMemoryPixelCount() {
        return maxSharedMemoryPixelCount;
    }

    /**
     * Sets the maximum pixel count copied to shared memory for one image tensor.
     *
     * @param maxSharedMemoryPixelCount
     *     positive maximum pixel count
     */
    public void setMaxSharedMemoryPixelCount(final long maxSharedMemoryPixelCount) {
        if (maxSharedMemoryPixelCount <= 0) {
            throw new IllegalArgumentException("Maximum shared-memory image pixel count must be positive.");
        }
        this.maxSharedMemoryPixelCount = maxSharedMemoryPixelCount;
    }

    /**
     * Emits a structured inference progress event. Progress reporting must not
     * affect inference execution.
     *
     * @param progress
     *     progress event
     */
    protected void emitProgress(final InferenceProgress progress) {
        if (inferenceProgressConsumer == null || progress == null) {
            return;
        }
        try {
            inferenceProgressConsumer.accept(progress);
        } catch (RuntimeException e) {
            // Progress listeners are observational and should not break model execution.
        }
    }

    /**
     * Loads the model into the Python service.
     *
     * @throws LoadModelException if the model cannot be loaded
     */
    @Override
    public void loadModel() throws LoadModelException {
    	if (python == null) {
            createPythonService();
            python.init("import numpy as np" + System.lineSeparator());
    	}
        if (loaded) {
            return;
        }
        if (closed) {
            throw new RuntimeException("Cannot load model after it has been closed");
        }

        try {
            emitProgress(InferenceProgress.modelLoading(weightsPath));
            String code = buildModelCode();
            code += RECOVER_OUTPUTS_CODE;
            final Task task = python.task(code);
            task.waitFor();
            ensureTaskSucceeded(task);
            emitProgress(InferenceProgress.modelLoaded(weightsPath));
        } catch (IOException | InterruptedException | TaskException e) {
        	python.close();
        	python = null;
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
                + "if 'torch' not in globals().keys():" + System.lineSeparator()
                + "  import torch" + System.lineSeparator()
                + "  task.export(torch=torch)" + System.lineSeparator()
                + "_jdll_requested_device = '" + device + "'" + System.lineSeparator()
                + "device = torch.device('cpu')" + System.lineSeparator()
                + "try:" + System.lineSeparator()
                + "  if _jdll_requested_device == 'cuda' and torch.cuda.is_available():" + System.lineSeparator()
                + "    device = torch.device('cuda')" + System.lineSeparator()
                + "  elif _jdll_requested_device == 'mps':" + System.lineSeparator()
                + "    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available():" + System.lineSeparator()
                + "      device = torch.device('mps')" + System.lineSeparator()
                + "except Exception:" + System.lineSeparator()
                + "  device = torch.device('cpu')" + System.lineSeparator()
                + "task.export(device=device)" + System.lineSeparator();

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
        code += "  device = torch.device('cpu')" + System.lineSeparator();
        code += "  task.export(device=device)" + System.lineSeparator();
        code += MODEL_VAR_NAME + ".to(device)" + System.lineSeparator();
        code += "try:" + System.lineSeparator()
                + "  " + MODEL_VAR_NAME + ".load_state_dict("
                + "torch.load(r'" + this.weightsPath + "', map_location=" + MODEL_VAR_NAME + ".device))"
                + System.lineSeparator()
                + "except:" + System.lineSeparator()
                + "  " + MODEL_VAR_NAME + ".load_state_dict("
                + "torch.load(r'" + this.weightsPath + "', map_location=device))"
                + System.lineSeparator();
        code += "task.export(" + MODEL_VAR_NAME + "=" + MODEL_VAR_NAME + ")" + System.lineSeparator();
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
        cancelCurrentInference();
        if (python != null && python.isAlive()) {
            python.close();
        }
        loaded = false;
        closed = true;
    }

    public void cancelCurrentInference() {
        inferenceCancellationRequested = true;
        Task task = currentTask;
        if (task != null) {
            task.cancel();
        }
    }

    private void throwIfInferenceCancelled() throws RunModelException {
        if (inferenceCancellationRequested || Thread.currentThread().isInterrupted()) {
            throw new RunModelException("Inference cancelled.");
        }
    }

    protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    Map<String, RandomAccessibleInterval<R>> executeCode(final String code) throws RunModelException {
        Throwable lastFailure = null;
        for (int attempt = 0; attempt <= MAX_TRANSIENT_TASK_RETRIES; attempt ++) {
        	Task task = null;
            try {
                throwIfInferenceCancelled();
                task = python.task(code);
                currentTask = task;
                python.debug((str) -> {});
                task.waitFor();
                throwIfInferenceCancelled();
                ensureTaskSucceeded(task);
                loaded = true;
                Map<String, RandomAccessibleInterval<R>> outMap = reconstructOutputs(task);
                try {
                	cleanShm();
                } catch (TaskException e) {
                	if (isApposeThreadDeath(e)) {
                        emitProgress(InferenceProgress.taskRetry("Appose thread death during shared-memory cleanup after inference."));
                	} else {
                        throw new RunModelException("Exception happened while cleaning shared memory: " +  Messages.stackTrace(lastFailure));
                	}
                }
                return outMap;
            } catch (InterruptedException e) {
                if (task != null) {
                    task.cancel();
                }
                Thread.currentThread().interrupt();
                cleanShmAfterFailure(e);
                lastFailure = e;
                break;
            } catch (TaskException | IOException e) {
                lastFailure = e;
                if (isApposeThreadDeath(e) && attempt < MAX_TRANSIENT_TASK_RETRIES) {
                    emitProgress(InferenceProgress.taskRetry("Appose thread death during inference; retrying task "
                            + (attempt + 1) + "/" + MAX_TRANSIENT_TASK_RETRIES + "."));
                    try {
						this.threadDeathCleanUp();
					} catch (InterruptedException | TaskException e1) {
				        throw new RunModelException(Messages.stackTrace(e1));
					}
                    continue;
                }
                cleanShmAfterFailure(e);
                break;
            } finally {
                if (currentTask == task) {
                    currentTask = null;
                }
            }
        }
        throw new RunModelException(lastFailure == null
                ? "Model execution failed."
                : Messages.stackTrace(lastFailure));
    }

    private void cleanShmAfterFailure(final Throwable original) throws RunModelException {
        try {
            cleanShm();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RunModelException(Messages.stackTrace(original) + System.lineSeparator()
                    + "Additionally interrupted while cleaning shared memory:"
                    + System.lineSeparator() + Messages.stackTrace(e));
        } catch (TaskException e) {
            throw new RunModelException(Messages.stackTrace(original) + System.lineSeparator()
                    + "Additionally failed to clean shared memory:"
                    + System.lineSeparator() + Messages.stackTrace(e));
        }
    }

    private static boolean isApposeThreadDeath(final Throwable failure) {
        String trace = Messages.stackTrace(failure);
        return trace != null && trace.toLowerCase(Locale.ROOT).startsWith("org.apposed.appose.TaskException: task failed: thread death".toLowerCase());
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
    List<Tensor<R>> inference(final Tensor<T>... inputs) throws RunModelException {
        inferenceCancellationRequested = false;
    	List<List<Tensor<T>>> inputBatches = new ArrayList<List<Tensor<T>>>();
    	for (Tensor<T> inp : inputs) {
    		List<Tensor<T>> inpList = new ArrayList<Tensor<T>>();
    		inpList.add(inp);
    		inputBatches.add(inpList);
    	}
    	List<List<Tensor<R>>> batchedOuts = backboneBatchInference(inputBatches);
    	List<Tensor<R>> singleOutput = new ArrayList<>();
    	for (List<Tensor<R>>batched : batchedOuts) {
    		singleOutput.add(batched.get(0));
    	}
    	return singleOutput;
    }

    protected String getOutputTensorAxes(int outputCount) {
		return "bcyx";
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
    List<List<Tensor<R>>> inferenceBatch(final List<Tensor<T>>... batchedInputs)
            throws RunModelException {
        inferenceCancellationRequested = false;
    	return backboneBatchInference(Arrays.asList(batchedInputs));
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<List<Tensor<R>>> backboneBatchInference(final List<List<Tensor<T>>> batchedInputs) throws RunModelException {

        if (!loaded) {
            throw new RuntimeException("Please load the model first.");
        }
        int batchInd = 0;
        int imsInBatch = batchedInputs.get(0).size();
        for (List<Tensor<T>> batch : batchedInputs) {
        	if (batch.size() != imsInBatch)
        		throw new IllegalArgumentException("All batches must have the same number of tensors. Batch of input 0 has "
        				+ imsInBatch +  " and batch of input " + batchInd + " has " + batch.size());
        	String name = batch.get(0).getName();
        	for (int i = 1; i < batch.size(); i ++) {
        		String nName = batch.get(i).getName();
        		if (nName.equals(name))
        			throw new IllegalArgumentException("All tensors of a batch should be named equally. For batch " + batchInd
        					+ "at least two different names where found: " + name + " (pos 0) and " + nName + " (pos" + i + ")");
        	}
        	batchInd ++;
        }
        
        List<List<Tensor<R>>> outputs = new ArrayList<List<Tensor<R>>>();
        for (int i = 0; i < batchedInputs.get(0).size(); i ++) {
            throwIfInferenceCancelled();
        	List<Tensor<T>> inputs = new ArrayList<Tensor<T>>();
        	for (int j = 0; j < batchedInputs.size(); j ++) {
        		inputs.add(batchedInputs.get(j).get(i));
        	}
        	List<Tensor<R>> aa = backboneSingleInference(inputs);
        	for (int k = 0; k < aa.size(); k ++) {
        		if (outputs.size() < k +1)
        			outputs.add(new ArrayList<Tensor<R>>());
        		outputs.get(k).add(aa.get(k));
        	}
        }
        return outputs;
    }
    
    protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    Merger<Tensor<T>, Tensor<R>> getTileMaker(final List<Tensor<T>> inputs) {
        Merger<Tensor<T>, Tensor<R>> merger = new NoTileMerger<T, R>();
        merger.configure(inputs);
        return merger;
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> backboneSingleInference(final List<Tensor<T>> inputs) throws RunModelException {

        throwIfInferenceCancelled();
        Merger<Tensor<T>, Tensor<R>> merger = getTileMaker(inputs);
        if (tileCounter == null) {
            tileCounter = new TilingConsumer();
        }

        int nPatches = merger.getNPatches();
        tileCounter.acceptTotal((long) nPatches);
        emitProgress(InferenceProgress.inferenceStart(nPatches));
        for (int i = 0; i < nPatches; i ++) {
            throwIfInferenceCancelled();
            emitProgress(InferenceProgress.patchStart(i + 1, nPatches));
            List<Tensor<R>> tiledOutputs = backboneSingleInferenceTile(merger.get(i));
            throwIfInferenceCancelled();
            merger.digest(i, tiledOutputs);
            tileCounter.acceptProgress((long) (i + 1));
            emitProgress(InferenceProgress.patchEnd(i + 1, nPatches));
        }
        throwIfInferenceCancelled();
        emitProgress(InferenceProgress.mergeStart());
        List<Tensor<R>> reconstructed = merger.getReconstructed();
        throwIfInferenceCancelled();
        emitProgress(InferenceProgress.inferenceEnd());
        return reconstructed;
    }
    
    private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
    List<Tensor<R>> backboneSingleInferenceTile(final List<Tensor<T>> inputs) throws RunModelException {

        throwIfInferenceCancelled();
        final List<String> names = IntStream.range(0, inputs.size())
                .mapToObj(i -> "var_" + UUID.randomUUID().toString().replace("-", "_"))
                .collect(Collectors.toList());

        final String code = createInputsCode(inputs, names);
        final Map<String, RandomAccessibleInterval<R>> map = executeCode(code);
        throwIfInferenceCancelled();
        outAxes = getOutputAxes(map.size());
        List<Tensor<R>> outTensors = new ArrayList<Tensor<R>>();
        int i = 0;
        for (Entry<String, RandomAccessibleInterval<R>> ee : map.entrySet()) {
            outTensors.add(Tensor.build("output_" + i, outAxes.get(i),
                    restoreOutputScale(ee.getValue(), outAxes.get(i))));
        	i ++;
        }
    	return outTensors;
    }

    protected List<String> getOutputAxes(final int outputCount) {
        final List<String> axes = new ArrayList<String>(outputCount);
        for (int i = 0; i < outputCount; i ++) {
            axes.add(getOutputTensorAxes(i));
        }
        return axes;
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
            final List<Tensor<T>> rais,
            final List<String> names) {
        String code = "created_shms.clear()" + System.lineSeparator();
        code += "task.outputs.clear()" + System.lineSeparator();
        code += SHM_NAMES_KEY + " = []" + System.lineSeparator();
        code += DTYPES_KEY + " = []" + System.lineSeparator();
        code += DIMS_KEY + " = []" + System.lineSeparator();
        final List<SharedMemoryArray> shmas = createSharedMemoryArraysForInputs(rais);
        for (int i = 0; i < rais.size(); i++) {
            final SharedMemoryArray shma = shmas.get(i);
            code += codeToConvertShmaToPython(shma, names.get(i));
            inShmaList.add(shma);
        }
        code += MODEL_VAR_NAME + ".eval()" + System.lineSeparator();
        code += "with torch.no_grad():" + System.lineSeparator();
        code += OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + "(";
        for (int i = 0; i < rais.size(); i++) {
            code += "torch.from_numpy(" + names.get(i) + ").to(device), ";
        }
        code = code.substring(0, code.length() - 2);
        code += ")" + System.lineSeparator();
        code += String.format("handle_output_list(%s, %s, %s, %s, %s)", OUTPUT_LIST_KEY,
        		SHMS_KEY, SHM_NAMES_KEY, DTYPES_KEY, DIMS_KEY)  + System.lineSeparator();
        code += taskOutputsCode();
        return code;
    }

    protected void resetInputTransferScales() {
        inputTransferScales = new ArrayList<InputTransferScale>();
    }

    protected <T extends RealType<T> & NativeType<T>>
    List<SharedMemoryArray> createSharedMemoryArraysForInputs(final List<Tensor<T>> tensors) {
        resetInputTransferScales();
        final long sharedXySubsampling = requiredXySubsampling(tensors);
        final List<SharedMemoryArray> shmas = new ArrayList<SharedMemoryArray>();
        for (Tensor<T> tensor : tensors) {
            final InputTransfer<T> transfer = prepareInputForSharedMemory(tensor, sharedXySubsampling);
            inputTransferScales.add(transfer.scale);
            shmas.add(SharedMemoryArray.createSHMAFromRAI(transfer.data, false, false));
        }
        return shmas;
    }

    private <T extends RealType<T> & NativeType<T>>
    long requiredXySubsampling(final List<Tensor<T>> tensors) {
        long sharedXySubsampling = 1L;
        for (Tensor<T> tensor : tensors) {
            sharedXySubsampling = Math.max(sharedXySubsampling, requiredXySubsampling(tensor));
        }
        return sharedXySubsampling;
    }

    private <T extends RealType<T> & NativeType<T>>
    long requiredXySubsampling(final Tensor<T> tensor) {
        final RandomAccessibleInterval<T> data = tensor.getData();
        final String axes = tensor.getAxesOrderString();
        final long[] dims = data.dimensionsAsLongArray();
        final int xAxis = axisIndex(axes, 'x');
        final int yAxis = axisIndex(axes, 'y');
        if (xAxis < 0 || yAxis < 0) {
            return 1L;
        }

        final int zAxis = axisIndex(axes, 'z');
        final double xSize = dims[xAxis];
        final double ySize = dims[yAxis];
        final double zSize = zAxis >= 0 ? dims[zAxis] : 1.0d;
        final double imageSize = xSize * ySize * zSize;
        final double maximumSize = maxSharedMemoryPixelCount;
        return Math.max(1L, (long) Math.ceil(Math.sqrt(imageSize / maximumSize)));
    }

    private <T extends RealType<T> & NativeType<T>>
    InputTransfer<T> prepareInputForSharedMemory(final Tensor<T> tensor, final long xySubsampling) {
        final RandomAccessibleInterval<T> data = tensor.getData();
        final String axes = tensor.getAxesOrderString();
        final long[] dims = data.dimensionsAsLongArray();
        final int xAxis = axisIndex(axes, 'x');
        final int yAxis = axisIndex(axes, 'y');
        if (xAxis < 0 || yAxis < 0) {
            return new InputTransfer<T>(data, InputTransferScale.noImage(axes, dims));
        }

        if (xySubsampling <= 1L) {
            return new InputTransfer<T>(data, InputTransferScale.identity(axes, dims, xAxis, yAxis));
        }

        final long[] subsampling = new long[dims.length];
        Arrays.fill(subsampling, 1L);
        subsampling[xAxis] = xySubsampling;
        subsampling[yAxis] = xySubsampling;
        final RandomAccessibleInterval<T> subsampled = Views.subsample(data, subsampling);
        return new InputTransfer<T>(subsampled, InputTransferScale.scaled(axes, dims,
                subsampled.dimensionsAsLongArray(), xAxis, yAxis, xySubsampling));
    }

    private <R extends RealType<R> & NativeType<R>>
    RandomAccessibleInterval<R> restoreOutputScale(final RandomAccessibleInterval<R> output, final String axes) {
        final InputTransferScale scale = referenceInputTransferScale();
        if (!scale.isScaled() || output == null || axes == null) {
            return output;
        }
        if ("bic".equals(axes)) {
            return restoreBicOutputScale(output, axes, scale);
        }
        if (axisIndex(axes, 'x') >= 0 && axisIndex(axes, 'y') >= 0) {
            return restoreImageOutputScale(output, axes, scale);
        }
        return output;
    }

    private <R extends RealType<R> & NativeType<R>>
    RandomAccessibleInterval<R> restoreImageOutputScale(final RandomAccessibleInterval<R> output,
            final String axes, final InputTransferScale scale) {
        final int xAxis = axisIndex(axes, 'x');
        final int yAxis = axisIndex(axes, 'y');
        final long[] sourceDims = output.dimensionsAsLongArray();
        if (xAxis < 0 || yAxis < 0
                || sourceDims[xAxis] != scale.scaledX()
                || sourceDims[yAxis] != scale.scaledY()) {
            return output;
        }

        final long[] targetDims = sourceDims.clone();
        targetDims[xAxis] = scale.originalX();
        targetDims[yAxis] = scale.originalY();
        final Img<R> restored = new ArrayImgFactory<R>(Util.getTypeFromInterval(output)).create(targetDims);
        final Cursor<R> cursor = restored.localizingCursor();
        final RandomAccess<R> sourceAccess = output.randomAccess();
        final long[] targetPosition = new long[targetDims.length];
        final long[] sourcePosition = new long[sourceDims.length];
        while (cursor.hasNext()) {
            cursor.fwd();
            cursor.localize(targetPosition);
            System.arraycopy(targetPosition, 0, sourcePosition, 0, sourcePosition.length);
            sourcePosition[xAxis] = Math.min(sourceDims[xAxis] - 1L,
                    (long) Math.floor(targetPosition[xAxis] * scale.scaledX() / (double) scale.originalX()));
            sourcePosition[yAxis] = Math.min(sourceDims[yAxis] - 1L,
                    (long) Math.floor(targetPosition[yAxis] * scale.scaledY() / (double) scale.originalY()));
            sourceAccess.setPosition(sourcePosition);
            cursor.get().set(sourceAccess.get());
        }
        return restored;
    }

    private <R extends RealType<R> & NativeType<R>>
    RandomAccessibleInterval<R> restoreBicOutputScale(final RandomAccessibleInterval<R> output,
            final String axes, final InputTransferScale scale) {
        final int cAxis = axisIndex(axes, 'c');
        final long[] dims = output.dimensionsAsLongArray();
        if (cAxis < 0 || dims[cAxis] < 4 || !bicCoordinatesFitScaledImage(output, cAxis, scale)) {
            return output;
        }

        final Img<R> restored = new ArrayImgFactory<R>(Util.getTypeFromInterval(output)).create(dims);
        final Cursor<R> sourceCursor = output.localizingCursor();
        final RandomAccess<R> targetAccess = restored.randomAccess();
        final long[] position = new long[dims.length];
        while (sourceCursor.hasNext()) {
            sourceCursor.fwd();
            sourceCursor.localize(position);
            targetAccess.setPosition(position);
            final long column = position[cAxis];
            final double value = sourceCursor.get().getRealDouble();
            if (column == 0L || column == 2L) {
                targetAccess.get().setReal(value * scale.xRatio());
            } else if (column == 1L || column == 3L) {
                targetAccess.get().setReal(value * scale.yRatio());
            } else {
                targetAccess.get().set(sourceCursor.get());
            }
        }
        return restored;
    }

    private <R extends RealType<R> & NativeType<R>>
    boolean bicCoordinatesFitScaledImage(final RandomAccessibleInterval<R> output,
            final int cAxis, final InputTransferScale scale) {
        double maxX = Double.NEGATIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        double maxCoordinate = Double.NEGATIVE_INFINITY;
        final Cursor<R> cursor = output.localizingCursor();
        final long[] position = new long[output.numDimensions()];
        while (cursor.hasNext()) {
            cursor.fwd();
            cursor.localize(position);
            final long column = position[cAxis];
            if (column < 0L || column > 3L) {
                continue;
            }
            final double value = cursor.get().getRealDouble();
            maxCoordinate = Math.max(maxCoordinate, value);
            if (column == 0L || column == 2L) {
                maxX = Math.max(maxX, value);
            } else {
                maxY = Math.max(maxY, value);
            }
        }
        if (maxCoordinate <= 1.0d) {
            return false;
        }
        return maxX <= scale.scaledX() + 1.0d && maxY <= scale.scaledY() + 1.0d;
    }

    private InputTransferScale referenceInputTransferScale() {
        for (InputTransferScale scale : inputTransferScales) {
            if (scale.hasXY()) {
                return scale;
            }
        }
        return InputTransferScale.noImage("", new long[0]);
    }

    private static int axisIndex(final String axes, final char axis) {
        return axes == null ? -1 : axes.toLowerCase(Locale.ROOT).indexOf(Character.toLowerCase(axis));
    }

    /**
     * Returns the code that copies the Python task outputs into Appose task outputs.
     *
     * @return the resulting Python code
     */
    protected String taskOutputsCode() {
        return ""
                + "task.outputs['" + SHM_NAMES_KEY + "'] = list(" + SHM_NAMES_KEY + ")" + System.lineSeparator()
                + "task.outputs['" + DTYPES_KEY + "'] = list(" + DTYPES_KEY + ")" + System.lineSeparator()
                + "task.outputs['" + DIMS_KEY + "'] = list(" + DIMS_KEY + ")" + System.lineSeparator();
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

    private void closeShm() {
        for (SharedMemoryArray shm : inShmaList) {
            shm.close();
        }
        inShmaList.clear();
    }

    private static final class InputTransfer<T extends RealType<T> & NativeType<T>> {
        private final RandomAccessibleInterval<T> data;
        private final InputTransferScale scale;

        private InputTransfer(final RandomAccessibleInterval<T> data, final InputTransferScale scale) {
            this.data = data;
            this.scale = scale;
        }
    }

    private static final class InputTransferScale {
        private final String axes;
        private final long[] originalDims;
        private final long[] scaledDims;
        private final int xAxis;
        private final int yAxis;
        private final long xySubsampling;
        private final boolean hasXY;

        private InputTransferScale(final String axes, final long[] originalDims, final long[] scaledDims,
                final int xAxis, final int yAxis, final long xySubsampling, final boolean hasXY) {
            this.axes = axes;
            this.originalDims = originalDims == null ? new long[0] : originalDims.clone();
            this.scaledDims = scaledDims == null ? new long[0] : scaledDims.clone();
            this.xAxis = xAxis;
            this.yAxis = yAxis;
            this.xySubsampling = xySubsampling;
            this.hasXY = hasXY;
        }

        private static InputTransferScale noImage(final String axes, final long[] dims) {
            return new InputTransferScale(axes, dims, dims, -1, -1, 1L, false);
        }

        private static InputTransferScale identity(final String axes, final long[] dims,
                final int xAxis, final int yAxis) {
            return new InputTransferScale(axes, dims, dims, xAxis, yAxis, 1L, true);
        }

        private static InputTransferScale scaled(final String axes, final long[] originalDims,
                final long[] scaledDims, final int xAxis, final int yAxis, final long xySubsampling) {
            return new InputTransferScale(axes, originalDims, scaledDims, xAxis, yAxis, xySubsampling, true);
        }

        private boolean hasXY() {
            return hasXY;
        }

        private boolean isScaled() {
            return hasXY && xySubsampling > 1L;
        }

        private long originalX() {
            return originalDims[xAxis];
        }

        private long originalY() {
            return originalDims[yAxis];
        }

        private long scaledX() {
            return scaledDims[xAxis];
        }

        private long scaledY() {
            return scaledDims[yAxis];
        }

        private double xRatio() {
            return originalX() / (double) scaledX();
        }

        private double yRatio() {
            return originalY() / (double) scaledY();
        }
    }

    /**
     * Cleans the shared memory resources used during execution.
     *
     * @throws InterruptedException if interrupted while waiting for cleanup
     * @throws TaskException if the cleanup task fails
     */
    protected void cleanShm() throws InterruptedException, TaskException {
        closeShm();
        final String cleanupCode = PlatformDetection.isWindows()
                ? CLEAN_SHM_CODE_WINDOWS
                : CLEAN_SHM_CODE_POSIX;
        final Task closeSHMTask = python.task(cleanupCode);
        closeSHMTask.waitFor();
        if (closeSHMTask.status == TaskStatus.FAILED || closeSHMTask.status == TaskStatus.CRASHED) {
            throw new TaskException("Unable to clean/close the opened shared memory arrays", closeSHMTask);
        }
    }

    /**
     * Cleans the shared memory resources that might have left open during thread death.
     *
     * @throws InterruptedException if interrupted while waiting for cleanup
     * @throws TaskException if the cleanup task fails
     */
    protected void threadDeathCleanUp() throws InterruptedException, TaskException {
        final Task closeSHMTask = python.task(CLEAN_SHM_CODE_THREAD_DEATH);
        closeSHMTask.waitFor();
        if (closeSHMTask.status == TaskStatus.FAILED || closeSHMTask.status == TaskStatus.CRASHED) {
            throw new TaskException("Unable to clean/close the opened shared memory arrays", closeSHMTask);
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
        code += varName + "_shm = shared_memory.SharedMemory(name='"
                + shma.getNameForPython() + "', size=" + shma.getSize() + ")"
                + System.lineSeparator();
        code += "created_shms.append(" + varName + "_shm)" + System.lineSeparator();

        long nElems = 1;
        for (long elem : shma.getOriginalShape()) {
            nElems *= elem;
        }

        code += varName + " = np.ndarray(" + nElems + ", dtype='"
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
        } catch (Exception e) {
            return false;
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
     */
    public static PixiEnvironmentSpec resolvePytorchEnv() {
    	String suffix = TOML_SUFFIX.get(PlatformDetection.getOs() + PlatformDetection.getArch() + PlatformDetection.isUsingRosseta());
        final String pixiTemplate = readClasspathResourceAsString(String.format(PIXI_TEMPLATE_RESOURCE, suffix));
        final String cudaVersion = GpuCompatibility.pickCudaVersion(CUDA_COMPAT_VERSIONS);

        final String pixiTomlContent;
        final String selectedEnvironment;
        final boolean installBiapyNoDeps;

        if (cudaVersion == null) {
        	if (!PlatformDetection.isMacOS())
        		pixiTomlContent = String.format(Locale.ROOT, pixiTemplate, COMMON_PYTORCH_ENV_NAME, "", "");
        	else
        		pixiTomlContent = String.format(Locale.ROOT, pixiTemplate, COMMON_PYTORCH_ENV_NAME);

            if (PlatformDetection.isLinux()) {
                selectedEnvironment = "linux-x86-64-no-cuda";
                installBiapyNoDeps = false;
            } else if (PlatformDetection.isWindows()) {
                selectedEnvironment = "win-x86-64-no-cuda";
                installBiapyNoDeps = false;
            } else if (isMacArmOrRosetta() && isLegacyMacOs()) {
                selectedEnvironment = "macos-arm64-legacy";
                installBiapyNoDeps = true;
            } else if (isMacArmOrRosetta()) {
                selectedEnvironment = "macos-arm64";
                installBiapyNoDeps = false;
            } else if (PlatformDetection.isMacOS() && !isMacArmOrRosetta() && isLegacyMacOs()) {
                selectedEnvironment = "macos-x86-64-legacy";
                installBiapyNoDeps = true;
            } else {
                selectedEnvironment = "macos-x86-64";
                installBiapyNoDeps = false;
            }
        } else {
            final String compactCuda = cudaVersion.replace(".", "");
        	if (!PlatformDetection.isMacOS())
                pixiTomlContent = String.format(Locale.ROOT, pixiTemplate, COMMON_PYTORCH_ENV_NAME, compactCuda, compactCuda);
        	else
        		pixiTomlContent = String.format(Locale.ROOT, pixiTemplate, COMMON_PYTORCH_ENV_NAME);
            if (PlatformDetection.isLinux()) {
                selectedEnvironment = "linux-x86-64-cuda";
            } else {
                selectedEnvironment = "win-x86-64-cuda";
            }
            installBiapyNoDeps = false;
        }

        final File environmentDirectory = new File(Environments.apposeEnvsDir(), COMMON_PYTORCH_ENV_NAME);
        return new PixiEnvironmentSpec(
                selectedEnvironment,
                pixiTomlContent,
                environmentDirectory,
                installBiapyNoDeps ? Arrays.asList("biapy==3.5.10") : new ArrayList<String>()
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
     * @throws RuntimeException
     *     if the resource cannot be found or read
     */
    protected static String readClasspathResourceAsString(final String absoluteResourcePath) {
        Objects.requireNonNull(absoluteResourcePath, "absoluteResourcePath");

        try (InputStream is = PixiEnvironmentManager.class.getClassLoader().getResourceAsStream(absoluteResourcePath)) {
            if (is == null) {
                throw new RuntimeException("Required resource not found on classpath: " + absoluteResourcePath);
            }
            return new String(readAllBytesJava8(is), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read resource: " + absoluteResourcePath, e);
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
