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
package io.bioimage.modelrunner.model.special.stardist;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Consumer;

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.builder.PixiBuilder;
import org.apposed.appose.TaskException;
import org.apposed.appose.Builder.ProgressConsumer;
import org.apposed.appose.util.Environments;
import org.apposed.appose.util.Messages;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.BaseModel;
import io.bioimage.modelrunner.model.processing.Processing;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentManager;
import io.bioimage.modelrunner.model.python.envs.PixiEnvironmentSpec;
import io.bioimage.modelrunner.system.GpuCompatibility;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.transformations.ScaleRangeTransformation;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.JSONUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Implementation of an API to run Stardist 2D models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *
 *@author Carlos Garcia
 */
public abstract class StardistAbstract extends DLModelPytorchProtected {
	
	private final String modelDir;
	
	protected final String name;
	
	protected final String basedir;
	
	protected String envString = "default";
	
	protected Double threshold = null;
	
	protected final int nChannels;
	
	protected final String axes;
	
	protected Map<String, Object> config;
	
	protected SharedMemoryArray shma;
	
	private ModelDescriptor descriptor;
		
	private Service python;
	
	/**
	 * Value used to scale the image wihtin the [0, 1] range.
	 * Using minimum percentile 0 is equivalent to use the minimum of the image as the max
	 * Every pixels is transformed as follows: new_pixel = (pixel - min) / (max - min)
	 */
	public double scaleRangeMaxPercentile = 99.8;
	
	/**
	 * Value used to scale the image wihtin the [0, 1] range.
	 * Using maximum percentile 100 is equivalent to use the maximum of the image as the max
	 * Every pixels is transformed as follows: new_pixel = (pixel - min) / (max - min)
	 */
	public double scaleRangeMinPercentile = 1;
	
	/**
	 * Channels along which the scalin is performed.
	 * Imagine a xyc image, if the axes specified are xyc, the image will be scaled all together.
	 * However if the axes specified are xy, each channel will be scaled differently.
	 * By default all the image is scaled together
	 * 
	 */
	public String scaleRangeAxes = null;
		
	private static final String PIXI_TOML = "tomls/cellcast-pixi.toml";
	
	private static final String COMMON_CELLCAST_ENV_NAME = "cellcast-jdll";

	private static final String CELLCAST_WHEEL_RESOURCE_DIR = "wheels/cellcast";

	private static final String JDLL_CACHE_DIR_NAME = "jdll";

	private static final String WHEELS_CACHE_DIR_NAME = "wheels";
	
	private static final String OUTPUT_MASK_KEY = "mask";
	
	private static final String SHM_NAME_KEY = "_shm_name";
	
	private static final String DTYPE_KEY = "_dtype";
	
	private static final String SHAPE_KEY = "_shape";
	
	private static final String KEYS_KEY = "keys";
	
	protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
			+ "if '%s' not in globals().keys():" + System.lineSeparator()
			+ "  from stardist.models import %s" + System.lineSeparator()
			+ "  globals()['%s'] = %s" + System.lineSeparator()
			+ "if 'np' not in globals().keys():" + System.lineSeparator()
			+ "  import numpy as np" + System.lineSeparator()
			+ "  globals()['np'] = np" + System.lineSeparator()
			+ "if 'os' not in globals().keys():" + System.lineSeparator()
			+ "  import os" + System.lineSeparator()
			+ "  globals()['os'] = os" + System.lineSeparator()
			+ "if 'shared_memory' not in globals().keys():" + System.lineSeparator()
			+ "  from multiprocessing import shared_memory" + System.lineSeparator()
			+ "  globals()['shared_memory'] = shared_memory" + System.lineSeparator()
			+ "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"-1\"" + System.lineSeparator()
			+ "model = %s(None, name='%s', basedir=r\"%s\")" + System.lineSeparator()
			+ "globals()['model'] = model" + System.lineSeparator();
	
	private static final String RUN_MODEL_CODE = ""
			+ "output = model.predict_instances(im, return_predict=False)" + System.lineSeparator()
			+ "if type(output) == np.ndarray:" + System.lineSeparator()
			+ "  im[:] = output" + System.lineSeparator()
			+ "  im[:] = output" + System.lineSeparator()
			+ "  if os.name == 'nt':" + System.lineSeparator()
			+ "    im_shm.close()" + System.lineSeparator()
			+ "    im_shm.unlink()" + System.lineSeparator()
			+ "if type(output) != list and type(output) != tuple:" + System.lineSeparator()
			+ "  raise TypeError('StarDist output should be a list of a np.ndarray')" + System.lineSeparator()
			+ "if type(output[0]) != np.ndarray:" + System.lineSeparator()
			+ "  raise TypeError('If the StarDist output is a list, the first entry should be a np.ndarray')" + System.lineSeparator()
			+ "if len(im.shape) == 3 and len(output[0].shape) == 2:" + System.lineSeparator()
			+ "  im[:, :, 0] = output[0]" + System.lineSeparator()
			+ "elif len(im.shape) == 4 and len(output[0].shape) == 3:" + System.lineSeparator()
			+ "  im[:, :, :, 0] = output[0]" + System.lineSeparator()
			+ "else:" + System.lineSeparator()
			+ "  im[:] = output[0]" + System.lineSeparator()
			+ "if len(output) > 1 and type(output[1]) != dict:" + System.lineSeparator()
			+ "  raise TypeError('If the StarDist output is a list, the second entry needs to be a dict.')" + System.lineSeparator()
			+ "task.outputs['" + KEYS_KEY + "'] = list(output[1].keys())" + System.lineSeparator()
			+ "shm_list = []" + System.lineSeparator()
			+ "np_list = []" + System.lineSeparator()
			+ "for kk, vv in output[1].items():" + System.lineSeparator()
			+ "  if type(vv) != np.ndarray:" + System.lineSeparator()
			+ "    task.update('Output ' + kk + ' is not a np.ndarray. Only np.ndarrays supported.')" + System.lineSeparator()
			+ "    continue" + System.lineSeparator()
			+ "  if output[1][kk].nbytes == 0:" + System.lineSeparator()
			+ "    task.outputs[kk] = None" + System.lineSeparator()
			+ "  else:" + System.lineSeparator()
			+ "    task.outputs[kk + '" + SHAPE_KEY + "'] = output[1][kk].shape" + System.lineSeparator()
			+ "    task.outputs[kk + '"+ DTYPE_KEY + "'] = str(output[1][kk].dtype)" + System.lineSeparator()
			+ "    shm = shared_memory.SharedMemory(create=True, size=output[1][kk].nbytes)" + System.lineSeparator()
			+ "    task.outputs[kk + '"+ SHM_NAME_KEY + "'] = shm.name" + System.lineSeparator()
			+ "    shm_list.append(shm)" + System.lineSeparator()
			+ "    aa = np.ndarray(output[1][kk].shape, dtype=output[1][kk].dtype, buffer=shm.buf)" + System.lineSeparator()
			+ "    aa[:] = output[1][kk]" + System.lineSeparator()
			+ "    np_list.append(aa)" + System.lineSeparator()
			+ "globals()['shm_list'] = shm_list" + System.lineSeparator()
			+ "globals()['np_list'] = np_list" + System.lineSeparator()
			
			
			+ "if os.name == 'nt':" + System.lineSeparator()
			+ "  im_shm.close()" + System.lineSeparator()
			+ "  im_shm.unlink()" + System.lineSeparator();
	
	private static final String CLOSE_SHM_CODE = ""
			+ "if 'np_list' in globals().keys():" + System.lineSeparator()
			+ "  for a in np_list:" + System.lineSeparator()
			+ "    del a" + System.lineSeparator()
			+ "if 'shm_list' in globals().keys():" + System.lineSeparator()
			+ "  for s in shm_list:" + System.lineSeparator()
			+ "    s.unlink()" + System.lineSeparator()
			+ "    del s" + System.lineSeparator();
	
	/**
	 * Creates imports code.
	 *
	 * @return the resulting string.
	 */
	protected abstract String createImportsCode();
	
	/**
	 * Checks input.
	 *
	 * @param image the image parameter.
	 */
	protected abstract <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image);
	
	/**
	 * Executes reconstruct mask.
	 *
	 * @return the resulting value.
	 * @throws IOException if an I/O error occurs.
	 */
	protected abstract <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask();

	/**
	 * 
	 * @return whether the model is stardist2d or not
	 */
	public abstract boolean is2D();
	
	/**
	 * 
	 * @return whether the model is stardist3d or not
	 */
	public abstract boolean is3D();
	
	/**
	 * Creates a new StardistAbstract.
	 *
	 * @param modelName the modelName parameter.
	 * @param baseDir the baseDir parameter.
	 * @param config the config parameter.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException 
	 */
	protected StardistAbstract(String modelName, String baseDir, Map<String, Object> config) throws BuildException, IOException {
		super(null, null, null, new File(baseDir, modelName).getAbsolutePath(), new HashMap<String, Object>(), true);
		this.name = modelName;
		this.basedir = baseDir;
		this.modelDir = new File(baseDir, modelName).getAbsolutePath();
		this.config = config;
		this.nChannels = ((Number) config.get("n_channel_in")).intValue();
		this.axes = inferAxes(config);
	}

	private String inferAxes(Map<String, Object> config) {
		if (config.get("axes") != null)
			return ((String) config.get("axes")).toLowerCase();
		return this.is2D() ? "yxc" : "zyxc";
	}
	
	/**
	 * Checks files present.
	 *
	 * @param modelDir the modelDir parameter.
	 * @throws IOException if an I/O error occurs.
	 */
	private static void checkFilesPresent(String modelDir) throws IOException {
		if (new File(modelDir, "config.json").isFile() == false && new File(modelDir, Constants.RDF_FNAME).isFile() == false)
			throw new IllegalArgumentException("No 'config.json' file found in the model directory");
		//else if (new File(modelDir, "config.json").isFile() == false)
		//	createConfigFromBioimageio(null, modelDir);
		if (new File(modelDir, "thresholds.json").isFile() == false && new File(modelDir, Constants.RDF_FNAME).isFile() == false)
			throw new IllegalArgumentException("No 'thresholds.json' file found in the model directory");
		//else if (new File(modelDir, "thresholds.json").isFile() == false)
		//	createThresholdsFromBioimageio(null, modelDir);
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
        final File cellcastWheel = resolveCellcastWheelInCache();
        final String pixiTemplate = readClasspathResourceAsString(PIXI_TOML);
        final String pixiTomlContent = String.format(Locale.ROOT, pixiTemplate,
                COMMON_CELLCAST_ENV_NAME, currentPixiPlatform(), toPixiPath(cellcastWheel));

        final String selectedEnvironment = "default";

        final File environmentDirectory = new File(Environments.apposeEnvsDir(), COMMON_CELLCAST_ENV_NAME);
        return new PixiEnvironmentSpec(
                selectedEnvironment,
                pixiTomlContent,
                environmentDirectory,
                new ArrayList<String>()
        );
    }

    private static File resolveCellcastWheelInCache() {
        final String resourcePath = selectCellcastWheelResource();
        final String wheelName = resourcePath.substring(resourcePath.lastIndexOf('/') + 1);
        final File wheelsDir = new File(userCacheDir(), WHEELS_CACHE_DIR_NAME);
        final File cachedWheel = new File(wheelsDir, wheelName);

        if (cachedWheel.isFile() && cachedWheel.length() > 0) {
            return cachedWheel;
        }

        if (!wheelsDir.isDirectory() && !wheelsDir.mkdirs()) {
            throw new RuntimeException("Could not create CellCast wheel cache directory: "
                    + wheelsDir.getAbsolutePath());
        }

        try (InputStream is = StardistAbstract.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new RuntimeException("Required CellCast wheel resource not found: " + resourcePath);
            }
            Files.copy(is, cachedWheel.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Could not cache CellCast wheel to: "
                    + cachedWheel.getAbsolutePath(), e);
        }
        return cachedWheel;
    }

    private static File userCacheDir() {
        final String base;
        if (PlatformDetection.isWindows()) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.trim().isEmpty()) {
                base = localAppData;
            } else {
                base = new File(System.getProperty("user.home"), "AppData" + File.separator + "Local").getAbsolutePath();
            }
        } else if (PlatformDetection.isMacOS()) {
            base = new File(System.getProperty("user.home"), "Library" + File.separator + "Caches").getAbsolutePath();
        } else {
            String xdgCache = System.getenv("XDG_CACHE_HOME");
            if (xdgCache != null && !xdgCache.trim().isEmpty()) {
                base = xdgCache;
            } else {
                base = new File(System.getProperty("user.home"), ".cache").getAbsolutePath();
            }
        }
        return new File(base, JDLL_CACHE_DIR_NAME);
    }

    private static String selectCellcastWheelResource() {
        final String wheelName;
        final String arch = PlatformDetection.getArch();
        if (PlatformDetection.isLinux()) {
            if (PlatformDetection.ARCH_X86_64.equals(arch)) {
                wheelName = "cellcast-0.2.1.dev0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
            } else if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
                wheelName = "cellcast-0.2.1.dev0-cp37-abi3-manylinux_2_28_aarch64.whl";
            } else {
                throw new RuntimeException("Unsupported Linux architecture for CellCast: " + arch);
            }
        } else if (PlatformDetection.isMacOS()) {
            if (PlatformDetection.ARCH_X86_64.equals(arch) || PlatformDetection.isUsingRosseta()) {
                wheelName = "cellcast-0.2.1.dev0-cp37-abi3-macosx_10_12_x86_64.whl";
            } else if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
                wheelName = "cellcast-0.2.1.dev0-cp37-abi3-macosx_11_0_arm64.whl";
            } else {
                throw new RuntimeException("Unsupported macOS architecture for CellCast: " + arch);
            }
        } else if (PlatformDetection.isWindows()) {
            if (PlatformDetection.ARCH_X86_64.equals(arch)) {
                wheelName = "cellcast-0.2.1.dev0-cp37-abi3-win_amd64.whl";
            } else {
                throw new RuntimeException("Unsupported Windows architecture for CellCast: " + arch);
            }
        } else {
            throw new RuntimeException("Unsupported operating system for CellCast: " + PlatformDetection.getOs());
        }
        return CELLCAST_WHEEL_RESOURCE_DIR + "/" + wheelName;
    }

    private static String currentPixiPlatform() {
        final String arch = PlatformDetection.getArch();
        if (PlatformDetection.isLinux()) {
            if (PlatformDetection.ARCH_X86_64.equals(arch)) {
                return "linux-64";
            }
            if (PlatformDetection.ARCH_ARM64.equals(arch) || PlatformDetection.ARCH_AARCH64.equals(arch)) {
                return "linux-aarch64";
            }
        } else if (PlatformDetection.isMacOS()) {
            if ((PlatformDetection.ARCH_ARM64.equals(arch)
                    || PlatformDetection.ARCH_AARCH64.equals(arch))
                    && !PlatformDetection.isUsingRosseta()) {
                return "osx-arm64";
            }
            return "osx-64";
        } else if (PlatformDetection.isWindows() && PlatformDetection.ARCH_X86_64.equals(arch)) {
            return "win-64";
        }
        throw new RuntimeException("Unsupported Pixi platform for CellCast: "
                + PlatformDetection.getOs() + "-" + arch);
    }

    private static String toPixiPath(File file) {
        return file.getAbsolutePath().replace('\\', '/');
    }
	
	/**
	 * Sets threshold.
	 *
	 * @param threshold the threshold parameter.
	 */
	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}
	
	/**
	 * Creates encode image script.
	 *
	 * @return the resulting string.
	 */
	protected String createEncodeImageScript() {
		String code = "";
		// This line wants to recreate the original numpy array. Should look like:
		// input0_appose_shm = shared_memory.SharedMemory(name=input0)
		// input0 = np.ndarray(size, dtype="float64", buffer=input0_appose_shm.buf).reshape([64, 64])
		code += "im_shm = shared_memory.SharedMemory(name='"
							+ shma.getNameForPython() + "', size=" + shma.getSize() 
							+ ")" + System.lineSeparator();
		long nElems = 1;
		for (long elem : shma.getOriginalShape()) nElems *= elem;
		code += "im = np.ndarray(" + nElems  + ", dtype='" + CommonUtils.getDataTypeFromRAI(Cast.unchecked(shma.getSharedRAI()))
			  + "', buffer=im_shm.buf).reshape([";
		for (int i = 0; i < shma.getOriginalShape().length; i ++)
			code += shma.getOriginalShape()[i] + ", ";
		code += "])" + System.lineSeparator();
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void preprocess(List<Tensor<T>> inputs) {
		if (descriptor != null) {
			Processing processing = Processing.init(descriptor);
			List<Tensor<T>> inputsProcessed = processing.preprocess(inputs, false);
			inputs.set(0, inputsProcessed.get(0));
		} else {
			ScaleRangeTransformation transform = new ScaleRangeTransformation();
			transform.setMaxPercentile(scaleRangeMaxPercentile);
			transform.setMinPercentile(scaleRangeMinPercentile);
			transform.setAxes(scaleRangeAxes);
			inputs.set(0, Cast.unchecked(transform.apply(inputs.get(0))));
		}
	}
	
	/**
	 * Gets nchannels.
	 *
	 * @return the resulting numeric value.
	 */
	public int getNChannels() {
		return nChannels;
	}
	
	/**
	 * Executes close.
	 */
	@Override
	public void close() {
		if (!loaded)
			return;
		python.close();
		loaded = false;
		closed = true;
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run( List< Tensor < T > > inTensors, List< Tensor < R > > outTensors ) throws RunModelException {
	}
	
	/**
	 * Loads model.
	 *
	 * @throws LoadModelException if a LoadModelException occurs while executing this method.
	 */
	@Override
	public void loadModel() throws LoadModelException {
		if (closed)
			throw new RuntimeException("Cannot load model after it has been closed");
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
		Task task;
		try {
			task = python.task(code);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException("Task canceled");
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED)
				throw new RuntimeException(task.error);
			loaded = true;
		} catch (InterruptedException | TaskException e) {
			throw new LoadModelException(Messages.stackTrace(e));
		}
	}
	
	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors)
			throws RunModelException {
		return null;
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runNoBatch(List<Tensor<R>> inputTensors)
			throws RunModelException  {
		try {
			Map<String, RandomAccessibleInterval<T>> outputs = run(inputTensors.get(0).getData());
			List<Tensor<T>> outTensors = new ArrayList<Tensor<T>>();
			for (Entry<String, RandomAccessibleInterval<T>> entry : outputs.entrySet()) {
				if (entry.getValue() == null)
					continue;
				String axesOrder = "xy";
				if (entry.getValue().dimensionsAsLongArray().length > 2 && this.is2D())
					axesOrder += "c";
				else if (entry.getValue().dimensionsAsLongArray().length == 3 && this.is3D())
					axesOrder += "z";
				else if (entry.getValue().dimensionsAsLongArray().length > 3 && this.is3D())
					axesOrder += "zc";
				else if (entry.getValue().dimensionsAsLongArray().length == 1)
					axesOrder = "i";
				Tensor<T> tt = Tensor.build(entry.getKey(), axesOrder, entry.getValue());
				// TODO
				if (!entry.getKey().equals("mask"))
					continue;
				outTensors.add(tt);
			}
			return outTensors;
		} catch (TaskException | InterruptedException | IOException e) {
			throw new RunModelException(Messages.stackTrace(e));
		}
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBatch(List<Tensor<R>> inputTensors) 
			throws RunModelException {
		int bPos = inputTensors.get(0).getAxesOrderString().indexOf("b");
		if (bPos != 0)
			throw new IllegalArgumentException("The Batch Size (b) should always be in the first dimension -> " + inputTensors.get(0).getAxesOrderString());
		int nBatches = inputTensors.get(0).getShape()[bPos];
		List<Tensor<T>> outTensors = runBatchPosOne(inputTensors, bPos, nBatches) ;
		try {
			for (int n = 1; n < nBatches; n ++ ) {
				RandomAccessibleInterval<R> data = Views.hyperSlice(inputTensors.get(0).getData(), bPos, n);
				Map<String, RandomAccessibleInterval<T>> outputs = run(data);
				for (Entry<String, RandomAccessibleInterval<T>> entry : outputs.entrySet()) {
					if (entry.getValue() == null)
						continue;
					// TODO
					if (!entry.getKey().equals("mask"))
						continue;
					Tensor<T> tensor = outTensors.stream()
							.filter(ten -> ten.getName().equals(entry.getKey())).findFirst().orElse(null);
					LoopBuilder.setImages( Views.hyperSlice(tensor.getData(), bPos, n), entry.getValue() )
						.multiThreaded().forEachPixel( ( i, o ) -> i.set( o ) );
				}
			}
			return outTensors;
		} catch (TaskException | InterruptedException | IOException e) {
			throw new RunModelException(Messages.stackTrace(e));
		}
	}

	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> runBatchPosOne(List<Tensor<R>> inputTensors, int bPos, int nBatches) 
			throws RunModelException {
		List<Tensor<T>> outTensors = new ArrayList<Tensor<T>>();
		try {
			RandomAccessibleInterval<R> data = Views.hyperSlice(inputTensors.get(0).getData(), bPos, 0);
			Map<String, RandomAccessibleInterval<T>> outputs = run(data);
			for (Entry<String, RandomAccessibleInterval<T>> entry : outputs.entrySet()) {
				if (entry.getValue() == null)
					continue;
				String axesOrder = "bxy";
				if (entry.getValue().dimensionsAsLongArray().length > 2 && this.is2D())
					axesOrder += "c";
				else if (entry.getValue().dimensionsAsLongArray().length == 3 && this.is3D())
					axesOrder += "z";
				else if (entry.getValue().dimensionsAsLongArray().length > 3 && this.is3D())
					axesOrder += "zc";
				else if (entry.getValue().dimensionsAsLongArray().length == 1)
					axesOrder = "bi";
				// TODO
				if (!entry.getKey().equals("mask"))
					continue;
				ArrayImgFactory<T> allBatchesOut = new ArrayImgFactory<>(entry.getValue().getType());
				long[] outDims = new long[1 + entry.getValue().dimensionsAsLongArray().length];
				outDims[0] = nBatches;
				for (int i = 1; i < outDims.length; i ++) outDims[i] = entry.getValue().dimensionsAsLongArray()[i - 1];
				ArrayImg<T, ?> output = allBatchesOut.create(outDims);
				Tensor<T> tt = Tensor.build(entry.getKey(), axesOrder, output);
				outTensors.add(tt);
				LoopBuilder.setImages( Views.hyperSlice(output, bPos, 0), entry.getValue() )
					.multiThreaded().forEachPixel( ( i, o ) -> i.set( o ) );
			}
			return outTensors;
		} catch (TaskException | InterruptedException | IOException e) {
			throw new RunModelException(Messages.stackTrace(e));
		}
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	Map<String, RandomAccessibleInterval<R>> run(RandomAccessibleInterval<T> img) throws InterruptedException, TaskException, IOException {
		checkInput(img);
		shma = SharedMemoryArray.createSHMAFromRAI(img, false, false);
		String code = "";
		if (!loaded) {
			code += createImportsCode() + System.lineSeparator();
		}
		
		code += createEncodeImageScript() + System.lineSeparator();
		if (this.threshold != null) {
			code += String.format("model.thresholds = dict (prob=%s, nms=model.thresholds.nms)", threshold) + System.lineSeparator();
		}
		code += RUN_MODEL_CODE + System.lineSeparator();
		
		Task task = python.task(code);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException("Task canceled");
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException(task.error);
		else if (task.status == TaskStatus.CRASHED)
			throw new RuntimeException(task.error);
		loaded = true;
		
		
		return reconstructOutputs(task);
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	Map<String, RandomAccessibleInterval<T>> reconstructOutputs(Task task) 
			throws IOException {
		
		Map<String, RandomAccessibleInterval<T>> outs = new LinkedHashMap<String, RandomAccessibleInterval<T>>();
		outs.put(OUTPUT_MASK_KEY, reconstructMask());
		
		if (task.outputs.get(KEYS_KEY) != null) {
			for (String kk : (List<String>) task.outputs.get(KEYS_KEY)) {
				outs.put(kk, reconstruct(task, kk));
			}
		}
		
		if (PlatformDetection.isWindows()) {
			Task closeSHMTask = python.task(CLOSE_SHM_CODE);
			// TODO closeSHMTask.waitFor();
		}
		return outs;
	}
	
	private <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> reconstruct(Task task, String key) throws IOException {

		String shm_name = (String) task.outputs.get(key + SHM_NAME_KEY);
		String coords_dtype = (String) task.outputs.get(key + DTYPE_KEY);
		List<Number> coords_shape = (List<Number>) task.outputs.get(key + SHAPE_KEY);
		if (coords_shape == null)
			return null;
		
		long[] coordsSh = new long[coords_shape.size()];
		for (int i = 0; i < coordsSh.length; i ++)
			coordsSh[i] = coords_shape.get(i).longValue();
		SharedMemoryArray shmCoords = SharedMemoryArray.readOrCreate(shm_name, coordsSh, 
				Cast.unchecked(CommonUtils.getImgLib2DataType(coords_dtype)), false, false);
		
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> coordsRAI = shmCoords.getSharedRAI();
		RandomAccessibleInterval<T> coordsCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(coordsRAI), 
				Util.getTypeFromInterval(Cast.unchecked(coordsRAI)));
		
		shmCoords.close();
		
		return coordsCopy;
	}

	public static void train(int epochs, String dataDir, String gtDir, String outputDir,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		Map<String, Object> config = defaultTrainingConfig(epochs);
		train(dataDir, gtDir, outputDir, true, "grayscale", "grayscale", 0.15d,
				config, progressConsumer, previewConsumer, logConsumer);
	}

	public static void train(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer)
			throws IOException, BuildException, InterruptedException, TaskException {
		validateTrainingArguments(dataDir, gtDir, outputDir, validFraction, config);
		File output = new File(outputDir);
		if (!output.isDirectory() && !output.mkdirs()) {
			throw new IOException("Could not create StarDist output directory: " + output.getAbsolutePath());
		}

		PixiEnvironmentSpec envSpec = resolvePytorchEnv();
		Environment env = Appose.pixi()
				.environment(envSpec.getSelectedEnvironment())
				.wrap(envSpec.getEnvironmentDirectory());
		Service python = env.python();
		python.init("import numpy as np");
		if (logConsumer != null) {
			python.debug(logConsumer);
		}
		try {
			Task task = python.task(buildTrainingCode(dataDir, gtDir, outputDir, gpu,
					imageChannels, labelColorMode, validFraction, config));
			task.listen(event -> handleTrainingEvent(event, progressConsumer, previewConsumer, logConsumer));
			task.waitFor();
		} finally {
			if (python.isAlive()) {
				python.close();
			}
		}
	}

	public static Map<String, Object> defaultTrainingConfig(int epochs) {
		Map<String, Object> config = new LinkedHashMap<String, Object>();
		config.put("n_rays", 32);
		config.put("grid", Arrays.asList(2, 2));
		config.put("patch_size", Arrays.asList(256, 256));
		config.put("batch_size", 4);
		config.put("epochs", epochs);
		config.put("steps_per_epoch", 100);
		config.put("validation_steps", 10);
		config.put("validation_preview_count", 20);
		config.put("learning_rate", 0.0003d);
		config.put("foreground_probability", 0.9d);
		config.put("background_reg", 1e-4d);
		config.put("loss_prob_weight", 1.0d);
		config.put("loss_dist_weight", 0.2d);
		config.put("lr_schedule", "reduce_on_plateau");
		config.put("lr_factor", 0.5d);
		config.put("lr_patience", 40);
		config.put("lr_min_delta", 0.0d);
		config.put("normalization", "percentile");
		config.put("normalization_percentiles", Arrays.asList(1.0d, 99.8d));
		config.put("seed", 42);
		return config;
	}

	private static void validateTrainingArguments(String dataDir, String gtDir,
			String outputDir, double validFraction, Map<String, Object> config) {
		if (dataDir == null || !new File(dataDir).isDirectory()) {
			throw new IllegalArgumentException("The StarDist dataset directory does not exist: " + dataDir);
		}
		if (gtDir != null && !gtDir.trim().isEmpty() && !new File(gtDir).isDirectory()) {
			throw new IllegalArgumentException("The StarDist ground-truth directory does not exist: " + gtDir);
		}
		if (outputDir == null || outputDir.trim().isEmpty()) {
			throw new IllegalArgumentException("The StarDist output directory cannot be empty.");
		}
		if (validFraction < 0.0d || validFraction >= 1.0d) {
			throw new IllegalArgumentException("The StarDist validation fraction must be in [0, 1).");
		}
		if (config == null || config.isEmpty()) {
			throw new IllegalArgumentException("The StarDist training config cannot be empty.");
		}
		Object epochs = config.get("epochs");
		if (!(epochs instanceof Number) || ((Number) epochs).intValue() <= 0) {
			throw new IllegalArgumentException("The StarDist config must define epochs > 0.");
		}
	}

	public static String buildTrainingCode(String dataDir, String gtDir, String outputDir,
			boolean gpu, String imageChannels, String labelColorMode, double validFraction,
			Map<String, Object> config) {
		String nl = System.lineSeparator();
		boolean hasGtDir = gtDir != null && !gtDir.trim().isEmpty();
		String gtDirCode = hasGtDir ? "gt_dir = r'" + py(new File(gtDir).getAbsolutePath()) + "'" + nl : "";
		String gtDirArgument = hasGtDir ? "gt_dir=gt_dir, " : "";
		String safeImageChannels = imageChannels == null || imageChannels.trim().isEmpty()
				? "grayscale" : imageChannels.trim();
		String safeLabelColorMode = labelColorMode == null || labelColorMode.trim().isEmpty()
				? "grayscale" : labelColorMode.trim();
		return ""
				+ "import contextlib, json, os, sys" + nl
				+ "from pathlib import Path" + nl
				+ "import numpy as np" + nl
				+ "_appose_stdout = sys.stdout" + nl
				+ "import cellcast.training.stardist_2d as train" + nl
				+ "data_dir = r'" + py(new File(dataDir).getAbsolutePath()) + "'" + nl
				+ gtDirCode
				+ "output_dir = Path(r'" + py(new File(outputDir).getAbsolutePath()) + "')" + nl
				+ "preview_dir = output_dir / 'previews'" + nl
				+ "preview_manifest_path = preview_dir / 'latest.json'" + nl
				+ "output_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "preview_dir.mkdir(parents=True, exist_ok=True)" + nl
				+ "stardist_log_path = output_dir / 'training.log'" + nl
				+ "config = json.loads(r'''" + toJson(config) + "''')" + nl
				+ "state = {'total_steps': int(config.get('epochs', 0)) * int(config.get('steps_per_epoch', 0)), 'total_epochs': int(config.get('epochs', 0))}" + nl
				+ "def _task_update(**kwargs):" + nl
				+ "  current_stdout = sys.stdout" + nl
				+ "  try:" + nl
				+ "    sys.stdout = _appose_stdout" + nl
				+ "    task.update(**kwargs)" + nl
				+ "  finally:" + nl
				+ "    sys.stdout = current_stdout" + nl
				+ "def _scalar(value):" + nl
				+ "  try:" + nl
				+ "    if value is None:" + nl
				+ "      return None" + nl
				+ "    return float(value)" + nl
				+ "  except Exception:" + nl
				+ "    return None" + nl
				+ "def _clean(values):" + nl
				+ "  return {str(k): float(v) for k, v in dict(values).items() if _scalar(v) is not None}" + nl
				+ "def _atomic_npy_save(path, array):" + nl
				+ "  tmp_path = str(path) + '.tmp'" + nl
				+ "  with open(tmp_path, 'wb') as f:" + nl
				+ "    np.save(f, array)" + nl
				+ "  os.replace(tmp_path, path)" + nl
				+ "def on_train_begin(plan):" + nl
				+ "  state['total_steps'] = int(plan.get('total_steps') or state['total_steps'])" + nl
				+ "  state['total_epochs'] = int(plan.get('epochs') or state['total_epochs'])" + nl
				+ "  info = {'type': 'progress', 'epoch': 0, 'step': 0, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': {}, 'metrics': _clean({'batch_size': plan.get('batch_size'), 'train_samples': plan.get('train_samples'), 'valid_samples': plan.get('valid_samples')})}" + nl
				+ "  _task_update(message='StarDist training started', current=0, maximum=state['total_steps'], info=info)" + nl
				+ "def on_step_end(step):" + nl
				+ "  global_step = int(step.get('global_step', 0))" + nl
				+ "  epoch = int(step.get('epoch', 0))" + nl
				+ "  losses = _clean({'train/total_loss': step.get('loss_total'), 'train/prob_loss': step.get('loss_prob'), 'train/dist_loss': step.get('loss_dist')})" + nl
				+ "  metrics = _clean({'learning_rate': step.get('learning_rate')})" + nl
				+ "  info = {'type': 'progress', 'epoch': epoch, 'step': global_step, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': losses, 'metrics': metrics}" + nl
				+ "  _task_update(message='StarDist training step %d/%d' % (global_step, state['total_steps']), current=global_step, maximum=state['total_steps'], info=info)" + nl
				+ "def on_validation_end(event):" + nl
				+ "  epoch = int(event.get('epoch', 0))" + nl
				+ "  step = min(state['total_steps'], epoch * int(config.get('steps_per_epoch', 1)))" + nl
				+ "  losses = _clean({'train/total_loss': event.get('train_total'), 'val/total_loss': event.get('valid_total')})" + nl
				+ "  metrics = _clean({'learning_rate': event.get('learning_rate')})" + nl
				+ "  info = {'type': 'progress', 'epoch': epoch, 'step': step, 'total_epochs': state['total_epochs'], 'total_steps': state['total_steps'], 'losses': losses, 'metrics': metrics}" + nl
				+ "  _task_update(message='StarDist epoch %d/%d' % (epoch, state['total_epochs']), current=step, maximum=state['total_steps'], info=info)" + nl
				+ "  samples = []" + nl
				+ "  for i, preview in enumerate(event.get('previews', [])[:20]):" + nl
				+ "    image_path = preview_dir / ('preview_%03d_image.npy' % i)" + nl
				+ "    label_path = preview_dir / ('preview_%03d_label.npy' % i)" + nl
				+ "    pred_path = preview_dir / ('preview_%03d_prediction.npy' % i)" + nl
				+ "    prob_path = preview_dir / ('preview_%03d_prob.npy' % i)" + nl
				+ "    sample = {'index': i}" + nl
				+ "    image = preview.get('image')" + nl
				+ "    label = preview.get('labels', preview.get('label'))" + nl
				+ "    prediction = preview.get('prediction')" + nl
				+ "    prob = preview.get('prob')" + nl
				+ "    if image is not None:" + nl
				+ "      _atomic_npy_save(image_path, image)" + nl
				+ "      sample['image_path'] = str(image_path)" + nl
				+ "    if label is not None:" + nl
				+ "      _atomic_npy_save(label_path, np.asarray(label, dtype=np.int32))" + nl
				+ "      sample['label_path'] = str(label_path)" + nl
				+ "    if prediction is not None:" + nl
				+ "      _atomic_npy_save(pred_path, np.asarray(prediction, dtype=np.int32))" + nl
				+ "      sample['prediction_path'] = str(pred_path)" + nl
				+ "    if prob is not None:" + nl
				+ "      _atomic_npy_save(prob_path, prob)" + nl
				+ "      sample['prob_path'] = str(prob_path)" + nl
				+ "    if 'image_path' in sample and 'prediction_path' in sample:" + nl
				+ "      samples.append(sample)" + nl
				+ "  if samples:" + nl
				+ "    manifest = {'epoch': epoch, 'samples': samples}" + nl
				+ "    with open(preview_manifest_path, 'w', encoding='utf-8') as f:" + nl
				+ "      json.dump(manifest, f)" + nl
				+ "    _task_update(message='StarDist validation preview epoch %d' % epoch, current=epoch, maximum=state['total_epochs'], info={'type': 'preview', 'epoch': epoch, 'preview_path': str(preview_manifest_path)})" + nl
				+ "with open(stardist_log_path, 'a', encoding='utf-8') as stardist_log, contextlib.redirect_stdout(stardist_log), contextlib.redirect_stderr(stardist_log):" + nl
				+ "  result = train.train_stardist_2d_folder(data_dir=data_dir, " + gtDirArgument + "output_dir=str(output_dir), gpu=" + (gpu ? "True" : "False") + ", image_channels='" + py(safeImageChannels) + "', label_color_mode='" + py(safeLabelColorMode) + "', valid_fraction=" + validFraction + ", config=config, on_train_begin=on_train_begin, on_step_end=on_step_end, on_validation_end=on_validation_end)" + nl
				//+ "result = train.train_stardist_2d_folder(data_dir=data_dir, " + gtDirArgument + "output_dir=str(output_dir), gpu=" + (gpu ? "True" : "False") + ", image_channels='" + py(safeImageChannels) + "', label_color_mode='" + py(safeLabelColorMode) + "', valid_fraction=" + validFraction + ", config=config, on_train_begin=on_train_begin, on_step_end=on_step_end, on_validation_end=on_validation_end)" + nl
				+ "task.outputs(result=str(result.get('output_dir', str(output_dir))))" + nl;
	}

	private static void handleTrainingEvent(TaskEvent event,
			Consumer<StardistTrainingProgress> progressConsumer,
			Consumer<StardistValidationPreview> previewConsumer,
			Consumer<String> logConsumer) {
		if (event.message != null && logConsumer != null) {
			logConsumer.accept(event.message);
		}
		if (!event.responseType.equals(ResponseType.UPDATE) || event.info == null) {
			return;
		}
		Object type = event.info.get("type");
		if ("progress".equals(type) && progressConsumer != null) {
			progressConsumer.accept(new StardistTrainingProgress(
					asInt(event.info.get("epoch"), (int) event.current),
					asInt(event.info.get("step"), (int) event.current),
					asInt(event.info.get("total_epochs"), 0),
					asInt(event.info.get("total_steps"), (int) event.maximum),
					asDoubleMap(event.info.get("losses")),
					asDoubleMap(event.info.get("metrics"))));
		} else if ("preview".equals(type) && previewConsumer != null) {
			Object previewPath = event.info.get("preview_path");
			previewConsumer.accept(new StardistValidationPreview(
					asInt(event.info.get("epoch"), (int) event.current),
					previewPath == null ? null : previewPath.toString()));
		}
	}

	private static int asInt(Object value, int fallback) {
		return value instanceof Number ? ((Number) value).intValue() : fallback;
	}

	private static Map<String, Double> asDoubleMap(Object value) {
		Map<String, Double> result = new LinkedHashMap<String, Double>();
		if (!(value instanceof Map)) {
			return result;
		}
		for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
			if (entry.getKey() != null && entry.getValue() instanceof Number) {
				result.put(entry.getKey().toString(), ((Number) entry.getValue()).doubleValue());
			}
		}
		return result;
	}

	private static String py(String path) {
		return path == null ? "" : path.replace("\\", "\\\\").replace("'", "\\'");
	}

	private static String toJson(Object value) {
		if (value == null) {
			return "null";
		}
		if (value instanceof String) {
			return "\"" + ((String) value).replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
		}
		if (value instanceof Boolean || value instanceof Number) {
			if (value instanceof Double && !Double.isFinite(((Double) value).doubleValue())) {
				return "null";
			}
			if (value instanceof Float && !Float.isFinite(((Float) value).floatValue())) {
				return "null";
			}
			return String.format(Locale.ROOT, "%s", value);
		}
		if (value instanceof Map) {
			StringBuilder builder = new StringBuilder("{");
			boolean first = true;
			for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
				if (!first) {
					builder.append(',');
				}
				first = false;
				builder.append(toJson(String.valueOf(entry.getKey()))).append(':').append(toJson(entry.getValue()));
			}
			return builder.append('}').toString();
		}
		if (value instanceof Iterable) {
			StringBuilder builder = new StringBuilder("[");
			boolean first = true;
			for (Object item : (Iterable<?>) value) {
				if (!first) {
					builder.append(',');
				}
				first = false;
				builder.append(toJson(item));
			}
			return builder.append(']').toString();
		}
		if (value.getClass().isArray()) {
			StringBuilder builder = new StringBuilder("[");
			int length = Array.getLength(value);
			for (int i = 0; i < length; i++) {
				if (i > 0) {
					builder.append(',');
				}
				builder.append(toJson(Array.get(value, i)));
			}
			return builder.append(']').toString();
		}
		return toJson(String.valueOf(value));
	}
	
	/**
	 * Executes init.
	 *
	 * @param modelDir the modelDir parameter.
	 * @return the resulting value.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException 
	 */
	public static StardistAbstract init(String modelDir) throws IOException, BuildException {
		File modelDirFile = new File(modelDir);
		String modelName = modelDirFile.getName();
		String baseDir = modelDirFile.getParentFile().getAbsolutePath();
		//checkFilesPresent(modelDir);
		Map<String, Object> configMap = JSONUtils.load(new File(baseDir, "config.json").getAbsolutePath());
		String axes = ((String) configMap.get("axes")).toUpperCase();
		if (axes.contains("Z"))
			return new Stardist3D(modelName, baseDir, configMap);
		else
			return new Stardist2D(modelName, baseDir, configMap);
	}

    /**
     * Checks whether the default protected PyTorch environment is installed.
     *
     * @return {@code true} if the environment appears to be fully installed
     */
    public static boolean isInstalled() {
        PixiEnvironmentSpec spec = null;
        try {
            spec = resolvePytorchEnv();
            return PixiEnvironmentManager.isInstalled(spec);
        } catch (Exception e) {
            return false;
        }
    }
    
    public static void main(String[] args) throws InterruptedException, BuildException {
        PixiEnvironmentSpec spec = resolvePytorchEnv();
        PixiEnvironmentManager.installRequirements(spec, (str) -> {System.out.println(str);});
    }
}
