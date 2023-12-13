/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.transformations;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.compress.archivers.ArchiveException;
import io.bioimage.modelrunner.apposed.appose.Conda;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.runmode.ops.GenericOp;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.YAMLUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;

public class PythonTransformation extends AbstractTensorTransformation
{
	public static final String NAME = "python";
	public static final String ENV_YAML_KEY = "env_yaml";
	public static final String SCRIPT_KEY = "script";
	public static final String N_OUTPUTS_KEY = "n_ouputs";
	public static final String KWARGS_KEY = "kwargs";
	public static final String METHOD_KEY = "kwargs";

	/**
	 * Path setting keys
	 */
	public static final String MAMBA_PATH_KEY = "mamba_path";
	public static final String SCRIPT_FILE_PATH_KEY = "script_file_path";
	public static final String ENV_YAML_FILE_PATH_KEY = "env_yaml_file_path";
	public static final String ENV_PATH_KEY = "env_path";
	
	private final static String MAMBA_RELATIVE_PATH = PlatformDetection.isWindows() ? 
			 File.separator + "Library" + File.separator + "bin" + File.separator + "micromamba.exe" 
			: File.separator + "bin" + File.separator + "micromamba";
	
	private final static String PYTHON_COMMAND = PlatformDetection.isWindows() ? "python.exe" : "bin/python";
	
	/**
	 * TODO adapt to conda to support conda
	 */
	private final static String CONDA_RELATIVE_PATH = PlatformDetection.isWindows() ? 
			 File.separator + "Library" + File.separator + "bin" + File.separator + "micromamba.exe" 
			: File.separator + "bin" + File.separator + "micromamba";
	
	private String envYaml = "stardist.yaml";
	
	private String script = "stardist_postprocessing.py";
	
	private String method = "stardist_postprocessing";
	
	private String envYamlFilePath = "";
	
	private String scriptFilePath = "";
	
	private String mambaPath = "";
	
	private String envPath = "";
	
	private int nOutputs = 1;
	
	private boolean install = false;
	
	private Map<String, Object> kwargs = new LinkedHashMap<String, Object>();
	{
		kwargs.put("prob_thresh", 0.6924782541382084);
		kwargs.put("nms_thresh", 0.3);
	}

	public PythonTransformation()
	{
		super(NAME);
	}
	
	public void setInstall(Object install) {
		if (install instanceof Boolean) {
			this.install = ((Boolean) install).booleanValue();
		} else if (install.getClass().equals(boolean.class)) {
			this.install = (boolean) install;
		} else {
			throw new IllegalArgumentException("'install' parameter has to be an instance of "
					+ Boolean.class
					+ ". The provided argument is an instance of: " + install.getClass());
		}
	}
	
	public void setEnvYamlFilePath(Object envYamlFilePath) {
		if (envYamlFilePath instanceof String) {
			this.envYamlFilePath = (String) envYamlFilePath;
		} else {
			throw new IllegalArgumentException("'envYamlFilePath' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYamlFilePath.getClass());
		}
	}
	
	public void setMambaPath(Object mambaDir) {
		if (mambaDir instanceof String) {
			this.mambaPath = (String) mambaDir;
		} else {
			throw new IllegalArgumentException("'mambaDir' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + mambaDir.getClass());
		}
	}
	
	public void setEnvPath(Object envDir) {
		if (envDir instanceof String) {
			this.envPath = (String) envDir;
		} else {
			throw new IllegalArgumentException("'envDir' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envDir.getClass());
		}
	}
	
	public void setScriptFilePath(Object scriptFilePath) {
		if (scriptFilePath instanceof String) {
			this.scriptFilePath = (String) scriptFilePath;
		} else {
			throw new IllegalArgumentException("'scriptFilePath' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + scriptFilePath.getClass());
		}
	}
	
	public void setEnvYaml(Object envYaml) {
		if (envYaml instanceof String) {
			this.envYaml = new File(String.valueOf(envYaml)).getAbsolutePath();
		} else {
			throw new IllegalArgumentException("'envYaml' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setScript(Object script) {
		if (script instanceof String) {
			this.script = new File(String.valueOf(script)).getAbsolutePath();
		} else {
			throw new IllegalArgumentException("'script' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + script.getClass());
		}
	}
	
	public void setMethod(Object method) {
		if (method instanceof String) {
			this.method = String.valueOf(method);
		} else {
			throw new IllegalArgumentException("'method' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + method.getClass());
		}
	}
	
	public void setNOutputs(Object nOutputs) {
		if (Number.class.isAssignableFrom(nOutputs.getClass()) 
				|| (nOutputs.getClass().isPrimitive() && !String.class.isAssignableFrom(nOutputs.getClass())) ) {
			this.nOutputs = (int) nOutputs;
		} else {
			throw new IllegalArgumentException("'nOutputs' parameter has to be a number"
					+ ". The provided argument is an instance of: " + nOutputs.getClass());
		}
	}
	
	public void setKwargs(Object kwargs) {
		if (kwargs == null) {
			this.kwargs = new LinkedHashMap<String, Object>();
			return;
		}
		if (kwargs instanceof Map) {
			this.kwargs = (Map<String, Object>) kwargs;
		} else {
			throw new IllegalArgumentException("'kwargs' parameter has to be an instance of "
					+ Map.class
					+ ". The provided argument is an instance of: " + kwargs.getClass());
		}
	}
	
	/**
	 * TODO fix the logic so when both envYaml and envYamlFilePath point to different valid files,
	 * an error is thrown because of the contradiction
	 * TODO same for script
	 * TODO fix the logic so when both envYaml and envYamlFilePath point to different valid files,
	 * an error is thrown because of the contradiction
	 * TODO same for script
	 * TODO fix the logic so when both envYaml and envYamlFilePath point to different valid files,
	 * an error is thrown because of the contradiction
	 * TODO same for script
	 * TODO fix the logic so when both envYaml and envYamlFilePath point to different valid files,
	 * an error is thrown because of the contradiction
	 * TODO same for script
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ArchiveException
	 * @throws URISyntaxException
	 */
	private void checkArgs() throws IOException, InterruptedException, ArchiveException, URISyntaxException {
		//Check that the path to the script of interest is correct
		if (!(new File(script).isFile()) && !(new File(this.scriptFilePath).exists()))
			throw new IllegalArgumentException();
		else if (!(new File(script).isFile()) && new File(this.scriptFilePath).isDirectory()
				&& !(new File(new File(this.scriptFilePath).getAbsolutePath(), new File(script).getName()).isFile()))
			throw new IllegalArgumentException();
		else if (new File(this.scriptFilePath).isFile())
			script = scriptFilePath;
		else if (new File(this.scriptFilePath).isDirectory()
				&& (new File(new File(this.scriptFilePath).getAbsolutePath(), new File(script).getName()).isFile()))
			script = new File(new File(this.scriptFilePath).getAbsolutePath(), new File(script).getName()).getAbsolutePath();
		 
		/**
		 * TODO think whether the envirornment shuold already be created or not
		 */
		// Check environment directory provided contains Python, if the env has been provided
		if (this.envPath != null && !(new File(this.envPath + File.separator + PYTHON_COMMAND)).isFile())
			throw new IllegalArgumentException();
		else if (this.envPath == null)
			throw new IllegalArgumentException();
		else if (this.envPath != null)
			return;
		// Check that the environment yaml file is correct
		if (!(new File(envYaml).isFile()) && !(new File(this.envYamlFilePath).exists()))
			throw new IllegalArgumentException();
		else if (!(new File(envYaml).isFile()) && new File(this.envYamlFilePath).isDirectory()
				&& !(new File(new File(this.envYamlFilePath).getAbsolutePath(), new File(envYaml).getName()).isFile()))
			throw new IllegalArgumentException();
		else if (new File(this.envYamlFilePath).isFile())
			envYaml = envYamlFilePath;
		else if (new File(this.envYamlFilePath).isDirectory()
				&& (new File(new File(this.envYamlFilePath).getAbsolutePath(), new File(envYaml).getName()).isFile()))
			envYaml = new File(new File(this.envYamlFilePath).getAbsolutePath(), new File(envYaml).getName()).getAbsolutePath();
		 
		// Check if the path to mamba is correct
		if (this.mambaPath == null && !install)
			throw new IllegalArgumentException();
		else if (this.mambaPath != null && !!(new File(this.mambaPath + MAMBA_RELATIVE_PATH).exists()) && !install)
			throw new IllegalArgumentException();
		else if (this.mambaPath == null || !(new File(this.mambaPath + MAMBA_RELATIVE_PATH).exists()))
			installMamba();

		String envName = null;
		try {
			envName = (String) YAMLUtils.load(this.envYaml).get("name");
		} catch (IOException e) {
			throw new IOException("Unable read the environemnt name from the environment .yaml file." 
						+ System.lineSeparator() + e.toString());
		}
		this.envPath = this.mambaPath + File.separator + "envs" + File.separator + envName;
		// Check if the env is installed
		if (!(new File(this.mambaPath + File.separator + "envs" + File.separator + envName).exists()) && !install)
			throw new IllegalArgumentException();
		else if (!(new File(this.mambaPath + File.separator + "envs" + File.separator + envName).exists()))
			installEnv();
	}

	public < R extends RealType< R > & NativeType< R > > Tensor<FloatType> apply( final Tensor< R > input )
	{
		try {
			checkArgs();
		} catch (IOException | InterruptedException | ArchiveException | URISyntaxException e) {
			e.printStackTrace();
			return Cast.unchecked(input);
		}
		GenericOp op = GenericOp.create(new File(envPath).getAbsolutePath(), new File(this.script).getAbsolutePath(), this.method, this.nOutputs);
		LinkedHashMap<String, Object> nMap = new LinkedHashMap<String, Object>();
		Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		nMap.put("input_" + dateString, input.getData());
		nMap.putAll(kwargs);
		op.setInputs(nMap);
		
		RunMode rm;
		try {
			rm = RunMode.createRunMode(op);
		} catch (Exception e1) {
			e1.printStackTrace();
			return Cast.unchecked(input);
		}
		Map<String, Object> resMap = rm.runOP();
		RandomAccessibleInterval<FloatType> outImg = (RandomAccessibleInterval<FloatType>) resMap.entrySet().stream()
				.map(e -> e.getValue()).collect(Collectors.toList()).get(0);
		return Tensor.build("output", "yx", outImg);
	}

	public void applyInPlace( final Tensor< FloatType > input )
	{
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		PythonTransformation pt = new PythonTransformation();
		//RandomAccessibleInterval<FloatType> img = ArrayImgs.floats(new long[] {1, 1024, 1024, 33});
		String fname = "C:\\Users\\angel\\OneDrive\\Documentos\\pasteur\\git\\deep-icy\\models\\stardist\\test_output.npy";
		RandomAccessibleInterval<FloatType> img = DecodeNumpy.retrieveImgLib2FromNpy(fname);
		Tensor<FloatType> tt = Tensor.build("output", "bcyx", img);
		Tensor<FloatType> out = pt.apply(tt);
		System.out.println();
	}
	
	public void installMamba() throws IOException, InterruptedException, ArchiveException, URISyntaxException {
		this.mambaPath = new File("appose_" + PlatformDetection.getArch()).getAbsolutePath();
		new Conda(mambaPath);
	}
	
	public static void installMamba(String dir) throws IOException, InterruptedException, ArchiveException, URISyntaxException {
		String mambaDir = new File(dir + File.separator + "appose_" + PlatformDetection.getArch()).getAbsolutePath();
		new Conda(mambaDir);
	}
	
	private void installEnv() throws RuntimeException, IOException, 
									InterruptedException, ArchiveException, URISyntaxException {
		Conda conda = new Conda(mambaPath);
		final List< String > cmd = 
				new ArrayList<>( Arrays.asList( "env", "create", "--prefix",
						envPath + File.separator + "envs", "--force", 
						"--file", envYaml, "-y" ) );
		conda.runConda( cmd.stream().toArray( String[]::new ) );
	}
}
