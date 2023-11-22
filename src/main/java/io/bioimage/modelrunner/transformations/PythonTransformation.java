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
import org.apposed.appose.Conda;

import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.runmode.ops.GenericOp;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.YAMLUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;

public class PythonTransformation extends AbstractTensorTransformation
{
	public static final String NAME = "python";
	public static final String ENV_YAML_KEY = "env_yaml";
	
	private String envYaml = "stardist.yaml";
	
	private String script = "stardist_postprocessing.py";
	
	private String method = "stardist_postprocessing";
	
	private String envYamlFilePath;
	
	private String scriptFilePath;
	
	private String mambaDir;
	
	private int nOutputs = 1;
	
	private LinkedHashMap<String, Object> kwargs = new LinkedHashMap<String, Object>();
	{
		kwargs.put("prob_thresh", 0.6924782541382084);
		kwargs.put("nms_thresh", 0.3);
	}

	public PythonTransformation()
	{
		super(NAME);
	}
	
	public void setEnvYamlFilePath(Object envYamlFilePath) {
		if (envYamlFilePath instanceof String) {
			this.envYamlFilePath = (String) envYamlFilePath;
		} else {
			throw new IllegalArgumentException("'envYamlFilePath' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setMambaDir(Object mambaDir) {
		if (mambaDir instanceof String) {
			this.mambaDir = (String) mambaDir;
		} else {
			throw new IllegalArgumentException("'mambaDir' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setScriptFilePath(Object scriptFilePath) {
		if (scriptFilePath instanceof String) {
			this.scriptFilePath = (String) scriptFilePath;
		} else {
			throw new IllegalArgumentException("'scriptFilePath' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setEnvYaml(Object envYaml) {
		if (envYaml instanceof String) {
			this.envYaml = new File(String.valueOf(envYaml)).getName();
		} else {
			throw new IllegalArgumentException("'envYaml' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setScript(Object script) {
		if (script instanceof String) {
			this.script = new File(String.valueOf(script)).getName();
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
		if (kwargs instanceof LinkedHashMap) {
			this.kwargs = (LinkedHashMap<String, Object>) kwargs;
		} else {
			throw new IllegalArgumentException("'kwargs' parameter has to be an instance of "
					+ LinkedHashMap.class
					+ ". The provided argument is an instance of: " + method.getClass());
		}
	}

	public < R extends RealType< R > & NativeType< R > > Tensor<FloatType> apply( final Tensor< R > input )
	{
		envYaml = this.scriptPath + File.separator + envYaml;
		String envName = null;
		try {
			envName = (String) YAMLUtils.load(envYaml).get("name");
		} catch (IOException e2) {
			e2.printStackTrace();
			return Cast.unchecked(input);
		}
		// TODO
		istallEnv();
		GenericOp op = GenericOp.create(envDir, this.script, this.method, this.nOutputs);
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
	
	public static void istallEnv() {
		String envDir = envPath + File.separator + "envs" + File.separator + envName;
		if (!(new File(envDir).isDirectory())) {
				try {
					Conda conda = new Conda(envPath);
					final List< String > cmd = 
							new ArrayList<>( Arrays.asList( "env", "create", "--prefix",
									envPath + File.separator + "envs", "--force", 
									"-n", envName, "--file", envYaml, "-y" ) );
					conda.runConda( cmd.stream().toArray( String[]::new ) );
				} catch (IOException | InterruptedException | ArchiveException | URISyntaxException e1) {
					e1.printStackTrace();
					return Cast.unchecked(input);
				}
		}
	}
}
