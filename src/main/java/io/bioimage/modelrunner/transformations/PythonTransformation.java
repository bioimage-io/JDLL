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
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.runmode.ops.GenericOp;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.YAMLUtils;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class PythonTransformation extends AbstractTensorTransformation
{
	private static String name = "python";
	
	private String envYaml;
	
	private String script;
	
	private String method;
	
	private int nOutputs;
	
	private LinkedHashMap<String, Object> kwargs;

	public PythonTransformation()
	{
		super(name);
	}
	
	public void setEnvYaml(Object envYaml) {
		if (envYaml instanceof String) {
			this.envYaml = String.valueOf(envYaml);
		} else {
			throw new IllegalArgumentException("'envYaml' parameter has to be an instance of "
					+ String.class
					+ ". The provided argument is an instance of: " + envYaml.getClass());
		}
	}
	
	public void setScript(Object script) {
		if (script instanceof String) {
			this.script = String.valueOf(script);
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
		String envName = (String) YAMLUtils.loadFromString(envYaml).get("name");
		String minicondaBase = Paths.get(System.getProperty("user.home"), ".local", "share", "appose", "miniconda").toString();
		String envPath = minicondaBase + File.separator + "envs" + File.separator + envName;
		if (!(new File(envPath).isDirectory())) {
				// TODO install env
		}
		
		GenericOp op = GenericOp.create(envPath, this.script, this.method, this.nOutputs);
		LinkedHashMap<String, Object> nMap = new LinkedHashMap<String, Object>();
		Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		nMap.put("input_" + dateString, input);
		nMap.putAll(kwargs);
		op.setInputs(nMap);
		
		RunMode rm;
		try {
			rm = RunMode.createRunMode(op);
		} catch (Exception e1) {
			e1.printStackTrace();
			return (Tensor<FloatType>) input;
		}
		Map<String, Object> resMap = rm.runOP();
		return (Tensor<FloatType>) resMap.entrySet().stream()
				.map(e -> e.getValue()).collect(Collectors.toList()).get(0);
	}

	public void applyInPlace( final Tensor< FloatType > input )
	{
	}
}
