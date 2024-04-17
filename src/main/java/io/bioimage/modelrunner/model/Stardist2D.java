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
package io.bioimage.modelrunner.model;

import java.io.File;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.apposed.appose.Mamba;
import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.runmode.RunMode;
import io.bioimage.modelrunner.runmode.ops.GenericOp;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.versionmanagement.InstalledEngines;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class Stardist2D {
	
	private static final List<String> STARDIST_DEPS = Arrays.asList(new String[] {"python=3.10", "stardist", "numpy"});
	
	private static final String STARDIST2D_PATH_IN_RESOURCES = "ops/stardist_postprocessing/";
	
	private static final String STARDIST2D_SCRIPT_NAME= "stardist_postprocessing.py";
	
	private static final String STARDIST2D_METHOD_NAME= "stardist_postprocessing";
	
	public static <T extends RealType<T> & NativeType<T>> 
	RandomAccessibleInterval<T> stardist(String modelPath, RandomAccessibleInterval<T> image, 
			float prob_threshold, float nms_threshold, boolean install) 
			throws Exception {
		
		boolean installed = InstalledEngines.buildEnginesFinder()
				.checkEngineWithArgsInstalledForOS("tensorflow", "1.15.0", null, null).size() != 0;
		if (!installed && !install)
			throw new IllegalArgumentException("Tensorflow is required but it is not installed. "
					+ "Install it manually or set argument 'install=true'");
		else if (!installed)
			EngineInstall.installEngineWithArgs("tensorflow", "1.15", true, true);
		
		EngineInfo tfJavaEngine  = EngineInfo.defineDLEngine("tensorflow", "1.15");
		Model model = Model.createDeepLearningModel(modelPath, modelPath, tfJavaEngine);
		
		Tensor<T> inputTensor = Tensor.build(modelPath, modelPath, image);
		
		List<Tensor<T>> inputs = new ArrayList<Tensor<T>>();
		model.runModel(null, null);
		
		Mamba mamba = new Mamba();
		
		boolean stardistPythonInstalled = false;
		try {
			stardistPythonInstalled = mamba.checkAllDependenciesInEnv("stardist", STARDIST_DEPS);
		} catch (MambaInstallException e) {
		}
		if (!stardistPythonInstalled)
			throw new IllegalArgumentException();
		
		String envPath = mamba.getEnvsDir() + File.separator + "stardist";
		String scriptPath = envPath + File.separator + STARDIST2D_SCRIPT_NAME;
		if (!Paths.get(scriptPath).toFile().isFile()) {
			
		}
		
		GenericOp op = GenericOp.create(envPath, scriptPath, STARDIST2D_METHOD_NAME, 1);
		LinkedHashMap<String, Object> nMap = new LinkedHashMap<String, Object>();
		Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("ddMMYYYY_HHmmss");
		String dateString = sdf.format(cal.getTime());
		nMap.put("input_" + dateString, image);
		Map<String, Object> kwargs = new LinkedHashMap<String, Object>();
		kwargs.put("nms_thresh", nms_threshold);
		kwargs.put("prob_thresh", prob_threshold);
		op.setInputs(nMap);
		
		RunMode rm;
		rm = RunMode.createRunMode(op);
		Map<String, Object> resMap = rm.runOP();
		
		List<RandomAccessibleInterval<T>> rais = resMap.entrySet().stream()
				.filter(e -> {
					Object val = e.getValue();
					if (val instanceof RandomAccessibleInterval) return true;
					return false;
				}).map(rai -> (RandomAccessibleInterval<T>) rai).collect(Collectors.toList());
		
		return rais.get(0);
	}
	
	public void checkStardistInstalled() {
		
	}
	
	private void install() {
		
	}
}
