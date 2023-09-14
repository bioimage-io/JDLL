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
package io.bioimage.modelrunner.example;

import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.engine.installation.EngineInstall;
import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.LongStream;

import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;

/**
 * This class tries to run every Bioimage.io model available.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ExampleLoadAndRunAllBmzModels {
	
	private static final String CWD = System.getProperty("user.dir");
	private static final String ENGINES_DIR = new File(CWD, "engines").getAbsolutePath();
	private static final String MODELS_DIR = new File(CWD, "models").getAbsolutePath();
	
	private static final List<String> SUPPORTED_FRAMEWORKS;
	static {
		SUPPORTED_FRAMEWORKS = new ArrayList<String>();
		SUPPORTED_FRAMEWORKS.add(ModelWeight.getTensorflowID());
		SUPPORTED_FRAMEWORKS.add(ModelWeight.getOnnxID());
		SUPPORTED_FRAMEWORKS.add(ModelWeight.getTorchscriptID());
	}
	
	/**
	 * Method that installs one engine compatible with the OS and Java version
	 * per DL framework and major version, this is installing Tf1, Tf2, Pytorch 1,
	 * Pytorch 2 and Onnx 17
	 */
	private static void installAllValidEngines() {
		EngineInstall installer = EngineInstall.createInstaller(ENGINES_DIR);
		installer.basicEngineInstallation();
	}

	/**
	 * 
	 * @param args
	 * 	main args, in this case nothing is needed
	 */
	public static void main(String[] args) {
		installAllValidEngines();
		
		BioimageioRepo br = BioimageioRepo.connect();
		Map<Path, ModelDescriptor> bmzModelList = br.listAllModels(false);
		int successModelCount = 0;
		
		for (Entry<Path, ModelDescriptor> modelEntry : bmzModelList.entrySet()) {
			try {
				checkModelCompatibleWithEngines(modelEntry.getValue());
				String modelFolder = br.downloadByName(modelEntry.getValue().getName(), MODELS_DIR);
				loadAndRunModel(modelFolder, modelEntry.getValue());
				successModelCount ++;
			} catch (IllegalArgumentException ex) {
				continue;
			} catch (IOException | InterruptedException e) {
				System.out.println(modelEntry.getValue().getName() 
						+ ": Error downloading model." + e.toString());
			} catch (Exception e) {
				System.out.println(modelEntry.getValue().getName() 
						+ ": Error loading/running model." + e.toString());
			}
		}
		
		System.out.println("Models run without any issue: " 
				+ successModelCount + "/" + bmzModelList.size());
	}
	
	public static void loadAndRunModel(String modelFolder, ModelDescriptor descriptor) throws Exception {
		Model model = Model.createBioimageioModel(modelFolder, ENGINES_DIR);
		model.loadModel();
		List<Tensor<?>> inputs = createInputs(descriptor);
		List<Tensor<?>> outputs = createOutputs(descriptor);
		model.runModel(inputs, outputs);
		for (Tensor<?> tt : outputs) {
			if (tt.isEmpty())
				throw new Exception(descriptor.getName() + ": Output tensor is empty");
		}
		
		model.closeModel();
		inputs.stream().forEach(t -> t.close());
		outputs.stream().forEach(t -> t.close());
	}
	
	private static List<Tensor<?>> createInputs(ModelDescriptor descriptor) {
		List<Tensor<?>> inputs = new ArrayList<Tensor<?>>();
		final ImgFactory< FloatType > imgFactory = new ArrayImgFactory<>( new FloatType() );
		
		for ( TensorSpec it : descriptor.getInputTensors()) {
			String axesStr = it.getAxesOrder();
			String name = it.getName();
			int[] min = it.getShape().getPatchMinimumSize();
			int[] step = it.getShape().getPatchPositionStep();
			long[] imSize = LongStream.range(0, step.length)
					.map(i -> min[(int) i] + step[(int) i]).toArray();
			Tensor<FloatType> tt = Tensor.build(name, axesStr, imgFactory.create(imSize));
			inputs.add(tt);
		}
		return inputs;
	}
	
	private static List<Tensor<?>> createOutputs(ModelDescriptor descriptor) {
		List<Tensor<?>> outputs = new ArrayList<Tensor<?>>();
		
		for ( TensorSpec ot : descriptor.getOutputTensors()) {
			String axesStr = ot.getAxesOrder();
			String name = ot.getName();
			Tensor<?> tt = Tensor.buildEmptyTensor(name, axesStr);
			outputs.add(tt);
		}
		return outputs;
	}
	
	private static void checkModelCompatibleWithEngines(ModelDescriptor descriptor) {
		List<WeightFormat> wws = descriptor.getWeights().getSupportedWeights();
		boolean supported = false;
		boolean pytorch2 = false;
		for (WeightFormat ww : wws) {
			if (!SUPPORTED_FRAMEWORKS.contains(ww.getFramework()))
				continue;
			if (ww.getFramework().equals(ModelWeight.getTorchscriptID()) 
					&& ww.getTrainingVersion().startsWith("2")) {
				pytorch2 = true;
				continue;
			}
			supported = true;
		}
		
		if (!supported && pytorch2)
			throw new IllegalArgumentException(descriptor.getName() 
					+ ": pytorch 2 models cannot run on this test");
		if (!supported)
			throw new IllegalArgumentException(descriptor.getName() 
					+ ": weights not supported");
	}
}
