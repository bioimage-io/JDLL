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
/**
 * 
 */
package io.bioimage.modelrunner.model.python;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.compress.archivers.ArchiveException;
import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.description.weights.ModelWeight;
import io.bioimage.modelrunner.bioimageio.description.weights.WeightFormat;
import io.bioimage.modelrunner.bioimageio.tiling.TileInfo;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;

/**
 * Class that contains the methods to use a Pytorch model from JDLL using the Bioimage.io model format.
 * The model should be compatible with the default environment (Biapy environment) or the environmemt
 * preferred.
 * 
 * The Bioimage.io model must have weights in the 'pytroch_state_dict' format
 * 
 */
public class BioimageIoModelPytorch extends BioimageIoModelPytorchProtected {
	
	/**
	 * Creates a new BioimageIoModelPytorch.
	 *
	 * @param modelFile the model file.
	 * @param callable the callable.
	 * @param importModule the import module.
	 * @param weightsPath the weights path.
	 * @param kwargs the kwargs.
	 * @param descriptor the descriptor.
	 * @param device the device.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	protected BioimageIoModelPytorch(String modelFile, String callable, String importModule, String weightsPath, Map<String, Object> kwargs,
			ModelDescriptor descriptor, String device) throws BuildException {
		super(modelFile, callable, importModule, weightsPath, kwargs, descriptor, device);
	}

	/**
	 * Create a Bioaimge.io Pytorch model that can be run from JDLL.
	 * The model should have weights in the 'pytroch_state_dict' format
	 *
	 * @param descriptor the descriptor.
	 * @param device the device.
	 * @return the created bioimage I/O model pytorch.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	public static BioimageIoModelPytorch create(ModelDescriptor descriptor, String device) throws BuildException {
		if (descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID()) == null)
			throw new IllegalArgumentException("The model provided does not have weights in the required format, "
					+ ModelWeight.getPytorchID() + ".");
		WeightFormat pytorchWeights = descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID());
		String modelFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getArchitecture().getSource();
		String callable = pytorchWeights.getArchitecture().getCallable();
		String importModule = pytorchWeights.getArchitecture().getImportModule();
		String weightsFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getSource();
		Map<String, Object> kwargs = pytorchWeights.getArchitecture().getKwargs();
		return new BioimageIoModelPytorch(modelFile, callable, importModule, weightsFile, kwargs, descriptor, device);
	}

	/**
	 * Returns the result of create.
	 *
	 * @param descriptor the descriptor.
	 * @return the created bioimage I/O model pytorch.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	public static BioimageIoModelPytorch create(ModelDescriptor descriptor) throws BuildException {
		return create(descriptor, "cpu");
	}


	/**
	 * Create a Bioaimge.io Pytorch model that can be run from JDLL.
	 * The model should have weights in the 'pytroch_state_dict' format
	 *
	 * @param modelPath the model path.
	 * @param device the device.
	 * @return the created bioimage I/O model pytorch.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	public static BioimageIoModelPytorch create(String modelPath, String device) throws IOException, BuildException {
		return create(ModelDescriptorFactory.readFromLocalFile(modelPath + File.separator + Constants.RDF_FNAME), device);
	}

	/**
	 * Returns the result of create.
	 *
	 * @param modelPath the model path.
	 * @return the created bioimage I/O model pytorch.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	public static BioimageIoModelPytorch create(String modelPath) throws IOException, BuildException {
		return create(modelPath, "cpu");
	}
	
	/**
	 * Executes main.
	 *
	 * @param <T> the T type parameter.
	 * @param args command-line arguments.
	 * @throws IOException if an I/O error occurs.
	 * @throws LoadEngineException if the engine cannot be loaded.
	 * @throws RunModelException if model inference cannot be run.
	 * @throws LoadModelException if the model cannot be loaded.
	 * @throws InterruptedException if the current thread is interrupted.
	 * @throws RuntimeException if the operation fails at runtime.
	 * @throws ArchiveException if archive occurs.
	 * @throws URISyntaxException if URI syntax occurs.
	 * @throws BuildException if the Python environment or service cannot be built.
	 */
	public static <T extends NativeType<T> & RealType<T>> void main(String[] args) throws IOException, LoadEngineException, RunModelException, LoadModelException, InterruptedException, RuntimeException, ArchiveException, URISyntaxException, BuildException {
		
		String mm = "/home/carlos/git/deepimagej-plugin/models/OC1 Project 11 Cellpose_24022025_131039";
		Img<T> im = Cast.unchecked(ArrayImgs.floats(new long[] {1, 1, 1024, 1024}));
		List<Tensor<T>> l = new ArrayList<Tensor<T>>();
		l.add(Tensor.build("input", "bcyx", im));
		//BioimageIoModelPytorch.installRequirements();
		BioimageIoModelPytorch model = create(mm, "cpu");
		model.loadModel();
		TileInfo tile = TileInfo.build(l.get(0).getName(), new long[] {1, 1, 512, 512}, 
				l.get(0).getAxesOrderString(), new long[] {1, 1, 512, 512}, l.get(0).getAxesOrderString());
		List<TileInfo> tileList = new ArrayList<TileInfo>();
		tileList.add(tile);
		System.out.println(false);
		
	}

}
