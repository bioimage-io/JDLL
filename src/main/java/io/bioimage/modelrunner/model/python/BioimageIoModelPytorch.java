/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
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

import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
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

public class BioimageIoModelPytorch extends BioimageIoModelPytorchProtected {
	
	protected BioimageIoModelPytorch(String modelFile, String callable, String weightsPath, Map<String, Object> kwargs,
			ModelDescriptor descriptor) throws IOException {
		super(modelFile, callable, weightsPath, kwargs, descriptor);
	}

	public static BioimageIoModelPytorch create(ModelDescriptor descriptor) throws IOException {
		if (descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID()) == null)
			throw new IllegalArgumentException("The model provided does not have weights in the required format, "
					+ ModelWeight.getPytorchID() + ".");
		WeightFormat pytorchWeights = descriptor.getWeights().getModelWeights(ModelWeight.getPytorchID());
		String modelFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getArchitecture().getSource();
		String callable = pytorchWeights.getArchitecture().getCallable();
		String weightsFile = descriptor.getModelPath() +  File.separator + pytorchWeights.getSource();
		Map<String, Object> kwargs = pytorchWeights.getArchitecture().getKwargs();
		return new BioimageIoModelPytorch(modelFile, callable, weightsFile, kwargs, descriptor);
	}
	
	public static BioimageIoModelPytorch create(String modelPath) throws IOException {
		return create(ModelDescriptorFactory.readFromLocalFile(modelPath + File.separator + Constants.RDF_FNAME));
	}
	
	/**
	 * 
	 * @param <T>
	 * 	nothing
	 * @param args
	 * 	nothing
	 * @throws IOException	nothing
	 * @throws LoadEngineException	nothing
	 * @throws RunModelException	nothing
	 * @throws LoadModelException	nothing
	 * @throws URISyntaxException 
	 * @throws ArchiveException 
	 * @throws MambaInstallException 
	 * @throws RuntimeException 
	 * @throws InterruptedException 
	 */
	public static <T extends NativeType<T> & RealType<T>> void main(String[] args) throws IOException, LoadEngineException, RunModelException, LoadModelException, InterruptedException, RuntimeException, MambaInstallException, ArchiveException, URISyntaxException {
		
		String mm = "/home/carlos/git/deepimagej-plugin/models/OC1 Project 11 Cellpose_24022025_131039";
		Img<T> im = Cast.unchecked(ArrayImgs.floats(new long[] {1, 1, 1024, 1024}));
		List<Tensor<T>> l = new ArrayList<Tensor<T>>();
		l.add(Tensor.build("input", "bcyx", im));
		//BioimageIoModelPytorch.installRequirements();
		BioimageIoModelPytorch model = create(mm);
		List<String> missing = model.findMissingDependencies();
		model.loadModel();
		TileInfo tile = TileInfo.build(l.get(0).getName(), new long[] {1, 1, 512, 512}, 
				l.get(0).getAxesOrderString(), new long[] {1, 1, 512, 512}, l.get(0).getAxesOrderString());
		List<TileInfo> tileList = new ArrayList<TileInfo>();
		tileList.add(tile);
		model.run(l);
		System.out.println(false);
		
	}

}
