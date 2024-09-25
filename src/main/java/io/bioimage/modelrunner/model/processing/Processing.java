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
/**
 * 
 */
package io.bioimage.modelrunner.model.processing;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.TransformSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Class that executes the pre-processing associated to a given tensor
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class Processing {
	/**
	 * Descriptor containing the info about the model
	 */
	private ModelDescriptor descriptor;
	/**
	 * Map containing all the needed input objects to make the processing.
	 * It has to contain the tensor of interest.
	 */
	private Map<String, List<TransformationInstance>> preMap;
	private Map<String, List<TransformationInstance>> postMap;
	//private static BioImageIoPython interp;
	private static String BIOIMAGEIO_PYTHON_TRANSFORMATIONS_WEB = 
						"https://github.com/bioimage-io/core-bioimage-io-python/blob/b0cea"
						+ "c8fa5b412b1ea811c442697de2150fa1b90/bioimageio/core/prediction_pipeline"
						+ "/_processing.py#L105";

	private Processing(ModelDescriptor descriptor) throws IllegalArgumentException, RuntimeException {
		this.descriptor = descriptor;
		buildPreprocessing();
		buildPostprocessing();
	}
	
	private void buildPreprocessing() throws IllegalArgumentException, RuntimeException {
		preMap = new HashMap<String, List<TransformationInstance>>();
		for (TensorSpec tt : this.descriptor.getInputTensors()) {
			List<TransformSpec> preprocessing = tt.getPreprocessing();
			List<TransformationInstance> list = new ArrayList<TransformationInstance>();
			for (TransformSpec transformation : preprocessing) {
				list.add(TransformationInstance.create(transformation));
			}
		}
	}
	
	private void buildPostprocessing() throws IllegalArgumentException, RuntimeException {
		postMap = new HashMap<String, List<TransformationInstance>>();
		for (TensorSpec tt : this.descriptor.getOutputTensors()) {
			List<TransformSpec> preprocessing = tt.getPostprocessing();
			List<TransformationInstance> list = new ArrayList<TransformationInstance>();
			for (TransformSpec transformation : preprocessing) {
				list.add(TransformationInstance.create(transformation));
			}
		}
	}
	
	public static Processing init(ModelDescriptor descriptor) throws IllegalArgumentException, RuntimeException {
		return new Processing(descriptor);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> preprocess(List<Tensor<T>> tensorList){
		return preprocess(tensorList, false);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> preprocess(List<Tensor<T>> tensorList, boolean inplace) {
		for (Entry<String, List<TransformationInstance>> ee : this.preMap.entrySet()) {
			Tensor<T> tt = tensorList.stream().filter(t -> t.getName().equals(ee.getKey())).findFirst().orElse(null);
			if (tt == null)
				continue;
			ee.getValue().forEach(trans -> {
					trans.run(tt, inplace);
			});
		}
		return null;
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> postprocess(List<Tensor<T>> tensorList, boolean inplace) {
		for (Entry<String, List<TransformationInstance>> ee : this.postMap.entrySet()) {
			Tensor<T> tt = tensorList.stream().filter(t -> t.getName().equals(ee.getKey())).findFirst().orElse(null);
			if (tt == null)
				continue;
			ee.getValue().forEach(trans -> {
					trans.run(tt, inplace);
			});
		}
		return null;
	}
}
