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
import net.imglib2.util.Cast;

/**
 * Class that executes the pre- or post-processing associated to a given tensor.
 * This class manages the preprocessing and postprocessing steps for tensors
 * based on the specifications provided in valid rdf.yaml Bioimage.io specs file.
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
			preMap.put(tt.getName(), list);
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
			postMap.put(tt.getName(), list);
		}
	}
    
    /**
     * Initializes and returns a new Processing object.
     * 
     * @param descriptor
     *  The {@link ModelDescriptor} object created from the Bioimage.io rdf spec file containing model information.
     * @return A new {@link Processing} object.
     * @throws IllegalArgumentException If there's an issue with the model descriptor.
     * @throws RuntimeException If there's an error during initialization.
     */
	public static Processing init(ModelDescriptor descriptor) throws IllegalArgumentException, RuntimeException {
		return new Processing(descriptor);
	}
    
    /**
     * Applies preprocessing to a list of tensors.
     * If the tensors do not correspond to the input tensors of the model, nothing happens to them.
     * 
     * The method output is a separate list of tensors. In order to apply the 
     * pre-processing in-place, use {@link #preprocess(List, boolean)}.
     * 
     * @param <T>
     *  The type of the input tensor elements.
     * @param <R>
     *  The type of the output tensor elements.
     * @param tensorList
     *  The list of tensors to preprocess.
     * @return A list of preprocessed tensors.
     */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> preprocess(List<Tensor<T>> tensorList){
		return preprocess(tensorList, false);
	}
	
	/**
     * Applies preprocessing to a list of tensors with an option for in-place operations.
     * 
     * @param <T> The type of the input tensor elements.
     * @param <R> The type of the output tensor elements.
     * @param tensorList The list of tensors to preprocess.
     * @param inplace If true, preprocessing is done in-place; otherwise, new tensors are created.
     * @return A list of preprocessed tensors.
     */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> preprocess(List<Tensor<T>> tensorList, boolean inplace) {
		List<Tensor<R>> outputs = new ArrayList<Tensor<R>>();
		if (preMap.entrySet().size() == 0) return Cast.unchecked(tensorList);
		for (Entry<String, List<TransformationInstance>> ee : this.preMap.entrySet()) {
			Tensor<T> tt = tensorList.stream().filter(t -> t.getName().equals(ee.getKey())).findFirst().orElse(null);
			if (tt == null)
				continue;
			if (ee.getValue().size() == 0)
				outputs.add(Cast.unchecked(tt));
			for (TransformationInstance trans : ee.getValue()) {
				List<Tensor<R>> outList = trans.run(tt, inplace);
				outputs.addAll(outList);
			}
		}
		return outputs;
	}
    
    /**
     * Applies postprocessing to a list of tensors.
     * If the tensors do not correspond to the output tensors of the model, nothing happens to them.
     * 
     * The method output is a separate list of tensors. In order to apply the 
     * pre-processing in-place, use {@link #postprocess(List, boolean)}.
     * 
     * @param <T>
     *  The type of the input tensor elements.
     * @param <R>
     *  The type of the output tensor elements.
     * @param tensorList
     *  The list of tensors to postprocess.
     * @return A list of postprocessed tensors.
     */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> postprocess(List<Tensor<T>> tensorList){
		return postprocess(tensorList, false);
	}

    /**
     * Applies postprocessing to a list of tensors with an option for in-place operations.
     * 
     * @param <T> The type of the input tensor elements.
     * @param <R> The type of the output tensor elements.
     * @param tensorList The list of tensors to postprocess.
     * @param inplace If true, postprocessing is done in-place; otherwise, new tensors are created.
     * @return A list of postprocessed tensors.
     */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> postprocess(List<Tensor<T>> tensorList, boolean inplace) {
		List<Tensor<R>> outputs = new ArrayList<Tensor<R>>();
		if (postMap.entrySet().size() == 0) return Cast.unchecked(tensorList);
		for (Entry<String, List<TransformationInstance>> ee : this.postMap.entrySet()) {
			Tensor<T> tt = tensorList.stream().filter(t -> t.getName().equals(ee.getKey())).findFirst().orElse(null);
			if (tt == null)
				continue;
			if (ee.getValue().size() == 0)
				outputs.add(Cast.unchecked(tt));
			for (TransformationInstance trans : ee.getValue()) {
				List<Tensor<R>> outList = trans.run(tt, inplace);
				outputs.addAll(outList);
			}
		}
		return outputs;
	}
}
