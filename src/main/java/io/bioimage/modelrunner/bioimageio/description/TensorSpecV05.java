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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.tiling.PatchGridCalculator;
import io.bioimage.modelrunner.utils.YAMLUtils;


/**
 * A tensor specification descriptor. It holds the information of an input or output tensor (name, shape, axis order, data type, halo, etc.).
 * It is built from a input or output tensor map element in the yaml file.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class TensorSpecV05 {
	/**
	 * Whether the tensor represents an input or an output
	 */
    private final boolean input;
    
    private final Axes axes;
    /**
     * The name of the tensor
     */
    private final String id;
    /**
     * The description of the tensor
     */
    private final String description;
    /**
     * The list of pre-processing routines
     */
    private List<TransformSpec> preprocessing;
    /**
     * The list of post-processing routines
     */
    private List<TransformSpec> postprocessing;

    /**
     * Builds the tensor specification instance from the tensor map and an input flag.
     * 
     * @param tensorSpecMap
     *        The map of elements describing the tensor.
     * @param input
     *        Whether it is an input (true) or an output (false) tensor.
     * @return The tensor specification instance.
     * @throws ModelSpecsException if any of the fields does not fulfill the requirements
     */
    protected TensorSpecV05(Map<String, Object> tensorSpecMap, boolean input) throws ModelSpecsException
    {
        id = (String) tensorSpecMap.get("name");
        if (tensorSpecMap.get("axes") == null || (tensorSpecMap.get("axes") instanceof List))
        	throw new IllegalArgumentException("Invalid tensor specifications for '" + tensor.id
        			+ "'. The axes are incorrectly specified. For more info, visit the Bioimage.io docs.");
        axes = new Axes((List<Object>) tensorSpecMap.get("axes"));
        description = (String) tensorSpecMap.get("description");
        this.input = input;

        List<?> preprocessingTensors = (List<?>) tensorSpecMap.get("preprocessing");
        if (preprocessingTensors == null)
        {
            tensor.preprocessing = new ArrayList<>(0);
        }
        else
        {
            tensor.preprocessing = new ArrayList<TransformSpec>(preprocessingTensors.size());
            for (Object elem : preprocessingTensors)
            {
                tensor.preprocessing.add(TransformSpec.build((Map<String, Object>) elem));
            }
        }

        List<?> postprocessingTensors = (List<?>) tensorSpecMap.get("postprocessing");
        if (postprocessingTensors == null)
        {
            tensor.postprocessing = new ArrayList<>(0);
        }
        else
        {
            tensor.postprocessing = new ArrayList<TransformSpec>(postprocessingTensors.size());
            for (Object elem : postprocessingTensors)
            {
                tensor.postprocessing.add(TransformSpec.build((Map<String, Object>) elem));
            }
        }

        return tensor;
    }
}
