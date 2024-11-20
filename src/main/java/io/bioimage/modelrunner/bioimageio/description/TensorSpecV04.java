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
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;


/**
 * A tensor specification descriptor. It holds the information of an input or output tensor (name, shape, axis order, data type, halo, etc.).
 * It is built from a input or output tensor map element in the yaml file.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class TensorSpecV04 implements TensorSpec {
	/**
	 * Whether the tensor represents an input or an output
	 */
    private final boolean input;
    
    private final Axes axes;
    
    // TODO
    private final InputData data = null;
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
    
    protected String sampleTensorName;
    
    protected String testTensorName;

    /**
     * Builds the tensor specification instance from the tensor map and an input flag.
     * 
     * @param tensorSpecMap
     *        The map of elements describing the tensor.
     * @param input
     *        Whether it is an input (true) or an output (false) tensor.
     */
    protected TensorSpecV04(Map<String, Object> tensorSpecMap, boolean input)
    {
        id = (String) tensorSpecMap.get("name");
        if (tensorSpecMap.get("axes") == null || (tensorSpecMap.get("axes") instanceof List))
        	throw new IllegalArgumentException("Invalid tensor specifications for '" + id
        			+ "'. The axes are incorrectly specified. For more info, visit the Bioimage.io docs.");
        axes = new AxesV04(tensorSpecMap, input);
        description = (String) tensorSpecMap.get("description");
        this.input = input;

        List<?> preprocessingTensors = (List<?>) tensorSpecMap.get("preprocessing");
        if (preprocessingTensors == null)
        {
            preprocessing = new ArrayList<>(0);
        }
        else
        {
            preprocessing = new ArrayList<TransformSpec>(preprocessingTensors.size());
            for (Object elem : preprocessingTensors)
            {
                preprocessing.add(TransformSpec.build((Map<String, Object>) elem));
            }
        }

        List<?> postprocessingTensors = (List<?>) tensorSpecMap.get("postprocessing");
        if (postprocessingTensors == null)
        {
            postprocessing = new ArrayList<>(0);
        }
        else
        {
            postprocessing = new ArrayList<TransformSpec>(postprocessingTensors.size());
            for (Object elem : postprocessingTensors)
            {
                postprocessing.add(TransformSpec.build((Map<String, Object>) elem));
            }
        }
    }
    
    protected void setSampleTensor(String sampleTensorName) {
    	this.sampleTensorName = sampleTensorName;
    }
    
    protected void setTestTensor(String testTensorName) {
    	this.testTensorName = testTensorName;
    }
    
    public String getName() {
    	return this.id;
    }
    
    public String getDescription() {
    	return this.description;
    }
    
    public List<TransformSpec> getPreprocessing(){
    	if (!this.input)
    		return new ArrayList<TransformSpec>();
    	return this.preprocessing;
    }
    
    public List<TransformSpec> getPostprocessing(){
    	if (this.input)
    		return new ArrayList<TransformSpec>();
    	return this.postprocessing;
    }
    
    public String getAxesOrder() {
    	return this.axes.getAxesOrder();
    }
    
    public String getSampleTensorName() {
    	return this.sampleTensorName;
    }
    
    public String getTestTensorName() {
    	return this.testTensorName;
    }
    
    public int[] getMinTileSizeArr() {
    	return this.axes.getMinTileSizeArr();
    }
    
    public int[] getTileStepArr() {
    	return this.axes.getTileStepArr();
    }
    
    public double[] getTileScaleArr() {
    	return this.axes.getTileScaleArr();
    }
    
    public Axes getAxesInfo() {
    	return this.axes;
    }
    
    public String getDataType() {
    	// TODO 
    	//return this.data.toString();
    	return "float32";
    }

	@Override
	public int[] getHaloArr() {
		return this.axes.getHaloArr();
	}

	@Override
	public boolean isImage() {
		if (axes.getAxesOrder().contains("x") && axes.getAxesOrder().contains("y"))
			return true;
		else
			return false;
	}
}
