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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.exceptions.ModelSpecsException;
import io.bioimage.modelrunner.utils.YAMLUtils;

/**
 * Holds the information of the shape of a tensor. i.e. The size of a patch when processing an image.
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos Garcia Lopez de Haro
 */
public class ShapeSpec
{
    private boolean input;

    private int[] patchRecomendedSize;
    private int[] patchMinimumSize;
    private int[] patchPositionStep;
    private int[] tensorShape;
    private boolean fixedSize = false;

    private String referenceInput;
    private float[] scale;
    private float[] offset;

    /**
     * Creates an instance of the shape specification included in the provided shape element.
     * It can be a {@link List} of {@link Integer} (fixed size) or a{@link Map} where the keys
     * are Strings and the values Objects (adjusted size).
     * The list of intergers contains the fixed size for the input. The map contains the minimum
     * size it can have and the step per dimension, giving instructions to build the input.
     * 
     * @param shapeElem
     *        The shape element object
     * @param input
     *        True if it is an input shape. False if it's an output shape descriptor.
     * @return The shape specification instance.
     * @throws ModelSpecsException if any of the rdf.yaml does not comply the requirements
     */
    @SuppressWarnings("unchecked")
    public static ShapeSpec build(Object shapeElem, boolean input) throws ModelSpecsException
    {
        ShapeSpec shape = new ShapeSpec();
        shape.input = input;
        if (shapeElem instanceof List<?>)
        {
            shape.patchRecomendedSize = YAMLUtils.castListToIntArray((List<?>) shapeElem);
            if (input)
            {
                shape.patchMinimumSize = Arrays.copyOf(shape.patchRecomendedSize, shape.patchRecomendedSize.length);
                shape.tensorShape = Arrays.copyOf(shape.patchRecomendedSize, shape.patchRecomendedSize.length);
                shape.patchPositionStep = new int[shape.patchRecomendedSize.length];
            }
            else
            {
            	// If scale and offset are not defined in the yaml, create the default (non-affecting) ones
                shape.scale = new float[shape.patchRecomendedSize.length];
                shape.offset = new float[shape.patchRecomendedSize.length];
                for (int i = 0; i < shape.scale.length; i ++) {shape.scale[i] = 1;}
            }
            shape.fixedSize = true;
        }
        else if (shapeElem instanceof HashMap<?, ?>)
        {
            Map<String, Object> map = (Map<String, Object>) shapeElem;
            if (input)
            {
                shape.patchMinimumSize = YAMLUtils.castListToIntArray((List<?>) map.get("min"));
                shape.patchPositionStep = YAMLUtils.castListToIntArray((List<?>) map.get("step"));
                shape.patchRecomendedSize = new int[shape.patchMinimumSize.length];
                shape.tensorShape = new int[shape.patchMinimumSize.length];
                for (int i = 0; i < shape.patchPositionStep.length; i++)
                {
                    if (shape.patchPositionStep[i] == 0)
                        shape.tensorShape[i] = shape.patchMinimumSize[i];
                    else
                        shape.tensorShape[i] = -1;
                }
            }
            else
            {
                shape.referenceInput = (String) map.get("reference_tensor");
                shape.scale = YAMLUtils.castListToFloatArray((List<?>) map.get("scale"));
                shape.setOffset(YAMLUtils.castListToFloatArray((List<Integer>) map.get("offset")));
            }
            shape.fixedSize = false;
        } else {
        	throw new IllegalArgumentException("The shape of a tensor has to defined either with a map "
        			+ "of specifications or with a fixed array shape.");
        }
        return shape;
    }

    private ShapeSpec()
    {
        // Only used by build method.
    }

    /**
     * @return Whether this is the shape of an input (true) or output (false) tensor.
     */
    public boolean isInput()
    {
        return input;
    }

    /**
     * @return The int array of the recommended tensor size.
     */
    protected int[] getTileRecomendedSize()
    {
        return patchRecomendedSize;
    }

    /**
     * @return The int array with the minimum valid tensor size.
     */
    public int[] getTileMinimumSize()
    {

        return patchMinimumSize;
    }

    /**
     * @return The int array with the valid tensor size step.
     */
    public int[] getTileStep()
    {
        return patchPositionStep;
    }

    /**
     * @return The reference input tensor associated to this output tensor.
     */
    public String getReferenceInput()
    {
        return referenceInput;
    }

    /**
     * @return The output/input pixel scale for each dimension.
     */
    public float[] getScale()
    {
        return scale;
    }
    
    /**
     * Set the output offset for the output shape specs. The ofset should
     * always be divisible by 0.5
     * @param offset
     * 	the output offset
     * @throws ModelSpecsException if the offset is not divisible by 0.5
     */
    public void setOffset(float[] offset) throws ModelSpecsException  {
    	for (float ff : offset) {
    		if (ff % 0.5 != 0) {
    			throw new ModelSpecsException("Invalid output specifications. "
    					+ "Every output offset should be divisible by 0.5");
    		}
    	}
    	this.offset = offset;
    }

    /**
     * @return The Position of the output patch origin with respect to input patch
     */
    public float[] getOffset()
    {
        return offset;
    }

    /**
     * @return Whether the input tensor is defined as a single patch size or a pair of min size and step size.
     */
    public boolean isFixedSize()
    {
        return fixedSize;
    }

    @Override
    public String toString()
    {
        return "ShapeSpec {input=" + input + ", tileRecommendedSize=" + Arrays.toString(patchRecomendedSize)
                + ", tileMinSize=" + Arrays.toString(patchMinimumSize) + ", tileStep="
                + Arrays.toString(patchPositionStep) + ", referenceInput=" + referenceInput + ", scale="
                + Arrays.toString(scale) + ", offset=" + Arrays.toString(offset) + ", fixedSize=" + fixedSize + "}";
    }

}
