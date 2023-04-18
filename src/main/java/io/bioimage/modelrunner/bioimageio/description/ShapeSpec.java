package io.bioimage.modelrunner.bioimageio.description;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
     * It can be a List<Integer> (fixed size) or a Map<String, Object> (adjusted size).
     * 
     * @param shapeElem
     *        The shape element object
     * @param input
     *        True if it is an input shape. False if it's an output shape descriptor.
     * @return The shape specification instance.
     * @throws Exception if any of the rdf.yaml does not comply the requirements
     */
    @SuppressWarnings("unchecked")
    public static ShapeSpec build(Object shapeElem, boolean input) throws Exception
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
    public int[] getPatchRecomendedSize()
    {
        return patchRecomendedSize;
    }

    /**
     * @return The int array with the minimum valid tensor size.
     */
    public int[] getPatchMinimumSize()
    {

        return patchMinimumSize;
    }

    /**
     * @return The int array with the valid tensor size step.
     */
    public int[] getPatchPositionStep()
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
     * @throws Exception if the offset is not divisible by 0.5
     */
    public void setOffset(float[] offset) throws Exception {
    	for (float ff : offset) {
    		if (ff % 0.5 != 0) {
    			throw new Exception("Invalid output specifications. "
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
        return "ShapeSpec {input=" + input + ", patchRecommendedSize=" + Arrays.toString(patchRecomendedSize)
                + ", patchMinSize=" + Arrays.toString(patchMinimumSize) + ", patchPositionStep="
                + Arrays.toString(patchPositionStep) + ", referenceInput=" + referenceInput + ", scale="
                + Arrays.toString(scale) + ", offset=" + Arrays.toString(offset) + ", fixedSize=" + fixedSize + "}";
    }

}
