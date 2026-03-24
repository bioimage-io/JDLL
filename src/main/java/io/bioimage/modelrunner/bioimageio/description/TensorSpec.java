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
package io.bioimage.modelrunner.bioimageio.description;

import java.util.List;


/**
 * A tensor specification descriptor. It holds the information of an input or output tensor (name, shape, axis order, data type, halo, etc.).
 * It is built from a input or output tensor map element in the yaml file.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public interface TensorSpec {
	

    
    /**
     * Gets name.
     *
     * @return the resulting string.
     */
    public String getName();

    /**
     * Gets description.
     *
     * @return the resulting string.
     */
    public String getDescription();
    
    /**
     * Gets preprocessing.
     *
     * @return the resulting list.
     */
    public List<TransformSpec> getPreprocessing();
    
    /**
     * Gets postprocessing.
     *
     * @return the resulting list.
     */
    public List<TransformSpec> getPostprocessing();
    
    /**
     * Gets axes order.
     *
     * @return the resulting string.
     */
    public String getAxesOrder();
    
    /**
     * Gets sample tensor name.
     *
     * @return the resulting string.
     */
    public String getSampleTensorName();
    
    /**
     * Gets test tensor name.
     *
     * @return the resulting string.
     */
    public String getTestTensorName();
    
    /**
     * Gets min tile size arr.
     *
     * @return the resulting array.
     */
    public int[] getMinTileSizeArr();
    
    /**
     * Gets tile step arr.
     *
     * @return the resulting array.
     */
    public int[] getTileStepArr();
    
    /**
     * Gets halo arr.
     *
     * @return the resulting array.
     */
    public int[] getHaloArr();
    
    /**
     * Gets tile scale arr.
     *
     * @return the resulting array.
     */
    public double[] getTileScaleArr();
    
    /**
     * Gets axes info.
     *
     * @return the resulting value.
     */
    public Axes getAxesInfo();
    
    /**
     * Gets data type.
     *
     * @return the resulting string.
     */
    public String getDataType();
    
    /**
     * Checks whether image.
     *
     * @return true if the operation succeeds; otherwise, false.
     */
    public boolean isImage();
    
    /**
     * Gets pixel size unit.
     *
     * @return the resulting string.
     */
    public String getPixelSizeUnit();
}
