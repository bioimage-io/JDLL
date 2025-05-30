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
	

    
    public String getName();

    public String getDescription();
    
    public List<TransformSpec> getPreprocessing();
    
    public List<TransformSpec> getPostprocessing();
    
    public String getAxesOrder();
    
    public String getSampleTensorName();
    
    public String getTestTensorName();
    
    public int[] getMinTileSizeArr();
    
    public int[] getTileStepArr();
    
    public int[] getHaloArr();
    
    public double[] getTileScaleArr();
    
    public Axes getAxesInfo();
    
    public String getDataType();
    
    public boolean isImage();
    
    public String getPixelSizeUnit();
}
