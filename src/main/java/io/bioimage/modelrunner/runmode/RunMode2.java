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
package io.bioimage.modelrunner.runmode;

import java.util.List;

import io.bioimage.modelrunner.tensor.Tensor;

/**
 * Interface for run modes that contain complex Deep Learning routines.
 *
 * @author Carlos Garcia Lopez de Haro
 *
 */
public interface RunMode2
{

	/**
	 * Applies the run mode transformation to the specified input/s tensor/s.
	 * 
	 * This method will execute the corresponding run mode on the wanted inputs
	 * and return the result provided by the run mode. Regard that the run mode
	 * can be anything. For example tiling + model inference + post-processing;
	 * simply model inference; pre-processing + model inference
	 * 
	 *
	 * @param inputs
	 *            the input/s tensor/s.
	 * @return a list of the result tensors in their corresponding datatypes.
	 */
	public List<Tensor<?>> apply( Tensor< ? >... inputs );

	/**
	 * Returns the name of this run mode.
	 *
	 * @return the name of this run mode.
	 */
	public String getName();
}
