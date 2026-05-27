/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.model.special.stardist;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apposed.appose.BuildException;
import org.apposed.appose.TaskException;

import io.bioimage.modelrunner.bioimageio.BioimageioDirectConnection;
import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Implementation of an API to run Stardist 2D models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *
 *@author Carlos Garcia
 */
public class Stardist2D extends StardistAbstract {
	
	private static String MODULE_NAME = "StarDist2D";
	
	private static final Map<String, String> PRETRAINED_EQUIVALENCE;
	static {
		PRETRAINED_EQUIVALENCE = new HashMap<String, String>();
		PRETRAINED_EQUIVALENCE.put("2D_versatile_he", "StarDist H&E Nuclei Segmentation");
		PRETRAINED_EQUIVALENCE.put("2D_versatile_fluo", "StarDist Fluorescence Nuclei Segmentation");
	}
	
	private static final Map<String, String> ID_EQUIVALENCE;
	static {
		ID_EQUIVALENCE = new HashMap<String, String>();
		ID_EQUIVALENCE.put("StarDist H&E Nuclei Segmentation", "chatty-frog");
		ID_EQUIVALENCE.put("StarDist Fluorescence Nuclei Segmentation", "fearless-crab");
	}
	
	/**
	 * Creates a new Stardist2D.
	 *
	 * @param modelName the modelName parameter.
	 * @param baseDir the baseDir parameter.
	 * @param config the config parameter.
	 * @throws IOException if an I/O error occurs.
	 * @throws BuildException 
	 */
	protected Stardist2D(String modelPath, Map<String, Object> configMap) throws IOException, BuildException {
		super(modelPath, configMap);
		this.scaleRangeAxes = "xyc";
	}

	/**
	 * Creates imports code.
	 *
	 * @return the resulting string.
	 */
	@Override
	protected String createImportsCode() {
		return String.format(LOAD_MODEL_CODE_ABSTRACT, MODULE_NAME, MODULE_NAME, 
				MODULE_NAME, MODULE_NAME, MODULE_NAME, this.name, this.basedir);
	}

	/**
	 * Checks input.
	 *
	 * @param image the image parameter.
	 */
	@Override
	protected <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image) {
		if (image.dimensionsAsLongArray().length == 2 && this.nChannels != 1)
			throw new IllegalArgumentException("Stardist2D needs an image with three dimensions: XYC");
		else if (image.dimensionsAsLongArray().length != 3 && this.nChannels != 1)
			throw new IllegalArgumentException("Stardist2D needs an image with three dimensions: XYC");
		else if (image.dimensionsAsLongArray().length != 2 && image.dimensionsAsLongArray()[2] != nChannels)
			throw new IllegalArgumentException("This Stardist2D model requires " + nChannels + " channels.");
		else if (image.dimensionsAsLongArray().length > 3 || image.dimensionsAsLongArray().length < 2)
			throw new IllegalArgumentException("Stardist2D model requires an image with dimensions XYC.");
	}
	
	/**
	 * Executes reconstruct mask.
	 *
	 * @return the resulting value.
	 * @throws IOException if an I/O error occurs.
	 */
	@Override
	protected <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() {
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> mask = shma.getSharedRAI();
		RandomAccessibleInterval<T> maskCopy;
		if (this.nChannels == 1) {
			maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(mask), 
					Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
			shma.close();
		} else {
			long[] maxPos = mask.maxAsLongArray();
			maxPos[2] = 0;
			IntervalView<T> maskInterval = Views.interval(mask, mask.minAsLongArray(), maxPos);
			maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(maskInterval), 
					Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
			shma.close();
		}
		return maskCopy;
	}

	/**
	 * Checks whether 2 d.
	 *
	 * @return true if the operation succeeds; otherwise, false.
	 */
	@Override
	public boolean is2D() {
		return true;
	}

	/**
	 * Checks whether 3 d.
	 *
	 * @return true if the operation succeeds; otherwise, false.
	 */
	@Override
	public boolean is3D() {
		return false;
	}
	
	/**
	 * Downloads pretrained.
	 *
	 * @param modelName the modelName parameter.
	 * @param downloadDir the downloadDir parameter.
	 * @return the resulting string.
	 * @throws ExecutionException if a ExecutionException occurs while executing this method.
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 * @throws IOException if an I/O error occurs.
	 */
	public static String downloadPretrained(String modelName, String downloadDir) 
			throws ExecutionException, InterruptedException, IOException {
		return downloadPretrained(modelName, downloadDir, null);
	}
	
	/**
	 * Downloads pretrained.
	 *
	 * @param modelName the modelName parameter.
	 * @param downloadDir the downloadDir parameter.
	 * @param progressConsumer the progressConsumer parameter.
	 * @return the resulting string.
	 * @throws InterruptedException if the current thread is interrupted while waiting for the operation to finish.
	 * @throws IOException if an I/O error occurs.
	 */
	public static String downloadPretrained(String modelName, String downloadDir, Consumer<Double> progressConsumer) throws InterruptedException, IOException {
		if (!PRETRAINED_EQUIVALENCE.keySet().contains(modelName))
			return downloadPretrainedBioimageio(modelName, downloadDir, progressConsumer);
		else
			return downloadPretrainedBioimageio(PRETRAINED_EQUIVALENCE.get(modelName), downloadDir, progressConsumer);
	}
	
	private static String downloadPretrainedBioimageio(String modelName, String downloadDir, Consumer<Double> progressConsumer) 
			throws InterruptedException, IOException {
		
		if (ID_EQUIVALENCE.get(modelName) != null)
			modelName = ID_EQUIVALENCE.get(modelName);
		ModelDescriptor descriptor = BioimageioDirectConnection.selectByID(modelName);
		if (descriptor == null) {
			throw new IllegalArgumentException("The model does not correspond to one of the available pretrained StarDist2D models."
					+ " To find a list of available cellpose models, please run StarDist2D.getPretrainedList()");
		}
		return BioimageioRepo.downloadModel(descriptor, downloadDir, progressConsumer);
	}
}
