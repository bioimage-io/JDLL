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
package io.bioimage.modelrunner.model.special.stardist;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
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

/**
 * Implementation of an API to run Stardist 3D models out of the box with little configuration.
 * 
 *TODO add fine tuning
 *
 *@author Carlos Garcia
 */
public class Stardist3D extends StardistAbstract {
	
	private static String MODULE_NAME = "StarDist3D";
	
	private static final Map<String, String> ID_EQUIVALENCE;
	static {
		ID_EQUIVALENCE = new HashMap<String, String>();
		ID_EQUIVALENCE.put("StarDist Plant Nuclei 3D ResNet", "modest-octopus");
	}
	
	protected Stardist3D(String modelName, String baseDir, Map<String, Object> config) throws IOException {
		super(modelName, baseDir, config);
		this.scaleRangeAxes = "zyxc";
	}
	
	private Stardist3D(String modelName, String baseDir) throws IOException {
		super(modelName, baseDir);
		String axes = ((String) config.get("axes")).toUpperCase();
		if (!axes.contains("Z"))
			throw new IllegalArgumentException("Trying to instantiate a StarDist2D model."
					+ " Please use Stardist2D instead of Stardist3D.");
		this.scaleRangeAxes = "xyzc";
	}
	
	private Stardist3D(ModelDescriptor descriptor) throws IOException {
		super(descriptor);
		String axes = ((String) config.get("axes")).toUpperCase();
		if (!axes.contains("Z"))
			throw new IllegalArgumentException("Trying to instantiate a StarDist2D model."
					+ " Please use Stardist2D instead of Stardist3D.");
		this.scaleRangeAxes = "xyzc";
	}

	@Override
	protected String createImportsCode() {
		return String.format(LOAD_MODEL_CODE_ABSTRACT, MODULE_NAME, MODULE_NAME, 
				MODULE_NAME, MODULE_NAME, MODULE_NAME, this.name, this.basedir);
	}

	@Override
	protected <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image) {
		if (image.dimensionsAsLongArray().length == 3 && this.nChannels != 1)
			throw new IllegalArgumentException("Stardist3D needs an image with four dimensions: XYZC");
		else if (image.dimensionsAsLongArray().length != 4 && this.nChannels != 1)
			throw new IllegalArgumentException("Stardist3D needs an image with four dimensions: XYZC");
		else if (image.dimensionsAsLongArray().length == 4 && image.dimensionsAsLongArray()[3] != nChannels)
			throw new IllegalArgumentException("This Stardist3D model requires " + nChannels + " channels.");
		else if (image.dimensionsAsLongArray().length > 4 || image.dimensionsAsLongArray().length < 2)
			throw new IllegalArgumentException("Stardist3D model requires an image with dimensions XYZC.");
	}
	

	
	@Override
	protected <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() throws IOException {
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(shma.getSharedRAI()), 
				Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
		shma.close();
		return maskCopy;
	}

	@Override
	public boolean is2D() {
		return false;
	}

	@Override
	public boolean is3D() {
		return true;
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param descriptor
	 * 	the bioimage.io model descriptor
	 * @return an instance of a Stardist2D model ready to be used     * @throws FileNotFoundException If the model file is not found.
     * @throws IOException If there's an I/O error.
	 */
	public static Stardist3D fromBioimageioModel(ModelDescriptor descriptor) throws IOException {
		if (!descriptor.getConfig().getSpecMap().keySet().contains("stardist"))
			throw new IllegalArgumentException("This Bioimage.io model does not correspond to a StarDist model.");
		if (!descriptor.getModelFamily().equals(ModelDescriptor.STARDIST))
			throw new RuntimeException("Please first install StarDist with 'StardistAbstract.installRequirements()'");
		if (!descriptor.getInputTensors().get(0).getAxesOrder().contains("z"))
			throw new IllegalArgumentException("This StarDist model is not 3D");
		return new Stardist3D(descriptor);
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 2D models.
	 * By default, the model will be installed in the "models" folder inside the application
	 * @param pretrainedModel
	 * 	the name of the pretrained model. 
	 * @param install
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 */
	public static Stardist3D fromPretained(String pretrainedModel, boolean install) throws IOException, InterruptedException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), install);
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 3D models
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param installDir
	 * 	the directory where the model wants to be installed
	 * @param install
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist3D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 */
	public static Stardist3D fromPretained(String pretrainedModel, String installDir, boolean install) throws IOException, 
																					InterruptedException {
		if (pretrainedModel.equals("StarDist Plant Nuclei 3D ResNet") && !install) {
			ModelDescriptor md = ModelDescriptorFactory.getModelsAtLocalRepo(installDir).stream()
					.filter(mm ->mm.getName().equals(pretrainedModel)).findFirst().orElse(null);
			if (md != null) return new Stardist3D(md);
			return null;
		} else if (pretrainedModel.equals("StarDist Plant Nuclei 3D ResNet")) {
			String path = BioimageioRepo.downloadModel(BioimageioDirectConnection.selectByID("modest-octopus"), installDir);
			return Stardist3D.fromBioimageioModel(ModelDescriptorFactory.readFromLocalFile(path));
		} else {
			throw new IllegalArgumentException("There is no Stardist3D model called: " + pretrainedModel);
		}
	}
	
	public static String downloadPretrained(String modelName, String downloadDir) 
			throws ExecutionException, InterruptedException, IOException {
		return downloadPretrained(modelName, downloadDir, null);
	}
	
	public static String downloadPretrained(String modelName, String downloadDir, Consumer<Double> progressConsumer) throws InterruptedException, IOException {
		return downloadPretrainedBioimageio(modelName, downloadDir, progressConsumer);
	}
	
	private static String downloadPretrainedBioimageio(String modelName, String downloadDir, Consumer<Double> progressConsumer) 
			throws InterruptedException, IOException {
		

		if (ID_EQUIVALENCE.get(modelName) != null)
			modelName = ID_EQUIVALENCE.get(modelName);
		ModelDescriptor descriptor = BioimageioDirectConnection.selectByID(modelName);
		if (descriptor == null) {
			throw new IllegalArgumentException("The model does not correspond to on of the available pretrained StarDist3D models."
					+ " To find a list of available cellpose models, please run StarDist3D.getPretrainedList()");
		}
		return BioimageioRepo.downloadModel(descriptor, downloadDir, progressConsumer);
	}
	
	
	
	/**
	 * Main method to check functionality
	 * @param args
	 * 	nothing
	 * @throws IOException nothing
	 * @throws InterruptedException nothing
	 * @throws RuntimeException nothing
	 * @throws MambaInstallException nothing
	 * @throws LoadEngineException nothing
	 * @throws RunModelException nothing
	 * @throws ArchiveException nothing
	 * @throws URISyntaxException nothing
	 * @throws LoadModelException nothing
	 */
	public static void main(String[] args) throws IOException, InterruptedException, 
													RuntimeException, MambaInstallException, 
													LoadEngineException, 
													RunModelException, ArchiveException, 
													URISyntaxException, LoadModelException {
		Stardist3D.installRequirements();
		Stardist3D model = Stardist3D.fromPretained("StarDist Plant Nuclei 3D ResNet", false);

		RandomAccessibleInterval<FloatType> img = ArrayImgs.floats(new long[] {116, 120, 66});
		
		Map<String, RandomAccessibleInterval<FloatType>> res = model.run(img);
		model.close();
		System.out.println(true);
	}
}
