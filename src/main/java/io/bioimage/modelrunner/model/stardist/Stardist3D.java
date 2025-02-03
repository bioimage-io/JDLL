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
package io.bioimage.modelrunner.model.stardist;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;

import org.apache.commons.compress.archivers.ArchiveException;

import io.bioimage.modelrunner.apposed.appose.MambaInstallException;
import io.bioimage.modelrunner.bioimageio.BioimageioRepo;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.utils.Constants;
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
	
	private Stardist3D(String modelName, String baseDir) throws IOException {
		super(modelName, baseDir);
	}
	
	private Stardist3D(ModelDescriptor descriptor) throws IOException {
		super(descriptor);
	}

	@Override
	protected String createImportsCode() {
		return String.format(LOAD_MODEL_CODE_ABSTRACT, MODULE_NAME, MODULE_NAME, 
				MODULE_NAME, MODULE_NAME, MODULE_NAME, this.name, this.basedir);
	}

	@Override
	protected <T extends RealType<T> & NativeType<T>>  void checkInput(RandomAccessibleInterval<T> image) {
		if (image.dimensionsAsLongArray().length == 3 && this.nChannels != 1)
			throw new IllegalArgumentException("Stardist3D needs an image with four dimensions: XYCZ");
		else if (image.dimensionsAsLongArray().length != 4 && this.nChannels != 1)
			throw new IllegalArgumentException("Stardist3D needs an image with four dimensions: XYCZ");
		else if (image.dimensionsAsLongArray().length == 4 && image.dimensionsAsLongArray()[2] != nChannels)
			throw new IllegalArgumentException("This Stardist3D model requires " + nChannels + " channels.");
		else if (image.dimensionsAsLongArray().length > 4 || image.dimensionsAsLongArray().length < 2)
			throw new IllegalArgumentException("Stardist3D model requires an image with dimensions XYCZ.");
	}
	

	
	@Override
	protected <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> reconstructMask() throws IOException {
		// TODO I do not understand why is complaining when the types align perfectly
		RandomAccessibleInterval<T> maskCopy = Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(shma.getSharedRAI()), 
				Util.getTypeFromInterval(Cast.unchecked(shma.getSharedRAI())));
		shma.close();
		return maskCopy;
	}
	
	/**
	 * Initialize a Stardist2D using the format of the Bioiamge.io model zoo.
	 * @param modelPath
	 * 	path to the Bioimage.io model
	 * @return an instance of a Stardist2D model ready to be used     * @throws FileNotFoundException If the model file is not found.
     * @throws IOException If there's an I/O error.
	 */
	public static Stardist3D fromBioimageioModel(String modelPath) throws FileNotFoundException, IOException {
		ModelDescriptor descriptor = ModelDescriptorFactory.readFromLocalFile(modelPath + File.separator + Constants.RDF_FNAME);
		return new Stardist3D(descriptor);
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 2D models.
	 * By default, the model will be installed in the "models" folder inside the application
	 * @param pretrainedModel
	 * 	the name of the pretrained model. 
	 * @param forceInstall
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist2D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 */
	public static Stardist3D fromPretained(String pretrainedModel, boolean forceInstall) throws IOException, InterruptedException {
		return fromPretained(pretrainedModel, new File("models").getAbsolutePath(), forceInstall);
	}
	
	/**
	 * Initialize one of the "official" pretrained Stardist 3D models
	 * @param pretrainedModel
	 * 	the name of the pretrained model.
	 * @param installDir
	 * 	the directory where the model wants to be installed
	 * @param forceInstall
	 * 	whether to force the installation or to try to look if the model has already been installed before
	 * @return an instance of a pretrained Stardist3D model ready to be used
	 * @throws IOException if there is any error downloading the model, in the case it is needed
	 * @throws InterruptedException if the download of the model is stopped
	 */
	public static Stardist3D fromPretained(String pretrainedModel, String installDir, boolean forceInstall) throws IOException, 
																					InterruptedException {
		if (pretrainedModel.equals("StarDist Plant Nuclei 3D ResNet") && !forceInstall) {
			ModelDescriptor md = ModelDescriptorFactory.getModelsAtLocalRepo().stream()
					.filter(mm ->mm.getName().equals(pretrainedModel)).findFirst().orElse(null);
			if (md != null) return new Stardist3D(md);
			String path = BioimageioRepo.connect().downloadByName("StarDist Plant Nuclei 3D ResNet", installDir);
			return Stardist3D.fromBioimageioModel(path);
		} else if (pretrainedModel.equals("StarDist Plant Nuclei 3D ResNet")) {
			String path = BioimageioRepo.connect().downloadByName("StarDist Plant Nuclei 3D ResNet", installDir);
			return Stardist3D.fromBioimageioModel(path);
		} else {
			throw new IllegalArgumentException("There is no Stardist3D model called: " + pretrainedModel);
		}
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
