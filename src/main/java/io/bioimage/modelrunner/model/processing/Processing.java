package io.bioimage.modelrunner.model.processing;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.TransformSpec;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Class that executes the pre-processing associated to a given tensor
 * 
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class Processing {
	/**
	 * Descriptor containing the info about the model
	 */
	private ModelDescriptor descriptor;
	/**
	 * Map containing all the needed input objects to make the processing.
	 * It has to contain the tensor of interest.
	 */
	private Map<String, List<TransformationInstance>> preMap;
	private Map<String, List<TransformationInstance>> postMap;
	//private static BioImageIoPython interp;
	private static String BIOIMAGEIO_PYTHON_TRANSFORMATIONS_WEB = 
						"https://github.com/bioimage-io/core-bioimage-io-python/blob/b0cea"
						+ "c8fa5b412b1ea811c442697de2150fa1b90/bioimageio/core/prediction_pipeline"
						+ "/_processing.py#L105";

	/**
	 * The object that is going to execute processing on the given image
	 * @param tensorSpec
	 * 	the tensor specifications
	 * @param seq
	 * 	the image corresponding to a tensor where processing is going to be executed
	 * @throws SecurityException 
	 * @throws NoSuchMethodException 
	 * @throws InvocationTargetException 
	 * @throws IllegalArgumentException 
	 * @throws IllegalAccessException 
	 * @throws InstantiationException 
	 * @throws ClassNotFoundException 
	 */
	private Processing(ModelDescriptor descriptor) throws ClassNotFoundException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		this.descriptor = descriptor;
		buildPreprocessing();
		buildPostprocessing();
	}
	
	private void buildPreprocessing() throws ClassNotFoundException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		preMap = new HashMap<String, List<TransformationInstance>>();
		for (TensorSpec tt : this.descriptor.getInputTensors()) {
			List<TransformSpec> preprocessing = tt.getPreprocessing();
			List<TransformationInstance> list = new ArrayList<TransformationInstance>();
			for (TransformSpec transformation : preprocessing) {
				list.add(TransformationInstance.create(transformation));
			}
		}
	}
	
	private void buildPostprocessing() throws ClassNotFoundException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		postMap = new HashMap<String, List<TransformationInstance>>();
		for (TensorSpec tt : this.descriptor.getOutputTensors()) {
			List<TransformSpec> preprocessing = tt.getPostprocessing();
			List<TransformationInstance> list = new ArrayList<TransformationInstance>();
			for (TransformSpec transformation : preprocessing) {
				list.add(TransformationInstance.create(transformation));
			}
		}
	}
	
	public static Processing init(ModelDescriptor descriptor) throws ClassNotFoundException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		return new Processing(descriptor);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> preprocess(List<Tensor<T>> tensorList){
		return preprocess(tensorList, false);
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> preprocess(List<Tensor<T>> tensorList, boolean inplace) {
		for (Entry<String, List<TransformationInstance>> ee : this.preMap.entrySet()) {
			Tensor<T> tt = tensorList.stream().filter(t -> t.getName().equals(ee.getKey())).findFirst().orElse(null);
			if (tt == null)
				continue;
			ee.getValue().forEach(trans -> {
				try {
					trans.run(tt, inplace);
				} catch (NoSuchMethodException | SecurityException | IllegalAccessException | IllegalArgumentException
						| InvocationTargetException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
		}
		return null;
	}
	
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<Tensor<R>> postprocess(List<Tensor<T>> tensorList, boolean inplace) {
		for (Entry<String, List<TransformationInstance>> ee : this.postMap.entrySet()) {
			Tensor<T> tt = tensorList.stream().filter(t -> t.getName().equals(ee.getKey())).findFirst().orElse(null);
			if (tt == null)
				continue;
			ee.getValue().forEach(trans -> {
				try {
					trans.run(tt, inplace);
				} catch (NoSuchMethodException | SecurityException | IllegalAccessException | IllegalArgumentException
						| InvocationTargetException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
		}
		return null;
	}
}
