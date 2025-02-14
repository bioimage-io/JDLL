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
package io.bioimage.modelrunner.bioimageio.description.stardist;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.Axes;
import io.bioimage.modelrunner.bioimageio.description.Axis;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorV04;
import io.bioimage.modelrunner.bioimageio.description.TensorSpec;
import io.bioimage.modelrunner.bioimageio.description.TransformSpec;
import io.bioimage.modelrunner.model.stardist.StardistAbstract;
import io.bioimage.modelrunner.numpy.DecodeNumpy;
import io.bioimage.modelrunner.tensor.Utils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;


/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ModelDescriptorStardistV04 extends ModelDescriptorV04
{

	private List<String> oldOrdersInp = new ArrayList<String>();
	private List<String> oldOrdersOut = new ArrayList<String>();
	
	private static final String STARDIST_TEST = "stardist_test";

	public ModelDescriptorStardistV04(Map<String, Object> yamlElements) {
		super(yamlElements);
		this.input_tensors = this.buildInputTensorsStardist();
		this.output_tensors = this.buildOutputTensorsStardist();
    	this.modifyTestInputs();
    	this.modifyTestOutputs();
	}

	@Override
	public boolean areRequirementsInstalled() {
		return StardistAbstract.isInstalled();
	}

	@Override
	public String getModelFamily() {
		return ModelDescriptor.STARDIST;
	}
	
	protected List<TensorSpec> buildInputTensorsStardist() {
		List<Map<String, Object>> tensors = new ArrayList<Map<String, Object>>();
		for (TensorSpec tt : this.input_tensors) {
			Map<String, Object> map = reverseAxesShape(tt);
			oldOrdersInp.add(tt.getAxesOrder());
			map.put("name", tt.getName());
			map.put("description", tt.getDescription());
			List<Map<String, Object>> preList = new ArrayList<Map<String, Object>>();
			for (TransformSpec prep : tt.getPreprocessing()) {
				preList.add(prep.getSpecMap());
			}
			map.put("preprocessing", preList);
			tensors.add(map);
		}
		yamlElements.put("inputs", tensors);
		return super.buildInputTensors();
	}
	
	protected List<TensorSpec> buildOutputTensorsStardist() {
		List<Map<String, Object>> tensors = new ArrayList<Map<String, Object>>();
		for (TensorSpec tt : this.output_tensors) {
			Map<String, Object> map = reverseAxesShape(tt);
			map.put("name", tt.getName());
			oldOrdersOut.add(tt.getAxesOrder());
			map.put("description", tt.getDescription());
			List<Map<String, Object>> postList = new ArrayList<Map<String, Object>>();
			for (TransformSpec prep : tt.getPostprocessing()) {
				postList.add(prep.getSpecMap());
			}
			map.put("postprocessing", postList);
			tensors.add(map);
		}
		yamlElements.put("outputs", tensors);
		return super.buildOutputTensors();
	}
    
    protected <T extends RealType<T> & NativeType<T>> void modifyTestInputs() {
    	if (this.localModelPath == null)
    		return;
    	for (int i = 0; i < this.oldOrdersInp.size(); i ++) {
    		TensorSpec tt = input_tensors.get(i);
    		String testName = tt.getTestTensorName();
			try {
				RandomAccessibleInterval<T> im = DecodeNumpy.loadNpy(this.localModelPath + File.separator + testName);
	    		String newImAxesOrder = removeExtraDims(im, oldOrdersInp.get(i), tt.getAxesOrder());
	    		im = transposeToAxesOrder(im, tt.getAxesOrder(), newImAxesOrder);
	    		String newTestName = STARDIST_TEST + "_input_" + i + ".npy";
	    		DecodeNumpy.saveNpy(this.localModelPath + File.separator + newTestName, im);
	    		setInputTestNpyName(i, newTestName);
			} catch (IOException e) {
				continue;
			}
    	}
    }
    
    protected <T extends RealType<T> & NativeType<T>> void modifyTestOutputs() {
    	if (this.localModelPath == null)
    		return;
    	for (int i = 0; i < this.oldOrdersInp.size(); i ++) {
    		TensorSpec tt = input_tensors.get(i);
    		String testName = tt.getTestTensorName();
			try {
				RandomAccessibleInterval<T> im = DecodeNumpy.loadNpy(this.localModelPath + File.separator + testName);
	    		String newImAxesOrder = removeExtraDims(im, oldOrdersInp.get(i), tt.getAxesOrder());
	    		im = transposeToAxesOrder(im, tt.getAxesOrder(), newImAxesOrder);
	    		String newTestName = STARDIST_TEST + "_input_" + i + ".npy";
	    		if (!(new File(localModelPath + File.separator + newTestName).isFile()))
	    			DecodeNumpy.saveNpy(localModelPath + File.separator + newTestName, im);
	    		setInputTestNpyName(i, newTestName);
			} catch (IOException e) {
				continue;
			}
    	}
    }
	
	private Map<String, Object> reverseAxesShape(TensorSpec tt) {
		Axes axes = tt.getAxesInfo();
		boolean is3d = axes.getAxesOrder().contains("z");
		String nAxesOrder = is3d ? "xyzc" : "xyc";
		Map<String, Object> shape = new HashMap<String, Object>();
		double[] scale = new double[nAxesOrder.length()];
		for (int i = 0; i < scale.length; i ++) scale[i] = 1;
		int[] minArr = new int[nAxesOrder.length()];
		for (int i = 0; i < minArr.length; i ++) minArr[i] = 1;
		int[] stepArr = new int[nAxesOrder.length()];
		int[] haloArr = new int[nAxesOrder.length()];
		double[] offsetArr = new double[nAxesOrder.length()];
		int c = 0;
		for (String ax : nAxesOrder.split("")) {
			Axis axis = axes.getAxesList().stream()
					.filter(aa -> aa.getAxis().equals(ax)).findFirst().orElse(null);
			if (axis == null)
				continue;
			scale[c] = axis.getScale();
			minArr[c] = axis.getMin();
			stepArr[c] = axis.getStep();
			haloArr[c] = axis.getHalo();
			offsetArr[c] = axis.getOffset();
			c ++;
		}
		shape.put("step", stepArr);
		shape.put("min", minArr);
		shape.put("scale", scale);
		shape.put("offset", offsetArr);
		shape.put("reference_tensor", axes.getAxesList().get(0).getReferenceTensor());
		HashMap<String, Object> axesMap = new HashMap<String, Object>();
		axesMap.put("halo", haloArr);
		axesMap.put("axes", nAxesOrder);
		axesMap.put("shape", shape);
		return axesMap;
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	String removeExtraDims(RandomAccessibleInterval<T> rai, String ogAxes, String targetAxes) {
		String nAxes = "";
		for (String ax : ogAxes.split("")) {
			if (!targetAxes.contains(ax))
				continue;
			nAxes += ax;
		}
		return nAxes;
	}
	
	private static <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> transposeToAxesOrder(RandomAccessibleInterval<T> rai, String ogAxes, String targetAxes) {
		int[] transformation = new int[ogAxes.length()];
		int c = 0;
		for (String ss : targetAxes.split("")) {
			transformation[c ++] = ogAxes.indexOf(ss);
		}
		return Utils.rearangeAxes(rai, transformation);
	}
}
