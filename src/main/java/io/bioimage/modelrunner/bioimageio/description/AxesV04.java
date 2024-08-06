package io.bioimage.modelrunner.bioimageio.description;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import io.bioimage.modelrunner.utils.YAMLUtils;

public class AxesV04 implements Axes {
	
	private final List<Axis> axesList;
	
	private final String axesOrder;
	
	private final double[] scaleArr;
	
	private final int[] minArr;
	
	private final int[] stepArr;
	
	private int[] haloArr;
	
	private final double[] offsetArr;
	
	private String reference;
	
	protected AxesV04(Map<String, Object> tensorSpecMap, boolean isInput) {
		axesOrder = (String) tensorSpecMap.get("axes");
		if (axesOrder == null)
			throw new IllegalArgumentException("Model rdf.yaml Bioimage.io specs does not contain information about the axes order.");
        
		Object shape = tensorSpecMap.get("shape");
		
		if (shape instanceof List) {
			int[] shapeArr = YAMLUtils.castListToIntArray((List<?>) shape);
			minArr = shapeArr;
			stepArr = new int[shapeArr.length];
			offsetArr = new double[shapeArr.length];
			double[] arr = new double[shapeArr.length];
			Arrays.fill(arr, -1);
			scaleArr = arr;
		} else if (shape instanceof Map && isInput) {
			Map<String, Object> shapeMap = (Map<String, Object>) shape;
			Object min = shapeMap.get("min");
			if (min == null) throw new IllegalArgumentException("Minimum size needs to be defined for every input.");
			minArr = YAMLUtils.castListToIntArray((List<?>) min);
			Object step = shapeMap.get("step");
			if (step == null) throw new IllegalArgumentException("Step size needs to be defined for every input.");
			this.stepArr = YAMLUtils.castListToIntArray((List<?>) step);
			this.offsetArr = new double[stepArr.length];
			double[] arr = new double[stepArr.length];
			Arrays.fill(arr, -1);
			scaleArr = arr;
		} else if (shape instanceof Map && !isInput) {
			stepArr = new int[axesOrder.length()];
			minArr = new int[axesOrder.length()];
			Map<String, Object> shapeMap = (Map<String, Object>) shape;
			reference = (String) shapeMap.get("reference_tensor");
			if (reference == null) 
				throw new IllegalArgumentException("An ouput tensor needs to reference an input tensors to find out its size.");
			Object scale = shapeMap.get("scale");
			if (scale == null) 
				throw new IllegalArgumentException("Output tensors need to define a scale with respect to the reference input.");
			scaleArr = YAMLUtils.castListToDoubleArray((List<?>) scale);
			Object offset = shapeMap.get("offset");
			if (offset == null) 
				throw new IllegalArgumentException("Output tensors need to define a scale with respect to the reference input.");
			offsetArr = YAMLUtils.castListToDoubleArray((List<?>) offset);
		} else {
			throw new IllegalArgumentException("Incorrect specs consfiguration for the tensors.");
		}
		
		if (!isInput) {
			Object halo = tensorSpecMap.get("halo");
			if (halo == null || !(halo instanceof List))
				haloArr = new int[axesOrder.length()];
			else
				haloArr = YAMLUtils.castListToIntArray((List<?>) halo);
		} else {
			this.haloArr = new int[stepArr.length];
		}
		
		this.axesList = new ArrayList<Axis>();
		for (int i = 0; i < this.axesOrder.length(); i ++) {
			Axis ax = new AxisV04(axesOrder.split("")[i], this.minArr[i], this.stepArr[i], this.haloArr[i], 
					this.offsetArr[i], this.scaleArr[i], this.reference);
			this.axesList.add(ax);
		}
		
	}
	
	public String getAxesOrder() {
		return this.axesOrder;
	}

	/**
	 * @return the axesList
	 */
	public List<Axis> getAxesList() {
		return axesList;
	}

	public int[] getMinTileSizeArr() {
		return this.minArr;
	}

	public int[] getTileStepArr() {
		return this.stepArr;
	}

	public double[] getTileScaleArr() {
		return this.scaleArr;
	}

	public double[] getOffsetArr() {
		return this.offsetArr;
	}

	public int[] getHaloArr() {
		haloArr = new int[this.axesList.size()];
		for (int i = 0; i < this.axesList.size(); i ++)
			haloArr[i] = this.axesList.get(i).getHalo();
		return this.haloArr;
	}
	
	public Axis getAxis(String abreviation) {
		return axesList.stream().filter(ax -> ax.getAxis().equals(abreviation)).findFirst().orElse(null);
	}

}
