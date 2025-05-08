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

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;



/**
 * A data structure holding a single Bioimage.io pretrained model description. This instances are created by opening a {@code model.yaml} file.
 * More info about the parameters can be found at:
 * https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ModelDescriptorV04 extends ModelDescriptor
{
	private String newModelID;

	protected ModelDescriptorV04(Map<String, Object> yamlElements)
    {
    	this.yamlElements = yamlElements;
    	buildModelDescription();
    	buildBadgeElements();
    	addSampleAndTestImages();
    	newModelID = findID();
    	modelID = findOldID();
        // TODO super mega ultra WORKAROUND until model is fixed
        // TODO super mega ultra WORKAROUND until model is fixed
        // TODO super mega ultra WORKAROUND until model is fixed
        // TODO super mega ultra WORKAROUND until model is fixed
        // TODO super mega ultra WORKAROUND until model is fixed
        if (this.getNickname().equals("committed-turkey")) {
        	((TensorSpecV04) this.output_tensors.get(0)).dataType = "float32";
        }
    }

    /**
     * @return The nickname of this model.
     */
    @Override
    public String getNickname()
    {
        return this.newModelID;
    }

	@Override
	public boolean areRequirementsInstalled() {
		return true;
	}

	@Override
	protected List<String> buildAttachments() {
		Object att = yamlElements.get("attachments");
		if (att == null)
			return new ArrayList<String>();
		if (att instanceof Map)
			return getAllStrings((Map<String, Object>) att);
		else if (att instanceof List)
			return getAllStrings((List<Object>) att);
		else if (att instanceof String)
			return Arrays.asList((String) att);
		System.err.println("Cannot build the attachments for: " + name);
		return new ArrayList<String>();
	}
	
    private static List<String> getAllStrings(Map<String, Object> map) {
    	List<String> strs = new ArrayList<String>();
    	for (Entry<String, Object> ee : map.entrySet()) {
    		if (ee.getValue() instanceof String)
    			strs.add((String) ee.getValue());
    		else if (ee.getValue() instanceof Map)
    			strs.addAll(getAllStrings((Map<String, Object>) ee.getValue()));
    		else if (ee.getValue() instanceof List)
    			strs.addAll(getAllStrings((List<Object>) ee.getValue()));
    	}
    	return strs;
    }
	
    private static List<String> getAllStrings(List<Object> list) {
    	List<String> strs = new ArrayList<String>();
    	for (Object ee : list) {
    		if (ee instanceof String)
    			strs.add((String) ee);
    		else if (ee instanceof Map)
    			strs.addAll(getAllStrings((Map<String, Object>) ee));
    		else if (ee instanceof List)
    			strs.addAll(getAllStrings((List<Object>) ee));
    	}
    	return strs;
    }

	@Override
	protected List<TensorSpec> buildInputTensors() {
		Object list = this.yamlElements.get("inputs");
		if (!(list instanceof List<?>))
    		return null;
        List<TensorSpec> tensors = new ArrayList<>(((List<?>) list).size());
        for (Object elem : (List<?>) list)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            tensors.add(new TensorSpecV04((Map<String, Object>) elem, true));
        }
        return tensors;
	}

	@Override
	protected List<TensorSpec> buildOutputTensors() {
		Object list = this.yamlElements.get("outputs");
		if (!(list instanceof List<?>))
    		return null;
        List<TensorSpec> tensors = new ArrayList<>(((List<?>) list).size());
        for (Object elem : (List<?>) list)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            tensors.add(new TensorSpecV04((Map<String, Object>) elem, false));
        }
        return tensors;
	}

	@Override
	protected void calculateTotalInputHalo() {
		for (TensorSpec out: output_tensors) {
			for (Axis ax : out.getAxesInfo().getAxesList()) {
				int axHalo = ax.getHalo();
				if (axHalo == 0)
					continue;
				String ref = ax.getReferenceTensor();
				if (ref == null) {
					this.input_tensors.stream().forEach( tt -> {
						AxisV04 inAx = (AxisV04) tt.getAxesInfo().getAxesList().stream()
						.filter(xx -> xx.getAxis().equals(ax.getAxis()))
						.findFirst().orElse(null);
						if (inAx == null || inAx.getHalo() > axHalo) return;
						inAx.halo = axHalo;
					});
					return;
				}
				
				double axScale = ax.getScale();
				double axOffset = ax.getOffset();
				double nHalo = (axHalo + axOffset) / axScale;
				AxisV04 inAx = (AxisV04) this.findInputTensor(ref).getAxesInfo().getAxis(ax.getReferenceAxis());

				if (inAx == null || inAx.getHalo() > nHalo) return;
				inAx.halo = (int) nHalo;
			}
		}
		
	}

	@Override
	protected String findID() {
		if (yamlElements.get("config") != null && yamlElements.get("config") instanceof Map) {
    		Map<String, Object> configMap = (Map<String, Object>) yamlElements.get("config");
    		if (configMap.get("bioimageio") != null && configMap.get("bioimageio") instanceof Map) {
    			Map<String, Object> bioimageMap = (Map<String, Object>) configMap.get("bioimageio");
    			if (bioimageMap.get("nickname") != null)
    				return (String) bioimageMap.get("nickname");
    		}
    	}
    	return (String) yamlElements.get("id");
	}

	@Override
	protected void addBioEngine() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getModelFamily() {
		return ModelDescriptor.BIOIMAGEIO;
	}
    
    protected void addSampleAndTestImages() {
        List<SampleImage> sampleInputs = buildSampleImages((List<?>) yamlElements.get("sample_inputs"));
        List<SampleImage> sampleOutputs = buildSampleImages((List<?>) yamlElements.get("sample_outputs"));

        List<TestArtifact> testInputs = buildTestArtifacts((List<?>) yamlElements.get("test_inputs"));
        List<TestArtifact> testOutputs = buildTestArtifacts((List<?>) yamlElements.get("test_outputs"));

        for (int i = 0; i < sampleInputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.input_tensors.get(i);
        	tt.sampleTensorName = sampleInputs.get(i).getName();
        }
        for (int i = 0; i < testInputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.input_tensors.get(i);
        	tt.testTensorName = testInputs.get(i).getName();
        }
        
        for (int i = 0; i < sampleOutputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.output_tensors.get(i);
        	tt.sampleTensorName = sampleOutputs.get(i).getName();
        }
        for (int i = 0; i < testOutputs.size(); i ++) {
        	TensorSpecV04 tt = (TensorSpecV04) this.output_tensors.get(i);
        	tt.testTensorName = testOutputs.get(i).getName();
        }
    }

    /**
     * REturns a list of {@link SampleImage} of the sample images that are packed in the model
     * folder as tifs and that are specified in the rdf.yaml file
     * @param coverElements
     * 	data from the yaml
     * @return the list of {@link SampleImage} with the sample images data
     */
    protected static List<SampleImage> buildSampleImages(Object coverElements)
    {
        List<SampleImage> covers = new ArrayList<>();
    	if ((coverElements instanceof List<?>)) {
    		List<?> elems = (List<?>) coverElements;
	        for (Object elem : elems)
	        {
	        	if (!(elem instanceof String))
	        		continue;
	        	covers.add(SampleImage.build((String) elem));
	        }
    	} else if ((coverElements instanceof String)) {
            covers.add(SampleImage.build((String) coverElements));
    	}   	
        return covers.stream().filter(i -> i != null).collect(Collectors.toList());
    }

    /**
     * REturns a list of {@link TestArtifact} of the npy artifacts that are packed in the model
     * folder as input and output test objects
     * @param coverElements
     * 	data from the yaml
     * @return the list of {@link TestArtifact} with the sample images data
     */
    protected static List<TestArtifact> buildTestArtifacts(Object coverElements)
    {
        List<TestArtifact> covers = new ArrayList<>();
    	if ((coverElements instanceof List<?>)) {
    		List<?> elems = (List<?>) coverElements;
	        for (Object elem : elems)
	        {
	        	if (!(elem instanceof String))
	        		continue;
	        	covers.add(TestArtifact.build((String) elem));
	        }
    	} else if ((coverElements instanceof String)) {
            covers.add(TestArtifact.build((String) coverElements));
    	}   	
        return covers.stream().filter(i -> i != null).collect(Collectors.toList());
    }

    private List<Badge> buildBadgeElements()
    {
    	Object coverElements = this.yamlElements.get("badges");
    	if (!(coverElements instanceof List<?>))
    		return null;
        List<Badge> badges = new ArrayList<>();
        for (Object elem : (List<?>) coverElements)
        {
            if (!(elem instanceof Map<?, ?>))
            	continue;
            Map<String, Object> dict = (Map<String, Object>) elem;
        	badges.add(Badge.build((String) dict.get("label"), (String) dict.get("icon"), (String) dict.get("url")));
        }
        return badges;
    }
    
	private String findOldID() {
		if (yamlElements.get("config") != null && yamlElements.get("config") instanceof Map) {
    		Map<String, Object> configMap = (Map<String, Object>) yamlElements.get("config");
    		if (configMap.get("_conceptdoi") != null && configMap.get("_conceptdoi") instanceof String) {
    			return (String) configMap.get("_conceptdoi");
    		} else if (configMap.get("_id") != null && configMap.get("_id") instanceof String) {
        		String id = (String) configMap.get("_id");
        		if (id.length() - id.replace("/", "").length() >= 2 
        				&& id.substring(id.indexOf("/") + 1).indexOf("/") - id.indexOf("/") > 2 )
        			return id.substring(0, id.indexOf("/") + id.substring(id.indexOf("/") + 1).indexOf("/") + 1);
        		else
        			return id;
    		}
    	}
    	if (yamlElements.get("id") != null && yamlElements.get("id") instanceof String) {
    		String id = (String) yamlElements.get("id");
    		if (id.length() - id.replace("/", "").length() >= 2 
    				&& id.substring(id.indexOf("/") + 1).indexOf("/") - id.indexOf("/") > 2 )
    			return id.substring(0, id.indexOf("/") + id.substring(id.indexOf("/") + 1).indexOf("/") + 1);
    		else
    			return id;
    	}
    	return null;
    }
	
	protected void setInputTestNpyName(int n, String name) {
    	TensorSpecV04 tt = (TensorSpecV04) this.input_tensors.get(n);
    	if (this.localModelPath != null)
    		name = localModelPath + File.separator + name;
    	tt.testTensorName = name;
	}
	
	protected void setOutputTestNpyName(int n, String name) {
    	TensorSpecV04 tt = (TensorSpecV04) this.output_tensors.get(n);
    	if (this.localModelPath != null)
    		name = localModelPath + File.separator + name;
    	tt.testTensorName = name;
	}
}
