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
package io.bioimage.modelrunner.gui;

import java.awt.Color;
import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import javax.swing.SwingUtilities;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.gui.adapter.GuiAdapter;

public class ModelSelectionPanel extends ModelSelectionPanelGui {

	private static final long serialVersionUID = 6264134076603842497L;
    

    private List<String> modelNames;
    private List<String> modelNicknames;
    private List<URL> modelImagePaths;
    private List<ModelDescriptor> models;

	protected ModelSelectionPanel(GuiAdapter adapter) {
        super(adapter);
        this.setBackground(new Color(236, 240, 241));
	}
    
    private void setCardsData() {
    	// TODO separate between not found and loading
    	this.modelNames = models.stream().map(mm -> mm == null ? Gui.NOT_FOUND_STR : mm.getName()).collect(Collectors.toList());
    	this.modelNicknames = models.stream().map(mm -> mm == null ? Gui.NOT_FOUND_STR : mm.getNickname()).collect(Collectors.toList());
    	this.modelImagePaths = models.stream().map(mm -> {
    		if (mm == null || mm.getCovers() == null || mm.getCovers().size() == 0) 
    			return this.getClass().getClassLoader().getResource(DefaultIcon.DIJ_ICON_PATH);
    		File imFile = new File(mm.getCovers().get(0));
    		if (!imFile.exists() && mm.getModelPath() != null)
    			imFile = new File(mm.getModelPath() + File.separator + mm.getCovers().get(0));
    		else if (mm.getModelPath() == null) {
    			try {
					return new URL(mm.getModelURL() + mm.getCovers().get(0));
				} catch (MalformedURLException e) {
				}
    		}
    		if (!imFile.exists()) 
    			return this.getClass().getClassLoader().getResource(DefaultIcon.DIJ_ICON_PATH);
    		try {
				return imFile.toURI().toURL();
			} catch (MalformedURLException e) {
				return this.getClass().getClassLoader().getResource(DefaultIcon.DIJ_ICON_PATH);
			}
    	}).collect(Collectors.toList());
    }
    
    protected void setModels(List<ModelDescriptor> models) {
    	this.models = models;
    	setCardsData();
    	if (SwingUtilities.isEventDispatchThread())
    		redrawModelCards(0);
    	else
    		SwingUtilities.invokeLater(() -> redrawModelCards(0));
    }
    
    protected void setModelAt(ModelDescriptor model, int pos) {
    	Objects.requireNonNull(model);
    	if (pos > models.size())
    		throw new IllegalArgumentException("Wanted position of the model (" + pos + ") out of range (" + models.size() + ").");
    	this.models.set(pos, model);
    	this.modelNames.set(pos, model.getName() == null ? defaultString : model.getName());
    	this.modelNicknames.set(pos, model.getNickname() == null ? defaultString : model.getNickname());

		if (model.getCovers() == null || model.getCovers().size() == 0) {
			modelImagePaths.set(pos, this.getClass().getClassLoader().getResource(DefaultIcon.DIJ_ICON_PATH));
			return;
		}
		File imFile = new File(model.getCovers().get(0));
		if (!imFile.exists() && model.getModelPath() != null)
			imFile = new File(model.getModelPath() + File.separator + model.getCovers().get(0));
		else if (model.getModelPath() == null) {
			try {
				modelImagePaths.set(pos, new URL(model.getModelURL() + model.getCovers().get(0)));
				return;
			} catch (MalformedURLException e) {
			}
		}
		if (!imFile.exists()) {
			modelImagePaths.set(pos, this.getClass().getClassLoader().getResource(DefaultIcon.DIJ_ICON_PATH));
			return;
		}
		try {
			modelImagePaths.set(pos, imFile.toURI().toURL());
		} catch (MalformedURLException e) {
			modelImagePaths.set(pos, this.getClass().getClassLoader().getResource(DefaultIcon.DIJ_ICON_PATH));
		}
    }
    
    protected void redrawModelCards(int currentIndex) {
    	boolean prevSupported = true;
    	boolean mainSupported = true;
    	boolean nextSupported = true;
    	int prevInd = getWrappedIndex(currentIndex - 1);
    	int nextInd = getWrappedIndex(currentIndex + 1);
    	if (models.get(prevInd) != null
    			&& models.get(prevInd).getModelFamily().equals(ModelDescriptor.BIOIMAGEIO))
    		prevSupported = !Gui.UNSUPPORTED_MODELS.contains(models.get(prevInd).getNickname()) 
    					&& (models.get(prevInd).getWeights().getAllSuportedWeightNames().size() != 0);
    	if (models.get(currentIndex) != null
    			&& models.get(currentIndex).getModelFamily().equals(ModelDescriptor.BIOIMAGEIO))
    		mainSupported = !Gui.UNSUPPORTED_MODELS.contains(models.get(currentIndex).getNickname()) 
					&& models.get(currentIndex).getWeights().getAllSuportedWeightNames().size() != 0;
    	if (models.get(getWrappedIndex(currentIndex + 1)) != null
    			&& models.get(nextInd).getModelFamily().equals(ModelDescriptor.BIOIMAGEIO))
    		nextSupported = !Gui.UNSUPPORTED_MODELS.contains(models.get(nextInd).getNickname()) 
					&& models.get(nextInd).getWeights().getAllSuportedWeightNames().size() != 0;
        prevModelPanel.updateCard(modelNames.get(getWrappedIndex(currentIndex - 1)),
                modelNicknames.get(getWrappedIndex(currentIndex - 1)),
                modelImagePaths.get(getWrappedIndex(currentIndex - 1)),
                prevSupported);
        selectedModelPanel.updateCard(modelNames.get(currentIndex),
                modelNicknames.get(currentIndex),
                modelImagePaths.get(currentIndex),
                mainSupported);
        nextModelPanel.updateCard(modelNames.get(getWrappedIndex(currentIndex + 1)),
                modelNicknames.get(getWrappedIndex(currentIndex + 1)),
                modelImagePaths.get(getWrappedIndex(currentIndex + 1)),
                nextSupported);
    }

    private int getWrappedIndex(int index) {
        int size = getModelNames().size();
        return size == 0 ? size : (index % size + size) % size;
    }
    
    public List<String> getModelNames() {
    	return this.modelNames;
    }
    
    public List<String> getModelNicknames() {
    	return this.modelNicknames;
    }
    
    public List<URL> getCoverPaths() {
    	return this.modelImagePaths;
    }
    
    public List<ModelDescriptor> getModels() {
    	return this.models;
    }
}
