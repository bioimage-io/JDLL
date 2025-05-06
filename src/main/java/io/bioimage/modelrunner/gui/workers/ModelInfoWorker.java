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
package io.bioimage.modelrunner.gui.workers;

import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;

import io.bioimage.modelrunner.gui.Gui;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;

public class ModelInfoWorker extends SwingWorker<String, Void> {

    private ModelDescriptor model;
    private TextLoadCallback callback;

    public ModelInfoWorker(ModelDescriptor model, TextLoadCallback callback) {
        this.model = model;
        this.callback = callback;
    }

    @Override
    protected String doInBackground() throws Exception {
        // Perform the time-consuming task of generating the info text
        if (model == null) {
            // Return default text if model is null
            return Gui.INSTALL_INSTRUCTIONS;
        } else {
            // Generate the info from the model
            return model.buildInfo();
        }
    }

    @Override
    protected void done() {
        try {
            String infoText = get();
            SwingUtilities.invokeLater(() -> callback.onTextLoaded(infoText));
        } catch (Exception e) {
            SwingUtilities.invokeLater(() -> callback.onTextLoadFailed(e));
        }
    }

    @FunctionalInterface
    public interface TextLoadCallback {
        void onTextLoaded(String text);
        
        default void onTextLoadFailed(Exception e) {
            System.err.println("Failed to load text: " + e.getMessage());
        }
    }
}
