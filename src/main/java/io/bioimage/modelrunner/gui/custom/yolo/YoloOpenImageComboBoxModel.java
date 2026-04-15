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
package io.bioimage.modelrunner.gui.custom.yolo;

import java.util.List;
import java.util.Set;

import javax.swing.DefaultComboBoxModel;

public class YoloOpenImageComboBoxModel extends DefaultComboBoxModel<YoloImageSelectionEntry> {

    private static final long serialVersionUID = 8586150079690419881L;

    public YoloOpenImageComboBoxModel() {
        super();
    }

    public void setEntries(List<YoloImageSelectionEntry> entries) {
        removeAllElements();
        for (YoloImageSelectionEntry entry : entries) {
            addElement(entry);
        }
        if (getSize() > 0) {
            setSelectedItem(getElementAt(0));
        }
    }
    
    public int getIndexById(String id) {
        if (id == null) {
            return -1;
        }
        for (int i = 0; i < getSize(); i++) {
            YoloImageSelectionEntry entry = getElementAt(i);
            if (id.equals(entry.getId())) {
                return i;
            }
        }
        return -1;
    }
    
    public void retainIds(Set<String> ids) {
        for (int i = getSize() - 1; i >= 0; i--) {
            YoloImageSelectionEntry entry = getElementAt(i);
            if (!ids.contains(entry.getId())) {
                removeElementAt(i);
            }
        }
    }


}
