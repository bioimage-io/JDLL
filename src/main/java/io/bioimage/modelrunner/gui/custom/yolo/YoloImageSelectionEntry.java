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

public class YoloImageSelectionEntry {

    private final String id;
    private final String title;
    private final Object image;

    /**
     * Creates a new YoloImageSelectionEntry instance.
     *
     * @param id the ID.
     * @param title the title.
     * @param image the image.
     */
    public YoloImageSelectionEntry(String id, String title, Object image) {
        this.id = id;
        this.title = title;
        this.image = image;
    }

    /**
     * Returns the ID.
     *
     * @return the ID.
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the title.
     *
     * @return the title.
     */
    public String getTitle() {
        return title;
    }

    /**
     * Returns the image.
     *
     * @return the image.
     */
    public Object getImage() {
        return image;
    }

    /**
     * Returns a string representation of this object.
     *
     * @return the string representation.
     */
    @Override
    public String toString() {
        return title;
    }
}
