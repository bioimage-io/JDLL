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
package io.bioimage.modelrunner.gui.custom;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.swing.JComponent;

import io.bioimage.modelrunner.model.detection.Detection;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * @author Carlos Garcia
 */
public abstract class ConsumerInterface {
	
	protected List<String> varNames;
	
	protected List<JComponent> componentsGui;
	
	/**
	 * Gets models dir.
	 *
	 * @return the resulting string.
	 */
	public abstract String getModelsDir();

    /** Host-specific guidance for supported image formats and numeric array axes. */
    public String getImageFormatDocumentationUrl() {
        return io.bioimage.modelrunner.numpy.NpyImageFile.DOCUMENTATION_URL;
    }
	
	/**
	 * Updates gui.
	 */
	public abstract void updateGUI();
	
	/**
	 * Sets components.
	 *
	 * @param components the components parameter.
	 */
	public abstract void setComponents(List<JComponent> components);
	
	/**
	 * Sets var names.
	 *
	 * @param componentNames the componentNames parameter.
	 */
	public abstract void setVarNames(List<String> componentNames);
	
	/**
	 * Executes notify params.
	 *
	 * @param args the args parameter.
	 */
	public abstract void notifyParams(LinkedHashMap<String, String> args);
	
	/**
	 * Gets focused image.
	 *
	 * @return the resulting value.
	 */
	public abstract Object getFocusedImage();
	
	/**
	 * Gets focused image name.
	 *
	 * @return the resulting string.
	 */
	public abstract String getFocusedImageName();
	
	/**
	 * Gets focused image channels.
	 *
	 * @return the resulting value.
	 */
	public abstract Integer getFocusedImageChannels();
	
	/**
	 * Gets focused image slices.
	 *
	 * @return the resulting value.
	 */
	public abstract Integer getFocusedImageSlices();
	
	/**
	 * Gets focused image frames.
	 *
	 * @return the resulting value.
	 */
	public abstract Integer getFocusedImageFrames();
	
	/**
	 * Gets focused image width.
	 *
	 * @return the resulting value.
	 */
	public abstract Integer getFocusedImageWidth();
	
	/**
	 * Gets focused image height.
	 *
	 * @return the resulting value.
	 */
	public abstract Integer getFocusedImageHeight();

	/**
	 * Returns the displayed channel position for a host image. Hosts that do not
	 * expose viewer positions may keep the default first channel.
	 *
	 * @param image host image object.
	 * @return zero-based channel position.
	 */
	public int getImageChannelPosition(Object image) {
		return 0;
	}

	/**
	 * Returns the displayed z position for a host image.
	 *
	 * @param image host image object.
	 * @return zero-based z position.
	 */
	public int getImageZPosition(Object image) {
		return 0;
	}

	/**
	 * Returns the displayed time position for a host image.
	 *
	 * @param image host image object.
	 * @return zero-based time position.
	 */
	public int getImageTimePosition(Object image) {
		return 0;
	}

	/**
	 * Gets focused image as rai.
	 *
	 * @param <T> the T type parameter.
	 * @return the focused image as RAI.
	 */
	public abstract < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval<T> getFocusedImageAsRai();

	/**
	 * Opens a file with the host application's image loader, without displaying it.
	 * Returns zero-min X,Y,C,Z,T data, including singleton axes, with the original
	 * numeric type and pixel values (not display/LUT values). The returned pixels
	 * must remain usable after this call; the host must release temporary readers
	 * without invalidating them. Do not change the selected/open host image.
	 *
	 * @param <T> native pixel type, determined by the source image.
	 * @param imageFile image to open; multi-series hosts use the first series.
	 * @return non-null five-dimensional image.
	 * @throws IOException if the file cannot be decoded.
	 */
	public abstract <T extends RealType<T> & NativeType<T>>
	RandomAccessibleInterval<T> openImageAsRai(File imageFile) throws IOException;

    /**
     * Opens native XYC preview pixels at the first Z/time position, off the EDT.
     * Hosts should override this to avoid decoding an entire stack for a preview.
     */
    public <T extends RealType<T> & NativeType<T>> RandomAccessibleInterval<T> openImagePreviewAsRai(File imageFile)
            throws IOException {
        return openImageAsRai(imageFile);
    }

    /**
     * Opens a lazy preview session for series zero, off the EDT. The host provides
     * Z/T metadata, native pyramid access and display colors without loading the
     * stack. JDLL serializes reads, caches RGB tiles and closes stale sessions off
     * the EDT. Returning null preserves the legacy ImageIO/first-plane fallback.
     */
    public PreviewReader openPreviewReader(File imageFile) throws IOException {
        return null;
    }

	/**
	 * Channel count for file-source controls. Hosts should override this with a
	 * metadata-only query; the default uses their image loader off the UI thread.
	 *
	 * @param imageFile image to inspect, using the same series as openImageAsRai.
	 * @return number of channels.
	 * @throws IOException if the image cannot be inspected.
	 */
	public int getImageFileChannels(File imageFile) throws IOException {
		return Math.toIntExact(openImageAsRai(imageFile).dimension(2));
	}
	
	/**
	 * Performs display image.
	 *
	 * @param <T> the T type parameter.
	 * @param rai the RAI.
	 * @param axes the axes.
	 * @param name the name.
	 */
	public abstract < T extends RealType< T > & NativeType< T > > 
	void displayImage(RandomAccessibleInterval<T> rai, String axes, String name);
	
	/**
	 * Performs display detections.
	 *
	 * @param detections the detections.
	 */
	public abstract void displayDetections(List<Detection> detections);

	/**
	 * Performs save detections.
	 *
	 * @param detectionsByImagePath the detections by image path.
	 */
	public void saveDetections(Map<String, List<Detection>> detectionsByImagePath) {
		// Optional hook for host applications to persist ROIs from file-system inference.
	}
	
	/**
	 * Sets variable names.
	 *
	 * @param varNames the varNames parameter.
	 */
	public void setVariableNames(List<String> varNames) {
		this.varNames = varNames;
	}

	protected abstract < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval<T> convertIntoRai(Object objIm);
}
