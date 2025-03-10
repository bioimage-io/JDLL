package io.bioimage.modelrunner.gui;

import java.util.List;

import javax.swing.JComponent;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * @author Carlos Garcia
 */
public abstract class ConsumerInterface {
	
	protected List<String> varNames;
	
	protected List<JComponent> componentsGui;
	
	public abstract String getModelsDir();
	
	public abstract void setComponents(List<JComponent> components);
	
	public abstract void setVarNames(List<String> componentNames);
	
	public abstract Object getFocusedImage();
	
	public abstract String getFocusedImageName();
	
	public abstract Integer getFocusedImageChannels();
	
	public abstract Integer getFocusedImageSlices();
	
	public abstract Integer getFocusedImageFrames();
	
	public abstract Integer getFocusedImageWidth();
	
	public abstract Integer getFocusedImageHeight();

	public abstract < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval<T> getFocusedImageAsRai();
	
	public abstract < T extends RealType< T > & NativeType< T > > 
	void display(RandomAccessibleInterval<T> rai, String axes, String name);
	
	public void setVariableNames(List<String> varNames) {
		this.varNames = varNames;
	}
}
