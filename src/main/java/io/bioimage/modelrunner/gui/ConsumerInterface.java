package io.bioimage.modelrunner.gui;

import java.util.List;

import javax.swing.JComponent;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Interface to be implemented by the imaging software that wants to use the default SAMJ UI.
 * Provides a list of the images open to the SAMJ GUI
 * @author Carlos Garcia
 */
public abstract class ConsumerInterface {
	
	protected List<String> varNames;

	
	public interface ConsumerCallback { 
		
		void validPromptChosen(boolean isValid);
		
		}
	
	protected ConsumerCallback callback;
	
	public abstract void setListenersForComponents(List<JComponent> components);
	
	public abstract Object getFocusedImage();

	public abstract < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval<T> getFocusedImageAsRai();
	
	public abstract < T extends RealType< T > & NativeType< T > > 
	void display(RandomAccessibleInterval<T> rai, String axes, String name);
	
	public void setVariableNames(List<String> varNames) {
		this.varNames = varNames;
	}
	
	public void setCallback(ConsumerCallback callback) {
		this.callback = callback;
	}
}
