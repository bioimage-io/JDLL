package io.bioimage.modelrunner.gui.adapter;


import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public interface GuiAdapter {
	
	public String getSoftwareName();
	
	public RunnerAdapter createRunner(ModelDescriptor descriptor);
	
	public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath);
	
	public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai, String axesOrder);
}
