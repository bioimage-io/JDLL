package io.bioimage.modelrunner.gui.adapter;


import java.io.IOException;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public interface GuiAdapter {
	
	public String getSoftwareName();
	
	public String getModelsDir();
	
	public String getEnginesDir();
	
	public RunnerAdapter createRunner(ModelDescriptor descriptor) throws IOException, LoadEngineException;
	
	public RunnerAdapter createRunner(ModelDescriptor descriptor, String enginesPath) throws IOException, LoadEngineException;
	
	public <T extends RealType<T> & NativeType<T>> void displayRai(RandomAccessibleInterval<T> rai, String axesOrder);
}
