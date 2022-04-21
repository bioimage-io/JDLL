package plugins.carlosuc3m.deeplearning.test;

import java.util.ArrayList;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;

import icy.main.Icy;
import icy.plugin.PluginLauncher;
import icy.plugin.PluginLoader;
import icy.sequence.Sequence;
import icy.sequence.SequenceUtil;
import icy.system.thread.ThreadUtil;
import icy.type.DataType;
import plugins.adufour.ezplug.EzPlug;
import plugins.adufour.ezplug.EzVarSequence;

public class RunModelAgnostic  extends EzPlug
{

    private EzVarSequence varInSequence;

    @Override
    protected void initialize()
    {
        varInSequence = new EzVarSequence("Sequence");
        addEzComponent(varInSequence);
    }

    @Override
    protected void execute()
    {
    	ThreadUtil.bgRun(new Runnable()
    	{
    	@Override
    	public void run()
    	{
	        try {
	    		Sequence sequence = varInSequence.getValue(true);
		        sequence = SequenceUtil.convertToType(sequence, DataType.FLOAT, false);
		        long tStart = 0;
		        tStart = System.currentTimeMillis();
		        List<Tensor> inTensors = new ArrayList<Tensor>();
		        inTensors.add(Tensor.build("input", "bcyx", sequence));
	            System.out.println("Input tennsor created");
		        List<Tensor> outTensors = new ArrayList<Tensor>();
		        outTensors.add(Tensor.buildEmptyTensor("output", "bcyx"));
		        List<Tensor> outTensors2 = new ArrayList<Tensor>();
		        outTensors2.add(Tensor.buildEmptyTensor("output", "bcyx"));
	            System.out.println("Output tennsor created");
	            
				// Find the URL that corresponds to the file
	    		String jarsDirectory = "/home/carlos/Documents/test";
	    		EngineInfo engineInfo = EngineInfo.defineDLEngine("Pytorch", "1.7.1", jarsDirectory);
	        	String path = "/home/carlos/Pictures/fiji-linux64/Fiji.app/models (copy)/arabidopsis-ovules-boundarymodel";
	        	String pytorchFile = "/home/carlos/Pictures/fiji-linux64/Fiji.app/models (copy)/weights-torchscript.pt";
	        	Model model = Model.createDeepLearningModel(path, pytorchFile, engineInfo);
	    		model.loadModel();
	            System.out.println("Model loaded");
	            outTensors = model.runModel(inTensors, outTensors);
	            Sequence rebuiltSequence = outTensors.get(0).getData();
	            rebuiltSequence.updateChannelsBounds();
	            addSequence(rebuiltSequence);
	            System.out.println("End Execution");
	    		engineInfo = EngineInfo.defineDLEngine("Pytorch", "1.7.0", jarsDirectory);
	        	model = Model.createDeepLearningModel(path, pytorchFile, engineInfo);
	    		model.loadModel();
	            System.out.println("Model loaded");
	            outTensors2 = model.runModel(inTensors, outTensors2);
	            Sequence rebuiltSequence2 = outTensors.get(0).getData();
	            rebuiltSequence2.updateChannelsBounds();
	            addSequence(rebuiltSequence2);
	            System.out.println("End Execution");
	            long tConversion1 = System.currentTimeMillis() - tStart;
	            System.out.println("Duration = " + tConversion1 + "msec");
	        } catch (Exception ex) {
	        	ex.printStackTrace();
	        }
    	}
    	});
        
    }

    @Override
    public void clean()
    {
    }

    public static void main(String[] args)
    {
        Icy.main(args);
        PluginLauncher.start(PluginLoader.getPlugin(RunModelAgnostic.class.getName()));
    }
}
