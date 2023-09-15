package io.bioimage.modelrunner;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Types;

import io.bioimage.modelrunner.model.Model;
import io.bioimage.modelrunner.tensor.ImgLib2ToArray;
import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.numeric.real.FloatType;

public class PythonConnection {
	
	private static final String NAME_KEY = "name";
	
	private static final String AXES_KEY = "axes";
	
	private static final String DATA_KEY = "data";
	
	private static final String CODE = 
			"task.update('before imports')" + System.lineSeparator()
			+ "import torch" + System.lineSeparator()
			+ "import numpy as np" + System.lineSeparator()
			+ "task.update('starting')" + System.lineSeparator()
			// + "listArr = list(tensor)" + System.lineSeparator()
			+ "img = np.array(tensor)" + System.lineSeparator()
			+ "task.update('array created')" + System.lineSeparator()
			+ "torch_sum = torch.from_numpy(img).sum()" + System.lineSeparator()
			+ "torch_sum_val = torch_sum.to(torch.int32).sum().item()" + System.lineSeparator()
			+ "task.outputs['value'] = torch_sum_val" + System.lineSeparator();
	
	
	public static void main(String[] args) {
		final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
		final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512 );
		// Create the input tensor with the nameand axes given by the rdf.yaml file
		// and add it to the list of input tensors
		Tensor<FloatType> inpTensor = Tensor.build("input0", "bcyx", img1);
		List<Tensor<FloatType>> inputs = new ArrayList<Tensor<FloatType>>();
		inputs.add(inpTensor);
		
		runModelInPython(inputs, null);
	}
	
	public static void runModelInPython(List<Tensor<FloatType>> inputTensors, 
			List<Tensor<FloatType>> outputTensors) {
		LinkedHashMap<String, Object> inputMap = new LinkedHashMap<>();
		for (Tensor<FloatType> input : inputTensors) {
			HashMap<String, Object> tensorMap = new HashMap<String, Object>();
			//tensorMap.put(AXES_KEY, input.getAxesOrderString());
			//tensorMap.put(DATA_KEY, ImgLib2ToArray.buildFloat(input.getData()));
			// inputMap.put(NAME_KEY, tensorMap);
			inputMap.put("tensor", ImgLib2ToArray.buildFloat(input.getData()));
		}
        
        Environment env = Appose.base(new File("C:\\Users\\angel\\git\\jep\\miniconda\\envs\\testAppose")).build();
        try (Service python = env.python()) {
        	python.debug(line -> {
        		System.err.println(line);
        	});
            Task task = python.task(CODE, inputMap);
            System.out.println("here");
            task.listen(event -> {
                switch (event.responseType) {
	                case UPDATE:
	                    System.out.println("Progress: " + task.message);
	                    break;
	                case COMPLETION:
	                    int numer = (Integer) task.outputs.get("value");
	                    System.out.println("Task complete. Result: " + numer);
	                    break;
	                case CANCELATION:
	                    System.out.println("Task canceled");
	                    break;
	                case FAILURE:
	                    System.out.println("Task failed: " + task.error);
	                    break;
                }
            });
            task.waitFor();
            System.out.println("here2");
            Object result = task.outputs.get("torch_sum_val");
            System.out.println("here3");
            if (result instanceof Integer)
            	System.out.print(result);
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
