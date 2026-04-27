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
package io.bioimage.modelrunner.model.special.yolo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import org.apposed.appose.BuildException;

import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.model.python.DLModelPytorchProtected;
import io.bioimage.modelrunner.model.python.methods.ConvertDims;
import io.bioimage.modelrunner.model.python.methods.LetterboxPreprocessing;
import io.bioimage.modelrunner.model.python.methods.UndoLetterboxProcessingBoundingBoxes;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

/**
 * Implementation of an API to run YOLO models out of the box with little configuration.
 * 
 *
 *@author Carlos Garcia
 */
public class Yolo extends DLModelPytorchProtected {
		
							
	private static final Map<String, Long> PRETRAINED_YOLO_MODELS;
	static {
		PRETRAINED_YOLO_MODELS = new HashMap<String, Long>();
		PRETRAINED_YOLO_MODELS.put("YOLO26n", 5_544_453L);
		PRETRAINED_YOLO_MODELS.put("YOLO26m", 44_255_705L);
		PRETRAINED_YOLO_MODELS.put("YOLO26x", 118_667_365L);
	}
	
	protected static final String LOAD_MODEL_CODE_ABSTRACT = ""
			+ "from multiprocessing import shared_memory" + System.lineSeparator()
			+ "import os" + System.lineSeparator()
			+ "import torch" + System.lineSeparator()
			+ "device = 'cpu'" + System.lineSeparator()
			+ "if %s and torch.cuda.is_available():" + System.lineSeparator()
			+ "  device = 'cuda'" + System.lineSeparator()
			+ "elif %s:" + System.lineSeparator()
			+ "  from torch.backends import mps" + System.lineSeparator()
			+ "  if mps.is_built() and mps.is_available():" + System.lineSeparator()
			+ "    device = 'mps'" + System.lineSeparator()
			+ "from ultralytics import YOLO" + System.lineSeparator()
			+ MODEL_VAR_NAME + " = YOLO(r'%s')" + System.lineSeparator()
			+ "task.export(shared_memory=shared_memory)" + System.lineSeparator()
			+ "task.export(YOLO=YOLO)" + System.lineSeparator()
			+ "task.export(torch=torch)" + System.lineSeparator()
			+ "task.export(os=os)" + System.lineSeparator()
			+ "task.export(device=device)" + System.lineSeparator()
			+ "task.export(" + MODEL_VAR_NAME + "=" + MODEL_VAR_NAME +")" + System.lineSeparator();

	/**
	 * Creates a new YOLO model.
	 *
	 * @param weightsPath the weightsPath parameter.
	 * @throws BuildException if there is any error building the environment
	 */
	protected Yolo(String weightsPath) throws BuildException {
		super(null, null, null, weightsPath, new HashMap<String, Object>(), true);
	}
	
	// TODO add 3D
	protected <R extends RealType<R> & NativeType<R>> 
	List<Tensor<R>> checkInputTensors(List<Tensor<R>> inputTensors) {
		if (inputTensors.size() > 1)
			throw new IllegalArgumentException("The input tensor list should contain just one tensor");
		if (!inputTensors.get(0).getAxesOrderString().equals("xy") && !inputTensors.get(0).getAxesOrderString().equals("xyc"))
			throw new IllegalArgumentException("The input axes should be 'xyc'");

		long[] dims = inputTensors.get(0).getData().dimensionsAsLongArray();
		if (dims.length == 2) {
			FinalInterval interval = new FinalInterval(new long[3], new long[] {dims[0], dims[1], 1});
			IntervalView<R> nData = Views.interval(inputTensors.get(0).getData(), interval);
			inputTensors.set(0, Tensor.build(inputTensors.get(0).getName(), "xyc", nData));
		} else if (dims.length == 3 && dims[2] != 3 && dims[2] != 1)
			throw new IllegalArgumentException("Only 1 and 3 channel images supported. The provided input has " + dims[2]);
		return inputTensors;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	List<Tensor<T>> checkOutputTensors(List<Tensor<T>> outputTensors) {
		// TODO 
		return outputTensors;
	}
	
	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @return the resulting tensors 
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	List<Tensor<T>> run(List<Tensor<R>> inputTensors) throws RunModelException {
		return super.run(checkInputTensors(inputTensors));
	}

	/**
	 * Run a Bioimage.io model and execute the tiling strategy in one go.
	 * The model needs to have been previously loaded with {@link #loadModel()}.
	 * This method does not execute pre- or post-processing, they
	 * need to be executed independently before or after
	 * 
	 * @param <T>
	 * 	ImgLib2 data type of the output images
	 * @param <R>
	 * 	ImgLib2 data type of the input images
	 * @param inputTensors
	 * 	list of the input tensors that are going to be inputed to the model
	 * @param outputTensors
	 * 	list of output tensors that are expected to be returned by the model
	 * @throws RunModelException if the model has not been previously loaded
	 * @throws IllegalArgumentException if the model is not a Bioimage.io model or if lacks a Bioimage.io
	 *  rdf.yaml specs file in the model folder. 
	 */
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> 
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		super.run(checkInputTensors(inputTensors), checkOutputTensors(outputTensors));
	}
	
	/**
	 * Builds model code.
	 *
	 * @return the resulting string.
	 * @throws IOException if an I/O error occurs.
	 */
	protected String buildModelCode() throws IOException {
		String code = String.format(LOAD_MODEL_CODE_ABSTRACT, "True", "True", weightsPath);
		return code;
	}
	
	protected <T extends RealType<T> & NativeType<T>> 
	String createInputsCode(List<Tensor<T>> inRais, List<String> names) {
		String code = "";
		code += ConvertDims.getMethodDeclaration() + System.lineSeparator();
		code += LetterboxPreprocessing.getMethodDeclaration() + System.lineSeparator();
		code += UndoLetterboxProcessingBoundingBoxes.getMethodDeclaration() + System.lineSeparator();
		code += "created_shms = []" + System.lineSeparator();
		code += "try:" + System.lineSeparator();
		for (int i = 0; i < inRais.size(); i ++) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(inRais.get(i).getData(), false, false);
			code += codeToConvertShmaToPython(shma, names.get(i));
			inShmaList.add(shma);
			code += "  print(" + names.get(i) + ".shape)" + System.lineSeparator();
			code += "  " + names.get(i) + "_torch, meta = " + LetterboxPreprocessing.getMethodName() 
			+ "(" + ConvertDims.getMethodName() + "(" + names.get(i)
			+ ", '" + inRais.get(i).getAxesOrderString().toLowerCase() + "',device=device))" + System.lineSeparator();
			code += "  print(" + names.get(i) + "_torch.shape)" + System.lineSeparator();
		}
		code += "  " + OUTPUT_LIST_KEY + " = " + MODEL_VAR_NAME + "(" + names.get(0) + "_torch, device=device)" + System.lineSeparator();;
		String closeEverythingWin = closeSHMWin();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "except Exception as e:" + System.lineSeparator();
		code += "  " + closeEverythingWin + System.lineSeparator();
		code += "  raise e" + System.lineSeparator();
		code += ""
				+ SHMS_KEY + " = []" + System.lineSeparator()
				+ SHM_NAMES_KEY + " = []" + System.lineSeparator()
				+ DTYPES_KEY + " = []" + System.lineSeparator()
				+ DIMS_KEY + " = []" + System.lineSeparator()
				+ "task.export(" + SHMS_KEY + " = " + SHMS_KEY + ")" + System.lineSeparator()
				+ "task.export(" + SHM_NAMES_KEY + " = " + SHM_NAMES_KEY + ")" + System.lineSeparator()
				+ "task.export(" + DTYPES_KEY + " = " + DTYPES_KEY + ")" + System.lineSeparator()
				+ "task.export(" + DIMS_KEY + " = " + DIMS_KEY + ")" + System.lineSeparator();
		code += "print(" + OUTPUT_LIST_KEY + "[0].boxes)" + System.lineSeparator();
		code += "max_box = max([(0 if r.boxes is None else len(r.boxes)) for r in " + OUTPUT_LIST_KEY + "])" + System.lineSeparator();
		code += "max_box = max([1, max_box])" + System.lineSeparator();
		code += "shm = shared_memory.SharedMemory(create=True, size=len(" + OUTPUT_LIST_KEY + ") * max_box * 6 * 4)" + System.lineSeparator();
		code += "box_tensor = np.ndarray((len(" + OUTPUT_LIST_KEY + "), max_box, 6), dtype='float32', buffer=shm.buf)" + System.lineSeparator();
		code += "box_tensor.fill(0)" + System.lineSeparator();
		code += "" + SHMS_KEY + ".append(shm)" + System.lineSeparator();
		code += "" + SHM_NAMES_KEY + ".append(shm.name)" + System.lineSeparator();
		code += "" + DTYPES_KEY + ".append(str(box_tensor.dtype))" + System.lineSeparator();
		code += "" + DIMS_KEY + ".append(box_tensor.shape)" + System.lineSeparator();
		code += "for i_r, r in enumerate(" + OUTPUT_LIST_KEY + "):" + System.lineSeparator()
				+ "  boxes = r.boxes.xyxy.detach().cpu().numpy()" + System.lineSeparator()
				+ "  confs = r.boxes.conf.detach().cpu().numpy()" + System.lineSeparator()
				+ "  clss = r.boxes.cls.detach().cpu().numpy()" + System.lineSeparator()
				+ "  for i_b, (box, conf, cl) in enumerate(zip(boxes, confs, clss)):" + System.lineSeparator()
				+ "    box_tensor[i_r, i_b, :4] = box" + System.lineSeparator()
				+ "    box_tensor[i_r, i_b, 4] = conf" + System.lineSeparator()
				+ "    box_tensor[i_r, i_b, 5] = cl" + System.lineSeparator()
				+ ""
				+ ""
				+ "box_tensor = " + UndoLetterboxProcessingBoundingBoxes.getMethodName() + "(box_tensor, meta)" + System.lineSeparator();
		code += taskOutputsCode();
		return code;
	}
	
	/**
	 * Initialize a Cellpose model with the path to the model weigths.
	 * @param weightsPath
	 * 	path to the weights of a pretrained cellpose model
	 * @return an instance of a Stardist2D model ready to be used
     * @throws IOException If there's an I/O error.
	 * @throws BuildException if there is any error building the environment
	 */
	public static Yolo init(String weightsPath) throws IOException, BuildException {
		File wFile = new File(weightsPath);
		if (!wFile.isFile())
			throw new IllegalArgumentException("The path provided does not correspond to an existing file: " + weightsPath);		        
        Yolo cellpose = new Yolo(weightsPath);
		return cellpose;
	}
	
	
	/**
	 * Example code that shows how to run a model with cellpose
	 * @param <T>
	 * 	method param
	 * @param args
	 * 	method param
	 * @throws IOException exception
	 * @throws InterruptedException exception
	 * @throws ExecutionException exception
	 * @throws LoadModelException exception
	 * @throws RunModelException exception
	 * @throws BuildException if there is any error launching the python process
	 */
	public static <R extends RealType<R> & NativeType<R>, T extends RealType<T> & NativeType<T>>
	void main(String[] args) throws IOException, InterruptedException, ExecutionException, LoadModelException, RunModelException, BuildException {
		Yolo model = Yolo.init("/home/carlos/git/JDLL/models/yolo/yolo26n.pt");
		model.installRequirements();
		model.loadModel();
		ArrayImg<FloatType, FloatArray> rai = ArrayImgs.floats(new long[] {2, 512, 512, 3});
		long tt = System.currentTimeMillis();
		Tensor<FloatType> tensor = Tensor.build("input", "bxyc", rai);
		List<Tensor<T>> res = model.inference(tensor);
		System.out.println(System.currentTimeMillis() - tt);
		tt = System.currentTimeMillis();
		
		List<List<Tensor<R>>> rees = model.inferenceBatch(Arrays.asList(tensor));
		System.out.println(System.currentTimeMillis() - tt);
		model.close();
		System.out.println(false);
	}
}
