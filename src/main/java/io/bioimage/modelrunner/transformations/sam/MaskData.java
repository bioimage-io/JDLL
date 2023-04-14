package io.bioimage.modelrunner.transformations.sam;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;

public class MaskData {
	private HashMap<String, Object> stats; 
	private static String ILLEGAL_ARG_MESSAGE = "MaskData only "
				+ "supports list, ImgLib2 RandomAccessibleIntervals "
				+ "or Tensors.";
	
	public MaskData(HashMap<String, Object> kwargs) {
		checkArgs(kwargs);
		stats = kwargs;
	}
	
	private void checkArgs(HashMap<String, Object> kwargs) {
		for (Entry<String, Object> entry : kwargs.entrySet()) {
			if (!checkArg(entry.getValue()))
				throw new IllegalArgumentException(ILLEGAL_ARG_MESSAGE + " Input '" + entry.getKey() 
						+ "' is not of those types.");
		}
	}
	
	private boolean checkArg(Object arg) {
		boolean isList = (arg instanceof ArrayList);
		boolean isImg = (arg instanceof RandomAccessibleInterval);
		boolean isTensor = (arg instanceof Tensor);
		if (!isList && !isImg && !isTensor)
			return false;
		return true;
			
	}
	
	public void set(String key, Object item) {
		if (!checkArg(item))
			throw new IllegalArgumentException(ILLEGAL_ARG_MESSAGE);
		stats.put(key, item);
	}
	
	public void del(String key) {
		if (this.stats.get(key) != null)
			this.stats.remove(key);
	}
	
	public Object get(String key) {
		return this.stats.get(key);
	}
	
	public void filter(boolean[] keep) {
		for (Entry<String, Object> entry : this.stats.entrySet()) {
			if (entry.getValue() == null) {
				return;
			} else if (entry.getValue() instanceof RandomAccessibleInterval) {
				stats.put(entry.getKey(), keepPlanes((RandomAccessibleInterval<FloatType>) entry.getValue(), keep));
			} else if (entry.getValue() instanceof Tensor) {
				RandomAccessibleInterval rai = ((Tensor) entry.getValue()).getData();
				Tensor nTensor = Tensor.build(((Tensor) entry.getValue()).getName(), ((Tensor) entry.getValue()).getAxesOrderString(), 
						keepPlanes(rai, keep));
				((Tensor) entry.getValue()).close();
				stats.put(entry.getKey(), nTensor);
			} else if (entry.getValue() instanceof List) {
				List<Object> list = (List<Object>) entry.getValue();
				List<Object> nList = new ArrayList<Object>();
				for (int i = 0; i < list.size(); i ++) {
					if (keep[i])
						nList.add(list.get(i));
				}
			}
		}
	}
	
	public static RandomAccessibleInterval<FloatType> keepPlanes(RandomAccessibleInterval<FloatType> rai, boolean[] keep) {
		long[] orShape = rai.dimensionsAsLongArray();
		List<Long> keptSlices = new ArrayList<Long>();
		long nSlices = 0;
		for (long i = 0; i < keep.length; i ++) {
			if (keep[(int) i])
				keptSlices.add(i);
		}
		orShape[0] = keptSlices.size();
		Img<FloatType> nRai = 
				new ArrayImgFactory<>(new FloatType()).create(orShape);
		Cursor<FloatType> cursor = nRai.cursor();
		while (cursor.hasNext()) {
			long[] oldPos = cursor.positionAsLongArray();
			oldPos[0] = keptSlices.get((int) oldPos[0]);
			cursor.get().set(rai.getAt(oldPos));
		}
		return nRai;
	}

}
