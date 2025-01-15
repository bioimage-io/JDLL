package io.bioimage.modelrunner.model.stardist_java_deprecate;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class Utils {
	
	protected static String[] axesCheckAndNormalize(String axes, Integer length, Boolean disallowed) {
		String allowed = "STCZYX";
		if (axes == null)
			throw new IllegalArgumentException("Axes cannot be null");
		axes = axes.toUpperCase();
		for (String a : axes.split("")) {
			if (!allowed.contains(a))
				throw new IllegalArgumentException("Invalid axis: " + a + ", it must be one of " + allowed);
			if (axes.replace(a, "").length() + 1 != axes.length())
				throw new IllegalArgumentException("Invalid axis: " + a + " can only appear once.");
		}
		if (length != null && axes.length() != length)
			throw new IllegalArgumentException("Axes (" + axes + ") must have length " + length);
		return new String[] {axes, allowed};
	}
	
	protected static <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> 
	moveImageAxes(RandomAccessibleInterval<T> x, String fr, String to, boolean adjustSingletons) {
		String[] strs = axesCheckAndNormalize(fr, x.numDimensions(), false);
		fr = strs[0];
		strs = axesCheckAndNormalize(to, null, false);
		to = strs[0];
		String frInitial = fr;
		long[] xShapeInitial = x.dimensionsAsLongArray();
		if (adjustSingletons) {
			// TODO
		}
		Set<String> toSet = Arrays.asList(to.split("")).stream().collect(Collectors.toSet());
		Set<String> frSet = Arrays.asList(fr.split("")).stream().collect(Collectors.toSet());
		
		if (!frSet.equals(toSet))
			throw new IllegalArgumentException("Image dims '" + fr + "' not compatible "
					+ "with target dims '" + to + "'.");
		Map<String, Integer> axFrom = axesDict(fr);
		Map<String, Integer> axTo = axesDict(to);
		if (fr.equals(to))
			return x;
		int[] src = new int[fr.length()];
		int[] dest = new int[fr.length()];
		for (int i = 0; i < fr.length(); i ++) {
			String a = fr.split("")[i];
			src[i] = axFrom.get(a);
			dest[i] = axTo.get(a);
		}
		int[] orderChange = new int[src.length];
		for (int i = 0; i < orderChange.length; i ++) {
			int position = Arrays.asList(src).indexOf(dest[i]);
			orderChange[i] = position;
		}
		
		return io.bioimage.modelrunner.tensor.Utils.rearangeAxes(x, orderChange);
	}
	
	protected static Map<String, Integer> axesDict(String axes) {
		String[] strs = Utils.axesCheckAndNormalize(axes, null, null);
		axes = strs[0];
		String allowed = strs[1];
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (String a : allowed.split(""))
			map.put(a, axes.indexOf(a) == -1 ? null : axes.indexOf(a));
		return map;
	}

}
