package io.bioimage.modelrunner.model.stardist_java_deprecate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class Resizer {

	
	private final Map<String, Integer> grid;
	
	private Map<String, int[]> pad;
	
	private Map<String, Integer> paddedShape;
	
	protected Resizer(Map<String, Integer> grid) {
		this.grid = grid;
	}
	
	protected <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> 
	before(RandomAccessibleInterval<T> x, String axes, int[] axesDivBy) {
		for (int i = 0; i < axes.length(); i ++) {
			String ax = axes.split("")[i];
			int g = grid.keySet().contains(ax) ? grid.get(ax) : 1;
			int a = axesDivBy[i];
			if (a % g != 0)
				throw new IllegalArgumentException();
		}
		
		String[] strs = Utils.axesCheckAndNormalize(axes, x.numDimensions(), null);
		axes = strs[0];
		pad = new HashMap<String, int[]>();
		for (int i = 0; i < axes.length(); i ++) {
			long val = (axesDivBy[i] - x.dimensionsAsLongArray()[i] % axesDivBy[i]) % axesDivBy[i];
			String a = axes.split("")[i];
			pad.put(a, new int[] {0, (int) val});
		}
		long[] minLim = x.minAsLongArray();
		long[] maxLim = x.maxAsLongArray();
		int i = 0;
		for (String a : axes.split("")) {
			maxLim[ i ++] += pad.get(a)[1];
		}
		RandomAccessibleInterval<T> xPad = Views.interval(
				Views.extendMirrorDouble(x), new FinalInterval( minLim, maxLim ));
		paddedShape = new HashMap<String, Integer>();
		for (int j = 0; j < axes.length(); j ++) {
			String ax = axes.split("")[j];
			if (ax.toUpperCase().equals("C"))
				continue;
			paddedShape.put(ax.toUpperCase(), (int) xPad.dimensionsAsLongArray()[j]);
		}
		return xPad;
	}
	
	protected <T extends NativeType<T> & RealType<T>> RandomAccessibleInterval<T> 
	after(RandomAccessibleInterval<T> x, String axes) {
		String[] strs = Utils.axesCheckAndNormalize(axes, x.numDimensions(), null);
		axes = strs[0];
		for (int i = 0; i < axes.length(); i ++) {
			String ax = axes.split("")[i];
			long s = x.dimensionsAsLongArray()[i];
			int g = this.grid.keySet().contains(ax) ? grid.get(ax) : 1;
			long sPad = this.paddedShape.keySet().contains(ax) ? paddedShape.get(ax) : s;
			if (sPad != s * g)
				throw new IllegalArgumentException();
		}
		
		int[] end = new int[axes.length()];
		for (int i = 0; i < axes.length(); i ++) {
			String ax = axes.split("")[i];
			int p = pad.keySet().contains(ax) ? pad.get(ax)[1] : 0;
			int g = grid.keySet().contains(ax) ? grid.get(ax) : 1;
			if (p < g)
				end[i] = 0;
			else
				end[i] = -Math.floorDiv(p, g);
		}
		long[] minLim = x.minAsLongArray();
		long[] maxLim = x.maxAsLongArray();
		for (int i = 0; i < maxLim.length; i ++) {
			maxLim[i] += end[i];
		}
		RandomAccessibleInterval<T> crop = Views.interval(
				x, new FinalInterval( minLim, maxLim ));
		return crop;
	}
	
	protected void filterPoints() {
		
	}
}
