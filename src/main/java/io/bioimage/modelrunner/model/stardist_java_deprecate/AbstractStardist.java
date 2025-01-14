package io.bioimage.modelrunner.model.stardist_java_deprecate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgs;

public class AbstractStardist {
	/*
	private StardistConfig config;
	
	private float probThres;
	
	private float nmsThres;
	
	private int[][] tileOverlap;
	
	private void predict_instances(RandomAccessibleInterval img, Map<String, Object> kwargs) {
		Map<String, Object> predictKwargs = new HashMap<String, Object>();
		if (kwargs.get("predictKwargs") != null && kwargs.get("predictKwargs") instanceof Map)
			predictKwargs = (Map<String, Object>) kwargs.get("predictKwargs");
		Map<String, Object> nmsKwargs = new HashMap<String, Object>();
		if (kwargs.get("nmsKwargs") != null && kwargs.get("nmsKwargs") instanceof Map)
			nmsKwargs = (Map<String, Object>) kwargs.get("nmsKwargs");
		boolean returnPredict = false;
		boolean sparse = true;
		if (kwargs.get("returnPredict") != null && kwargs.get("returnPredict") instanceof Boolean)
			returnPredict = (boolean) kwargs.get("returnPredict");
		if (kwargs.get("sparse") != null && kwargs.get("sparse") instanceof Boolean)
			sparse = (boolean) kwargs.get("sparse");
		if (returnPredict && sparse)
			sparse = false;
		String axes = "";
		if (kwargs.get("axes") != null && kwargs.get("axes") instanceof String)
			axes = (String) kwargs.get("axes");
		
		String _axes = normalizeAxes(img, axes);
		String axesNet = config.axes;
		long[] shape = permuteAxes(null, null, _axes, axesNet, null, true).dimensionsAsLongArray();
		int[] shapeInst = new int[_axes.toUpperCase().replace("C", "").length()];
		int i = 0;
		for (long s : shape) {
			if (_axes.toUpperCase().split("")[i].equals("C"))
				continue;
			shapeInst[i ++] = (int) s;
		}
		
		Number scale = null;
		if (kwargs.get("scale") != null && kwargs.get("scale") instanceof Number)
			scale = (Number) kwargs.get("scale");
		
		if (scale != null) {
			// TODO
			for (String ax : _axes.split("")) {
			}
		}

		Normalizer normalizer = null;
		if (kwargs.get("normalizer") != null && kwargs.get("normalizer") instanceof Normalizer)
			normalizer = (Normalizer) kwargs.get("normalizer");
		List<Integer> nTiles = null;
		if (kwargs.get("nTiles") != null && kwargs.get("nTiles") instanceof List)
			nTiles = ((List<Integer>) kwargs.get("nTiles"));
		Float probThresh = null;
		if (kwargs.get("probThresh") != null && kwargs.get("probThresh") instanceof Number)
			probThresh = ((Number) kwargs.get("probThresh")).floatValue();
		
		if (sparse) {
			predictSparseGenerator(img, axes, normalizer, nTiles, probThresh);
		} else {
			
		}
		
		
	}
	
	private void predictSparseGenerator(RandomAccessibleInterval img, String axes, Normalizer normalizer, List<Integer> nTiles,
			Float probThresh) {
		if (probThresh == null) probThresh = this.probThres;
		
		Map<String, Object> returns = predictSetup(img, axes, normalizer, nTiles);
		
		RandomAccessibleInterval x = (RandomAccessibleInterval) returns.get("x");
		nTiles = (List<Integer>) returns.get("nTiles");
		axes = (String) returns.get("axes");
		String axesNet = (String) returns.get("axesNet");
		int[] axesNetDivBy = (int[]) returns.get("axesNetDivBy");
		int[] grid = (int[]) returns.get("grid");
		Map<String, Integer> gridDict = (Map<String, Integer>) returns.get("gridDict");
		Resizer resizer = (Resizer) returns.get("resizer");
		int channel = (int) returns.get("channel");
		int product = nTiles.stream().reduce(1, (a, b) -> a * b);
		if (product > 1) {
			// TODO
		} else {
			// TODO
			RandomAccessibleInterval prob = null; //(256, 256, 1)
			RandomAccessibleInterval dist = null; //(256, 256, 32)
		}
		
	}
	
	private Map<String, Object> predictSetup(RandomAccessibleInterval img, String axes, Normalizer normalizer, List<Integer> nTiles) {
		if (nTiles == null) {
			nTiles = new ArrayList<Integer>();
			for (int i = 0; i < img.dimensionsAsLongArray().length; i ++) nTiles.add(1);
		}
		if (nTiles.size() != img.dimensionsAsLongArray().length)
			throw new IllegalArgumentException("The number of image dimensions (" + img.dimensionsAsLongArray().length
					+ ") should be the same as the tile list lenght (" + nTiles.size() + ").");
		axes = normalizeAxes(img, axes);
		String axesNet = this.config.axes;
		RandomAccessibleInterval x = permuteAxes(img, axes, axesNet, null, null, true);
		int channel = Utils.axesDict(axesNet).get("C");
		if (this.config.n_channel_in != x.dimensionsAsLongArray()[channel])
			throw new IllegalArgumentException("The number of channels of the image ("
					+ x.dimensionsAsLongArray()[channel] + ") should be the same as the model config ("
					+ config.n_channel_in + ").");
		int[] axesNetDivBy = axesDivBy(axesNet);
		int[] grid = config.grid;
		if (grid.length != axesNet.length() - 1)
			throw new IllegalArgumentException();
		Map<String, Integer> gridDict = new HashMap<String, Integer>();
		int i = 0;
		for (String a : axesNet.toUpperCase().split("")) {
			if (a == "C")
				continue;
			gridDict.put(a, grid[i ++]);
		}
		
		Resizer resizer = new Resizer(gridDict);
		x = resizer.before(x, axesNet, axesNetDivBy);
		Map<String, Object> returns = new HashMap<String, Object>();
		returns.put("x", x);
		returns.put("axes", axes);
		returns.put("axesNet", axesNet);
		returns.put("axesNetDivBy", axesNetDivBy);
		returns.put("resizer", resizer);
		returns.put("nTiles", nTiles);
		returns.put("grid", grid);
		returns.put("gridDict", gridDict);
		returns.put("channel", channel);
		return returns;
	}
	
	private Integer[] sh;
	
	private void tilingSetup(RandomAccessibleInterval x, List<Integer> nTiles, int[] axesNetDivBy,
			String axesNet, Map<String, Integer> gridDict, int channel) {
		String tilingAxes = axesNet.replace("C", "");
		int[] xTilingAxes = new int[tilingAxes.length()];
		int c = 0;
		for (String a : tilingAxes.split("")) {
			xTilingAxes[c ++] = Utils.axesDict(axesNet).get(a);
		}
		int[] axesNetTileOverlaps = axesTileOverlap(axesNet);
		// TODO nTiles = permuteAxes();
		sh = new Integer[axesNet.length()];
		for (int i = 0; i < axesNet.length(); i ++) {
			String a = axesNet.split("")[i];
			sh[i] = (int) Math.floorDiv(x.dimensionsAsLongArray()[i], 
					gridDict.keySet().contains(a) ? gridDict.get(a) : 1);
		}
		sh[channel] = null;
		int[] nBlockOverlaps = new int[axesNetTileOverlaps.length];
		for (int i = 0; i < axesNetTileOverlaps.length; i ++) {
			nBlockOverlaps[i] = (int) Math.ceil(axesNetTileOverlaps[i] / (double) axesNetDivBy[i]);
		}
	}
	
	private void indProbThresh(RandomAccessibleInterval prob, float probThresh, Integer b) {
		if (b == null)
			b = 2;
		
	}
	
	private RandomAccessibleInterval createEmptyOutput(int nChannel, int channel) {
		sh[channel] = nChannel;
		long[] dims = new long[sh.length];
		for (int i = 0; i < sh.length; i ++)
			dims[i] = sh[i].longValue();
		return ArrayImgs.floats(dims);
	}
	
	private int[] axesTileOverlap(String queryAxes) {
		String[] strs = Utils.axesCheckAndNormalize(queryAxes, null, null);
		queryAxes = strs[0];
		if (this.tileOverlap != null) {
			tileOverlap = computeReceptiveField();
		}
		int i = 0;
		Map<String, Integer> overlap = new HashMap<String, Integer>();
		for (String ax : config.axes.split("")) {
			if (ax.equals("C"))
				continue;
			overlap.put(ax, Math.max(tileOverlap[i][0], tileOverlap[i][1]));
		}
		int[] arr = new int[queryAxes.length()];
		i = 0;
		for (String ax : queryAxes.split(""))
			arr[i ++] = overlap.keySet().contains(ax) ? overlap.get(ax) : 0;
		return arr;
	}
	
	private int[][] computeReceptiveField() {
		return null;
	}
	
	private int[] axesDivBy(String queryAxes) {
		if (this.config.backbone.equals("unet"))
			throw new IllegalArgumentException("Backbone '" + config.backbone + "' not implemented.");
		String[] strs = Utils.axesCheckAndNormalize(queryAxes, null, null);
		queryAxes = strs[0];
		
		if (config.unet_pool.length != config.grid.length)
			throw new IllegalArgumentException();
		int i = 0;
		Map<String, Integer> divBy = new HashMap<String, Integer>();
		for (String a : config.axes.split("")) {
			if (a.toUpperCase().equals("C"))
				continue;
			int val = (int) (Math.pow(config.unet_pool[i], config.unet_n_depth) * config.grid[i]);
			divBy.put(a.toUpperCase(), val);
			i ++;
		}
		int[] arr = new int[queryAxes.length()];
		i = 0;
		for (String a : queryAxes.split("")) {
			arr[i ++] = divBy.keySet().contains(a) ? divBy.get(a) : 1;
		}
		return arr;
	}

	private String normalizeAxes(RandomAccessibleInterval img, String axes) {
		if (axes == null) {
			axes = this.config.axes;
			if (!axes.toUpperCase().contains("C"))
				throw new IllegalArgumentException();
			if (img.numDimensions() == axes.length() - 1 && this.config.n_channel_in == 1)
				axes = axes.replace("C", "");
		}
		return Utils.axesCheckAndNormalize(axes, img.numDimensions(), null)[0];
	}
	
	private RandomAccessibleInterval permuteAxes(RandomAccessibleInterval data, 
			String imgAxesIn, String netAxesIn, String netAxesOut, String imgAxesOut, 
			boolean undo) {
		if (data == null)
			return null;
		if (netAxesOut == null)
			netAxesOut = netAxesIn;
		if (imgAxesOut == null)
			imgAxesOut = imgAxesIn;
		if (imgAxesIn.toUpperCase().contains("C") || imgAxesOut.toUpperCase().contains("C"))
			throw new IllegalArgumentException();
		if (!netAxesIn.toUpperCase().contains("C") || !netAxesOut.toUpperCase().contains("C"))
			throw new IllegalArgumentException();
		
		if (undo && imgAxesIn.toUpperCase().contains("C")) {
			return Utils.moveImageAxes(data, netAxesOut, imgAxesOut, true);
		} else if (undo) {
			// TODO
		} else {
			return Utils.moveImageAxes(data, imgAxesIn, netAxesIn, true);
		}
		return null;
	}
	*/
}
