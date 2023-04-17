package io.bioimage.modelrunner.transformations.sam;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;

public class AutomaticMaskGenerator {
	
	private int minMaskRegionArea = 1;
	private float boxNmsThres = 2;
	private float cropNmsThres = 2;
	private String outputMode;
	
	private static final String COCO_MODE = "coco_rle";
	private static final String BINARY_MODE = "binary_mask";
	private static final String RLES_KEY = "rles";
	private static final String SEGMENTATIONS_KEY = "segmentations";
	private static final String AREA_KEY = "area";
	private static final String BOXES_KEY = "boxes";
	private static final String IP_KEY = "iou_preds";
	private static final String POINTS_KEY = "points";
	private static final String SS_KEY = "stability_score";
	private static final String CP_KEY = "crop_boxes";
	
	
	private static final String ANN_SEGMENTATION_KEY = "segmentation";
	private static final String ANN_AREA_KEY = "area";
	private static final String ANN_BBOX_KEY = "bbox";
	private static final String ANN_PRED_IOU_KEY = "predicted_iou";
	private static final String ANN_PC_KEY = "point_coords";
	private static final String ANN_SS_KEY = "stability_score";
	private static final String ANN_CB_KEY = "crop_box";



	public < R extends RealType< R > & NativeType< R > > List<HashMap<String, Object>> generate( final Tensor< R > input ) {
		MaskData maskData = generateMasks(input);
		
		if (minMaskRegionArea > 0) {
			maskData = postprocessSmallRegions(maskData, minMaskRegionArea, Math.max(boxNmsThres, cropNmsThres));
		}
		
		if (outputMode.equals(COCO_MODE)) {
			// TODO
			maskData.set(SEGMENTATIONS_KEY, null);
		} else if (outputMode.equals(BINARY_MODE)) {
			List<Img<IntType>> list = new ArrayList<Img<IntType>>();
			List<HashMap<String, Object>> mapsList = (List<HashMap<String, Object>>) maskData.get(RLES_KEY); 
			for (HashMap<String, Object> rle : mapsList) {
				list.add(rleToMask(rle));
			}
			maskData.set(SEGMENTATIONS_KEY, list);
		} else {
			maskData.set(SEGMENTATIONS_KEY, maskData.get(RLES_KEY));
		}
		
		List<HashMap<String, Object>> currAnns = new ArrayList<HashMap<String, Object>>();
		int size = ((List<Object>) maskData.get(SEGMENTATIONS_KEY)).size();
		for (int i = 0; i < size; i ++) {
			HashMap<String, Object> ann = new HashMap<String, Object>();
			ann.put(ANN_SEGMENTATION_KEY, ((List<Object>) maskData.get(SEGMENTATIONS_KEY)).get(i));
			ann.put(ANN_AREA_KEY, areaFromRle(((List<HashMap<String, Object>>) maskData.get(RLES_KEY)).get(i)));
			ann.put(ANN_BBOX_KEY, boxXYXYToXYWH(((List<double[]>) maskData.get(BOXES_KEY)).get(i)));
			ann.put(ANN_PRED_IOU_KEY, ((float[]) maskData.get(IP_KEY))[i]);
			double[][] points = new double[1][2];
			points[0] = ((double[][]) maskData.get(POINTS_KEY))[i];
			ann.put(ANN_PC_KEY, points);
			ann.put(ANN_SS_KEY, ((float[]) maskData.get(SS_KEY))[i]);
			ann.put(ANN_CB_KEY, boxXYXYToXYWH(((List<double[]>) maskData.get(CP_KEY)).get(i)));
			currAnns.add(ann);
		}
		return currAnns;
	}

	private < R extends RealType< R > & NativeType< R > > MaskData generateMasks( final Tensor< R > input ) {
		long[] dims = input.getData().dimensionsAsLongArray();
		int hInd = input.getAxesOrderString().toLowerCase().indexOf("y");
		int wInd = input.getAxesOrderString().toLowerCase().indexOf("x");
		int[] origSize = new int[] {(int) dims[hInd], (int) dims[wInd]};
		Object[] cropBoxesLayerIdxs = generateCropBoxes(origSize, this.cropNLayers, this.cropOverlapRatio);
		List<int[]> cropBoxes = (List<int[]>) cropBoxesLayerIdxs[0];
		List<Integer> layerIdxs = (List<Integer>) cropBoxesLayerIdxs[1];
		
		MaskData data = new MaskData();
		for (int i = 0; i < cropBoxes.size(); i ++) {
			processCrop(input, cropBoxes.get(i), layerIdxs.get(i), origSize);
		}
		
		if (cropBoxes.size() > 1) {
			List<Double> scores = boxArea((List<Double[]>) data.get(ANN_CB_KEY));
			for (int i = 0; i < scores.size(); i ++)
				scores.set(i, 1 / (scores.get(i) + 1e-20));
			int[] keepByNms = batchedNms(data.get(BOXES_KEY), scores,
					new double[((List<double[]>) data.get(BOXES_KEY)).size()], 
					this.cropNmsThres);
			data.filter(keepByNms);
		}
	}
	
	public static int[] batchedNms(List<double[]> boxes, List<Double> scores,
										double[] idxs, float iouThreshold) {
		return batchedNmsVanilla(boxes, scores, idxs, iouThreshold);
		// TODO batchedNmsCoordinateTrick();
	}
	
	private static int[] batchedNmsVanilla(List<double[]> boxes, List<Double> scores,
			double[] idxs, float iouThreshold) {
		boolean[] keepMask = new boolean[scores.size()];
		List<Double> uniqueVals = Arrays.stream(idxs)
	            .distinct().boxed().collect(Collectors.toList());
		for (int i = 0; i < uniqueVals.size(); i ++) {
			final Double val = uniqueVals.get(i);
			int[] currIndices = IntStream.range(0, idxs.length)
					.filter(j -> idxs[j] == val).toArray();
			List<double[]> newBoxes = Arrays.stream(currIndices)
					.mapToObj(boxes::get).collect(Collectors.toList());
			List<Double> newScores = Arrays.stream(currIndices)
					.mapToObj(scores::get).collect(Collectors.toList());
			List<Integer> currKeepIndices = NMS.greedyNMS(newBoxes, newScores, iouThreshold);
			for (int ii : currKeepIndices) {
				keepMask[currIndices[ii]] = true;
			}
		}
		List<Integer> keepIndices = new ArrayList<Integer>();
		for (int i = 0; i < keepMask.length; i ++) {
			if (keepMask[i])
				keepIndices.add(i);
		}
		List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < keepIndices.size(); i++) {
            indices.add(i);
        }
        List<Double> newScores = keepIndices.stream()
				.map(scores::get).collect(Collectors.toList());
        // Sort the indices list based on the values in list2
        Collections.sort(indices, (i, j) -> -Double.compare(newScores.get(i), newScores.get(j)));
        
        // Create a new list that is sorted based on list2
        int[] sortedList = new int[indices.size()];
        for (int i = 0; i < indices.size(); i ++) 
        	sortedList[i] = keepIndices.get(indices.get(i));
		return sortedList;
	}
	
	private MaskData postprocessSmallRegions(MaskData maskData, int minArea, float nmsThres) {
		return null;
	}
	
	
	public static Img<IntType> rleToMask(HashMap<String, Object> rle) {
		int[] hw = (int[]) rle.get("size");
		Img<IntType> mask = new ArrayImgFactory<>(new IntType()).create(hw[0], hw[1]);
		int idx = 1;
		int parity = 0;
		int[] countArr = (int[]) rle.get("counts");
		for (int count : countArr) {
			int idx2 = idx + count;
			for (int i = idx; i < idx2; i ++) {
				int y = i / hw[0];
				int x = i % hw[1];
				mask.getAt(new int[] {y, x}).setInt((parity + 1) / 2);
			}
			idx += count;
			parity *= -1;
		}
		
		return mask;
	}
	
	public static int areaFromRle(HashMap<String, Object> rle) {
		int[] rleCounts = (int[]) rle.get("counts");
		List<Integer> cc = new ArrayList<Integer>();
		int idx = 1;
		while (idx < rleCounts.length) {
			cc.add(rleCounts[idx]);
			idx += 2;
		}
		int sum = 0;
		for (int i : cc) {
			sum += i;
		}
		return sum;
	}
	
	public static double[] boxXYXYToXYWH(double[] xyxy) {
		double[] xywh = xyxy;
		xywh[2] = xywh[2] - xywh[0];
		xywh[3] = xywh[3] - xywh[1];
		return xywh;
	}
	
	public static List<Double> boxArea(List<Double[]> vertices) {
		List<Double> areas = new ArrayList<Double>();
		for (int i = 0; i < vertices.size(); i ++) {
			Double[] arr = vertices.get(i);
			areas.add((arr[2] - arr[0]) * (arr[3] - arr[1]));
		}
		return areas;
	}
}
