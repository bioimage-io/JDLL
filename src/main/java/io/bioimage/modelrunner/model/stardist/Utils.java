package io.bioimage.modelrunner.model.stardist;

public class Utils {
	
	protected static String[] axesCheckAndNormalize(String axes, Integer length, String disallowed) {
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

}
