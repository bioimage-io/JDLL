package org.bioimageanalysis.icy.deeplearning.transformations;


public interface ImageTransformation extends TransformationSpecification {

	enum Mode {
		FIXED("fixed"), PER_DATASET("per_dataset"), PER_SAMPLE("per_sample");
		private final String name;
		Mode(String name) {
			this.name = name;
		}
		public String getName() {
			return name;
		}
	}

	default void setMode(String mode) {
		for (Mode value : Mode.values()) {
			if(value.getName().equals(mode)) {
				setMode(value);
				return;
			}
		}
	}

	void setMode(Mode mode);

	Mode getMode();

}
