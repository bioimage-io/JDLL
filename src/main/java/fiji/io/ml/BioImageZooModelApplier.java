package fiji.io.ml;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.utils.ModelDescription;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.ReadOnlyCachedCellImgFactory;
import net.imglib2.cache.img.ReadOnlyCachedCellImgOptions;
import net.imglib2.type.Type;

public class BioImageZooModelApplier {
    private final Model model;
    private final Type dataType;
    private final ModelDescription modelDescription;
    private final RandomAccessibleInterval<?> data;
    private final int[] dimensionsAsLongArray;

    public BioImageZooModelApplier(Model model, Type dataType, ModelDescription modelDescription, RandomAccessibleInterval<?> data, int... dimensionsAsLongArray) {
        this.model = model;
        this.dataType = dataType;
        this.modelDescription = modelDescription;
        this.data = data;
        this.dimensionsAsLongArray = dimensionsAsLongArray;
    }

    public RandomAccessibleInterval<?> applyModel() {
        return new ReadOnlyCachedCellImgFactory().create(
            data.dimensionsAsLongArray(),
            dataType,
            new DataLoader(model, modelDescription, data),
            ReadOnlyCachedCellImgOptions.options().cellDimensions(dimensionsAsLongArray)
        );
    }
}
