package fiji.io.ml;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.ModelDescription;

import lombok.extern.slf4j.Slf4j;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.SingleCellArrayImg;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

@Slf4j
public class DataLoader implements CellLoader<FloatType> {
    private final Model model;
    private final ModelDescription modelDescription;
    private final RandomAccessibleInterval<?> input;

    public DataLoader(Model model, ModelDescription modelDescription, RandomAccessibleInterval<?> input) {
        this.model = model;
        this.modelDescription = modelDescription;
        this.input = input;
    }

    @Override
    public void load(SingleCellArrayImg<FloatType, ?> cell) {
        loadModel();
        log.info("Try run model");
        List<Tensor> result = null;
        try {
            result = model.runModel(prepareInput(), prepareOutput());
        } catch (Exception e) {
            log.error("Failed to run model", e);
        }
        log.info(Arrays.toString(result.get(0).getData().maxAsDoubleArray()));
        model.closeModel();
    }

    private void loadModel() {
        try {
            model.loadModel();
        } catch (LoadModelException e) {
            log.error("Failed to load model", e);
        }
    }

    private List<Tensor> prepareInput() {
        log.info("create input");
        RandomAccessibleInterval<?> crop = Views.interval(input, new long[]{0, 0, 0}, new long[]{255, 255, 7});
        RandomAccessibleInterval<FloatType> rr = Tensor.createCopyOfRaiInWantedDataType((RandomAccessibleInterval) crop, new FloatType());
        RandomAccessibleInterval<FloatType> rr2 = Views.addDimension(rr, 0, 0);
        rr2 = Views.moveAxis(rr2, rr.numDimensions(), 0);
        rr2 = Views.addDimension(rr2, 0, 0);
        Tensor<FloatType> inpTensor = Tensor.build(modelDescription.getInputs().getName(), modelDescription.getInputs().getAxes(), rr2);
        return Collections.singletonList(inpTensor);
    }

    private List<Tensor> prepareOutput() {
        log.info("create output");
        ArrayImg<FloatType, FloatArray> arr2 = ArrayImgs.floats(new long[]{1, 256, 256, 8, 1});
        Tensor<FloatType> otpTensor = Tensor.build(modelDescription.getOutputs().getName(), modelDescription.getOutputs().getAxes(), arr2);
        return Collections.singletonList(otpTensor);
    }
}
