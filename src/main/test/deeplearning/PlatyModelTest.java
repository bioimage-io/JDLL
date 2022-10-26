package deeplearning;

import java.util.Collections;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import bdv.util.BdvFunctions;
import bdv.util.BdvStackSource;
import bdv.util.volatiles.SharedQueue;
import bdv.viewer.Source;
import fiji.io.n5.openers.N5S3Opener;
import lombok.extern.slf4j.Slf4j;
import mpicbg.spim.data.SpimData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

@Slf4j
public class PlatyModelTest {

    @Test
    void simpleTest() throws Exception {
        Model model = loadModel();
        SharedQueue sharedQueue = new SharedQueue(7);
        SpimData spimData = N5S3Opener.readURL("https://raw.githubusercontent.com/platybrowser/platybrowser/main/data/1.0.1/images/remote/sbem-6dpf-1-whole-raw.xml",
            sharedQueue);
        final BdvStackSource<?> stackSource = BdvFunctions.show(spimData).get(0);
        final Source<?> source = stackSource.getSources().get(0).getSpimSource();
        final RandomAccessibleInterval<?> randomAccessibleInterval = source.getSource(0, 3);

        RandomAccessibleInterval<FloatType> predictions = (RandomAccessibleInterval<FloatType>) getPredictions(model, createInput(randomAccessibleInterval)).get(0).getData();
//        RandomAccessibleInterval< FloatType > res = new ReadOnlyCachedCellImgFactory().create(
//            randomAccessibleInterval.dimensionsAsLongArray(),
//            new FloatType(),
//            predictions,
//            ReadOnlyCachedCellImgOptions.options().cellDimensions( 256, 256, 8 )
//        );
//        final RandomAccessibleInterval< Volatile< FloatType > > volatilePredictions = VolatileViews.wrapAsVolatile( predictions );
        BdvFunctions.show(predictions, "prediction");
        Thread.sleep(5000);
        log.info("done all");
    }

    private List<Tensor> createInput(RandomAccessibleInterval<?> randomAccessibleInterval) {

//        RandomAccessibleInterval<FloatType> randomAccessibleInterval = (RandomAccessibleInterval<FloatType>)
//            spimData.getSequenceDescription().getImgLoader().getSetupImgLoader(0).getImage(0);
        RandomAccessibleInterval<?> crop = Views.interval(randomAccessibleInterval, new long[]{0, 0, 0}, new long[]{255, 255, 7});
        RandomAccessibleInterval<FloatType> rr = Tensor.createCopyOfRaiInWantedDataType((RandomAccessibleInterval) crop, new FloatType());

        RandomAccessibleInterval<FloatType> rr2 = Views.addDimension(rr, 1, 1);
        rr2 = Views.moveAxis(rr2, rr.numDimensions(), 0);
        rr2 = Views.addDimension(rr2, 1, 1);
        Tensor<FloatType> inpTensor = Tensor.build("input", "bxyzc", rr2);
        return Collections.singletonList(inpTensor);
    }

    private List<Tensor> getPredictions(Model model, List<Tensor> inputTensors) throws Exception {

        ArrayImg<FloatType, FloatArray> arr2 = ArrayImgs.floats(new long[]{1, 256, 256, 8, 1});
        Tensor<FloatType> otpTensor = Tensor.build("output", "bxyzc", arr2);
        List<Tensor> output = Collections.singletonList(otpTensor);
        List<Tensor> result = model.runModel(inputTensors, output);
        model.closeModel();
        return result;
    }

    @NotNull
    private Model loadModel() throws Exception {
        String enginesDir = "/Users/ekaterina.moreva/Documents/Embl/engine2";
        boolean cpu = true;
        boolean gpu = false;
        EngineInfo engineInfo = EngineInfo.defineDLEngine("tensorflow_saved_model_bundle", "1.15.0", enginesDir, cpu, gpu);

        String modelFolder = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel";
        String modelSource = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel/saved_model.pb";
        Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
        model.loadModel();
        log.info("done load");
        return model;
    }

}
