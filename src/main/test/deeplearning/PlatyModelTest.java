package deeplearning;

import java.util.Collections;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import bdv.img.cache.VolatileCachedCellImg;
import bdv.util.BdvFunctions;
import bdv.util.volatiles.SharedQueue;
import fiji.io.n5.openers.N5S3Opener;
import lombok.extern.slf4j.Slf4j;
import mpicbg.spim.data.SpimData;
import mpicbg.spim.data.sequence.MultiResolutionSetupImgLoader;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.View;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.StackView;
import net.imglib2.view.Views;

@Slf4j
public class PlatyModelTest {

    @Test
    public void test() throws Exception {
        SharedQueue sharedQueue = new SharedQueue(7);
        SpimData spimData = N5S3Opener.readURL("https://raw.githubusercontent.com/platybrowser/platybrowser/main/data/1.0.1/images/remote/sbem-6dpf-1-whole-raw.xml",
            sharedQueue);
        BdvFunctions.show(spimData);
        MultiResolutionSetupImgLoader<?> imageLoader = (MultiResolutionSetupImgLoader<?>) spimData.
            getSequenceDescription().getImgLoader().getSetupImgLoader(0);
        VolatileCachedCellImg lowestResN5 = (VolatileCachedCellImg) imageLoader.getImage(0, imageLoader.numMipmapLevels() - 1);
        log.info(imageLoader.getMipmapResolutions().toString());

    }

    @Test
    void simpleTest() throws Exception {
        String enginesDir = "/Users/ekaterina.moreva/Documents/Embl/engine2";
        boolean cpu = true;
        boolean gpu = false;
        EngineInfo engineInfo = EngineInfo.defineDLEngine("tensorflow_saved_model_bundle", "1.15.0", enginesDir, cpu, gpu);

        String modelFolder = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel";
        String modelSource = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel/saved_model.pb";
        Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
        model.loadModel();
        log.info("done load");
        SharedQueue sharedQueue = new SharedQueue(7);
        SpimData spimData = N5S3Opener.readURL("https://raw.githubusercontent.com/platybrowser/platybrowser/main/data/1.0.1/images/remote/sbem-6dpf-1-whole-raw.xml",
            sharedQueue);
        RandomAccessibleInterval<FloatType> randomAccessibleInterval = (RandomAccessibleInterval<FloatType>)
            spimData.getSequenceDescription().getImgLoader().getSetupImgLoader(0).getImage(0);
        RandomAccessibleInterval<FloatType> rr = Views.interval(randomAccessibleInterval, new long[]{0, 0, 0}, new long[]{255, 255, 7});
        //insert
        RandomAccessibleInterval<FloatType> rr2 = Views.addDimension(rr, 1, 1);
        rr2 = Views.moveAxis(rr2, 3, 0);
        rr2 = Views.addDimension(rr2, 1, 1);
        Tensor<FloatType> inpTensor = Tensor.build("input", "bxyzc", rr2);
        List<Tensor> input = Collections.singletonList(inpTensor);


        ArrayImg<FloatType, FloatArray> arr2 = ArrayImgs.floats(new long[]{1, 256, 256, 8, 1});
        Tensor<FloatType> otpTensor = Tensor.build("output", "bxyzc", arr2);
        List<Tensor> output = Collections.singletonList(otpTensor);
        List<Tensor> result = model.runModel(input, output);
        Assertions.assertFalse(result.get(0).isEmpty());
        Assertions.assertTrue(result.get(0).isImage());
//
        model.closeModel();
        log.info("done all");
        System.out.println("SOS");
    }

    @Test
    public void loadModel() throws Exception {
//        String enginesDir = "/Users/ekaterina.moreva/Documents/Embl/engine2";
//        boolean cpu = true;
//        boolean gpu = false;
//        EngineInfo engineInfo = EngineInfo.defineDLEngine("tensorflow_saved_model_bundle", "1.15.0", enginesDir, cpu, gpu);
//
//        String modelFolder = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel";
//        String modelSource = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel/saved_model.pb";
//        Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
//        model.loadModel();
//        log.info("done load");

        SharedQueue sharedQueue = new SharedQueue(7);
        SpimData spimData = N5S3Opener.readURL("https://raw.githubusercontent.com/platybrowser/platybrowser/main/data/1.0.1/images/remote/sbem-6dpf-1-whole-raw.xml",
            sharedQueue);

        long[] dim = new long[]{256, 256, 8};
//        - axes: bxyzc
//        data_range: [0.0, 255.0]
//        data_type: uint8
//        name: input
//        preprocessing:
//        - kwargs: {axes: xyc, max_percentile: 100.0, min_percentile: 0, mode: per_sample}
//        name: scale_range
//        shape: [1, 256, 256, 8, 1]
//        license: MIT
//        name: PlatyNucleusData

        long[] imageDimensions = spimData.getSequenceDescription().getImgLoader().getSetupImgLoader(0).getImage(0).dimensionsAsLongArray();
        System.out.println(imageDimensions);
        RandomAccessibleInterval<?> randomAccessibleInterval = spimData.getSequenceDescription().getImgLoader().getSetupImgLoader(0).getImage(0);
        VolatileCachedCellImg volatileCachedCellImg = (VolatileCachedCellImg) randomAccessibleInterval;
        CellGrid cellGrid = volatileCachedCellImg.getCellGrid();


        RandomAccessibleInterval<?> rr = Views.interval(randomAccessibleInterval, new long[]{0, 0, 0}, new long[]{255, 255, 7});
        Tensor<FloatType> inpTensor = Tensor.build("input", "bxyzc", (VolatileCachedCellImg) rr);
        ArrayImg<FloatType, FloatArray> arr2 = ArrayImgs.floats(new long[]{1, 256, 256, 8, 1});


        MultiResolutionSetupImgLoader<?> imageLoader = (MultiResolutionSetupImgLoader<?>) spimData.
            getSequenceDescription().getImgLoader().getSetupImgLoader(0);
        System.out.println(imageLoader.numMipmapLevels());

//        for (int i = 0; i < imageLoader.numMipmapLevels(); i++) {
//            VolatileCachedCellImg lowestResN5 = (VolatileCachedCellImg) imageLoader.getImage(0, i);
//            Tensor<FloatType> inpTensor = Tensor.build("input0", "zyx", lowestResN5);
//            long[] array = lowestResN5.dimensionsAsLongArray();
//            System.out.println(Arrays.toString(array));
//        }
//        RandomAccessibleInterval lowestResN5 = imageLoader.getImage(0, imageLoader.numMipmapLevels() - 1);

//        final ImgFactory< FloatType > imgFactory = new CellImgFactory<>( new FloatType(), 5 );
        // create an 3d-Img with dimensions 20x30x40 (here cellsize is 5x5x5)Ø
//        final Img< FloatType > img1 = imgFactory.create( 1, 1, 512, 512);

//        for (int i =0; i<imageLoader.numMipmapLevels(); i++) {
//            final RandomAccessibleIntervalCellLoader cellLoader = new RandomAccessibleIntervalCellLoader(cellKeyToSource, i);
//            System.out.println(cellLoader.);
//        }
//
//        final CachedCellImg< T, ? > cachedCellImg =
//            new ReadOnlyCachedCellImgFactory().create(
//                mergedDimensions,
//                type,
//                cellLoader,
//                ReadOnlyCachedCellImgOptions.options().cellDimensions( cellDimensions[ level ] ) );
//
//        RandomAccessibleInterval rAI = Views.tiles(lowestResN5, new long[]{256, 256, 8, 1});


    }

    private void runModel(Model model, List<Tensor> inTensors, List<Tensor> outTensors) throws Exception {
        List<Tensor> res = model.runModel(inTensors, outTensors);
        res.forEach(tensor -> {
            tensor.getData();
        });
    }
}

//
//        ArrayImg<FloatType, FloatArray> arr2 = (ArrayImg<FloatType, FloatArray>) RaiArrayUtils.createCopyOfRaiInWantedDataType(lowestResN5, "float32");
//
//        Img arr = RaiArrayUtils.createCopyOfRaiInWantedDataType(lowestResN5, "float32");
//

//        ArrayImg<FloatType, FloatArray> im = ArrayImgs.floats(floatArr, new long[]{1, 256, 256, 8, 1});

//        Tensor tt = Tensor.build("input", "bxyzc", arr2);
//        System.out.println(tt.getShape().toString());
//        List<Tensor> input = Collections.singletonList(tt);
//        log.info(Arrays.toString(tt.getShape()));
//        log.info(tt.getDataType().toString());
//
//
//        AffineTransform ft= new AffineTransform();
//        ft.
//        Tensor tt2 = Tensor.build("output", "bxyzc", arr2);
//        List<Tensor> output = Collections.singletonList(tt2);
//        log.info(tt2.getDataType().toString());
//        log.info(Arrays.toString(tt2.getShape()));
//
//        log.info("done create arrays");
////
//        List<Tensor> result = model.runModel(input, output);
//        Assertions.assertFalse(result.get(0).isEmpty());
//        Assertions.assertTrue(result.get(0).isImage());
//
//        model.closeModel();
//        System.out.println("SOS");
