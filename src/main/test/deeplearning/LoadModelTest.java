package deeplearning;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.Model;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bioimageanalysis.icy.deeplearning.utils.EngineInfo;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import lombok.extern.slf4j.Slf4j;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.real.FloatType;

@Slf4j
public class LoadModelTest {

    @Test
    public void loadModel() throws Exception {
        String enginesDir = "/Users/ekaterina.moreva/Documents/Embl/engine2";
        boolean cpu = true;
        boolean gpu = false;
        EngineInfo engineInfo = EngineInfo.defineDLEngine("tensorflow_saved_model_bundle", "1.15.0", enginesDir, cpu, gpu);

        String modelFolder = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel";
        String modelSource = "/Users/ekaterina.moreva/Documents/Embl/model/PlatyTensorflowModel/saved_model.pb";
        Model model = Model.createDeepLearningModel(modelFolder, modelSource, engineInfo);
        model.loadModel();
        log.info("done load");

        float[] floatArr = new float[524288];
        ArrayImg<FloatType, FloatArray> im = ArrayImgs.floats(floatArr, new long[]{1, 256, 256, 8, 1});
        ArrayImg<FloatType, FloatArray> arr2 = ArrayImgs.floats(new long[]{1, 256, 256, 8, 1});
        Tensor tt = Tensor.build("input", "bxyzc", im);
        List<Tensor> input = Collections.singletonList(tt);
        log.info(Arrays.toString(tt.getShape()));
        log.info(tt.getDataType().toString());


        Tensor tt2 = Tensor.build("output", "bxyzc", arr2);
        List<Tensor> output = Collections.singletonList(tt2);
        log.info(tt2.getDataType().toString());
        log.info(Arrays.toString(tt2.getShape()));

        log.info("done create arrays");

        List<Tensor> result = model.runModel(input, output);
        Assertions.assertFalse(result.get(0).isEmpty());
        Assertions.assertTrue(result.get(0).isImage());

        model.closeModel();
        log.info("done all");
        System.out.println("SOS");
    }
}
