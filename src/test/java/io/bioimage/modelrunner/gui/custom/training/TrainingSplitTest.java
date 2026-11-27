package io.bioimage.modelrunner.gui.custom.training;

import static org.junit.Assert.*;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class TrainingSplitTest {
    @Rule public TemporaryFolder tmp = new TemporaryFolder();

    private File source(String name, boolean annotated) throws Exception {
        File file = tmp.newFile(name);
        if (annotated) Files.write(file.toPath(), new byte[]{1});
        return file;
    }

    private void assign(TrainingSplit split, List<File> train, List<File> val) throws Exception {
        split.assign(train, val, file -> file, file -> file, file -> file.length() > 0, .2, 42);
    }

    @Test public void explicitBackgroundHoldoutGainsAnAnnotatedSource() throws Exception {
        File output = tmp.newFolder("output");
        List<File> train = new ArrayList<>(Arrays.asList(source("first",true),source("second",true)));
        List<File> val = new ArrayList<>(Collections.singletonList(source("background",false)));
        assign(new TrainingSplit(tmp.getRoot(),output,null,null),train,val);
        assertEquals(1, train.size());
        assertEquals(2, val.size());
        assertTrue(Collections.disjoint(train,val));
        assertFalse(new File(output,"split.json").exists());
    }

    @Test public void retainOneExistingValidationSourceAndAssignOnlyNewSources() throws Exception {
        File parent = tmp.newFolder("parent");
        List<File> originals = new ArrayList<>();
        for (int i=0; i<10; i++) originals.add(source("field"+i,true));
        List<File> train = new ArrayList<>(originals), val = new ArrayList<>();
        assign(new TrainingSplit(tmp.getRoot(),parent,null,null),train,val);
        assertEquals(2,val.size());
        Path dir = parent.toPath();
        Files.createFile(dir.resolve("best.pt"));
        Files.writeString(dir.resolve("continuation.json"), "{\"status\":\"cancelled\",\"epochs\":100,"
                + "\"checkpoints\":{\"best\":{\"weights\":\"best.pt\",\"epoch\":38}}}");
        File survivor = val.get(0);
        originals.remove(val.get(1));
        train = new ArrayList<>(originals);
        val.clear();
        assign(new TrainingSplit(tmp.getRoot(),tmp.newFolder("reduced"),parent.toString(),null),train,val);
        assertEquals(Collections.singletonList(survivor),val);

        train = new ArrayList<>(originals);
        for (int i=0; i<5; i++) train.add(source("new"+i,true));
        val.clear();
        assign(new TrainingSplit(tmp.getRoot(),tmp.newFolder("grown"),parent.toString(),null),train,val);
        assertEquals(2,val.size());
        assertTrue(val.contains(survivor));
        assertTrue(val.stream().anyMatch(f -> f.getName().startsWith("new")));
        assertTrue(Collections.disjoint(train,val));

        Files.write(survivor.toPath(), new byte[0]);
        train = new ArrayList<>(originals);
        val.clear();
        assign(new TrainingSplit(tmp.getRoot(),tmp.newFolder("background"),parent.toString(),null),train,val);
        assertTrue(val.contains(survivor));
        assertEquals(2,val.size());
        assertTrue(val.stream().anyMatch(f -> f.length() > 0));
        assertTrue(Collections.disjoint(train,val));
    }
}
