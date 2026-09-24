/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2026 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * #L%
 */
package io.bioimage.modelrunner.gui.custom;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import javax.swing.SwingUtilities;
import javax.swing.Timer;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationBackend;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingConfig;
import io.bioimage.modelrunner.gui.custom.unet.DenseSegmentationTrainingService;
import io.bioimage.modelrunner.gui.custom.unet.UnetBackend;
import io.bioimage.modelrunner.gui.custom.unet.UnetDatasetInspector.Dimensionality;
import io.bioimage.modelrunner.gui.custom.unet.UnetModelRegistry;
import io.bioimage.modelrunner.model.special.unet.UnetTrainingProgress;
import io.bioimage.modelrunner.model.special.unet.UnetValidationPreview;

public class UNetPluginUIDatasetReviewTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    private UNetPluginUI ui;
    private final Queue<Review> pendingReviews = new ConcurrentLinkedQueue<Review>();
    private final List<Review> reviews = new ArrayList<Review>();
    private final CountDownLatch trainingStarted = new CountDownLatch(1);
    private final CountDownLatch finishTraining = new CountDownLatch(1);
    private volatile DenseSegmentationTrainingConfig trainingConfig;
    private Thread trainingThread;
    private int appliedReviews;
    private int configReads;

    @Before
    public void createUi() throws Exception {
        UnetBackend delegate = new UnetBackend();
        File models = temporaryFolder.newFolder("models");
        DenseSegmentationTrainingService trainer = new DenseSegmentationTrainingService() {
            @Override
            public void train(DenseSegmentationTrainingConfig config,
                    Consumer<UnetTrainingProgress> progress, Consumer<UnetValidationPreview> preview,
                    Consumer<String> log) throws InterruptedException {
                trainingConfig = config;
                trainingThread = Thread.currentThread();
                trainingStarted.countDown();
                finishTraining.await(10, TimeUnit.SECONDS);
            }

            @Override public void close() { finishTraining.countDown(); }
        };
        DenseSegmentationBackend backend = (DenseSegmentationBackend) Proxy.newProxyInstance(
                DenseSegmentationBackend.class.getClassLoader(), new Class<?>[] {DenseSegmentationBackend.class},
                (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "inspectDataset":
                            Review review = pendingReviews.remove();
                            review.thread = Thread.currentThread();
                            review.started.countDown();
                            if (!review.release.await(10, TimeUnit.SECONDS)) {
                                throw new IllegalStateException("Test did not release dataset review");
                            }
                            if (review.failure != null) {
                                throw review.failure;
                            }
                            return review.result;
                        case "applyDatasetReview":
                            appliedReviews++;
                            break;
                        case "createTrainingService":
                            return trainer;
                        case "createTrainingConfig":
                            assertTrue("Training fields must be read on the EDT", SwingUtilities.isEventDispatchThread());
                            configReads++;
                            args[1] = models.getAbsolutePath();
                            break;
                        default:
                            break;
                    }
                    try {
                        return method.invoke(delegate, args);
                    } catch (InvocationTargetException ex) {
                        throw ex.getCause();
                    }
                });
        SwingUtilities.invokeAndWait(() -> {
            ui = new UNetPluginUI(null, null, backend);
            ui.getTrainPanel().getModelNameField().setText("test-model");
            ui.getTrainPanel().getEpochsField().setText("1");
        });
    }

    @After
    public void closeUi() throws Exception {
        if (ui != null) {
            SwingUtilities.invokeAndWait(ui::close);
        }
        for (Review review : reviews) {
            review.release.countDown();
            if (review.thread != null) {
                review.thread.join(5000);
            }
        }
        if (trainingThread != null) {
            trainingThread.join(5000);
        }
        SwingUtilities.invokeAndWait(() -> { });
    }

    @Test
    public void waitsForReviewThenTrainsWithTheChosen3dArchitecture() throws Exception {
        setDataset(temporaryFolder.newFolder("volumes"));
        assertTrainingBlocked(); // Debounce period, before the inspector even starts.
        Review review = startReview(Dimensionality.THREE_D);
        assertTrainingBlocked();
        completeReview(review);
        SwingUtilities.invokeAndWait(() -> {
            assertTrue(ui.getTrainPanel().getTrainActionPanel().getRunButton().isEnabled());
            assertTrue(ui.getTrainPanel().getScratchArchitectureComboBox().isEnabled());
            assertEquals(UnetModelRegistry.SMALL_FAST_3D,
                    ui.getTrainPanel().getSelectedScratchArchitectureValue());
            assertTrue(ui.getTrainPanel().selectScratchArchitectureValue(UnetModelRegistry.SMALL_TRUE_3D));
            // Field validation rewrites the same path; it must not restart review/reset this choice.
            ui.trainUnet();
            assertEquals(1, configReads);
        });
        assertTrue(trainingStarted.await(5, TimeUnit.SECONDS));
        assertEquals(UnetModelRegistry.SMALL_TRUE_3D, trainingConfig.getScratchArchitecture());
        finishTraining.countDown();
        trainingThread.join(5000);
        assertFalse(trainingThread.isAlive());
        SwingUtilities.invokeAndWait(() -> {
            assertTrue(ui.getTrainPanel().getTrainActionPanel().getRunButton().isEnabled());
            assertEquals(1, appliedReviews);
        });
    }

    @Test
    public void ignoresOldResultsEvenDuringTheNextDatasetsDebounce() throws Exception {
        setDataset(temporaryFolder.newFolder("planes"));
        Review oldReview = startReview(Dimensionality.TWO_D);
        setDataset(temporaryFolder.newFolder("volumes"));
        completeReview(oldReview);
        SwingUtilities.invokeAndWait(() -> assertEquals(0, appliedReviews));
        assertTrainingBlocked();
        completeReview(startReview(Dimensionality.THREE_D));
        SwingUtilities.invokeAndWait(() -> {
            assertEquals(1, appliedReviews);
            assertEquals(UnetModelRegistry.SMALL_FAST_3D,
                    ui.getTrainPanel().getSelectedScratchArchitectureValue());
        });
    }

    @Test
    public void ignoresReviewResultsAfterClosing() throws Exception {
        setDataset(temporaryFolder.newFolder("volumes"));
        Review review = startReview(Dimensionality.THREE_D);
        SwingUtilities.invokeAndWait(ui::close);
        completeReview(review);
        SwingUtilities.invokeAndWait(() -> {
            assertEquals(0, appliedReviews);
            ui.trainUnet();
            assertEquals(0, configReads);
        });
    }

    @Test
    public void failedReviewCanBeRetriedWithoutLaunchingTheDefault2dModel() throws Exception {
        setDataset(temporaryFolder.newFolder("volumes"));
        Review review = startReview(Dimensionality.THREE_D);
        review.failure = new IllegalArgumentException("Unreadable dataset");
        completeReview(review);
        SwingUtilities.invokeAndWait(() -> {
            assertTrue(ui.getTrainPanel().getTrainActionPanel().getRunButton().isEnabled());
            ui.trainUnet();
            reviewTimer().stop();
            assertEquals(0, configReads);
        });
        assertTrainingBlocked();
        completeReview(startReview(Dimensionality.THREE_D));
        SwingUtilities.invokeAndWait(() -> assertTrue(
                ui.getTrainPanel().getTrainActionPanel().getRunButton().isEnabled()));
    }

    private void setDataset(File dataset) throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            ui.getTrainPanel().getDatasetField().setText(dataset.getAbsolutePath());
            reviewTimer().stop();
        });
    }

    private Review startReview(Dimensionality result) throws Exception {
        Review review = new Review(result);
        reviews.add(review);
        pendingReviews.add(review);
        SwingUtilities.invokeAndWait(() -> {
            for (java.awt.event.ActionListener listener : reviewTimer().getActionListeners()) {
                listener.actionPerformed(null);
            }
        });
        assertTrue(review.started.await(5, TimeUnit.SECONDS));
        return review;
    }

    private void completeReview(Review review) throws Exception {
        review.release.countDown();
        review.thread.join(5000);
        assertFalse(review.thread.isAlive());
        SwingUtilities.invokeAndWait(() -> { }); // Drain the posted result, without sleeps.
    }

    private void assertTrainingBlocked() throws Exception {
        SwingUtilities.invokeAndWait(() -> {
            assertFalse(ui.getTrainPanel().getTrainActionPanel().getRunButton().isEnabled());
            assertFalse(ui.getTrainPanel().getScratchArchitectureComboBox().isEnabled());
            assertFalse(ui.getTrainPanel().getBaseModelComboBox().isEnabled());
            assertFalse(ui.getTrainPanel().getScratchRadio().isEnabled());
            assertFalse(ui.getTrainPanel().getFineTuneRadio().isEnabled());
            ui.trainUnet();
            assertEquals(0, configReads);
            assertNull(trainingConfig);
        });
    }

    private Timer reviewTimer() {
        try {
            Field field = UNetPluginUI.class.getDeclaredField("datasetReviewTimer");
            field.setAccessible(true);
            return (Timer) field.get(ui);
        } catch (ReflectiveOperationException ex) {
            throw new AssertionError(ex);
        }
    }

    private static final class Review {
        final Dimensionality result;
        final CountDownLatch started = new CountDownLatch(1);
        final CountDownLatch release = new CountDownLatch(1);
        Thread thread;
        RuntimeException failure;

        Review(Dimensionality result) { this.result = result; }
    }
}
