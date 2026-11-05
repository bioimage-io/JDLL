# Removed JDLL Test Inventory

Recorded on 2026-11-05, before removing the test suite at the user's request.
This is a historical reconstruction reference, not an executable test suite or
a requirement to restore all of these tests.

## Scope And Recovery

The source snapshot is commit `6b3be8f026fd03359de6a10e9e3d648d6e24baba`.
The inventory covers this JDLL repository only, not the independent Python
backend repositories. It contains 188 JUnit test methods in 47 Java classes,
86 named Python unittest methods in nine files, three additional script-style
Python bridge checks, and the model/environment smoke checks described below.
Java methods that invoke Python overlap with the corresponding Python coverage;
these counts are source entry points, not independent scenarios or a coverage score.

Original files can be inspected without restoring the suite using
`git show 6b3be8f026fd03359de6a10e9e3d648d6e24baba:<original-path>`.
Fixtures, mocks, supporting methods, and exact assertions remain in that history.
All paths below are original paths, not links to files still present.

Removed infrastructure includes `src/test/`, the explicit JUnit dependency,
Pixi test scripts/helpers, the Pixi smoke composite action/workflow, and the
automated example/engine-download workflows. Test-only README instructions were
removed. Normal build/deployment and engine-version maintenance remain.
Production `TestArtifact.java` models BioImage.IO test artifacts; it is not a
test implementation and remains. SDK usage examples and runtime validation,
dataset validation, GPU probing, and error checks also remain.

## Execution Context And Known Limitations

- Java tests used JUnit 4.13.2, temporary directories, synthetic ImgLib2 arrays,
  tiny generated PNG/TIFF/NumPy files, and fake services where possible.
- Swing tests used the EDT; some used displayed windows, asynchronous waits,
  Robot/mouse actions, or timing-sensitive assertions. The real-window
  installation-log following check skipped in headless environments.
- Several Java tests checked generated Python text rather than executing it.
  Python-backed tests required an appropriate installed environment and were
  opt-in through the environment variables noted below.
- `JDLL_TEST_PYTHON` enabled StarDist decoder checks;
  `JDLL_STARDIST_NUMPY_SMOKE=1` additionally enabled generated NumPy training.
- `JDLL_STARDIST_TEST_PYTHON` enabled the real StarDist validation suite.
- `JDLL_UNET_TEST_PYTHON` enabled generated UNet inference/validation scripts.
  The validation script needs NumPy, TIFF, PyTorch, and jdll-unet; the inference
  output-selection script uses only Python's standard library and a fake API.
- `JDLL_CROSS_GOOSE_INTEGRATION=true` enabled real Cross-GOOSE worker tests.
  `JDLL_PIXI_SMOKE_MANIFEST` and optional `JDLL_PIXI_SMOKE_ENVIRONMENT` selected
  the actual Pixi prefix for the Appose shared-memory check.
- YOLO NumPy checks used `JDLL_PYTHON` when provided and the installed Python
  dependencies required by their helpers. Synthetic model checks are not
  accuracy or representative performance benchmarks.
- `outputNamesPreserveExtensionsWhenBasenamesCollide` supplied both
  `sample.png` and `Sample.PNG` as if they were distinct inputs. This fixture
  fails on Windows because Java File keys are case-insensitive there. Future
  coverage must distinguish real distinct files from aliases of one file.
- This document records what was checked, not a guarantee that every scenario
  passed on every OS. Removing the suite removes automated regression coverage.

## Shared Inference And UI

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/`.

### InferenceRunFilesTest.java

- `singleFileAndFolderUseOneSiblingRootAndFreshModelRuns`: File/folder inference shares a sibling results root, reserves a fresh model run, leaves inputs/prior outputs unchanged, and rejects outputs for unknown inputs.
- `concurrentRunsCannotReserveTheSameDirectory`: Eight concurrent run reservations receive distinct directories.
- `outputNamesPreserveExtensionsWhenBasenamesCollide`: Retains source extensions and adds disambiguation when output basenames collide; see the Windows fixture limitation above.
- `everySegmentationFolderSelectorSkipsAnnotationsButAllowsExplicitFiles`: All segmentation UIs exclude known annotation/output suffixes during folder discovery but accept explicitly selected masks; verifies exclusion logging and allowed ordinary names.
- `yoloExportsStayInRunDirectoryAndKeepOriginalSourceIdentity`: CSV/GeoJSON detections go into the run directory while retaining original image paths.
- `occupiedRootFailsWithoutChangingInputsAndModelNamesCannotEscapeRunRoot`: An occupied results-root file fails safely; sanitization prevents model names escaping the results directory.

### InferenceStackTest.java

- `planarInferenceKeepsEveryZAndTimeAndAggregatesProgress`: A 2D model processes every Z/T plane, preserving output axes and aggregating progress.
- `volumeInferenceKeepsCompleteZForEveryTimePoint`: Volumetric inference processes a complete Z volume for each T entry.
- `singletonAxesAndSinglePlanesRetainTheirMeaning`: Singleton dimensions do not cause incorrect plane/volume interpretation.
- `cancellationBetweenEntriesDoesNotStartAnotherInference`: Cancellation prevents the next plane/time-point invocation.
- `tiffExportPreservesAllPlanesTimeMetadataAndLargeLabels`: TIFF saving preserves Z/T layout, metadata, and label values beyond uint16 range.
- `detectionsAndExportsRetainExplicitZTimeAndClassNames`: Detection structures and exports retain plane/time positions and class names.
- `yoloServiceRunsEveryPlaneAndPositionsTheDetections`: The YOLO service iterates all planes and associates detections with their Z/T positions.

### SizeHintsTest.java

- `sharedHintsAndIndependentExceptionsNeverDependOnZ`: Shared hints and per-file/time exceptions resolve independently of Z navigation.
- `runSnapshotsAreIndependentAndCellposeResolvesDiameterPerTimePoint`: Running tasks receive independent hint snapshots and Cellpose resolves the correct time-point diameter.

### UNetPluginUIDatasetReviewTest.java

- `waitsForReviewThenTrainsWithTheChosen3dArchitecture`: Dataset review finishes before training and the selected 3D architecture reaches the training service.
- `ignoresOldResultsEvenDuringTheNextDatasetsDebounce`: Results from an older asynchronous review cannot overwrite a newly selected dataset.
- `ignoresReviewResultsAfterClosing`: Late review completion cannot update a closed UI.
- `failedReviewCanBeRetriedWithoutLaunchingTheDefault2dModel`: A failed metadata review permits retry without accidentally training the default 2D architecture.

## Dataset Preparation And Image Readers

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/training/`.

### ImageReaderCompatibilityTest.java

- `sharedAndIndividualSizeHintsFollowFilesAndTimeButNeverZ`: Preview navigation preserves shared/per-entry size hints and does not introduce slice-specific hints.
- `aSingleVolumeKeepsItsLatestRectangleOnEverySliceWithoutASharingCheckbox`: One volume retains the most recently drawn rectangle across Z without an unnecessary sharing control.
- `outsideClicksCommitThroughTheInstalledListenerAndDetachWithThePanel`: Clicking outside the path field commits it, and the installed listener is removed with the panel.
- `rapidSliceChangesDoNotInterruptOrPoisonTheSharedDiskReader`: Fast Z navigation does not interrupt the shared decoder or leave future previews unusable.
- `sourceSelectionAndFocusLossPreserveNavigatedSystemPreviewAcrossAllWorkflows`: Host-image events/focus changes cannot reset an active filesystem preview or its selected plane.
- `diskPreviewNavigatesZTimeAndExactInferenceFileOrderOffTheEdt`: Filesystem previews follow inference ordering and navigate Z/T without decoding on the EDT.
- `staleMetadataCannotReplaceNewerFileAndItsReaderIsClosed`: Stale asynchronous metadata is ignored and its reader is closed.
- `openImageNavigationPreservesXyCzTAxes`: Navigation through host-opened images preserves XY/C/Z/T semantics.
- `hostImageReaderExposesOneStepArrowsAndDoesNotLoadAWholeStack`: Host-backed controls support single-step navigation and request only the displayed plane.
- `imageJLookupTableFailureUsesOneCachedHostPreviewOffTheEdt`: A Java ImageIO LUT failure falls back to one cached host preview, off the EDT.
- `failedPreviewShowsAnErrorInsteadOfRetryingForever`: Decoder failure produces a visible error rather than an infinite rendering/retry state.
- `inferenceReadersDelegateWithoutConvertingPixelsOrDroppingCztAxes`: Host reader delegation preserves pixels and channel/Z/T dimensions.
- `trainingMaskReadersPreserveSixteenBitLabelsWithWrongExtensions`: Mask decoding preserves 16-bit labels even when the extension disagrees with the image contents.

### SegmentationDatasetPreparerTest.java

- `reusesBackendCompatibleRootAliases`: Reuses supported image/mask folder aliases without unnecessary dataset generation.
- `normalizesNamesOutsideBackendSuffixContract`: Unsupported pairing names produce a normalized, linked dataset with backend-compatible names.
- `linksMetadataWithRenamedImagesAndMasks`: Companion JSON metadata follows renamed image/mask links.
- `linksMetadataIntoGeneratedTrainingAndValidationSplits`: Sidecars follow samples into generated train/validation splits.
- `reusesCompatibleDatasetWithExistingMetadata`: Already compatible datasets and sidecars remain in place.
- `reusesEveryBackendSupportedSuffixCombination`: Supported image/mask suffix combinations do not trigger unnecessary normalization.
- `supportsAndNormalizesValidationAliases`: Handles validation-folder aliases, normalizing only when required by the backend.
- `supportsImagesFirstSplitLayoutThroughLinks`: Converts images/train plus masks/train layouts through links.
- `supportsRootAndSplitMixedFoldersWhenMasksHaveSuffixes`: Handles mixed image/mask folders at the root and within supplied splits.
- `rejectsMixedFoldersWithoutMaskSuffix`: Does not infer image-versus-mask roles from extension differences alone.
- `rejectsMissingMasksAndMismatchedShapes`: Reports missing masks or unusable geometry rather than silently passing invalid pairs onward.
- `selectsStarDistChannelsFromImageContent`: Resolves StarDist channel requirements from actual grayscale/multichannel content.
- `createsChannelStratifiedSplitForMixedStarDistDataset`: Generated StarDist splits retain representation of mixed channel types.
- `detectsVolumesAndRejectsMixedDimensionality`: Detects volumetric StarDist data and rejects mixed 2D/3D StarDist datasets.
- `unetMixedDatasetsKeepAllSourcesAndLeaveSplittingToPython`: Keeps UNet planar/volumetric sources and delegates automatic eligibility-aware splitting.
- `unetPreservesExplicitMixedSplitsAndMetadata`: Preserves supplied mixed-dimensional splits and their metadata links.
- `unetSingleVolumeIsNotReusedAsValidation`: Does not create a misleading duplicate validation source from a single volume.
- `unetRecognizesPlanarChannelStacksAndSingletonZ`: Distinguishes channel stacks and Z=1 images from true volumes.
- `unetRecognizesMultichannelVolumesWithFewerMaskPages`: Allows valid multichannel volumes whose masks have fewer TIFF pages.
- `unetUsesDeclaredSpatialAxesForChannelLastMasks`: Uses declared axes instead of assuming page count equals Z for channel-last masks.
- `unetDefersAmbiguousPageCountsButDropsKnownSpatialMismatches`: Defers ambiguous geometry to Python while rejecting definite spatial mismatch.
- `unetReviewFindsVolumeAfterFirst48ImagesAndIgnoresUnpairedNestedFiles`: Dataset review is not limited to the first 48 files and ignores unrelated nested/unpaired sources.
- `unetDefersUnreadableAxesMetadataToPython`: Unreadable axis metadata does not cause premature Java rejection.

### SegmentationNpyDatasetTest.java

- `normalizesMixedNumpyRasterPairsUsingLinksOnly`: Mixed NumPy/raster pairs are normalized without copying pixel arrays.
- `reviewsVolumesAndRejectsTheFirstInvalidMask`: NumPy geometry review recognizes volumes and stops at the first unsupported mask with a useful error.
- `ignoresSameStemNumpyImageCachesBeforeReadingHeaders`: Raster companions take precedence over same-stem NumPy image caches before expensive header inspection.

### ValidationRequestsTest.java

- `tokensAreRunScopedAndReadyGatedAndCleanedUp`: Full-validation request tokens are scoped to a run, require a ready backend, and are cleaned up at closure.

## UNet Training UI And Service

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/unet/`.

### UnetTrainPanelTest.java

- `offersExpectedArchitecturesAndDefaultsOnTheEdt`: Architecture options/defaults match planar, mixed, and volumetric dataset review results.
- `defaultsToCompatibleCustomConfigsOnlyAfterReview`: A compatible custom config becomes default only after geometry review.
- `compatibilityPreservesSourceDimensionality`: Fine-tuning compatibility respects source-model dimensionality.
- `metadataReviewShowsActivityAndLocksOnlyModelSelectionAndTraining`: Review feedback is visible and disables only the affected controls.
- `sharedCrossGoosePanelKeepsItsFixedArchitectures`: Reusing the panel does not expose UNet architecture choices for Cross-GOOSE.

### UnetTrainingServiceTest.java

- `customScratchConfigProvidesArchitectureAndLearningRate`: Scratch training forwards the custom architecture and explicit learning rate.
- `fineTuningLeavesArchitectureAndLearningRateResolutionToBackend`: Fine-tuning sends source-model intent rather than reconstructing architecture/LR in Java.
- `forwardsGeometrySettingsAndNestedPoliciesWithoutResolvingThem`: Geometry, sampling, and nested policies survive Java forwarding unchanged.
- `checksDatasetCompatibilityForScratchCustomAndFineTuning`: Compatibility checks apply consistently to default scratch, custom scratch, and source-model fine-tuning.

## Shared Training UI, YOLO And Preview Rendering

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/yolo/`.

### TrainingUiTest.java

- `inferenceModelLabelsAreConsistentAcrossWorkflows`: Model-selection labels use the shared Model wording across workflows.
- `inferenceStartsWithImmediateFeedbackAndTheTimerAlreadyRunning`: Inference shows a log entry and starts elapsed-time feedback immediately.
- `trainingViewButtonsHighlightOnlyTheVisibleView`: Loss/metric/validation/log controls highlight only the active view.
- `startingTrainingRestoresLogCaretAndFollowing`: Starting a run resets the log caret and follow-to-bottom state.
- `logLifecycleAndFollowingRespectUserNavigation`: Log clearing, copying, completion display, and automatic following respect manual navigation.
- `annotationColorsAreIndependentOfReadableTitles`: Annotation color changes do not affect the readable white-on-dark image title.

### ValidationPreviewColorTest.java

- `pngMasksCanBeRecoloredWithoutReloadingOrMovingTheViewport`: Recolors binary/multiclass/instance raster previews without reloading images or resetting viewport state.
- `overlayOnlyLegacyPreviewsStillLoadAndEmptyMasksRemainEditable`: Overlay-only previews still display; empty masks do not disable color changes.

### PreviewTileCacheTest.java

- `readsOnlyVisiblePyramidTilesAndReusesThem`: Only visible pyramid tiles are read and repeat navigation reuses cached tiles.
- `cacheBoundsBothPositionsAndRgbBytes`: Tile cache admission/eviction respects both entry and byte limits.

### StardistTrainPanelTest.java

- `enablesFineTuneControlsWhenSelected`: Fine-tuning selection enables the correct source-model controls.
- `fullValidationIsOptInAndOneShotDuringTraining`: Full validation defaults off and a user request is one-shot during a run.
- `completingOldPassDoesNotDeselectNewRequest`: Completion of an older pass cannot clear a newly queued request.

### YoloAccelerationCheckBoxTest.java

- `unavailableGpuCannotBeReenabledAfterInferenceOrMethodChange`: Workflow unlocks cannot re-enable acceleration when no supported GPU is available.
- `availableGpuKeepsSelectionWhileTemporarilyLocked`: Temporary run locking preserves a valid user's acceleration selection.
- `completingProbeDoesNotOverrideAnActiveRunLock`: Asynchronous availability completion does not unlock a busy UI.
- `failedNativeProbeLeavesAccelerationDisabled`: Failed native device probing leaves GPU selection disabled.

### YoloNpyDatasetTest.java

- `preparesReferencedNumpyImagesAndUint16MasksWithoutCopies`: YOLO preparation references NumPy inputs, preserves uint16 mask values, and exercises numeric-image training/inference/preview helpers when Python is available.
- `handlesFortranOrderAndStopsAtFirstInvalidMask`: Handles Fortran-order arrays and stops on the first invalid NumPy mask.

### YoloTrainingConfigTest.java

- `writesPortableFormatZeroConfig`: Writes format-0 training/runtime/software settings without mixing preview/output state into the model config.
- `appliesCustomTrainingOptionsAndRelativeArchitecture`: Resolves relative architecture paths and forwards custom LR, augmentation, image size, and batch settings while honoring requested epochs.
- `exposesAllFiveScratchArchitectures`: Scratch training offers official n/s/m/l/x architectures.

## StarDist UI And Fine-Tuning

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/stardist/`.

### DenseValidationPreviewTest.java

- `volumetricChannelsAreMovedWithoutConvertingLabelPrecision`: Dense preview axis conversion preserves volume channels and label precision.
- `fast3dPreviewShowsOnlyTheCentralContextPlanePerChannel`: 2.5D previews display the central target plane, not the full context stack as predictions.

### StardistFineTuneTest.java

- `resolvesOnlyCompleteTwoDimensionalModels`: Selects complete 2D StarDist models and rejects incomplete sources.
- `generatedTaskAdaptsChannelsAndPreservesLastCheckpoint`: Generated fine-tuning code adapts channels and preserves the last-checkpoint behavior.
- `resolvesThreeDimensionalModelsForThreeDimensionalDatasetsOnly`: Requires volumetric dataset compatibility for a 3D source model.
- `generatedThreeDimensionalTaskPreservesVolumesAndResolvesAnisotropy`: Generated 3D tasks preserve volume layout and apply anisotropy resolution.

## Cellpose

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/cellpose/`.

### CellposeImageIOTest.java

- `savesAllZTPlanesAsFloatLabelsAndDoesNotReprocessOrOverwriteThem`: Saves every Z/T plane with float label precision, excludes generated outputs from discovery, and avoids overwrites.

### CellposeInferenceTest.java

- `packsArbitrarySourceChannelsWithoutRescalingIntensity`: Packs selected source channels without altering intensity values.
- `selectorsPreserveValidSelectionsAndRejectUnavailableChannels`: Channel selectors preserve valid choices and reject unavailable channels.
- `preservesAllPlanesAndEstimatesDiameterIndependentlyWithoutCarryOver`: Processes all planes with independent diameter estimation where appropriate.
- `manualDiameterAndOptionalIntermediateShapesAreStable`: Explicit diameter and optional intermediate outputs retain their expected shapes.
- `generatedTasksAlwaysIncludeCurrentDiameterAndOnlyRequestedOutputs`: Generated Python requests use current settings and return only requested outputs.
- `runCopiesChannelSettingsBeforeBackgroundProcessing`: Background jobs use a snapshot rather than mutable UI channel selections.

### CellposeLayoutTest.java

- `usesSingleStandardInferenceTab`: The migrated Cellpose UI uses the standard inference-only structure.
- `placesCellposeOptionsBetweenImageSelectionAndPreview`: Channel/output options occupy the intended rows without disrupting preview layout.

## Cross-GOOSE

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/crossgoose/`.

### CrossGooseModelRegistryTest.java

- `discoversOnlyCompleteModelDirectories`: Model discovery filters out incomplete Cross-GOOSE folders.
- `acceptsCheckpointWhenItsModelDirectoryHasConfig`: An explicit checkpoint is valid when its associated config is present.
- `trainingUsesTheSelectedFinalDirectoryAndNeverOverwritesAnExistingRun`: Training uses the requested final run location and protects occupied runs.
- `normalizesOnlyLayoutsAndNamesTheCrossGooseBackendCannotRead`: Dataset preparation links/renames only when the Python backend cannot directly digest the original layout.

### CrossGooseIntegrationTest.java

These two opt-in tests used the real revision-pinned Pixi worker, tiny synthetic
datasets, and a five-minute timeout per Java test.

- `trainsFineTunesReloadsAndInfersThroughTheActualJavaBridge`: Trains, receives AP@0.75/preview/checkpoint/full-validation events, reloads weights, performs repeated inference, and fine-tunes without changing source weights.
- `cancellationAndBackendFailureNeverReportSuccessOrOverwriteSavedWeights`: Worker cancellation fails the pending task rather than reporting success; saved last weights remain intact and an occupied run is rejected.

## Denoising

Original directory: `src/test/java/io/bioimage/modelrunner/gui/custom/denoise/`.

### DenoisingConfigTest.java

- `usesProductionDefaultsAndNormalizesDevice`: Default method/effort/strength and device normalization match production requests.
- `nestsStructN2VConfigurationUnderMethodConfig`: StructN2V options are placed in the backend's method-specific config.
- `forwardsAllZsN2NEffortsWithoutDuplicatingPythonPresets`: Forwards effort IDs while leaving preset implementation in Python.
- `normalizesBalancedHighId`: Normalizes the balanced-high effort spelling to its backend identifier.
- `preservesOtherMethodsNormalization`: ZS-N2N-specific normalization changes do not leak into other methods.
- `rejectsBalancedHighForOtherMethods`: The additional ZS-N2N effort cannot be used by unsupported algorithms.

### DenoisingImageIOTest.java

- `readsSixteenBitImageAndUsesCollisionSafeNames`: Preserves 16-bit source intensities and avoids result-name collisions.
- `hostPixelsPreserveBitDepthChannelsAndAllZtPlanesWithoutAnEagerFloatCopy`: Host input preserves dtype/C/Z/T without a full eager float conversion.
- `hostInputMustSpecifyCztEvenForSinglePlanes`: Host input requires explicit channel/Z/T semantics, including singleton cases.
- `publishesCompletedFolderAndRemovesTemporaryTrees`: Publishes completed output folders and cleans temporary directories.

### DenoisingImagesTest.java

- `blendsInOriginalTypeAndIntensityDomain`: Strength blending occurs in the original intensity domain and returns the expected dtype.
- `extractsRequestedCztPlane`: Extracts the requested channel/Z/T plane from host data.

### DenoisingInputTest.java

- `singletonZDoesNotTurnAPlanarImageIntoAVolumeForAnyMethod`: Z=1 inputs remain planar for every method.
- `realVolumesKeepTheirZAxis`: True volumes retain their spatial Z axis.

### DenoisingLayoutTest.java

- `placesAccelerationInNoiseRowAndLogAboveActions`: Acceleration, noise controls, log, and actions follow the intended layout.
- `keepsProgressVisibleAndUsesTheWholeRow`: The progress bar remains present and spans its assigned row.
- `keepsImageSourceHelpIconsInsideTheirRowsWhenResized`: Image-source help icons remain correctly positioned after resizing.

### DenoisingOptionsPanelTest.java

- `showsBalancedHighOnlyForZsN2NAndResetsWhenSwitchingMethod`: Effort availability updates correctly when changing denoising methods.
- `notifiesEffortChangesAndDisablesAllChoicesDuringWork`: Effort listeners fire and all options lock during execution.
- `fitsThreeOrFourOptionsInTheSameRowAtDifferentWidths`: Three- and four-choice effort rows fit at different panel widths.

### DenoisingViewportTest.java

- `appliesSharedViewportState`: Original/denoised views apply the shared zoom/pan viewport consistently.

## Python Environments And Installation Logging

Original directory: `src/test/java/io/bioimage/modelrunner/model/python/envs/`.

### BackendDependenciesTest.java

- `allPlatformsInheritTheSamePinnedBackendsAndExtras`: Platform manifests share exact backend Git revisions and required extras.
- `localSourcesAreExplicitOverridesOnly`: Local development checkout paths are used only through explicit overrides.

### EnvironmentSetupLogTest.java

- `setupIsAnnouncedBeforeProgressAndNotRepeatedForTheSameReadyEnvironment`: Environment setup is announced promptly but not repeated for the same ready installation.
- `phasesUseHonestUnitsAndOnlyArchiveSummaries`: Download/package/resolve phases report meaningful units; the disk log records summaries rather than animation frames.
- `scopesAreNestedAndWorkerLocal`: Nested logging scopes restore correctly and stay isolated between workers.
- `currentPixiLockLayoutAndSingularPackageMessagesHaveCounts`: Current lockfile layouts and singular package messages produce correct package counts.

### InstallationLogTest.java

- `followSurvivesAutomaticLayoutDuringEnvironmentSetup`: In a displayed window, layout changes do not disable follow-to-bottom during installation.
- `completedSetupClearsLiveLineWhileTheTaskContinues`: Finishing setup removes its live progress row even if model work continues.
- `bothLogsFillTheViewportWithWhiteForEmptyShortAndLongContent`: Inference/training logs fill the full viewport with white regardless of text width.
- `trainingReplacesLiveLineAndKeepsOnlySummariesOnDisk`: Live progress replaces one UI row and only stable summaries are persisted.
- `inferencePreservesScrollUpAndResumesFollowingAtBottom`: Manual upward scrolling pauses following; returning to the bottom resumes it.
- `inferenceRowsDoNotWrapOrRebuildTheDocumentWhenBounded`: Timestamp/message rows stay unwrapped and bounded-log updates avoid complete document replacement.
- `trainingLiveProgressPreservesSelectionAndRestoresCaretAtBottom`: Live updates preserve manual selection/navigation and restore end-caret following when appropriate.

### PixiEnvironmentOperationTest.java

- `resolutionIsSharedWithinAnOperationButNotBetweenThreadsOrRuns`: Resolution is reused inside one operation, not across unrelated threads/runs.
- `installerAndWorkerReuseOneSuccessfulBuildUntilTheOperationCloses`: Installer and worker reuse one environment build during the operation.
- `changedOrDeletedEnvironmentFilesInvalidateReuse`: Manifest/installation changes invalidate reuse.
- `manualDependenciesAndSelectedEnvironmentArePartOfTheIdentity`: Manual dependency steps and selected environment affect the reuse key.
- `failuresAreNotCachedAndConcurrentOperationsAreIndependent`: Failed builds can be retried and concurrent operations do not share mutable state.

### PixiEnvironmentResolverTest.java

- `crossGooseAcceptsModelDirectoriesAndUsesTheMainEnvironment`: Cross-GOOSE model directories resolve into the common PyTorch environment.
- `platformsAndCudaBuildsResolveWithoutHardwareDetection`: Explicit Linux/Windows/macOS and CUDA build combinations render the correct templates without probing hardware.
- `oldIntelProfilesApplyToEveryBackend`: Intel compatibility profiles apply to all backends sharing that environment.
- `intelMacVersionBoundariesSelectTheRightSharedTemplate`: OS-version boundaries select Catalina, Big Sur, or the newer shared template correctly.
- `intelMacVersionsBelowCatalinaFailClearly`: Unsupported older Intel macOS releases fail explicitly.
- `legacyIsExplicitAndStillUsesTheOriginalManifests`: Legacy environment selection is opt-in and retains the original templates.
- `unsupportedCudaIsNotSilentlyMappedToAnotherBuild`: Unsupported CUDA builds are rejected instead of substituted.
- `unknownEnvironmentIsRejected`: Invalid environment-family selection fails clearly.

### PixiEnvironmentSmokeTest.java

- `apposeWorkerExecutesNumpyAndSharedMemory`: Opens the selected real Pixi prefix, verifies environment reuse/installation, confirms NumPy initialized before the worker input loop, and checks shared-memory creation, summation, and cleanup.

### PythonWorkerInitializationTest.java

- `registersNumpyInitializationBeforeReturningAnUnstartedWorker`: NumPy initialization is registered before the worker is returned or started.
- `backendInitializationKeepsNumpyFirstInOneInitCall`: Backend eager imports follow NumPy in one initialization script.
- `reportsStartupBeforeCreatingTheWorker`: Worker startup feedback precedes potentially slow process creation.

## Generated Model Bridges And GPU Parsing

Original bridge directories are under
`src/test/java/io/bioimage/modelrunner/model/special/`; the GPU test was under
`src/test/java/io/bioimage/modelrunner/system/`.

### crossgoose/CrossGooseBridgeTest.java

- `explicitCheckpointIsPreservedAndDirectoryDefaultsToBest`: Keeps explicit checkpoint selection; a directory resolves to its best checkpoint.
- `forwardsMetricsPreviewsAndCheckpointEventsWithoutChangingTheirMeaning`: Forwards Cross-GOOSE metrics, previews, and lifecycle events intact.
- `generatedTrainingUsesCallbacksRatherThanBackendSpecificTaskSerialization`: Generated requests use ordinary callbacks adapted by Java rather than requiring Appose-specific backend serialization.

### denoise/DenoisingBridgeTest.java

- `forwardsEveryEffortToBothPublicEntryPointsWithCallbacks`: Preview and full-denoising entry points receive each effort ID and event callbacks.
- `guardsAgainstOldPackagesAndMissingEfforts`: Backend capability checks reject packages lacking required methods/efforts with clear errors.

### stardist/StarDistImageLoadingTest.java

- `usesSeparateReadersForTiffAndRasterFormats`: Generated loader dispatches TIFF and ordinary raster formats to the appropriate readers.
- `preservesTiffSeriesAxesAndAllAcceptedImageExtensions`: Generated loading retains TIFF axis metadata and recognizes the accepted extension set.
- `decodesRealImagesWithTheGeneratedPythonLoader`: Optional Python decoder regression suite uses real files; optionally runs NumPy-volume training as well.

### stardist/StarDistScaleTest.java

- `metadataScaleMatchesEquivalentDiameterAndIsBounded`: Object-size metadata gives the expected bounded inference scale.
- `legacyVolumeScaleTargetsThreeDimensionalEquivalentOfAreaRule`: Without size metadata, 3D scaling follows the volumetric equivalent of the original area heuristic.
- `legacyVolumeScaleUsesAnisotropyToEstimateDepth`: The fallback estimates reference-object depth using anisotropy.

### stardist/StarDistShutdownTest.java

- `removesWorkspaceAfterGracefulExitWithoutKilling`: A clean worker exit removes temporary workspace without a forced kill.
- `alreadyStoppedWorkerNeedsNoCloseOrKill`: An already terminated worker is not closed/killed redundantly.
- `timeoutKillsButDefersDeletionUntilExit`: Shutdown timeout triggers a kill but does not delete files while the process still owns them.
- `interruptedCallerKeepsInterruptAndStillStopsWorker`: Shutdown preserves the caller's interruption state while stopping the worker.

### stardist/StarDistValidationTest.java

- `generatedTrainingUsesSharedValidationWithoutAutomaticFinalFullPass`: Generated training uses the shared validation implementation, exports object F1, and does not force final-volume inference.
- `objectF1IsSeparateFromLearningRateAndLoss`: Progress models keep object F1 distinct from validation loss and learning rate.
- `smallCpuBackendRegressionSuite`: Runs the real CPU Python validation tests plus four Java-generated planar/volumetric and single/multi-source task fixtures.

### unet/UnetInferenceBridgeTest.java

- `outputSelectionFollowsTheTaskRatherThanAvailableAuxiliaryOutputs`: Generated code selects labels only for instance tasks, masks for binary/semantic, and forwards optional object size without a removed boolean override.
- `generatedInferencePreservesBinaryClassAndInstanceValues`: Executes the generated call with a fake Python API and verifies binary values, class IDs, and large instance IDs are selected correctly.

### unet/UnetValidationBridgeTest.java

- `preloadsScipySubmodulesOnWindowsOnly`: Windows eager-import initialization includes the SciPy modules needed to avoid worker freezes.
- `forwardsControlSeparatelyFromTrainingConfiguration`: Full-validation control is transported separately from the model training config.
- `structuredEventsPreserveExistingPreviewConsumers`: New event families preserve existing preview listeners and do not increment training iterations during validation.
- `generatedTaskRunsAgainstCurrentPythonContract`: Runs generated 2.5D/3D training tasks against jdll-unet, including full-validation requests and persisted preview assets.

### GpuCompatibilityTest.java

- `windowsDriverVersionsMapToNvidiaVersions`: Converts Windows WMI driver versions into NVIDIA versions, including 31.0.15.3742 -> 537.42, leading zeros, and newer driver branches.
- `incompleteOrInvalidDriverVersionsAreNotConverted`: Invalid strings and incomplete WMI versions do not become apparently valid NVIDIA versions.

## Python Test Methods

Original directory: `src/test/python/`, unless explicitly prefixed `.github/`.
These scripts generally import the production resources or execute generated
Java task code; their mocks/fixtures were not production backend implementations.

### test_cellpose_inference.py

Eight methods in `DiameterTest`, using fake size estimators and temporary model files.

- `test_official_model_prefers_its_size_file_and_caches_estimator`: Picks the official matching size model and reuses the estimator.
- `test_custom_selection_is_sorted_and_ambiguity_is_reported`: Custom size-file selection is deterministic and ambiguity is explained.
- `test_no_estimator_uses_model_diameter_and_warns`: Missing size estimation falls back to model diameter with a warning.
- `test_official_model_does_not_use_unrelated_size_weights`: Official models never use another model's size weights.
- `test_invalid_estimate_falls_back_explicitly`: Invalid estimated diameters produce an explicit fallback.
- `test_manual_diameter_skips_estimation`: A user size hint bypasses automatic diameter estimation.
- `test_rejects_invalid_diameter`: Invalid manual diameter values fail validation.
- `test_helper_survives_appose_task_namespace`: The helper remains callable in Appose's task/global namespace arrangement.

### test_device_selection.py

Eleven methods using fake PyTorch/TensorFlow devices, kernels, and task updates.

- `test_cpu_does_not_probe_gpu`: Explicit CPU avoids GPU availability/kernel probing.
- `test_unavailable_cuda_and_mps_warn`: Unsupported requested accelerators warn before CPU fallback.
- `test_explicit_no_fallback_policy_warns_before_failing`: A strict device policy reports the reason before raising.
- `test_gpu_boolean_is_not_enough_if_driver_or_kernel_fails`: A successful availability boolean is insufficient if a real device operation fails.
- `test_successful_cuda_and_mps_are_preserved`: Working accelerator requests keep their intended devices.
- `test_automatic_pytorch_requests_remain_supported`: Automatic PyTorch device selection still works.
- `test_tensorflow_no_gpu_warns`: TensorFlow without a GPU reports fallback.
- `test_tensorflow_soft_cpu_placement_warns`: Soft placement onto CPU is detected and reported.
- `test_tensorflow_cuda_and_metal_report_resolved_device`: TensorFlow CUDA/Metal selection emits the actual resolved device.
- `test_tensorflow_cpu_does_not_create_gpu_tensor`: Explicit TensorFlow CPU execution does not probe through a GPU tensor.
- `test_tensorflow_plugin_without_memory_growth_still_uses_gpu`: Unsupported memory-growth configuration does not incorrectly disable a functioning GPU plugin.

### test_stardist_image_loading.py

Twenty-one methods using real raster/TIFF/NumPy fixtures and decoder instrumentation.

- `test_supported_formats_do_not_need_extra_signature_probes`: Normal accepted formats avoid redundant signature inspection.
- `test_sixteen_bit_png_and_palette_mask_labels_are_preserved`: PNG precision and palette label IDs survive reading.
- `test_cache_is_scoped_and_mixed_contents_still_work`: Per-operation reader preferences do not break files with mixed actual encodings.
- `test_png_mislabeled_as_tiff_and_uppercase_extensions`: Wrong suffixes and uppercase names decode correctly.
- `test_tiff_axes_precision_and_spacing_survive_wrong_extensions`: TIFF axes, pixel precision, and spacing survive extension fallback.
- `test_corruption_or_decoder_failure_does_not_switch_readers`: Corrupt data is not misclassified as a different format merely because decoding failed.
- `test_failed_fallback_does_not_poison_cache`: A failed reader fallback does not alter future dispatch incorrectly.
- `test_jpeg_masks_rejected_even_with_png_extension`: Lossy JPEG masks remain rejected when disguised by an accepted suffix.
- `test_multiple_series_and_animated_png_are_not_silently_truncated`: Unsupported multi-series/animated content is not silently reduced to one image.
- `test_numpy_images_and_masks_stay_mapped_and_reuse_patch_normalization`: NumPy images/masks retain memory mapping and cached normalization.
- `test_invalid_numpy_rank_objects_and_time_axes_name_the_source`: Unsupported NumPy rank/object arrays/time axes identify the failing source.
- `test_auto_cache_limit_and_explicit_overrides`: Automatic cache sizing and explicit cache limits resolve correctly.
- `test_compressed_tiff_crops_preserve_axes_values_and_normalization`: Compressed TIFF crop reading preserves geometry, values, and normalization behavior.
- `test_tiled_and_stripped_tiffs_decode_only_selected_regions`: Region-readable TIFF requests do not decode unrelated image regions.
- `test_float_normalization_and_ignored_masks_match_cached_full_image_loading`: Crop-first floating-point normalization and ignored-mask behavior match full-image reference results.
- `test_pyramidal_tiff_reads_the_same_full_resolution_level_as_before`: Pyramid support does not accidentally train from a reduced-resolution level.
- `test_region_array_composed_views_match_numpy_without_decoding_unused_pixels`: Composed indexing/transposition views match NumPy and remain lazy.
- `test_png_ram_cache_is_bounded_reused_and_never_spilled_to_disk`: PNG decoded-image caching is bounded/reusable and creates no disk spill cache.
- `test_tiff_region_backend_fallback_is_explicit_and_memory_guarded`: Region-backend failure gives a warning and respects full-decode memory safeguards.
- `test_uncompressed_tiff_maps_original_pixels_and_remains_read_only`: Mappable TIFFs expose original read-only pixels without rewriting them.
- `test_raster_images_take_precedence_over_numpy_caches_without_filtering_masks`: Raster source precedence avoids duplicate cached images without accidentally removing genuine NumPy masks.

### test_stardist_validation.py

Twenty-eight methods in `ValidationTest`, using tiny real TensorFlow/StarDist
models, synthetic 2D/3D cases, controlled validation requests, and temporary outputs.

- `test_periodic_manual_phase_and_one_shot`: Manual validation shifts the periodic schedule and remains one-shot when periodic evaluation is disabled.
- `test_object_metric_pools_objects_not_image_averages`: Object F1 aggregates object counts rather than averaging image-level F1 values.
- `test_request_during_active_pass_survives_and_is_deduplicated`: Requests arriving during a pass survive its completion and duplicate IDs do not execute twice.
- `test_planner_relaxes_sparse_foreground_and_covers_sources`: Bounded planning relaxes unattainable foreground requirements and reports/source-balances achieved coverage.
- `test_partial_final_batch_and_compatible_plan_reuse`: Supports partial final batches and reuses compatible deterministic plans.
- `test_object_extent_estimation_uses_bounded_deterministic_sample`: Object-size estimation uses a bounded reproducible sample.
- `test_region_readable_sources_and_training_only_holdout_normalization`: Patch reading stays regional and normalization statistics exclude validation holdout regions.
- `test_full_failure_preserves_checkpoints_and_does_not_abort_training`: Optional full-validation failure leaves saved regular checkpoints intact and reports failure separately.
- `test_defaults_resolve_real_sample_budget`: Default validation sample/batch budgets use the actual microbatch size.
- `test_reuses_predictions_and_loss_matches_keras`: Each regular batch is forwarded once and its reconstructed losses match Keras definitions.
- `test_four_enlarged_previews_need_only_twelve_extra_tiles`: Four enlarged previews reuse anchors and require only twelve additional tile predictions.
- `test_empty_small_capacity_keeps_real_background`: Limited-capacity planning keeps real empty/background validation examples rather than forcing duplicates.
- `test_enlarged_overlap_stitches_maps_in_their_original_coordinates`: Preview stitching blends overlapping continuous maps in the correct coordinates.
- `test_two_dimensional_budget_and_checkpoint_training`: 2D training honors validation budgeting and writes regular checkpoints.
- `test_two_dimensional_explicit_budget_uses_shuffled_source_cycles`: Explicit 2D budgets cycle through shuffled sources instead of repeatedly favoring the same samples.
- `test_two_dimensional_automatic_budget_weights_image_capacity`: Automatic 2D expansion allocates patches according to usable image capacity.
- `test_two_dimensional_automatic_budget_is_bounded_and_shape_first`: Capacity planning is bounded and inspects shapes before loading pixels.
- `test_two_dimensional_expansion_does_not_repeat_forced_foreground_crop`: Expanded 2D plans avoid repeating the same forced-positive crop.
- `test_two_dimensional_validation_keeps_coordinates_across_epochs`: 2D validation coordinates stay fixed across epochs.
- `test_three_dimensional_training_and_cancel`: Tiny volumetric training supports cancellation.
- `test_checkpoint_events_follow_native_saves_and_resume_best`: Checkpoint events occur after actual saves and retain the best criterion on resume.
- `test_checkpoint_precedes_full_validation_and_preview_retention_is_bounded`: Regular checkpoints precede optional full evaluation and preview assets have bounded retention.
- `test_full_volume_metric_is_separate_from_regular_validation`: Full-domain scores do not replace regular validation history/selection.
- `test_crop_first_full_prediction_matches_native_sparse_tiling`: Crop-first full validation matches native sparse tiling results.
- `test_resume_reuses_coordinates_optimizer_and_full_validation_phase`: Compatible resume restores validation coordinates, optimizer state, and periodic scheduling phase.
- `test_invalid_resume_preserves_saved_plan`: Incompatible resume does not overwrite the existing plan.
- `test_request_after_final_boundary_is_reported_and_cleared`: A too-late request is closed/reported without inventing another epoch.
- `test_generated_java_training_tasks`: Executes Java-provided generated training fixtures; skips when no fixture payload is supplied.

### test_unet_inference_bridge.py

Three standard-library-only methods using the exact generated Java inference
snippet and a fake API returning deliberately different mask/label values.

- `test_binary_ignores_connected_component_labels_even_if_returned`: Binary inference selects the 0/1 mask despite auxiliary object labels.
- `test_semantic_keeps_class_ids_instead_of_auxiliary_instance_ids`: Semantic output preserves class IDs rather than auxiliary instance IDs.
- `test_instance_keeps_object_ids_instead_of_a_binary_mask`: Instance output selects object labels, including IDs greater than 65535.

### test_yolo_data.py

Two methods using real Ultralytics datasets/loaders with fixed random seeds.

- `test_shuffled_epochs_match_native_including_fortran_and_duplicate_labels`: PNG and equivalent NumPy datasets produce byte-identical shuffled/augmented batch tensors over three epochs, including Fortran arrays, duplicate labels, empty labels, and a partial batch; source arrays remain unchanged.
- `test_float_training_statistics_virtual_crops_and_shared_preview`: Training-only float statistics, virtual crops, TIFF/NumPy numeric reading, in-memory intensity conversion, and preview decoding share the same preprocessing without modifying originals.

## Script-Style Python Bridge Checks

These three original `src/test/python/` files execute assertions at script scope
rather than defining unittest methods; they are included separately from the 86.

- `test_stardist_numpy_training.py`: Takes a generated Java task, injects a tiny one-epoch 3D configuration and mapped float/uint32 NumPy sources, verifies best/last weights and result path, and hashes every source to ensure it was not modified.
- `test_unet_validation_bridge.py`: Executes Java-generated 2.5D/3D jobs on synthetic ZYX TIFFs; submits one request at ready and another during the first pass; verifies epoch-1/2 full-validation completion, regular checkpoints, preview manifests/NumPy asset shapes, emitted validation events, and token cleanup.
- `test_yolo_bridge.py`: Executes generated YOLO training or inference tasks with a fake Appose task collector. Training verifies one completed epoch, exported weights, persisted intensity statistics, checkpoints, and rendered NumPy-backed validation previews. Inference verifies statistics reuse, a normalized 1x3x640x640 input, and shared-memory cleanup.

## CI Contract Tests

Original directory: `.github/scripts/`.

### test_pixi_ci.py

- `test_streaming_wheel_timeout_retries_the_command`: A transient wheel-unzip timeout retries the same installation command after a five-second delay.
- `test_dependency_conflicts_fail_without_retry`: A deterministic solver conflict fails immediately, without backoff/retry.
- `test_repeated_network_failures_are_bounded`: Repeated transient transport failures stop after three attempts with bounded delays.

### test_pixi_dependencies.py

- `test_single_headless_conda_provider_with_both_metadata_names_is_valid`: Accepts both OpenCV metadata names when one headless conda implementation owns them.
- `test_split_conda_packages_can_own_the_metadata_and_bindings_separately`: Accepts matching conda library/bindings packages jointly owning those files.
- `test_wheel_overwriting_a_conda_install_is_rejected`: Rejects a wheel that overwrote a conda-owned OpenCV provider.
- `test_unowned_cv2_module_is_rejected`: Rejects cv2 bindings outside the claimed conda package ownership.
- `test_non_headless_or_mismatched_conda_provider_is_rejected`: Rejects GUI-enabled builds or library/bindings version mismatches.

### test_pixi_manifests.py

- `test_all_platform_templates_render`: Every JDLL template renders valid TOML with a workspace name and platforms.
- `test_cuda_indices_are_explicit_and_do_not_share_cpu_wheels`: Linux/Windows CUDA 11.8/12.1/12.4 select matching Torch/Torchvision indexes; CPU profiles use CPU indexes.
- `test_unknown_cuda_is_rejected`: Unsupported CUDA versions fail instead of rendering an unintended environment.
- `test_model_specific_environments_are_rejected`: Invalid model-specific profile requests cannot replace shared environment profiles.
- `test_old_intel_cli_profiles_resolve_to_explicit_versioned_templates`: Old Intel CLI profile aliases render the corresponding explicit templates.

## Real Model And Installation Smoke Checks

Original file: `.github/scripts/smoke_jdll_models.py`.
Synthetic weights/data avoided pretrained downloads; these checked runnable
kernels, finite outputs, expected dimensions, dependency consistency, and gradients,
not biological prediction quality.

- `unet`: Verifies compressed TIFF region reading through Zarr; trains/infers one tiny instance model; checks foreground/boundary/distance/label outputs; exercises 2.5D and 3D forward/backward kernels.
- `denoise`: Runs Noise2Fast, ZS-N2N, StructN2V, and BM3D on synthetic data; invokes the native BM4D backend on a small volume.
- `cellpose`: Builds restoration plus segmentation wrappers with synthetic weights and checks label/flow/restored outputs.
- `yolo`: Verifies native torchvision NMS, random-weight prediction, and a training loss/backward pass; uses CPU NMS for MPS inputs as Ultralytics does.
- `cross_goose`: Creates/reloads a synthetic checkpoint, performs instance inference, and exercises model maps/loss/backward.
- `stardist`: Checks 2D/3D probability/distance prediction, instance reconstruction, and TensorFlow gradients, with an explicit requested-device check.
- `check_dependencies` and `shared_conda_opencv_metadata`: Check installed requirement versions/markers and reject conflicting OpenCV providers while allowing one correctly owned conda implementation.
- Runner/reporting: Checks requested CUDA/MPS availability, records package/platform versions, per-model duration/failure details, and exits unsuccessfully on failed model checks.

Removed `.github/actions/jdll-pixi-smoke/action.yml` rendered a host manifest,
locked/installed Pixi, ran the above checks, optionally exercised the Java Appose
shared-memory test, and archived manifests/locks/package inventories/results.
Its helpers were `.github/scripts/render_jdll_pixi.py` (template rendering) and
`.github/scripts/pixi_ci.py` (bounded retries for transient installation failures).
Those helpers were CI/test support rather than production environment selection.

Removed `.github/workflows/jdll-pixi-install.yml` ran contract tests plus Linux,
Windows, Apple Silicon, Intel Mac compatibility profiles, separate StarDist
profiles, and CUDA 11.8/12.1/12.4 wheel installations. Hosted CUDA-wheel jobs
checked CPU execution only. Actual CUDA, MPS/Metal, and older-macOS execution
required opt-in self-hosted runners. A newer Mac compatibility profile was not
evidence that the corresponding older macOS release ran successfully.

## Other Removed Automated Verification Workflows

- `.github/workflows/example_run_all_bmz_models.yml`: Scheduled/manual/release-driven multi-OS model-zoo exercise through `ExampleLoadAndRunAllBmzModels`; installed engines and reported models with loading/running errors.
- `.github/workflows/example_run_pytorch1_2_model.yml`: Built JDLL and ran the PyTorch 1/2 coexistence example across an OS matrix.
- `.github/workflows/example_run_tf1_and_tf2_model.yml`: Built JDLL and ran the TensorFlow 1/2 coexistence example across an OS matrix.
- `.github/workflows/download_all_engines.yml`: Exercised the standalone engine downloader on Linux, Windows, and macOS.

The underlying SDK examples and `download-engines.py` remain usage/maintenance
tools, but are no longer invoked by automated verification workflows.

## Guidance For Selective Reconstruction

Choose a small set of meaningful contracts rather than restoring every old
assertion. Separate pure unit checks, displayed-Swing checks, installed-backend
integration, and clean-machine/accelerator checks so requirements and costs are
visible. Use portable fixtures, explicit capability requirements, bounded waits,
and immutable source data. Do not silently interpret a skipped accelerator test
as successful GPU coverage. Revisit timing-sensitive GUI assertions and duplicated
Java-string/Python-execution checks before reintroducing them.
