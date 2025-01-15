/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2024 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.model.stardist_java_deprecate;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;

import io.bioimage.modelrunner.utils.JSONUtils;

/**
 * Implementation of the Stardist Config instance in Java.
 *
 *@author Carlos Garcia
 */
public class StardistConfig {

	final public int n_dim;
	final public String axes;
	final public int n_channel_in;
	final public int n_channel_out;
	final public String train_checkpoint;
	final public String train_checkpoint_last;
	final public String train_checkpoint_epoch;
	final public int n_rays;
	final public int[] grid;
	final public String backbone;
	final public int n_classes;
	final public int unet_n_depth;
	final public int[] unet_kernel_size;
	final public int unet_n_filter_base;
	final public int unet_n_conv_per_depth;
	final public int[] unet_pool;
	final public String unet_activation;
	final public String unet_last_activation;
	final public boolean unet_batch_norm;
	final public double unet_dropout;
	final public String unet_prefix;
	final public int net_conv_after_unet;
	final public int[] net_input_shape;
	final public int[] net_mask_shape;
	final public boolean train_shape_completion;
	final public int train_completion_crop;
	final public int[] train_patch_size;
	final public float train_background_reg;
	final public float train_foreground_only;
	final public boolean train_sample_cache;
	final public String train_dist_loss;
	final public double[] train_loss_weights;
	final public double[] train_class_weights;
	final public int train_epochs;
	

	final public int train_steps_per_epoch;
	final public double train_learning_rate;
	final public int train_batch_size;
	final public Integer train_n_val_patches;
	final public boolean train_tensorboard;
	final public Map<String, Object> train_reduce_lr;
	final public int patience;
	final public double min_delta;
	final public double factor;
	final public boolean use_gpu;
	
	public static final String FNAME = "config.json";
	
	private StardistConfig(String path) throws IOException {
		Map<String, Object> config = JSONUtils.load(path);
		n_dim = (int) config.get("n_dim");
		axes = (String) config.get("axes");
		n_channel_in = (int) config.get("n_channel_in");
		n_channel_out = (int) config.get("n_channel_out");
		train_checkpoint = (String) config.get("train_checkpoint");
		train_checkpoint_last = (String) config.get("train_checkpoint_last");
		train_checkpoint_epoch = (String) config.get("train_checkpoint_epoch");
		n_rays = (int) config.get("n_rays");
		grid = (int[]) config.get("grid");
		backbone = (String) config.get("backbone");
		n_classes = (int) config.get("n_classes");
		unet_n_depth = (int) config.get("unet_n_depth");
		unet_kernel_size = (int[]) config.get("unet_kernel_size");
		unet_n_filter_base = (int) config.get("unet_n_filter_base");
		unet_n_conv_per_depth = (int) config.get("unet_n_conv_per_depth");
		unet_pool = (int[]) config.get("unet_pool");
		unet_activation = (String) config.get("unet_activation");
		unet_last_activation = (String) config.get("unet_last_activation");
		unet_batch_norm = (boolean) config.get("unet_batch_norm");
		unet_dropout = (double) config.get("unet_dropout");
		unet_prefix = (String) config.get("unet_prefix");
		net_conv_after_unet = (int) config.get("net_conv_after_unet");
		net_input_shape = (int[]) config.get("net_input_shape");
		net_mask_shape = (int[]) config.get("net_mask_shape");
		train_shape_completion = (boolean) config.get("train_shape_completion");
		train_completion_crop = (int) config.get("train_completion_crop");
		train_patch_size = (int[]) config.get("train_patch_size");
		train_background_reg = (float) config.get("train_background_reg");
		train_foreground_only = (float) config.get("train_foreground_only");
		train_sample_cache = (boolean) config.get("train_sample_cache");
		train_dist_loss = (String) config.get("train_dist_loss");
		train_loss_weights = (double[]) config.get("train_loss_weights");
		train_class_weights = (double[]) config.get("train_class_weights");
		train_epochs = (int) config.get("train_epochs");
		train_steps_per_epoch = (int) config.get("train_steps_per_epoch");
		train_learning_rate = (double) config.get("train_learning_rate");
		train_batch_size = (int) config.get("train_batch_size");
		train_n_val_patches = (Integer) config.get("train_n_val_patches");
		train_tensorboard = (boolean) config.get("train_tensorboard");
		train_reduce_lr = (Map<String, Object>) config.get("train_reduce_lr");
		use_gpu = (boolean) config.get("use_gpu");
		factor = (double) train_reduce_lr.get("factor");
		patience = (int) train_reduce_lr.get("patience");
		min_delta = (double) train_reduce_lr.get("min_delta");
	}
	
	public static StardistConfig create(String name, String baseDir) throws IOException {
		if (new File(baseDir + File.separator + name + File.separator + FNAME).isFile() == false)
			throw new IllegalArgumentException("No '" + FNAME + "' found at " + baseDir + File.separator + name);
		return new StardistConfig(baseDir + File.separator + name + File.separator + FNAME);
	}
	
}
