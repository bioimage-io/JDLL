/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
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
package io.bioimage.modelrunner.runmode;

import io.bioimage.modelrunner.bioimageio.download.DownloadModel;

public class RunModeScripts {
	
	protected static final String AXES_KEY = "axes";
	
	protected static final String SHAPE_KEY = "shape";
	
	protected static final String DATA_KEY = "data";
	
	protected static final String NAME_KEY = "name";
	
	protected static final String DTYPE_KEY = "dtype";
	
	protected static final String NP_METHOD = "convertNpIntoDic";
	
	protected static final String XR_METHOD = "convertXrIntoDic";
	
	protected static final String LIST_METHOD = "convertListIntoSupportedList";
	
	protected static final String DICT_METHOD = "convertDicIntoDic";
	
	protected static final String APPOSE_DT_KEY = DownloadModel.addTimeStampToFileName("appose_data_type_", true);
	
	protected static final String TENSOR_KEY = "tensor";

	protected static final String IS_FORTRAN_KEY = "is_fortran";
	
	protected static final String NP_ARR_KEY = "np_arr";
	
	protected static final String SHARED_MEM_PACKAGE_NAME = "shared_memory";
	
	protected static final String NP_PACKAGE_NAME = "np";
	
	/**
	 * Script that contains all the methods neeed to convert python types 
	 * into Appose supported types (primitive types and dics and lists of them)
	 */
	protected static final String TYPE_CONVERSION_METHODS_SCRIPT = ""
			+ "shm_out_list = []" + System.lineSeparator()
			+ "def " + NP_METHOD + "(np_arr):" + System.lineSeparator()
			+ "  shm = shared_memory.SharedMemory(create=True, size=np_arr.nbytes)" + System.lineSeparator()
			+ "  aux_np_arr = np.ndarray((np_arr.size), dtype=np_arr.dtype, buffer=shm.buf)" + System.lineSeparator()
			+ "  aux_np_arr[:] = np_arr.flatten()" + System.lineSeparator()
			+ "  shm_out_list.append(shm)" + System.lineSeparator()
			+ "  shm.unlink()" + System.lineSeparator()
			+ "  return {\"" + DATA_KEY + "\": shm.name, \"" + SHAPE_KEY 
							+ "\": np_arr.shape, \"" + APPOSE_DT_KEY + "\": \"" 
							+ NP_ARR_KEY + "\", "
							+ "\"" + IS_FORTRAN_KEY + "\": np.isfortran(np_arr), "
							+ "\"" + DTYPE_KEY + "\": str(np_arr.dtype)}" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + XR_METHOD + "(xr_arr):" + System.lineSeparator()
			+ "  shm = shared_memory.SharedMemory(create=True, size=xr_arr.values.nbytes)" + System.lineSeparator()
			+ "  aux_np_arr = np.ndarray((xr_arr.values.size), dtype=xr_arr.values.dtype, buffer=shm.buf)" + System.lineSeparator()
			+ "  aux_np_arr[:] = xr_arr.values.flatten()" + System.lineSeparator()
			+ "  shm_out_list.append(shm)" + System.lineSeparator()
			+ "  shm.unlink()" + System.lineSeparator()
			+ "  return {\"" + DATA_KEY + "\": shm.name, \"" + SHAPE_KEY 
							+ "\": xr_arr.shape, \"" + AXES_KEY + "\": \"\".join(xr_arr.dims),\"" + NAME_KEY 
							+ "\": xr_arr.name, \"" + APPOSE_DT_KEY + "\": \"" + TENSOR_KEY + "\", "
							+ "\"" + IS_FORTRAN_KEY + "\": np.isfortran(xr_arr.values), "
							+ "\"" + DTYPE_KEY + "\": str(xr_arr.values.dtype)}" 
							+ System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + LIST_METHOD + "(list_ob):" + System.lineSeparator()
			+ "  n_list = []" + System.lineSeparator()
			+ "  for value in list_ob:" + System.lineSeparator()
			+ "    if str(type(value)) == \"<class 'xarray.core.dataarray.DataArray'>\":" + System.lineSeparator()
			+ "      n_list.append(" + XR_METHOD + "(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.ndarray'>\":" + System.lineSeparator()
			+ "      n_list.append(" + NP_METHOD + "(value))" + System.lineSeparator()
			+ "    elif isinstance(value, dict):" + System.lineSeparator()
			+ "      n_list.append(" + DICT_METHOD + "(value))" + System.lineSeparator()
			+ "    elif isinstance(value, list):" + System.lineSeparator()
			+ "      n_list.append(" + LIST_METHOD + "(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.float32'>\""
					+ " or str(type(value)) == \"<class 'numpy.float16'>\""
					+ " or str(type(value)) == \"<class 'numpy.float64'>\":" + System.lineSeparator()
			+ "      n_list.append(float(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.int8'>\" or str(type(value)) == \"<class 'numpy.uint8'>\""
					+ " or str(type(value)) == \"<class 'numpy.int16'>\" or str(type(value)) == \"<class 'numpy.uint16'>\""
					+ " or str(type(value)) == \"<class 'numpy.int32'>\" or str(type(value)) == \"<class 'numpy.uint32'>\"" 
					+ " or str(type(value)) == \"<class 'numpy.int64'>\":" + System.lineSeparator()
			+ "      n_list.append(int(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.str_'>\":" + System.lineSeparator()
			+ "      n_list.append(str(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.bool_'>\":" + System.lineSeparator()
			+ "      n_list.append(bool(value))" + System.lineSeparator()
			+ "    else:" + System.lineSeparator()
			+ "      n_list.append(value)" + System.lineSeparator()
			+ "  return n_list" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + DICT_METHOD + "(dic):" + System.lineSeparator()
			+ "  n_dic = {}" + System.lineSeparator()
			+ "  for key, value in dic.items():" + System.lineSeparator()
			+ "    if str(type(value)) == \"<class 'xarray.core.dataarray.DataArray'>\":" + System.lineSeparator()
			+ "      n_dic[key] = " + XR_METHOD + "(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.ndarray'>\":" + System.lineSeparator()
			+ "      n_dic[key] = " + NP_METHOD + "(value)" + System.lineSeparator()
			+ "    elif isinstance(value, dict):" + System.lineSeparator()
			+ "      n_dic[key] = " + DICT_METHOD + "(value)" + System.lineSeparator()
			+ "    elif isinstance(value, list):" + System.lineSeparator()
			+ "      n_dic[key] = " + LIST_METHOD + "(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.float32'>\""
					+ " or str(type(value)) == \"<class 'numpy.float16'>\""
					+ " or str(type(value)) == \"<class 'numpy.float64'>\":" + System.lineSeparator()
			+ "      n_dic[key] = float(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.int8'>\" or str(type(value)) == \"<class 'numpy.uint8'>\""
					+ " or str(type(value)) == \"<class 'numpy.int16'>\" or str(type(value)) == \"<class 'numpy.uint16'>\""
					+ " or str(type(value)) == \"<class 'numpy.int32'>\" or str(type(value)) == \"<class 'numpy.uint32'>\"" 
					+ " or str(type(value)) == \"<class 'numpy.int64'>\":" + System.lineSeparator()
			+ "      n_dic[key] = int(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.str_'>\":" + System.lineSeparator()
			+ "      n_dic[key] = str(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.bool_'>\":" + System.lineSeparator()
			+ "      n_dic[key] = bool(value)" + System.lineSeparator()
			+ "    else:" + System.lineSeparator()
			+ "      n_dic[key] = value" + System.lineSeparator()
			+ "  return n_dic" + System.lineSeparator()
			+ "globals()['" + XR_METHOD + "'] = " + XR_METHOD +  System.lineSeparator()
			+ "globals()['" + NP_METHOD + "'] = " + NP_METHOD +  System.lineSeparator()
			+ "globals()['" + DICT_METHOD + "'] = " + DICT_METHOD +  System.lineSeparator()
			+ "globals()['" + LIST_METHOD + "'] = " + LIST_METHOD +  System.lineSeparator()
			+ "globals()['shm_out_list'] = shm_out_list" +  System.lineSeparator()
			+ "globals()['" + NP_PACKAGE_NAME + "'] = " + NP_PACKAGE_NAME +  System.lineSeparator()
			+ "globals()['" + SHARED_MEM_PACKAGE_NAME + "'] = " + SHARED_MEM_PACKAGE_NAME +  System.lineSeparator();
	
	/**
	 * Script that contains all the methods neeed to convert python types 
	 * into Appose supported types (primitive types and dics and lists of them)
	 * TODO delete, does not use shm
	 */
	protected static final String TYPE_CONVERSION_METHODS_SCRIPT_NO_SHM = ""
			+ "def " + NP_METHOD + "(np_arr):" + System.lineSeparator()
			+ "  return {\"" + DATA_KEY + "\": np_arr.flatten().tolist(), \"" + SHAPE_KEY 
							+ "\": np_arr.shape, \"" + APPOSE_DT_KEY + "\": \"" 
							+ NP_ARR_KEY + "\", \"" + DTYPE_KEY + "\": str(np_arr.dtype)}" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + XR_METHOD + "(xr_arr):" + System.lineSeparator()
			+ "  return {\"" + DATA_KEY + "\": xr_arr.values.flatten().tolist(), \"" + SHAPE_KEY 
							+ "\": xr_arr.shape, \"" + AXES_KEY + "\": \"\".join(xr_arr.dims),\"" + NAME_KEY 
							+ "\": xr_arr.name, \"" + APPOSE_DT_KEY + "\": \"" + TENSOR_KEY + "\", "
							+ "\"" + DTYPE_KEY + "\": str(xr_arr.values.dtype)}" 
							+ System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + LIST_METHOD + "(list_ob):" + System.lineSeparator()
			+ "  n_list = []" + System.lineSeparator()
			+ "  for value in list_ob:" + System.lineSeparator()
			+ "    if str(type(value)) == \"<class 'xarray.core.dataarray.DataArray'>\":" + System.lineSeparator()
			+ "      n_list.append(" + XR_METHOD + "(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.ndarray'>\":" + System.lineSeparator()
			+ "      n_list.append(" + NP_METHOD + "(value))" + System.lineSeparator()
			+ "    elif isinstance(value, dict):" + System.lineSeparator()
			+ "      n_list.append(" + DICT_METHOD + "(value))" + System.lineSeparator()
			+ "    elif isinstance(value, list):" + System.lineSeparator()
			+ "      n_list.append(" + LIST_METHOD + "(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.float32'>\""
					+ " or str(type(value)) == \"<class 'numpy.float16'>\""
					+ " or str(type(value)) == \"<class 'numpy.float64'>\":" + System.lineSeparator()
			+ "      n_list.append(float(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.int8'>\" or str(type(value)) == \"<class 'numpy.uint8'>\""
					+ " or str(type(value)) == \"<class 'numpy.int16'>\" or str(type(value)) == \"<class 'numpy.uint16'>\""
					+ " or str(type(value)) == \"<class 'numpy.int32'>\" or str(type(value)) == \"<class 'numpy.uint32'>\"" 
					+ " or str(type(value)) == \"<class 'numpy.int64'>\":" + System.lineSeparator()
			+ "      n_list.append(int(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.str_'>\":" + System.lineSeparator()
			+ "      n_list.append(str(value))" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.bool_'>\":" + System.lineSeparator()
			+ "      n_list.append(bool(value))" + System.lineSeparator()
			+ "    else:" + System.lineSeparator()
			+ "      n_list.append(value)" + System.lineSeparator()
			+ "  return n_list" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + DICT_METHOD + "(dic):" + System.lineSeparator()
			+ "  n_dic = {}" + System.lineSeparator()
			+ "  for key, value in dic.items():" + System.lineSeparator()
			+ "    if str(type(value)) == \"<class 'xarray.core.dataarray.DataArray'>\":" + System.lineSeparator()
			+ "      n_dic[key] = " + XR_METHOD + "(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.ndarray'>\":" + System.lineSeparator()
			+ "      n_dic[key] = " + NP_METHOD + "(value)" + System.lineSeparator()
			+ "    elif isinstance(value, dict):" + System.lineSeparator()
			+ "      n_dic[key] = " + DICT_METHOD + "(value)" + System.lineSeparator()
			+ "    elif isinstance(value, list):" + System.lineSeparator()
			+ "      n_dic[key] = " + LIST_METHOD + "(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.float32'>\""
					+ " or str(type(value)) == \"<class 'numpy.float16'>\""
					+ " or str(type(value)) == \"<class 'numpy.float64'>\":" + System.lineSeparator()
			+ "      n_dic[key] = float(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.int8'>\" or str(type(value)) == \"<class 'numpy.uint8'>\""
					+ " or str(type(value)) == \"<class 'numpy.int16'>\" or str(type(value)) == \"<class 'numpy.uint16'>\""
					+ " or str(type(value)) == \"<class 'numpy.int32'>\" or str(type(value)) == \"<class 'numpy.uint32'>\"" 
					+ " or str(type(value)) == \"<class 'numpy.int64'>\":" + System.lineSeparator()
			+ "      n_dic[key] = int(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.str_'>\":" + System.lineSeparator()
			+ "      n_dic[key] = str(value)" + System.lineSeparator()
			+ "    elif str(type(value)) == \"<class 'numpy.bool_'>\":" + System.lineSeparator()
			+ "      n_dic[key] = bool(value)" + System.lineSeparator()
			+ "    else:" + System.lineSeparator()
			+ "      n_dic[key] = value" + System.lineSeparator()
			+ "  return n_dic" + System.lineSeparator()
			+ "globals()['" + XR_METHOD + "'] = " + XR_METHOD +  System.lineSeparator()
			+ "globals()['" + NP_METHOD + "'] = " + NP_METHOD +  System.lineSeparator()
			+ "globals()['" + DICT_METHOD + "'] = " + DICT_METHOD +  System.lineSeparator()
			+ "globals()['" + LIST_METHOD + "'] = " + LIST_METHOD +  System.lineSeparator();
}
