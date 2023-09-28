package io.bioimage.modelrunner.runmode;

public class RunModeScripts {
	
	protected static final String AXES_KEY = "axes";
	
	protected static final String SHAPE_KEY = "shape";
	
	protected static final String DATA_KEY = "data";
	
	protected static final String NAME_KEY = "name";
	
	protected static final String NP_METHOD = "convertNpIntoDic";
	
	protected static final String XR_METHOD = "convertXrIntoDic";
	
	protected static final String LIST_METHOD = "convertListIntoSupportedList";
	
	protected static final String DICT_METHOD = "convertDicIntoDic";
	
	/**
	 * Script that contains all the methods neeed to convert python types 
	 * into Appose supported types (primitive types and dics and lists of them)
	 */
	protected static final String TYPE_CONVERSION_METHODS_SCRIPT = ""
			+ "def " + NP_METHOD + "(np_arr):" + System.lineSeparator()
			+ "  return {\"" + DATA_KEY + "\": np_arr.flatten().tolist(), \"" + SHAPE_KEY + "\": np_arr.shape}" + System.lineSeparator()
			+ "" + System.lineSeparator()
			+ "def " + XR_METHOD + "(xr_arr):" + System.lineSeparator()
			+ "  return {\"" + DATA_KEY + "\": xr_arr.values.flatten().tolist(), \"" + SHAPE_KEY + "\": xr_arr.shape, \"" + AXES_KEY + "\": xr_arr.dims,\"" + NAME_KEY + "\": xr_arr.name}" + System.lineSeparator()
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
			+ "    else:" + System.lineSeparator()
			+ "      n_dic[key] = value" + System.lineSeparator()
			+ "  return n_dic" + System.lineSeparator()
			+ "globals()['" + XR_METHOD + "'] = " + XR_METHOD +  System.lineSeparator()
			+ "globals()['" + NP_METHOD + "'] = " + NP_METHOD +  System.lineSeparator()
			+ "globals()['" + DICT_METHOD + "'] = " + DICT_METHOD +  System.lineSeparator()
			+ "globals()['" + LIST_METHOD + "'] = " + LIST_METHOD +  System.lineSeparator();

}
