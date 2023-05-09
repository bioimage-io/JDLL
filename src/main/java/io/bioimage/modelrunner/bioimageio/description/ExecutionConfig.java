/*-
 * #%L
 * Use deep learning frameworks from Java in an agnostic and isolated way.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.bioimageio.description;

import java.util.Map;

import io.bioimage.modelrunner.bioimageio.description.execution.DeepImageJExecutionConfig;

/**
 * Execution configuration element. Comes from the config section in the yaml file.
 * 
 * @author Daniel Felipe Gonzalez Obando and Carlos GArcia Lopez de Haro
 */
public class ExecutionConfig
{

    /**
     * Creates an execution configuration instance with the config element map from the yaml file.
     * 
     * @param yamlFieldElements
     *        The config map.
     * @return The execution configuration instance.
     */
    public static ExecutionConfig build(Map<String, Object> yamlFieldElements)
    {
        if (yamlFieldElements == null)
            return null;

        ExecutionConfig config = new ExecutionConfig();
        config.specMap = yamlFieldElements;
        return config;
    }

    private Map<String, Object> specMap;

    /**
     * Retrieves the yaml config element map.
     * 
     * @return The config element map.
     */
    public Map<String, Object> getSpecMap()
    {
        return specMap;
    }

    private DeepImageJExecutionConfig deepImageJExecutionConfig;

    /**
     * Retrieves the configuration instance for DeepImageI.
     * 
     * @return The DeepImageJ execution configuration instance.
     */
    public DeepImageJExecutionConfig getDeepImageJ()
    {
        if (deepImageJExecutionConfig == null)
        {
            @SuppressWarnings("unchecked")
            Map<String, Object> deepImageJEntry = (Map<String, Object>) specMap.get("deepimagej");
            if (deepImageJEntry != null)
            {
                deepImageJExecutionConfig = DeepImageJExecutionConfig.build(deepImageJEntry);
            }
        }
        return deepImageJExecutionConfig;
    }
    
    /**
     * Key for the model id in the ocnfig field
     */
    private static String idKey = "_id";
    
    /**
     * Get the model ID specified inside the config field. This is the same as the ID that appears
     * in the rd.yaml in the collections repo
     * @return the model ID
     */
    public String getID() {
    	return (String) specMap.get(idKey);
    }

    

    @Override
    public String toString()
    {
        return "ExecutionConfig {specMap=" + specMap + "}";
    }

}
