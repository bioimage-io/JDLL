package org.bioimageanalysis.icy.deeplearning.utils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.gson.Gson;
import com.google.gson.internal.LinkedTreeMap;
import com.google.gson.reflect.TypeToken;

/**
 * Class to create an object that contains all the information about a
 * Deep Learning framework (engine) that is needed to launch the engine
 * in an independent ClassLoader
 *
 * @author Carlos Garcia Lopez de Haro
 */
public class SupportedVersions {
    /**
     * Variable containing the name used to refer to Tensorflow
     * in the program
     */
    private static String tensorflowEngineName = "tensorflow";
    /**
     * Variable containing the name used to refer to Pytorch
     * in the program
     */
    private static String pytorchEngineName = "pytorch";
    /**
     * Key for the Java equivalent version in the JSON file
     */
    private static String javaVersionsKey = "javaVersion";
    /**
     * HashMap containing all the versions supported for a specific
     * engine and their corresponding Java versions
     */
    private LinkedTreeMap<String, Object> versionsDic;
    /**
     * Set of Strings where each entry corresponds to a supported
     * Deep Learning framework version
     */
    private Set<String> versionSet;

    /**
     * Class to find the version of Deep Learning framework (engine) equivalent or compatible
     * with the one used to train the model. This is done because sometimes APIs for
     * different languages are named differently
     *
     * @param engine Deep Learning framework which is going to be loaded
     */
    public SupportedVersions(String engine) {
        this.versionsDic = getSpecificEngineVersionsJson(engine);
        this.versionSet = this.versionsDic.keySet();
    }

    /**
     * Obtain from the plugin resources json the currently supported Deep Learning
     * frameworks and versions
     *
     * @return a hashmap containing the json file with the all supported versions
     */
    public static HashMap<String, Object> readVersionsJson() {
        BufferedReader br = new BufferedReader(new InputStreamReader(
            SupportedVersions.class.getClassLoader().getResourceAsStream("supportedVersions.json")));
        Gson g = new Gson();
        // Create the type that we want to read from the json file
        Type mapType = new TypeToken<HashMap<String, Object>>() {
        }.getType();
        HashMap<String, Object> supportedVersions = g.fromJson(br, mapType);
        return supportedVersions;
    }

    /**
     * Get the supported versions for an specific Deep Learning framework (engine)
     *
     * @param specificEngine the Deep Learning framework we want
     * @return a HashMap containing all the supported versions for a Deep Learning
     * framework
     */
    public static LinkedTreeMap<String, Object> getSpecificEngineVersionsJson(String specificEngine) {
        HashMap<String, Object> allVersions = readVersionsJson();
        LinkedTreeMap<String, Object> engineVersions = (LinkedTreeMap<String, Object>) allVersions.get(specificEngine);
        return engineVersions;
    }

    // TODO add exception for not supported engine, engine that is not
    // in the JSON file

    /**
     * Get the closest Deep Learning framework (engine) version to the one provided.
     * If no version coincides exactly with the ones allowed, retrieve the closest one.
     * For example if there is no version 2.7.1, retrieve 2.7.5. In the worst case scenario
     * However, at least the major version has to coincide. For example version 2.1.0 will not
     * be never retrieved if the wanted version is 1.2.2. In the case there is no version 1.X.Y
     * an exception will be thrown.
     *
     * @param version    The wanted version of the Deep Learning framework
     * @param versionSet Set of all the versions supported by the program
     * @return the closest version to the available one
     * @throws Exception if the version is totally incompatible with the wanted one
     */
    public static String findVersionInJSON(String version, Set<String> versionSet) throws Exception {
        // Get the version with only major and minor version numbers, no revision number
        // For example 2.8.1 -> 2.8. If the version already has not the revision number
        // leave it as it is.
        if (version.indexOf(".") != -1 && version.indexOf(".") != version.lastIndexOf(".")) {
            int secondDotPos = version.substring(version.indexOf(".")).indexOf(".");
            version = version.substring(0, version.indexOf(".") + secondDotPos);
        }
        List<String> auxVersionList = versionSet.stream().map(s -> s.substring(0, s.lastIndexOf(".")))
            .collect(Collectors.toList());
        if (auxVersionList.contains(version)) {
            return (String) versionSet.toArray()[auxVersionList.indexOf(version)];
        }
        // If there is still no coincidence, just look for the major version.
        // For example, in 2.3.4 just look for the most recent 2 version
        if (version.indexOf(".") != -1)
            version = version.substring(0, version.indexOf("."));
        auxVersionList = auxVersionList.stream().map(s -> s.substring(0, s.lastIndexOf(".")))
            .collect(Collectors.toList());
        if (auxVersionList.contains(version)) {
            return (String) versionSet.toArray()[auxVersionList.indexOf(version)];
        } else {
            // TODO create exception
            throw new Exception();
        }
    }

    /**
     * Retrieve the JavaVersion key from the SupportedVErsions HashMap read from the JSON
     *
     * @param version     version of the Deep Learning framework as it is written in the JSON file.
     *                    It is the same as the original one but it might contain only
     * @param allVersions list of all supported versions
     * @return get the Java version for a specific Pyrhon version
     */
    public static String getJavaVersionFromVersionJSON(String version, LinkedTreeMap<String, Object> allVersions) {
        LinkedTreeMap<String, String> versionJSON = (LinkedTreeMap<String, String>) allVersions.get(version);
        return versionJSON.get(javaVersionsKey);
    }

    /**
     * Find the corresponding version of the API to run a Deep Learning
     * framework in Java.
     *
     * @param version version of the Deep Learning framework (engine) used to create the model
     * @return the corresponding Java version
     * @throws Exception throw exception in the case the version wanted is not
     *                   supported
     */
    public String getCorrespondingJavaVersion(String version) throws Exception {
        if (this.versionSet.contains(version)) {
            return getJavaVersionFromVersionJSON(version, this.versionsDic);
        } else {
            version = findVersionInJSON(version, versionSet);
            // TODO warn that the version used is not the exact same one as the one created
            return getJavaVersionFromVersionJSON(version, this.versionsDic);
        }
    }

    /**
     * Get the supported versions for the wanted Deep Learning framework (engine)
     *
     * @return set of supported versions
     */
    public Set<String> getSupportedVersions() {
        return this.versionSet;
    }
}