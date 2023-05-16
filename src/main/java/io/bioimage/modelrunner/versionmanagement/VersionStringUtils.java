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
package io.bioimage.modelrunner.versionmanagement;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.utils.IndexingUtils;

/**
 * Class that contains methods to manipulate and compare the version Strings.
 * It has methods that identify the closest
 * version or identify if two versions are equal for example.
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class VersionStringUtils {	
	/**
	 * Version number given if there is no version known
	 */
	private static final int WRONG_VERSION = -10000000;
	
	public static void main(String[] args) {
		List<String> list = new ArrayList<String>();
		list.add("15");
		list.add("17");
		list.add("28");
		list.add("31");
		list.add("22");
		List<String> aa = getCompatibleEngineVersionsInOrder("20", list, "onnx");
		System.out.print(false);
	}

	/** TODO clean a little bit the code
	 * Return an ordered list of the most compatible engine versions to the engine version of interest.
	 * We define compatible engines as those of the same DL framework whose major version is the same.
	 * Pytorch 1.13.1 and 1.9.0 are compatible but Tensorflow 1.15.0 and Tensorflow 2.3.0 are not.
	 * 
	 * The criteria to order the list is first being a bigger version (1.15 is more compatible with 1.25 than 
	 * with 1.13) and then distance (1.15 is more compatible with 1.16 than with 1.25).
	 * 
	 * @param version
	 * 	the version of interest
	 * @param versionList
	 * 	list of all the versions that are installed
	 * @param engine
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @return a list ordered from more compatible to less compatible
	 */
	public static List<String> getCompatibleEngineVersionsInOrder(String version, List<String> versionList, String engine) {
		List<String> vs = new ArrayList<String>();
		if (version == null || versionList == null || versionList.size() == 0) {
			String miss = missingArgument(version, versionList, engine);
			if (miss != null)
				vs.add(miss);
			return vs;
		}
		int nPoints = getNumberOfPoints(Arrays.asList(new String[]{version}));
		nPoints = nPoints > getNumberOfPoints(versionList) ? nPoints : getNumberOfPoints(versionList);
		versionList = uniformVersionList(versionList, nPoints);
		version = uniformVersionList(Arrays.asList(new String[]{version}), nPoints).get(0);
		List<Integer> intVersionList = listOfStringVersionsIntoListOfIntVersions(versionList);
		// If there are no downloaded versions return null
		if (intVersionList.size() == 0)
			return vs;
		int intVersion = convertVersionIntoIntegerOrGetFromList(version, intVersionList);
		// Substract the version of interest to the version list to find the closest higher version
		List<Integer> versionDists = intVersionList.stream()
												.map(v -> v = v - intVersion)
												.collect(Collectors.toList());
		List<Integer> absDists = versionDists.stream().map(i -> Math.abs(i)).collect(Collectors.toList());
		Integer[] absArgSort = IndexingUtils.argsort(absDists);
		List<String> posList = new ArrayList<String>();
		List<String> negList = new ArrayList<String>();
		for (int i = 0; i < absDists.size(); i ++) {
			if (!engine.toLowerCase().equals(EngineInfo.getOnnxKey())  
					&& version.split("\\.")[0].equals(versionList.get(absArgSort[i]).split("\\.")[0]))
				continue;
			else if ( versionDists.get(absArgSort[i]) >= 0) {
				posList.add(versionList.get(absArgSort[i]));
			} else if ( versionDists.get(absArgSort[i]) < 0) {
				negList.add(versionList.get(absArgSort[i]));
			}
		}
		posList.addAll(negList);
		return posList;
	}

	/** TODO clean a little bit the code
	 * Return the most convenient engine version to load the model trained with
	 * the version specified in the yaml file. The most convenient is either the 
	 * actual training version, the closest higher version existing or in the case 
	 * only one version is downloaded, the only one downloaded.
	 * @param version
	 * 	the version of interest
	 * @param versionList
	 * 	list of all the versions that are installed
	 * @param engine
	 * 	Deep Learning framework (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @return the most convenient version
	 */
	public static String getMostCompatibleEngineVersion(String version, List<String> versionList, String engine) {
		if (version == null || versionList == null || versionList.size() == 0) {
			return missingArgument(version, versionList, engine);
		}
		int nPoints = getNumberOfPoints(Arrays.asList(new String[]{version}));
		nPoints = nPoints > getNumberOfPoints(versionList) ? nPoints : getNumberOfPoints(versionList);
		versionList = uniformVersionList(versionList, nPoints);
		version = uniformVersionList(Arrays.asList(new String[]{version}), nPoints).get(0);
		List<Integer> intVersionList = listOfStringVersionsIntoListOfIntVersions(versionList);
		// If there are no downloaded versions return null
		if (intVersionList.size() == 0)
			return null;
		int intVersion = convertVersionIntoIntegerOrGetFromList(version, intVersionList);
		// Substract the version of interest to the version list to find the closest higher version
		List<Integer> versionDists = intVersionList.stream()
												.map(v -> v = v - intVersion)
												.collect(Collectors.toList());
		// Find the closest version if the wanted version is not available
		int closestBiggerInd = indexOfBiggerClosestVersion(versionDists);
		if (closestBiggerInd != -1 ) {
			String possibleVersion = versionList.get(closestBiggerInd);
			// Make sure that for every engine, there is no interference between major versions
			if (!engine.toLowerCase().equals(EngineInfo.getOnnxKey())  
					&& version.split("\\.")[0].equals(possibleVersion.split("\\.")[0]))
				return possibleVersion;
			else if (engine.toLowerCase().equals(EngineInfo.getOnnxKey()))
				return possibleVersion;
		}
		int closestSmallerInd = indexOfSmallerClosestVersion(versionDists);
		if (closestSmallerInd != -1 ) {
			String possibleVersion = versionList.get(closestSmallerInd);
			// Make sure that for every engine, there is no interference between major versions
			if (!engine.toLowerCase().equals(EngineInfo.getOnnxKey()) 
					&& version.split("\\.")[0].equals(possibleVersion.split("\\.")[0]))
				return possibleVersion;
			else if (engine.toLowerCase().equals(EngineInfo.getOnnxKey()))
				return possibleVersion;
		}
		return null;
	}
	
	/**
	 * In case any of the arguments of the {@link #getMostCompatibleEngineVersion(String, List, String)}
	 * is null or empty, find the highest version available and return it. IF there is no available
	 * versions or engine throw an exception
	 * @param version
	 * 	version of interest
	 * @param versionList
	 * 	list of available versions
	 * @param engine
	 * 	engine of interest (tensorflow, pytorch, onnx...) as defined with the engine tag 
	 * at https://raw.githubusercontent.com/bioimage-io/model-runner-java/main/src/main/resources/availableDLVersions.json
	 * @return version wanted
	 */
	private static String missingArgument(String version, List<String> versionList, String engine) {
		if (versionList == null || versionList.size() == 0) {
			return null;
		} else if (engine == null) {
			throw new NullPointerException("In order to find the most compatible version for a "
					+ "framework, please introduce a non-null Deep Learning framework wanted.");
		}
		// The other option is version == null
		List<Integer> intVs = listOfStringVersionsIntoListOfIntVersions(versionList);
		int v = 0;
		int ind = 0;
		for (int i = 0; i < intVs.size(); i ++) {
			if (intVs.get(i) > v) {
				v = intVs.get(i);
				ind = i;
			}
		}
		return versionList.get(ind);
	}
	
	/**
	 * Convert list of String versions into list of Integer versions
	 * @param strs
	 * 	list of String versions
	 * @return list of int versions
	 */
	private static List<Integer> listOfStringVersionsIntoListOfIntVersions(List<String> strs){
		List<Integer> intVersionList = strs.stream()
				.map(VersionStringUtils::convertVersionIntoInteger)
				.collect(Collectors.toList());
		return intVersionList;
	}
	
	/**
	 * This method checks if two different Strings represent the same version
	 * @param v1
	 * 	one version
	 * @param v2
	 * 	another version
	 * @return true if the versions are the same and false otherwise
	 */
	public static boolean areTheyTheSameVersion(String v1, String v2) {
		if (v1 == null || v2 == null)
			return false;
		int nPoints = getNumberOfPoints(Arrays.asList(new String[]{v1, v2}));
		List<String> versionList = uniformVersionList(Arrays.asList(new String[]{v1, v2}), nPoints);
		return versionList.get(0).equals(versionList.get(1));
	}
	
	/**
	 * This method checks if two different Strings represent the same version
	 * @param v1
	 * 	one version
	 * @param v2
	 * 	another version
	 * @param point
	 * 	until which point the comparison should be made
	 * @return true if the versions are the same and false otherwise
	 */
	public static boolean areTheyTheSameVersionUntilPoint(String v1, String v2, int point) {
		if (v1 == null || v2 == null)
			return false;
		v1 = getUntilPoint(v1, point);
		v2 = getUntilPoint(v2, point);
		int nPoints = getNumberOfPoints(Arrays.asList(new String[]{v1, v2}));
		List<String> versionList = uniformVersionList(Arrays.asList(new String[]{v1, v2}), nPoints);
		return versionList.get(0).equals(versionList.get(1));
	}
	
	/**
	 * Return a version String until the selected point.
	 * For version 1.15.0.3 -> if point = 1, 1; point = 2, 1.15; 
	 * point = 3, 1.15.0; point >= 4, 1.15.0.3
	 * @param v
	 * 	string version
	 * @param point
	 * 	up to which point the string is going to be evaluated
	 * @return the wanted substring
	 */
	private static String getUntilPoint(String v, int point) {
		int ind = 0;
		for (int i = 0; i < point; i ++) {
			ind = v.indexOf(".", ind + 1);
			if (ind == -1)
				return v;
		}
		return v.substring(0, ind);
	}
	
	/**
	 * MEthod that makes uniform the number of points accross a version list by adding ".0"
	 * the needed times to the end of each version
	 * @param versionList
	 * 	the list with versions
	 * @param nPoints
	 * 	the target number of points per version
	 * @return a list of versions where each version has at least the wanted amount of points
	 */
	public static List<String> uniformVersionList(List<String> versionList, int nPoints){
		for (int i = 0; i < versionList.size(); i ++) {
			// Add a 0 if the version ends in ".", for example 1.5. -> 1.5.0
			if (versionList.get(i).endsWith("."))
				versionList.set(i, versionList.get(i) + "0");
			// Add ".0" to the version number until it has the required number of points
			while (countNumberOfOccurences(versionList.get(i), ".") < nPoints) {
				versionList.set(i, versionList.get(i) + ".0");
			}
		}
		return versionList;
	}
	
	/**
	 * 
	 * REturn the maximum number of points that any version in a list of versions has 
	 * @param versionList
	 * 	the list of versions
	 * @return the maximum number of points that any single version of the list has
	 */
	private static int getNumberOfPoints(List<String> versionList) {
		int points = 0;
		for (String vv : versionList) {
			int pp = countNumberOfOccurences(vv, ".");
			if (pp > points)
				points = pp;
		}
		return points;
	}
	
	/**
	 * Count number of occurences of a String in another String
	 * @param str
	 * 	the String where the occurences are counted
	 * @param occurence
	 * 	the String that we need to look for in the other String
	 * @return number of times that one String appears in the other
	 */
	public static int countNumberOfOccurences(String str, String occurence) {
		return str.length() - str.replace(occurence, "").length();
	}
	
	/**
	 * Retrieves the version integer from a String and in the case that the String does not
	 * contain an integer, get the highest version from the list
	 * @param version
	 * 	version of interest
	 * @param intVersionList
	 * 	list of available versions
	 * @return version of interest in int format
	 */
	private static int convertVersionIntoIntegerOrGetFromList(String version, List<Integer> intVersionList){
		int versionInt = convertVersionIntoInteger(version);
		if (versionInt == WRONG_VERSION)
			versionInt = getHighestAvailable(intVersionList);
		return versionInt;
	}
	
	/**
	 * Returns the highest number of a list of versions
	 * @param versions
	 * 	the list of versions
	 * @return the highest version
	 */
	private static int getHighestAvailable(List<Integer> versions) {
		ArrayList<Integer> versionsAux = new ArrayList<Integer>(versions);
		Collections.sort(versionsAux);
		int highestVersion = versionsAux.get(versionsAux.size() - 1);
		return highestVersion;
	}
	
	/**
	 * Get the position of the absolute smallest number in the list that
	 * is bigger than 0
	 * @param list
	 * 	list of integers
	 * @return the position of the wanted number, -1 if it does not exist
	 */
	private static int indexOfBiggerClosestVersion(List<Integer> list){
		int ind = -1;
		int biggerClosestDist = Integer.MAX_VALUE;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) < 0)
				continue;
			if (list.get(i) < biggerClosestDist) {
				ind = i;
				biggerClosestDist = list.get(i);
			}
		}
		return ind;		
	}
	
	/**
	 * Get the position of the absolute smallest number in the list that
	 * is smaller than 0
	 * @param list
	 * 	list of integers
	 * @return the position of the wanted number, -1 if it does not exist
	 */
	private static int indexOfSmallerClosestVersion(List<Integer> list){
		int ind = -1;
		int smallerClosestDist = Integer.MIN_VALUE;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) > 0)
				continue;
			if (list.get(i) > smallerClosestDist) {
				ind = i;
				smallerClosestDist = list.get(i);
			}
		}
		return ind;		
	}
	
	/**
	 * Convert an String version identifier into an integer version identifier.
	 * The integer obtained always has 3 figures. For example 1, 1.0 and 1.0.0 are
	 * all converted to 100, 2.2 to 220 and 2.2.3 to 223
	 * @param version
	 * 	the version string
	 * @return the version written in integer format. If the version is not a number,
	 * 	-1 is returned
	 */
	public static int convertVersionIntoInteger(String version) {
		if (version == null || version.toLowerCase().contains("unknown"))
			return WRONG_VERSION;
		String[] separated = version.split("\\.");
		int scaleFactor = separated.length;
		int intV = 0;
		try {
			for (int i = 0; i < scaleFactor; i ++) {
				intV += Integer.parseInt(separated[i]) * Math.pow(10, (scaleFactor - 1 - i) * 4);
			}
		} catch (NumberFormatException ex) {
			return WRONG_VERSION;
		}
		return intV;
	}
}
