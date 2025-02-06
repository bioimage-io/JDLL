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
/**
 * 
 */
package io.bioimage.modelrunner.engine;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.exceptions.LoadEngineException;
import io.bioimage.modelrunner.versionmanagement.DeepLearningVersion;

/**
 * @author Carlos Garcia Lopez de Haro
 */
public class EngineLoader extends ClassLoader
{

	/**
	 * Child ClassLoader of the System ClassLoader that adds the JARs needed to
	 * the classes
	 */
	private ClassLoader engineClassloader;

	/**
	 * Child ClassLoader of the System ClassLoader that adds the JARs needed to
	 * the classes
	 */
	private ClassLoader baseClassloader;

	/**
	 * Path to the folder containing all the jars needed to load the
	 * corresponding engine
	 */
	private String enginePath;

	/**
	 * Engine name of the Deep Learning framework.
	 */
	private String engine;

	/**
	 * Engine name plus major version of the Deep Learning framework.
	 *
	 * Key for the cache of loaded engines.
	 */
	private String versionedEngine;

	/**
	 * Instance of the class from the wanted Deep Learning engine that is used
	 * to call all the needed methods to execute a model
	 */
	private DeepLearningEngineInterface engineInstance;

	/**
	 * Name of the interface all the engines have to implement
	 */
	private static final String ENGINE_INTERFACE_NAME = 
			"io.bioimage.modelrunner.engine.DeepLearningEngineInterface";

	/**
	 * HashMap containing all the already loaded ClassLoader. This variables
	 * avoids reloading classes that have already been loaded. Reloading already
	 * existing ClassLaoders can be a problem if the loaded class opens an
	 * external native library because this library will not be freed until the
	 * Classes from the ClassLoader are Garbage Collected
	 */
	private static HashMap< String, ClassLoader > loadedEngines = new HashMap< String, ClassLoader >();

	/**
	 * HashMap containing all the already loaded engine versions. 
	 * The key corresponds to the framework + the major version and the value
	 * corresponds to the precise string version that was loaded.
	 * This variable is informative to know which framework native libraries
	 * have been loaded.
	 */
	private static HashMap< String, String > loadedVersions = new HashMap< String, String >();

	/**
	 * Create a ClassLaoder that contains the classes of the parent ClassLoader
	 * given as an input and the classes found in the String path given
	 * 
	 * @param classloader
	 *            parent ClassLoader of the wanted ClassLoader
	 * @param engineInfo
	 *            object containing all the needed info to load a Deep LEarning
	 *            framework
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the engines has been incorrectly modified
	 * @throws LoadEngineException if there is any error loading the DL framework
	 */
	private EngineLoader( ClassLoader classloader, EngineInfo engineInfo ) throws LoadEngineException, MalformedURLException, IllegalStateException, IOException
	{
		super();
		this.baseClassloader = classloader;
		this.engine = engineInfo.getFramework();
		this.enginePath = engineInfo.getDeepLearningVersionJarsDirectory();
		this.versionedEngine = this.engine + engineInfo.getMajorVersion();
		loadClasses();
		setEngineClassLoader();
		setEngineInstance();
		setBaseClassLoader();
	}

	/**
	 * Returns the ClassLoader of the corresponding Deep Learning framework
	 * (engine)
	 * 
	 * @param classloader
	 *            parent ClassLoader of the wanted ClassLoader
	 * @param engineInfo
	 *            the path to the directory where all the JARs needed to load
	 *            the corresponding Deep Learning framework (engine) are stored
	 * @return the ClassLoader corresponding to the wanted Deep Learning version
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the engines has been incorrectly modified
	 * @throws LoadEngineException if there is any error loading the engines
	 */
	public static EngineLoader createEngine( ClassLoader classloader, EngineInfo engineInfo ) throws MalformedURLException, IllegalStateException, LoadEngineException, IOException
	{
		return new EngineLoader( classloader, engineInfo );
	}

	/**
	 * Load the needed JAR files into a child ClassLoader of the
	 * ContextClassLoader.The JAR files needed are the JARs that contain the
	 * engine and the JAR containing this class
	 * @throws MalformedURLException if the JAR files are not well defined in the .json file
	 * @throws IOException if there is any error finding the engines in the system
	 * @throws IllegalStateException if any of the engines has been incorrectly modified
	 */
	private void loadClasses() throws MalformedURLException, IllegalStateException, IOException
	{
		// If the ClassLoader was already created, use it.
		// As tf2 is loaded in a separate process, as many versions as we want can be loaded
		if ( loadedEngines.get( versionedEngine ) != null && !versionedEngine.equals("tensorflow2"))
		{
			this.engineClassloader = loadedEngines.get( versionedEngine );
			return;
		}
		ArrayList<URL> urlList = new ArrayList<URL>();
		if ( !(new File( this.enginePath ).isDirectory()) )
			throw new IllegalArgumentException("Engine directory does not exist: " + enginePath);
		DeepLearningVersion dlv = DeepLearningVersion.fromFile(new File( this.enginePath ));
		for ( File ff : new File( this.enginePath ).listFiles() )
		{
			if (!ff.getName().endsWith(".jar") || !dlv.doesJarBelongToEngine(ff.getAbsolutePath()))
					continue;
				
			urlList.add(ff.toURI().toURL());
		}
		URL[] urls = new URL[urlList.size()];
		urlList.toArray(urls);
		this.engineClassloader = new ParentLastURLClassLoader( urls, baseClassloader );
		
		loadedEngines.put( this.versionedEngine, this.engineClassloader );
		loadedVersions.put(this.versionedEngine, dlv.getPythonVersion());
	}

	/**
	 * Set the ClassLoader containing the engines classes as the Thread
	 * classloader.
	 * 
	 */
	public void setEngineClassLoader()
	{
		Thread.currentThread().setContextClassLoader( engineClassloader );
	}

	/**
	 * Set the parent ClassLoader as the Thread classloader
	 */
	public void setBaseClassLoader()
	{
		Thread.currentThread().setContextClassLoader( this.baseClassloader );
	}

	/**
	 * Find the wanted interface {@link DeepLearningEngineInterface} from the entries
	 * of a JAR file. REturns null if the interface is not in the entries
	 * 
	 * @param entries
	 *            entries of a JAR executable file
	 * @return the wanted class implementing the interface or null if it is not
	 *         there
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws SecurityException 
	 * @throws NoSuchMethodException 
	 * @throws InvocationTargetException 
	 * @throws IllegalArgumentException 
	 */
	private static DeepLearningEngineInterface getEngineClassFromEntries( Enumeration< ? extends ZipEntry > entries,
			ClassLoader engineClassloader )
			throws ClassNotFoundException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException
	{
		while ( entries.hasMoreElements() )
		{
			ZipEntry entry = ( ZipEntry ) entries.nextElement();
			String file = entry.getName();
			if ( file.endsWith( ".class" ) && !file.contains( "$" ) && !file.contains( "-" ) )
			{
				String className = getClassNameInJAR( file );
				Class< ? > c = engineClassloader.loadClass( className );
				Class[] intf = c.getInterfaces();
				for ( int j = 0; j < intf.length; j++ )
				{
					if ( intf[ j ].getName().equals( ENGINE_INTERFACE_NAME ) )
					{
						// Assume that DeepLearningInterface has no arguments
						// for the constructor
						return ( DeepLearningEngineInterface ) c.getDeclaredConstructor().newInstance();
					}
				}
				// REmove references
				intf = null;
				c = null;
			}
		}
		return null;
	}

	/**
	 * Return the name of the class as seen by the ClassLoader from the name of
	 * the file entry in the JAR file. Basically removes the .class suffix and
	 * substitutes "/" by ".".
	 * 
	 * @param entryName
	 *            String containing the name of the file compressed inside the
	 *            JAR file
	 * @return the Class name as seen by the ClassLoader
	 */
	private static String getClassNameInJAR( String entryName )
	{
		String className = entryName.substring( 0, entryName.indexOf( "." ) );
		className = className.replace( "/", "." );
		return className;
	}

	/**
	 * Finds the class that implements the interface that connects the java
	 * framework with the deep learning libraries
	 * 
	 * @throws LoadEngineException
	 *             if there is any error finding and loading the DL libraries
	 */
	private void setEngineInstance() throws LoadEngineException
	{
		// Load all the classes in the engine folder and select the wanted
		// interface
		ZipFile jarFile;
		String jarPrefix = "dl-modelrunner-" + this.engine;
		DeepLearningVersion dlv = null;
		try {
			dlv = DeepLearningVersion.fromFile(new File( this.enginePath ));
		} catch (IllegalStateException | IOException e) {
		}
		String errMsg = "Missing " + jarPrefix + " jar file that implements the 'DeepLearningInterface";
		try
		{
			for ( File ff : new File( this.enginePath ).listFiles() )
			{
				// Find the correct dl-modelrunner-<engine> JAR file.
				if (!ff.getName().endsWith(".jar")
						|| !ff.getName().startsWith(jarPrefix + "-") 
						|| !dlv.doesJarBelongToEngine(ff.getAbsolutePath()))
				{
					continue;
				}
				jarFile = new ZipFile( ff );
				Enumeration< ? extends ZipEntry > entries = jarFile.entries();
				this.engineInstance = getEngineClassFromEntries( entries, engineClassloader );
				if ( this.engineInstance != null )
				{
					jarFile.close();
					return;
				}
				jarFile.close();
			}
		}
		catch (Exception | Error ex) {
			errMsg = Types.stackTrace(ex);
		}
		// As no interface has been found create an exception
		throw new LoadEngineException( new File( this.enginePath ), errMsg );
	}

	/**
	 * Return the engine instance from where to call the corresponging engine.
	 * 
	 * 
	 * @return engine instance
	 */
	public DeepLearningEngineInterface getEngineInstance()
	{
		return this.engineInstance;
	}

	/**
	 * Close the created ClassLoader
	 */
	// TODO is it necessary??
	public void close()
	{
		engineInstance.closeModel();
		setBaseClassLoader();
	}
	
	/**
	 * 
	 * @return a map where the keys are the engines that have been loaded and
	 * 	the values the specific versions
	 */
	public static HashMap<String, String> getLoadedVersions() {
		return loadedVersions;
	}
}
