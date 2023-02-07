/**
 * 
 */
package io.bioimage.modelrunner.engine;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import io.bioimage.modelrunner.exceptions.LoadEngineException;

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
	private ClassLoader icyClassloader;

	/**
	 * Path to the folder containing all the jars needed to load the
	 * corresponding engine
	 */
	private String enginePath;

	/**
	 * Major version of the Deep Learning framework. This is the first number of
	 * the version until the first dot.
	 */
	private String majorVersion;

	/**
	 * Instance of the class from the wanted Deep Learning engine that is used
	 * to call all the needed methods to execute a model
	 */
	private DeepLearningEngineInterface engineInstance;

	/**
	 * Keyword that the jar file that implements the needed interface has to
	 * implement to be able to be recognized by the DL manager
	 */
	private static final HashMap<String, String> ENGINE_INTERFACE_JARS_MAP = getDlInterfaceJarsMap();

	/**
	 * Name of the interface all the engines have to implement
	 */
	private static final String ENGINE_INTERFACE_NAME = 
			"package io.bioimage.modelrunner.engine.DeepLearningEngineInterface";

	/**
	 * HashMap containing all the already loaded ClassLoader. This variables
	 * avoids reloading classes that have already been loaded. Reloading already
	 * existing ClassLaoders can be a problem if the loaded class opens an
	 * external native library because this library will not be freed until the
	 * Classes from the ClassLoader are Garbage Collected
	 */
	private static HashMap< String, ClassLoader > loadedEngines = new HashMap< String, ClassLoader >();

	/**
	 * Create a ClassLaoder that contains the classes of the parent ClassLoader
	 * given as an input and the classes found in the String path given
	 * 
	 * @param classloader
	 *            parent ClassLoader of the wanted ClassLoader
	 * @param engineInfo
	 *            object containing all the needed info to load a Deep LEarning
	 *            framework
	 * @throws LoadEngineException
	 *             if there are errors loading the DL framework
	 * @throws Exception
	 *             if the DL engine does not contain all the needed libraries
	 */
	private EngineLoader( ClassLoader classloader, EngineInfo engineInfo ) throws LoadEngineException, Exception
	{
		super();
		this.icyClassloader = classloader;
		this.enginePath = engineInfo.getDeepLearningVersionJarsDirectory();
		serEngineAndMajorVersion( engineInfo );
		loadClasses();
		setEngineClassLoader();
		setEngineInstance();
		setIcyClassLoader();
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
	 * @throws LoadEngineException
	 *             if there are errors loading the DL framework
	 * @throws Exception
	 *             if the DL engine does not contain all the needed libraries
	 */
	public static EngineLoader createEngine( ClassLoader classloader, EngineInfo engineInfo )
			throws LoadEngineException, Exception
	{
		return new EngineLoader( classloader, engineInfo );
	}

	/**
	 * Load the needed JAR files into a child ClassLoader of the
	 * ContextClassLoader.The JAR files needed are the JARs that contain the
	 * engine and the JAR containing this class
	 * 
	 * @throws URISyntaxException
	 *             if there is an error creating an URL
	 * @throws MalformedURLException
	 *             if theURL is incorrect
	 * @throws InvocationTargetException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 * @throws SecurityException
	 * @throws NoSuchMethodException
	 * @throws ClassNotFoundException
	 */
	private void loadClasses()
			throws URISyntaxException, MalformedURLException, ClassNotFoundException, NoSuchMethodException,
			SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException
	{
		// If the ClassLoader was already created, use it
		if ( loadedEngines.get( majorVersion ) != null )
		{
			this.engineClassloader = loadedEngines.get( majorVersion );
			return;
		}
		ArrayList<URL> urlList = new ArrayList<URL>();
		// TODO  remove URL[] urls = new URL[ new File( this.enginePath ).listFiles().length ];
		// TODO remove int c = 0;
		for ( File ff : new File( this.enginePath ).listFiles() )
		{
			if (!ff.getName().endsWith(".jar"))
					continue;
			urlList.add(ff.toURI().toURL());
		}
		URL[] urls = new URL[urlList.size()];
		urlList.toArray(urls);
		this.engineClassloader = new URLClassLoader( urls, icyClassloader );

		loadedEngines.put( this.majorVersion, this.engineClassloader );
	}

	/**
	 * Set the ClassLoader containing the engines classes as the Thread
	 * classloader
	 */
	public void setEngineClassLoader()
	{
		Thread.currentThread().setContextClassLoader( this.engineClassloader );
	}

	/**
	 * Set the original Icy ClassLoader as the Thread classloader
	 */
	public void setIcyClassLoader()
	{
		Thread.currentThread().setContextClassLoader( this.icyClassloader );
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

	private void serEngineAndMajorVersion( EngineInfo engineInfo )
	{
		String engine = engineInfo.getEngine();
		String vv = engineInfo.getMajorVersion();
		this.majorVersion = ( engine + vv ).toLowerCase();

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
	public static String getClassNameInJAR( String entryName )
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
		String jarKeyword = getCorrespondingEngineJar();
		String errMsg = "Missing '" + jarKeyword + "' jar file that implements the 'DeepLearningInterface";
		try
		{
			for ( File ff : new File( this.enginePath ).listFiles() )
			{
				// Only allow .jar files
				if ( !ff.getName().endsWith( ".jar" ) || !ff.getName().startsWith( jarKeyword ) )
					continue;
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
		catch ( IOException | ClassNotFoundException | InstantiationException
				| IllegalAccessException | IllegalArgumentException 
				| InvocationTargetException | NoSuchMethodException | SecurityException e )
		{
			errMsg = e.getCause().toString();
		}
		// As no interface has been found create an exception
		throw new LoadEngineException( new File( this.enginePath ), errMsg );
	}

	/**
	 * Return the engine instance from where to call the corresponging engine
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
		setIcyClassLoader();
		System.out.println( "Exited engine ClassLoader" );
	}
	
	/**
	 * Create the static map that associates each of the supported DL engines with the 
	 * corresponding engine JAR needed that implements {@link DeepLearningEngineInterface}
	 * @return
	 */
	private static HashMap<String, String> getDlInterfaceJarsMap(){
		HashMap<String, String> map = new HashMap<String, String>();
		map.put(EngineInfo.getTensorflowKey() + "2", "tensor-flow-2-interface-");
		map.put(EngineInfo.getTensorflowKey() + "1", "tensor-flow-1-interface-");
		map.put(EngineInfo.getOnnxKey(), EngineInfo.getOnnxKey() + "-interface-");
		map.put(EngineInfo.getPytorchKey(), EngineInfo.getPytorchKey() + "-interface-");
		return map;
	}
	
	/**
	 * Helper to extract the JAR name from the {@link #ENGINE_INTERFACE_JARS_MAP}. A helper is
	 * neeed because Tensorflow needs to specify the major version, but Pytorch does not.
	 * @return
	 */
	private String getCorrespondingEngineJar() {
		if (this.majorVersion.startsWith(EngineInfo.getTensorflowKey()))
			return ENGINE_INTERFACE_JARS_MAP.get(this.majorVersion);
		else if (this.majorVersion.startsWith(EngineInfo.getPytorchKey()))
			return ENGINE_INTERFACE_JARS_MAP.get(EngineInfo.getPytorchKey());
		else if (this.majorVersion.startsWith(EngineInfo.getOnnxKey()))
			return ENGINE_INTERFACE_JARS_MAP.get(EngineInfo.getOnnxKey());
		else
			throw new IllegalArgumentException("Selected Deep Learning framework (" + this.majorVersion + ") "
					+ "not supported at the moment.");
	}
}
