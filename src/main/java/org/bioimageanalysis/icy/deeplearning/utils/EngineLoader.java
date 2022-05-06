/**
 * 
 */
package org.bioimageanalysis.icy.deeplearning.utils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.bioimageanalysis.icy.deeplearning.exceptions.LoadEngineException;

import ai.djl.ndarray.NDManager;

/**
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class EngineLoader  extends ClassLoader{

	/**
	 * Child ClassLoader of the System ClassLoader that 
	 * adds the JARs needed to the classes 
	 */
	private ClassLoader engineClassloader;
	/**
	 * Child ClassLoader of the System ClassLoader that 
	 * adds the JARs needed to the classes 
	 */
	private ClassLoader icyClassloader;
	/**
	 * Path to the folder containing all the jars 
	 * needed to load the corresponding engine
	 */
	private String enginePath;
	/**
	 * Major version of the Deep Learning framework. This is the first
	 * number of the version until the first dot.
	 */
	private String majorVersion;
	/**
	 * Instance of the class from the wanted Deep Learning engine
	 * that is used to call all the needed methods to execute a model
	 */
	private DeepLearningInterface engineInstance;
	/**
	 * Keyword that the jar file that implements the needed interface has to implement
	 * to be able to be recognized by the DL manager
	 */
	private static final String jarKeyword = "DlEngine";
	/**
	 * Name of the interface all the engines have to implement
	 */
	private static final String interfaceName = "org.bioimageanalysis.icy.deeplearning.utils.DeepLearningInterface";
	/**
	 * HashMap containing all the already loaded ClassLoader. This variables
	 * avoids reloading classes that have already been loaded. Reloading already
	 * existing ClassLaoders can be a problem if the loaded class opens an
	 * external native library because this library will not be freed until
	 * the Classes from the ClassLoader are Garbage Collected
	 */
	private static HashMap<String, ClassLoader> loadedEngines = new HashMap<String, ClassLoader>();
	
	/**
	 * Create a ClassLaoder that contains the classes of the parent
	 * ClassLoader given as an input and the classes found in the 
	 * String path given
	 * 
	 * @param classloader
	 * 	parent ClassLoader of the wanted ClassLoader
	 * @param enginePath
	 * 	String path where the new JARs containing the wanted classes
	 * 	should be located
	 * @throws LoadEngineException if there are errors loading the DL framework
	 * @throws Exception if the DL engine does not contain all the needed libraries
	 */
	private EngineLoader(ClassLoader classloader, EngineInfo engineInfo) throws LoadEngineException,
																				Exception
	{
		super();
		this.icyClassloader = classloader;
		this.enginePath = engineInfo.getDeepLearningVersionJarsDirectory();
		serEngineAndMajorVersion(engineInfo);
		loadClasses();
	    setEngineInstance();
	}
	
	/**
	 * Create an EngineLoader which creates an URLClassLoader
	 * with all the jars in the provided String path
	 * @throws LoadEngineException if there are errors loading the DL framework
	 * @throws Exception if the DL engine does not contain all the needed libraries
	 */
	private EngineLoader(EngineInfo engineInfo) throws LoadEngineException, Exception
	{
		this(Thread.currentThread().getContextClassLoader(), engineInfo);
	}
	
	/**
	 * Returns the ClassLoader of the corresponding Deep Learning framework (engine)
	 * @param enginePath
	 * 	the path to the directory where all the JARs needed to load the corresponding
	 * 	Deep Learning framework (engine) are stored
	 * @return the ClassLoader corresponding to the wanted Deep Learning version
	 * @throws LoadEngineException if there are errors loading the DL framework
	 * @throws Exception if the DL engine does not contain all the needed libraries
	 */
	public static EngineLoader createEngine(EngineInfo engineInfo) throws LoadEngineException, Exception
	{
		return new EngineLoader(engineInfo);
	}
	
	/**
	 * Load the needed JAR files into a child ClassLoader of the 
	 * ContextClassLoader.The JAR files needed are the JARs that
	 * contain the engine and the JAR containing this class
	 * @throws URISyntaxException if there is an error creating an URL
	 * @throws MalformedURLException if theURL is incorrect
	 * @throws InvocationTargetException 
	 * @throws IllegalArgumentException 
	 * @throws IllegalAccessException 
	 * @throws SecurityException 
	 * @throws NoSuchMethodException 
	 * @throws ClassNotFoundException 
	 */
	private void loadClasses() throws URISyntaxException, MalformedURLException, ClassNotFoundException, NoSuchMethodException, SecurityException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		// If the ClassLoader was already created, use it
		if (loadedEngines.get(majorVersion) != null) {
			this.engineClassloader = loadedEngines.get(majorVersion);
			return;
		}
		URL[] urls = new URL[new File(this.enginePath).listFiles().length + 1];
		int c = 0;
		for (File ff : new File(this.enginePath).listFiles()) {
			urls[c ++] = ff.toURI().toURL();
		}
		// TODO the following line tried to add the JAR of the model runner without repetition but remove it for now
		urls[c] = EngineLoader.class.getProtectionDomain().getCodeSource().getLocation().toURI().toURL();
	    this.engineClassloader = new URLClassLoader(urls);
		
		//loadedEngines.put(this.majorVersion, this.engineClassloader);
	}
	
	/**
	 * Set the ClassLoader containing the engines classes as the Thread classloader
	 */
	public void setEngineClassLoader() {
	    Thread.currentThread().setContextClassLoader(this.engineClassloader);
	}
	
	/**
	 * Set the original Icy ClassLoader as the Thread classloader
	 */
	public void setIcyClassLoader() {
	    Thread.currentThread().setContextClassLoader(this.icyClassloader);
	}
	
	/**
	 * Find the wanted interface {@link DeepLearningInterface} from the entries
	 * of a JAR file. REturns null if the interface is not in the entries
	 * @param entries
	 * 	entries of a JAR executable file
	 * @return the wanted class implementing the interface or null if it is not there
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 * @throws InstantiationException 
	 */
	private static DeepLearningInterface getEngineClassFromEntries(Enumeration<? extends ZipEntry> entries, 
										ClassLoader engineClassloader) 
										throws ClassNotFoundException, 
										InstantiationException, 
										IllegalAccessException
	{
		while (entries.hasMoreElements()) {
            ZipEntry entry = (ZipEntry) entries.nextElement();
            String file = entry.getName();
            if (file.endsWith(".class") && !file.contains("$") && !file.contains("-")) {
            	String className = getClassNameInJAR(file);
            	Class<?> c = engineClassloader.loadClass(className);
            	Class[] intf = c.getInterfaces();
            	for (int j=0; j<intf.length; j++) {
					if (intf[j].getName().equals(interfaceName)){
						// Assume that DeepLearningInterface has no arguments for the constructor
						return (DeepLearningInterface) c.newInstance();
					}
				}
            	// REmove references
            	intf = null;
            	c = null;
            }
        }
		return null;
	}
	
	private void serEngineAndMajorVersion(EngineInfo engineInfo) {
		String engine = engineInfo.getEngine();
		String vv = engineInfo.getMajorVersion();
		this.majorVersion = (engine + vv).toLowerCase();
		
	}
	
	/**
	 * Return the name of the class as seen by the ClassLoader from
	 * the name of the file entry in the JAR file. Basically removes the
	 * .class suffix and substitutes "/" by ".".
	 * 
	 * @param entryName
	 * 	String containing the name of the file compressed inside the JAR file
	 * @return the Class name as seen by the ClassLoader
	 */
	public static String getClassNameInJAR(String entryName)
	{
		String className = entryName.substring(0, entryName.indexOf("."));
		className = className.replace("/", ".");
		return className;
	}
	
	/**
	 * Finds the class that implements the interface that connects the java framework 
	 * with the deep learning libraries
	 * @throws LoadEngineException if there is any error finding and loading the DL libraries
	 */
	private void setEngineInstance() throws LoadEngineException
	{
		// Load all the classes in the engine folder and select the wanted interface
		ZipFile jarFile;
		String errMsg = "Missing '" + jarKeyword + "' jar file that implements the 'DeepLearningInterface";
		try {
			for (File ff : new File(this.enginePath).listFiles()) {
				// Only allow .jar files
				if (!ff.getName().endsWith(".jar") || !ff.getName().contains(jarKeyword))
					continue;
				jarFile = new ZipFile(ff);
				Enumeration<? extends ZipEntry> entries = jarFile.entries();
				this.engineInstance = getEngineClassFromEntries(entries, engineClassloader);
            	if (this.engineInstance != null) {
					jarFile.close();
					return;
				}
				jarFile.close();
			}
		} catch (IOException e) {
			errMsg = e.getMessage();
		} catch (ClassNotFoundException e) {
			errMsg = e.getMessage();
		} catch (InstantiationException e) {
			errMsg = e.getMessage();
		} catch (IllegalAccessException e) {
			errMsg = e.getMessage();
		}
		// As no interface has been found create an exception
		throw new LoadEngineException(new File(this.enginePath), errMsg);
	}
	
	/**
	 * Return the engine instance from where to call the corresponging engine
	 * @return engine instance
	 */
	public DeepLearningInterface getEngineInstance()
	{
		return this.engineInstance;
	}
	
	/**
	 * Close the created ClassLoader
	 * @throws IOException if closing any file opened 
	 * 	by this class loader resulted in an IOException. Any 
	 * 	such exceptions are caught internally. If only one is caught,
	 * 	then it is re-thrown. If more than one exception is caught, then
	 * 	the second and following exceptions are added as suppressed exceptions of 
	 * the first one caught, which is then re-thrown.
	 */
	// TODO is it necessary??
	public void close() {
		engineInstance.closeModel();
	    setIcyClassLoader();
	    System.out.println("Exited engine ClassLoader");
	}
}
