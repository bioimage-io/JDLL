package org.bioimageanalysis.icy.deeplearning.yaml;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.bioimageanalysis.icy.deeplearning.transformations.TensorTransformation;
import org.bioimageanalysis.icy.deeplearning.transformations.TensorTransformation.Mode;
import org.yaml.snakeyaml.Yaml;

public class DeserializePreProcessingTestDrive
{

	private static final String PREPROCESSING_KEY = "preprocessing";

	private static final String PREPROCESSING_NAME_KEY = "name";

	private static final String PREPROCESSING_KWARGS_KEY = "kwargs";

	private static final String KWARGS_AXES_KEY = "axes";

	private static final String KWARGS_MODE_KEY = "mode";

	private static final String KWARGS_EPS_KEY = "eps";

	private static final String KWARGS_MAXPERCENTILE_KEY = "max_percentile";

	private static final String KWARGS_MINPERCENTILE_KEY = "min_percentile";

	public static void main( final String[] args )
	{
		final String yamlPath = "samples/rdf.yaml";
		final Yaml yaml = new Yaml();

		try (InputStream stream = new FileInputStream( yamlPath ))
		{
			final Map< String, Object > map = yaml.load( stream );
			System.out.println( map );

			/*
			 * List of pre-processing.
			 */
			final Object preprocessingListObj = map.get( PREPROCESSING_KEY );
			if ( !( preprocessingListObj instanceof List ) )
			{
				System.err.println( "Invalid pre-processing specifications in " + yamlPath
						+ "\nExpected pre-processing commands to be a list but was: "
						+ preprocessingListObj.getClass().getName() );
				return;
			}

			@SuppressWarnings( "unchecked" )
			final List< Object > preprocessingList = ( List< Object > ) preprocessingListObj;
			int index = 0;
			for ( final Object preprocessingObj : preprocessingList )
			{
				/*
				 * Pre-processing.
				 */
				if ( !( preprocessingObj instanceof Map ) )
				{
					System.err.println( "Invalid pre-processing specifications in " + yamlPath
							+ "\nAt index " + index + " in the command list, expected the command to be a map, but was: "
							+ preprocessingObj.getClass().getName() );
					return;
				}

				@SuppressWarnings( "unchecked" )
				final Map< String, Object > preprocessing = ( Map< String, Object > ) preprocessingObj;

				/*
				 * Pre-processing name.
				 */
				final Object nameObj = preprocessing.get( PREPROCESSING_NAME_KEY );
				if ( nameObj == null )
				{
					System.err.println( "Invalid pre-processing specifications in " + yamlPath
							+ "\nAt index " + index + " in the command list, could not find the name of the preprocessing command. " );
					return;
				}
				if ( !( nameObj instanceof String ) )
				{
					System.err.println( "Invalid pre-processing specifications in " + yamlPath
							+ "\nAt index " + index + " in the command list, exepcted the name of the preprocessing command to be a string, but was: "
							+ nameObj.getClass().getName() );
					return;
				}
				final String name = ( String ) nameObj;

				/*
				 * Pre-processing args.
				 */
				final Object kwargsObj = preprocessing.get( PREPROCESSING_KWARGS_KEY );
				if ( kwargsObj == null )
				{
					System.err.println( "Invalid pre-processing specifications in " + yamlPath
							+ "\nAt index " + index + " in the command list, could not find the arguments (kwargs) of the preprocessing command. " );
					return;
				}
				if ( !( kwargsObj instanceof Map ) )
				{
					System.err.println( "Invalid pre-processing specifications in " + yamlPath
							+ "\nAt index " + index + " in the command list, exepcted the arguments of the preprocessing command to be a map, but was: "
							+ kwargsObj.getClass().getName() );
					return;
				}
				@SuppressWarnings( "unchecked" )
				final Map< String, Object > kwargs = ( Map< String, Object > ) kwargsObj;

				try
				{
					final TensorTransformation transform = makeTransformation( name, kwargs );
				}
				catch ( final IllegalArgumentException e )
				{
					System.err.println( "Problem parsing the pre-processing list at index " + index + ":\n" + e.getMessage() );
					return;
				}
				index++;
			}

		}
		catch ( final FileNotFoundException e )
		{
			System.err.println( "Could not find YAML file at specified path: " + yamlPath );
			e.printStackTrace();
		}
		catch ( final IOException e )
		{
			System.err.println( "Problem reading the YAML file: " + yamlPath );
			e.printStackTrace();
		}
	}

	private static TensorTransformation makeTransformation( final String name, final Map< String, Object > kwargs ) throws IllegalArgumentException
	{
		switch ( name )
		{
		case "scale_range":
			return makeScaleRange( kwargs );
		default:
			throw new IllegalArgumentException( "Unknown transformation name: " + name );
		}
	}

	private static TensorTransformation makeScaleRange( final Map< String, Object > kwargs )
	{
		// Axes, no default value, do not accept batch axis.
		final boolean bValid = false;
		final String axes = getAxesArg( kwargs, bValid );

		// Mode, no default value, accept only 'per_dataset' and 'per_sample'.
		final Mode mode = getModeArg( kwargs, Mode.PER_DATASET, Mode.PER_SAMPLE );

		// Epsilon arg. Has default value.
		final double eps = getEpsArg( kwargs, 1e-6 );

		// Max percentile. Has default value.
		final double maxPercentile = getMaxPercentileArg( kwargs, 100. );

		// Min percentile. Has default value.
		final double minPercentile = getMinPercentileArg( kwargs, 0. );

		if (minPercentile <= maxPercentile)
			throw new IllegalArgumentException( "Max percentile must be strictly larger than min percentile. "
					+ "Values were: " + maxPercentile + " and " + minPercentile );

		return null; // TODO make the scale range transform.

	}

	private static double getMinPercentileArg( final Map< String, Object > kwargs, final double defaultValue )
	{
		return getArg( KWARGS_MINPERCENTILE_KEY, kwargs, Double.class, defaultValue );
	}

	private static double getMaxPercentileArg( final Map< String, Object > kwargs, final double defaultValue )
	{
		return getArg( KWARGS_MAXPERCENTILE_KEY, kwargs, Double.class, defaultValue );
	}

	private static double getEpsArg( final Map< String, Object > kwargs, final double defaultValue )
	{
		return getArg( KWARGS_EPS_KEY, kwargs, Double.class, defaultValue );
	}

	private static Mode getModeArg( final Map< String, Object > kwargs, final Mode... acceptedModes )
	{
		final String modeStr = getArg( KWARGS_MODE_KEY, kwargs );

		Mode mode = null;
		for ( final Mode m : Mode.values() )
		{
			if ( m.toString().equals( modeStr ) )
			{
				mode = m;
				break;
			}
		}
		if ( mode == null )
			throw new IllegalArgumentException( "Unknown 'mode' value: " + modeStr );

		final List< Mode > modeList = Arrays.asList( acceptedModes );
		if ( !modeList.isEmpty() && !modeList.contains( mode ) )
			throw new IllegalArgumentException( "Mode can only be one of " + modeList + ". Was: " + mode );

		return mode;
	}

	private static String getAxesArg( final Map< String, Object > kwargs, final boolean bValid )
	{
		final String axes = getArg( KWARGS_AXES_KEY, kwargs );
		if ( !bValid && axes.contains( "b" ) )
			throw new IllegalArgumentException( "The 'axes' specification contains the batch axis 'b' which is not valid here." );

		if ( !axes.matches( "[bitcxyz]*" ) )
			throw new IllegalArgumentException( "The 'axes' specification contains unknown characters: " + axes );

		return axes;
	}

	private static String getArg( final String key, final Map< String, Object > kwargs )
	{
		return getArg( key, kwargs, String.class );
	}

	private static < T > T getArg( final String key, final Map< String, Object > map, final Class< T > klass )
	{
		if ( !map.containsKey( key ) )
			throw new IllegalArgumentException( "Could not find the specification for the '" + key + "' argument and no defautl value specified." );

		return getAndCast( key, map, klass );
	}

	private static < T > T getArg( final String key, final Map< String, Object > map, final Class< T > klass, final T defaultValue )
	{
		if ( !map.containsKey( key ) )
			return defaultValue;

		return getAndCast( key, map, klass );
	}

	private static < T > T getAndCast( final String key, final Map< String, Object > map, final Class< T > klass )
	{
		final Object obj = map.get( key );
		if ( klass.isInstance( obj ) )
			throw new IllegalArgumentException( "Expected the value for the '" + key + "' argument to be of class "
					+ klass.getSimpleName() + " but was " + obj.getClass().getSimpleName() );

		final T value = klass.cast( obj );
		return value;
	}
}
