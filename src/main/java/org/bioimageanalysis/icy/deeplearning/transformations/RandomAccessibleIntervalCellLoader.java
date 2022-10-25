package org.bioimageanalysis.icy.deeplearning.transformations;

import java.util.Map;

import bdv.viewer.Source;
import bdv.viewer.SourceAndConverter;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.SingleCellArrayImg;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.view.Views;

public class RandomAccessibleIntervalCellLoader< T extends NativeType< T >> implements CellLoader< T >
{
    private final Map< String, SourceAndConverter< T >> cellKeyToSource;
    private final int level;
    private final boolean encodeSource;


    public RandomAccessibleIntervalCellLoader( Map< String, SourceAndConverter< T > > cellKeyToSource, int level, boolean encodeSource )
    {
        this.cellKeyToSource = cellKeyToSource;
        this.level = level;
        this.encodeSource = encodeSource;
    }

    @Override
    public void load( SingleCellArrayImg< T, ? > cell ) throws Exception
    {
        final String cellKey = getCellKey( cell.minAsLongArray() );

//        Views.interval();
//        Views.addDimension();


        if ( ! cellKeyToSource.containsKey( cellKey ) )
        {
            return;
        }
        else
        {
            // Get the RAI for this cell
            final Source< T > source = cellKeyToSource.get( cellKey ).getSpimSource();
            RandomAccessibleInterval< T > data = source.getSource( 0, level );

            // Create a view that is shifted to the cell position
            final long[] offset = computeTranslation( asInts( cell.dimensionsAsLongArray() ), cell.minAsLongArray(), data.dimensionsAsLongArray() );
            data = Views.translate( Views.zeroMin( data ), offset );

            // copy RAI into cell
            Cursor< T > sourceCursor = Views.iterable( data ).cursor();
            RandomAccess< T > targetAccess = cell.randomAccess();

            if ( encodeSource )
            {
                final String name = source.getName();
                while ( sourceCursor.hasNext() )
                {
                    sourceCursor.fwd();
                    // copy the sourceCursor in order not to modify it by the source name encoding
                    targetAccess.setPositionAndGet( sourceCursor ).set( sourceCursor.get().copy() );
                    SourceNameEncoder.encodeName( (UnsignedIntType) targetAccess.get(), name );
                }
            }
            else
            {
                while ( sourceCursor.hasNext() )
                {
                    sourceCursor.fwd();
                    targetAccess.setPositionAndGet( sourceCursor ).set( sourceCursor.get() );
                }
            }
        }
    }

    public static int[] asInts( long[] longs) {
        int[] ints = new int[longs.length];

        for(int i = 0; i < longs.length; ++i)
        {
            ints[i] = (int)longs[i];
        }

        return ints;
    }

    private long[] computeTranslation( int[] cellDimensions, long[] cellMin, long[] dataDimensions )
    {
        final long[] translation = new long[ cellMin.length ];
        for ( int d = 0; d < 2; d++ )
        {
            // position of the cell + offset for margin
            translation[ d ] = cellMin[ d ] + (long) ( ( cellDimensions[ d ] - dataDimensions[ d ] ) / 2.0 );
        }
        return translation;
    }

    private static String getCellKey( long[] cellMins )
    {
        StringBuilder key = new StringBuilder("_");
        for ( int d = 0; d < 2; d++ )
            key.append(cellMins[d]).append("_");

        return key.toString();
    }
}
