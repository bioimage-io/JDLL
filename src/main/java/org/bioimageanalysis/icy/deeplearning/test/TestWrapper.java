
package org.bioimageanalysis.icy.deeplearning.test;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** {@code PtEngineProvider} is the PyTorch implementation of {@link EngineProvider}. */
public class TestWrapper implements EngineProvider {

    private static volatile Engine engine; // NOPMD
    private static EngineProvider originalEngineProvider;

    
    public TestWrapper(EngineProvider ep) {
    	originalEngineProvider = ep;
    }
    
    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return "PyTorch";
    }

    /** {@inheritDoc} */
    @Override
    public int getEngineRank() {
        return 2;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        if (engine == null) {
            synchronized (this) {
                engine = originalEngineProvider.getEngine();
            }
        }
        return engine;
    }

	public void setEngine(Engine eng) {
		engine = eng;
		
	}
}
