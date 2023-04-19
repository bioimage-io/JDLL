package io.bioimage.modelrunner.engine.installation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;

public class FileDownloader {
	private ReadableByteChannel rbc;
	private FileOutputStream fos;
	
	public FileDownloader(ReadableByteChannel rbc, FileOutputStream fos) {
		this.rbc = rbc;
		this.fos = fos;
	}

	public void call() throws IOException  {
		fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
	}

}
