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

import java.net.MalformedURLException;
import java.net.URL;

/**
 * The citation related to this specific model.`
 * It is composed the textual citation and the DOI reference (resolved as a URL).
 * 
 * @author Daniel Felipe Gonzalez Obando
 */
public class Cite
{
    /**
     * Creates a {@link Cite} instance with the text and DOI.
     * 
     * @param text
     *        The citation text.
     * @param doi
     *        The DOI url. It can be null.
     * @param url
     *        url useful for the citation. It can be null.
     * @return The creates instance.
     */
    public static Cite build(String text, String doi, String url)
    {
        Cite cite = new Cite();
        cite.text = text;
        try {
			cite.doi = new URL(doi);
		} catch (MalformedURLException e) {
			cite.doi = null;
		}
        try {
			cite.setUrl(new URL(url));
		} catch (MalformedURLException e) {
			cite.setUrl(null);
		}
        return cite;
    }

    private String text;
    private URL doi;
    private URL url;

    /**
     * @return Citation text
     */
    public String getText()
    {
        return text;
    }

    /**
     * Sets the citation text.
     * 
     * @param text
     *        Citation text.
     */
    public void setText(String text)
    {
        this.text = text;
    }

    /**
     * @return The DOI url.
     */
    public URL getDoi()
    {
        return doi;
    }

    /**
     * Sets the citation DOI url.
     * 
     * @param doi
     *        DOI url.
     */
    public void setDoi(URL doi)
    {
        this.doi = doi;
    }

    /**
	 * @return the url
	 */
	public URL getUrl() {
		return url;
	}

	/**
	 * @param url the url to set
	 */
	public void setUrl(URL url) {
		this.url = url;
	}

	@Override
    public String toString()
    {
		String str = "Cite {";
		if (text != null)
			str += " text=" + text;
		if (doi != null)
			str += " doi=" + doi.toString();
		if (url != null)
			str += " url=" + url.toString();
		str += " }";
        return str;
    }
}
