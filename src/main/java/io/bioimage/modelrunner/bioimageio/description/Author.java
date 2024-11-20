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
package io.bioimage.modelrunner.bioimageio.description;


/**
 * The author field related to this specific model.`
 * It is composed the textual citation and the DOI reference (resolved as a URL).
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class Author
{
    /**
     * Creates a {@link Author} instance with the text and DOI.
     * All the fields are optional
     * 
	 * @param affiliation
	 * 	affiliation of the author
	 * @param email
	 * 	email of the author
	 * @param githubUser
	 * 	github user of the author
	 * @param name
	 * 	name of the author
	 * @param orcid
	 * 	orcid of the author
     * @return The creates instance.
     */
    public static Author build(String affiliation, String email,
    						String githubUser, String name, String orcid)
    {
        Author author = new Author();
        author.affiliation = affiliation;
        author.email = email;
        author.setGithubUser(githubUser);
        author.setName(name);
        author.setOrcid(orcid);
        return author;
    }

    private String affiliation;
    private String email;
    private String githubUser;
    private String name;
    private String orcid;

    /**
     * @return Affiliation
     */
    public String getAffiliation()
    {
        return affiliation;
    }

    /**
     * Sets the affiliation
     * 
     * @param affiliation
     *        affiliation of the author.
     */
    public void setText(String affiliation)
    {
        this.affiliation = affiliation;
    }

    /**
     * @return The email of the author
     */
    public String getEmail()
    {
        return email;
    }

    /**
     * Sets the email of the author
     * 
     * @param email
     *        email of the author.
     */
    public void setDoi(String email)
    {
        this.email = email;
    }

    /**
	 * @return the githubUser
	 */
	public String getGithubUser() {
		return githubUser;
	}

	/**
	 * @param githubUser 
	 * 	the githubUser of the author
	 */
	public void setGithubUser(String githubUser) {
		this.githubUser = githubUser;
	}

	/**
	 * @return the name of the author
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name tof the author
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return the orcid of the author
	 */
	public String getOrcid() {
		return orcid;
	}

	/**
	 * @param orcid the orcid of the author
	 */
	public void setOrcid(String orcid) {
		this.orcid = orcid;
	}
	
	@Override
    public String toString()
    {
		String str = "Author {";
		if (name != null)
			str += " name=" + name;
		if (affiliation != null)
			str += " affiliation=" + affiliation;
		if (orcid != null)
			str += " orcid=" + orcid;
		if (email != null)
			str += " email=" + email;
		if (githubUser != null)
			str += " github_user=" + githubUser;
		str += " }";
        return str;
    }
}
