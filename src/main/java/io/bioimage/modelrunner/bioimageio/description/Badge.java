package io.bioimage.modelrunner.bioimageio.description;


/**
 * Custom badges specified in some Bioimage.io models' rdf.yaml.
 * For more info go to:
 *  https://github.com/bioimage-io/spec-bioimage-io/blob/gh-pages/model_spec_latest.md
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class Badge
{
    /**
     * Creates a {@link Badge} instance.
     * 
     * @param label
     * 	label of the badge, e.g. 'Open in Colab' 
     * @param icon
     * 	optional String e.g. 'https://colab.research.google.com/assets/colab-badge.svg'
     * @param url
     * 	e.g. 'https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/U-net_2D_ZeroCostDL4Mic.ipynb' 
     * @return The creates instance.
     */
    public static Badge build(String label, String icon, String url)
    {
        Badge badge = new Badge();
        badge.label = label;
        badge.icon = icon;
        badge.url = url;
        return badge;        
    }

    private String label;
    private String icon;
    private String url;

    /**
     * @return label of the badge
     */
    public String getLabel()
    {
        return label;
    }

    /**
     * Sets the label of the badge
     * 
     * @param label
     *        label of the badge.
     */
    public void setText(String label)
    {
        this.label = label;
    }

    /**
     * @return The icon url as a String
     */
    public String getIcon()
    {
        return icon;
    }

    /**
     * Sets the icon url as a String
     * 
     * @param icon
     * 	icon url as a String
     */
    public void setIcon(String icon)
    {
        this.icon = icon;
    }

    /**
	 * @return the url
	 */
	public String getUrl() {
		return url;
	}

	/**
	 * @param url the url to set
	 */
	public void setUrl(String url) {
		this.url = url;
	}

	@Override
    public String toString()
    {
		String str = "Cite {";
		label += " label=" + label;
		if (icon != null)
			str += " icon=" + icon;
		if (url != null)
			str += " url=" + url;
		str += " }";
        return str;
    }
}
