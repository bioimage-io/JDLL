package org.bioimageanalysis.icy.deeplearning.utils.model.description;

import java.util.List;

public class ModelDescription {
    private List<AuthorsDescription> authors;
    private List<CiteDescription> cite;
    private List<String> covers;
    private String description;
    private String documentation;
    private String format_version;
    private List<InputTensorDescription> inputs;
    private String license;
    private String name;
    private List<OutputTensorDescription> outputs;
    private List<String> tags;
    private List<String> test_inputs;
    private List<String> test_outputs;
    private String timestamp;
    private String type;
    private WeightsDescription weights;

    public String getDocumentation() {
        return documentation;
    }

    public void setDocumentation(String documentation) {
        this.documentation = documentation;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getFormatVersion() {
        return format_version;
    }

    public List<InputTensorDescription> getInputs() {
        return inputs;
    }

    public void setInputs(List<InputTensorDescription> inputs) {
        this.inputs = inputs;
    }

    public List<OutputTensorDescription> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<OutputTensorDescription> outputs) {
        this.outputs = outputs;
    }

    public List<AuthorsDescription> getAuthors() {
        return authors;
    }

    public void setAuthors(List<AuthorsDescription> authors) {
        this.authors = authors;
    }

    public List<CiteDescription> getCite() {
        return cite;
    }

    public void setCite(List<CiteDescription> cite) {
        this.cite = cite;
    }

    public List<String> getCovers() {
        return covers;
    }

    public void setCovers(List<String> covers) {
        this.covers = covers;
    }

    public String getFormat_version() {
        return format_version;
    }

    public void setFormat_version(String format_version) {
        this.format_version = format_version;
    }

    public String getLicense() {
        return license;
    }

    public void setLicense(String license) {
        this.license = license;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public List<String> getTest_inputs() {
        return test_inputs;
    }

    public void setTest_inputs(List<String> test_inputs) {
        this.test_inputs = test_inputs;
    }

    public List<String> getTest_outputs() {
        return test_outputs;
    }

    public void setTest_outputs(List<String> test_outputs) {
        this.test_outputs = test_outputs;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public WeightsDescription getWeights() {
        return weights;
    }

    public void setWeights(WeightsDescription weights) {
        this.weights = weights;
    }
}
