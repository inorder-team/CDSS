package com.inorder.clinical.acs.model;

public class CagFinding {
    private String caseId;
    private Boolean diagnosticCagCompleted = Boolean.FALSE;
    private LesionCategory lesionCategory;
    private Double maxDiameterStenosisPercent;
    private Integer numberOfEpicardialVesselsWithSignificantDisease;
    private Boolean culpritVessel = Boolean.FALSE;
    private Boolean ffrPerformed = Boolean.FALSE;
    private Double ffrValue;
    private Boolean stressImagingPerformed = Boolean.FALSE;
    private Boolean stressImagingPositiveForIschaemia;

    public CagFinding() {
    }

    public String getCaseId() {
        return caseId;
    }

    public void setCaseId(String caseId) {
        this.caseId = caseId;
    }

    public Boolean getDiagnosticCagCompleted() {
        return diagnosticCagCompleted;
    }

    public void setDiagnosticCagCompleted(Boolean diagnosticCagCompleted) {
        this.diagnosticCagCompleted = diagnosticCagCompleted;
    }

    public LesionCategory getLesionCategory() {
        return lesionCategory;
    }

    public void setLesionCategory(LesionCategory lesionCategory) {
        this.lesionCategory = lesionCategory;
    }

    public Double getMaxDiameterStenosisPercent() {
        return maxDiameterStenosisPercent;
    }

    public void setMaxDiameterStenosisPercent(Double maxDiameterStenosisPercent) {
        this.maxDiameterStenosisPercent = maxDiameterStenosisPercent;
    }

    public Integer getNumberOfEpicardialVesselsWithSignificantDisease() {
        return numberOfEpicardialVesselsWithSignificantDisease;
    }

    public void setNumberOfEpicardialVesselsWithSignificantDisease(Integer numberOfEpicardialVesselsWithSignificantDisease) {
        this.numberOfEpicardialVesselsWithSignificantDisease = numberOfEpicardialVesselsWithSignificantDisease;
    }

    public Boolean getCulpritVessel() {
        return culpritVessel;
    }

    public void setCulpritVessel(Boolean culpritVessel) {
        this.culpritVessel = culpritVessel;
    }

    public Boolean getFfrPerformed() {
        return ffrPerformed;
    }

    public void setFfrPerformed(Boolean ffrPerformed) {
        this.ffrPerformed = ffrPerformed;
    }

    public Double getFfrValue() {
        return ffrValue;
    }

    public void setFfrValue(Double ffrValue) {
        this.ffrValue = ffrValue;
    }

    public Boolean getStressImagingPerformed() {
        return stressImagingPerformed;
    }

    public void setStressImagingPerformed(Boolean stressImagingPerformed) {
        this.stressImagingPerformed = stressImagingPerformed;
    }

    public Boolean getStressImagingPositiveForIschaemia() {
        return stressImagingPositiveForIschaemia;
    }

    public void setStressImagingPositiveForIschaemia(Boolean stressImagingPositiveForIschaemia) {
        this.stressImagingPositiveForIschaemia = stressImagingPositiveForIschaemia;
    }
}
