package com.inorder.clinical.acs.model;

public class AcsCase {
    private String caseId;
    private AcsType acsType;
    private Integer timiScore;
    private Integer graceScore;
    private Boolean haemodynamicInstability = Boolean.FALSE;
    private Boolean electricalInstability = Boolean.FALSE;
    private Boolean recurrentIschaemia = Boolean.FALSE;
    private Boolean dynamicStOrTChanges = Boolean.FALSE;
    private Boolean largeTroponinRise = Boolean.FALSE;
    private Boolean primaryPciFacilityAvailableCloseBy = Boolean.TRUE;
    private Integer expectedFmcToBalloonMinutes;
    private Boolean delayedPresentation = Boolean.FALSE;
    private Boolean lvDysfunction = Boolean.FALSE;
    private Boolean viableMyocardium;

    public AcsCase() {
    }

    public String getCaseId() {
        return caseId;
    }

    public void setCaseId(String caseId) {
        this.caseId = caseId;
    }

    public AcsType getAcsType() {
        return acsType;
    }

    public void setAcsType(AcsType acsType) {
        this.acsType = acsType;
    }

    public Integer getTimiScore() {
        return timiScore;
    }

    public void setTimiScore(Integer timiScore) {
        this.timiScore = timiScore;
    }

    public Integer getGraceScore() {
        return graceScore;
    }

    public void setGraceScore(Integer graceScore) {
        this.graceScore = graceScore;
    }

    public Boolean getHaemodynamicInstability() {
        return haemodynamicInstability;
    }

    public void setHaemodynamicInstability(Boolean haemodynamicInstability) {
        this.haemodynamicInstability = haemodynamicInstability;
    }

    public Boolean getElectricalInstability() {
        return electricalInstability;
    }

    public void setElectricalInstability(Boolean electricalInstability) {
        this.electricalInstability = electricalInstability;
    }

    public Boolean getRecurrentIschaemia() {
        return recurrentIschaemia;
    }

    public void setRecurrentIschaemia(Boolean recurrentIschaemia) {
        this.recurrentIschaemia = recurrentIschaemia;
    }

    public Boolean getDynamicStOrTChanges() {
        return dynamicStOrTChanges;
    }

    public void setDynamicStOrTChanges(Boolean dynamicStOrTChanges) {
        this.dynamicStOrTChanges = dynamicStOrTChanges;
    }

    public Boolean getLargeTroponinRise() {
        return largeTroponinRise;
    }

    public void setLargeTroponinRise(Boolean largeTroponinRise) {
        this.largeTroponinRise = largeTroponinRise;
    }

    public Boolean getPrimaryPciFacilityAvailableCloseBy() {
        return primaryPciFacilityAvailableCloseBy;
    }

    public void setPrimaryPciFacilityAvailableCloseBy(Boolean primaryPciFacilityAvailableCloseBy) {
        this.primaryPciFacilityAvailableCloseBy = primaryPciFacilityAvailableCloseBy;
    }

    public Integer getExpectedFmcToBalloonMinutes() {
        return expectedFmcToBalloonMinutes;
    }

    public void setExpectedFmcToBalloonMinutes(Integer expectedFmcToBalloonMinutes) {
        this.expectedFmcToBalloonMinutes = expectedFmcToBalloonMinutes;
    }

    public Boolean getDelayedPresentation() {
        return delayedPresentation;
    }

    public void setDelayedPresentation(Boolean delayedPresentation) {
        this.delayedPresentation = delayedPresentation;
    }

    public Boolean getLvDysfunction() {
        return lvDysfunction;
    }

    public void setLvDysfunction(Boolean lvDysfunction) {
        this.lvDysfunction = lvDysfunction;
    }

    public Boolean getViableMyocardium() {
        return viableMyocardium;
    }

    public void setViableMyocardium(Boolean viableMyocardium) {
        this.viableMyocardium = viableMyocardium;
    }
}
