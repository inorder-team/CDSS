package com.inorder.clinical.acs.model;

public class Recommendation {
    private RecommendationType type;
    private String code;
    private String message;
    private String rationale;
    private String urgency;
    private String priority = "HIGH";

    public Recommendation() {
    }

    public Recommendation(RecommendationType type, String code, String message, String rationale, String urgency) {
        this.type = type;
        this.code = code;
        this.message = message;
        this.rationale = rationale;
        this.urgency = urgency;
        this.priority = inferPriority(type, urgency);
    }

    private String inferPriority(RecommendationType type, String urgency) {
        if (type == RecommendationType.PRIMARY_PCI || type == RecommendationType.FIBRINOLYSIS_PHARMACOINVASIVE_PCI) {
            return "CRITICAL";
        }
        String normalizedUrgency = urgency == null ? "" : urgency.toLowerCase();
        if (normalizedUrgency.contains("24") || normalizedUrgency.contains("unstable")) {
            return "URGENT";
        }
        if (type == RecommendationType.DELAYED_ELECTIVE_CAG || type == RecommendationType.MEDICAL_THERAPY) {
            return "MEDIUM";
        }
        return "HIGH";
    }

    public RecommendationType getType() {
        return type;
    }

    public void setType(RecommendationType type) {
        this.type = type;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getRationale() {
        return rationale;
    }

    public void setRationale(String rationale) {
        this.rationale = rationale;
    }

    public String getUrgency() {
        return urgency;
    }

    public void setUrgency(String urgency) {
        this.urgency = urgency;
    }

    public String getPriority() {
        return priority;
    }

    public void setPriority(String priority) {
        this.priority = priority;
    }
}
