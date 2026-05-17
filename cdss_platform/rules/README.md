# CDSS Clinical Rules Engine

This folder houses all deterministic rule definitions used by the CDSS platform.
Rules are executed via the **python-drools-sdk** against a KIE Server instance.

## Folder Layout

```
rules/
├── drools-rules/          # Drools Rule Language (.drl) files executed by KIE Server
│   ├── acs_triage.drl         ACS Triage pathway (STEMI / NSTEMI / UA risk stratification)
│   ├── cag_decision.drl       Post-CAG revascularisation decision tree (PCI / CABG / Medical)
│   └── post_mi_viability.drl  Delayed STEMI & post-MI LV dysfunction viability pathway
│
├── arden-rules/           # Arden Syntax MLM files (future – local rule authoring)
│
└── regex-rules/           # Pattern-based text extraction rules (medication / allergy parsing)
```

## Rule Execution Flow

```
POST /clinical-decision-support/acs/recommendations
        │
        ▼
  ACS Rules Engine (app/services/rules/drools_client.py)
        │
        ├── Sends payload to KIE Server  POST /kie/services/rest/server/containers/{containerId}/dmn
        │   (or stateless rules:        POST /kie/services/rest/server/containers/{containerId}/ksession)
        │
        ├── KIE Server fires matching .drl rules
        │
        └── Returns structured Recommendation objects
                │
                ▼
          NLP formatter → human-readable clinical text → Streamlit UI
```

## Supported ACS Types

| acsType         | Primary DRL            | Typical Outcome                              |
|-----------------|------------------------|----------------------------------------------|
| `STEMI`         | acs_triage.drl         | Primary PCI or Pharmacoinvasive strategy     |
| `NSTEMI`        | acs_triage.drl         | Early invasive or Medical optimisation       |
| `Diagnostic CAG`| cag_decision.drl       | PCI / CABG / FFR / Medical therapy           |

## Running KIE Server via Docker

```bash
docker-compose up kie_server -d
```

The `kie_server` service in docker-compose.yml mounts this `rules/` folder and
auto-loads the ACS container on startup.
