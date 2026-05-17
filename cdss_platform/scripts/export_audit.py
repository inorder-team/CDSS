"""
CDSS Platform – Audit Log Export Script
Exports audit log to CSV for compliance reporting.
Run: python scripts/export_audit.py [--output report.csv]
"""
import csv
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from app.core.config import get_settings
settings = get_settings()


def export_audit(output_path: str = "audit_export.csv"):
    audit_path = Path(settings.audit_log_path)
    if not audit_path.exists():
        print(f"No audit log found at {audit_path}")
        return

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("Audit log is empty.")
        return

    # Collect all unique keys
    records = []
    all_keys: set[str] = set()
    for line in lines:
        try:
            entry = json.loads(line)
            entry.pop("_hash", None)  # exclude hash from export
            records.append(entry)
            all_keys.update(entry.keys())
        except json.JSONDecodeError:
            continue

    fieldnames = sorted(all_keys)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"Exported {len(records)} audit entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CDSS audit log to CSV")
    parser.add_argument("--output", default="audit_export.csv", help="Output CSV path")
    args = parser.parse_args()
    export_audit(args.output)
