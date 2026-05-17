"""
CDSS Platform – CLI Management Script
Run with: python scripts/manage.py [command]
Commands:
  ingest     – Ingest all guidelines into ChromaDB
  test-rag   – Test RAG retrieval
  test-pipeline – Run full pipeline with sample NSTEMI request (requires ANTHROPIC_API_KEY)
  audit      – Display recent audit log entries
"""
import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

console = Console()


def cmd_ingest():
    """Ingest guidelines into ChromaDB."""
    console.rule("[bold cyan]CDSS RAG Ingestion[/]")
    from app.rag.rag_engine import rag_engine
    guideline_dir = Path("./data/guidelines")
    if not guideline_dir.exists():
        console.print(f"[red]Error: {guideline_dir} does not exist[/]")
        return
    console.print(f"[yellow]Ingesting from: {guideline_dir.absolute()}[/]")
    count = rag_engine.ingest_guidelines_directory(guideline_dir)
    console.print(f"[green]✓ Ingested {count} chunks into ChromaDB[/]")
    console.print(f"[dim]Collection: {rag_engine.collection_count()} total chunks[/]")


def cmd_test_rag():
    """Test RAG retrieval."""
    console.rule("[bold cyan]CDSS RAG Retrieval Test[/]")
    from app.rag.rag_engine import rag_engine

    queries = [
        "NSTEMI management aspirin allergy alternative antiplatelet",
        "eGFR renal impairment antithrombotic dose adjustment CKD",
        "high risk NSTEMI coronary angiography invasive strategy",
        "human review cardiologist AI decision support",
    ]

    for query in queries:
        console.print(f"\n[cyan]Query:[/] {query}")
        docs = rag_engine.retrieve(query, top_k=3)
        if docs:
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Doc ID", style="cyan", width=30)
            table.add_column("Score", width=8)
            table.add_column("Tag", width=20)
            table.add_column("Snippet", width=60)
            for d in docs:
                table.add_row(
                    d.doc_id,
                    f"{d.similarity_score:.3f}",
                    d.relevance_tag or "-",
                    d.content_snippet[:80] + "...",
                )
            console.print(table)
        else:
            console.print("[yellow]No results (RAG not ingested yet – run 'ingest' first)[/]")


def cmd_test_pipeline():
    """Run a full pipeline test with the sample NSTEMI patient."""
    console.rule("[bold cyan]CDSS Full Pipeline Test – NSTEMI[/]")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        console.print("[red]Error: ANTHROPIC_API_KEY not set in .env file[/]")
        return

    from app.models.schemas import CDSSRecommendationRequest
    from app.services.pipeline import pipeline

    payload = {
        "patientId": "PAT-CARD-001",
        "encounterId": "ENC-CARD-001",
        "userId": "cardiologist.local",
        "userRole": "CARDIOLOGIST",
        "query": "Recommend guideline-based considerations for NSTEMI management for this patient.",
        "consentVerified": True,
        "patientContext": {
            "age": 68,
            "sex": "male",
            "encounterType": "cardiology-consult",
            "diagnoses": ["NSTEMI", "type-2-diabetes", "chronic-kidney-disease"],
            "ecgFindings": ["ST depression in lateral leads"],
            "labs": {"troponin": "elevated and rising", "eGFR": "42", "potassium": "4.8"},
            "vitals": {"systolicBp": "138", "heartRate": "92"},
            "currentMedications": ["atorvastatin", "metoprolol"],
            "allergies": ["aspirin"],
            "contraindications": ["documented aspirin allergy"],
            "cardiacHistory": ["prior PCI"],
        },
    }

    import uuid
    correlation_id = str(uuid.uuid4())
    console.print(f"[dim]Correlation ID: {correlation_id}[/]\n")

    try:
        request = CDSSRecommendationRequest(**payload)
        response = pipeline.run(request=request, correlation_id=correlation_id)

        rec = response.recommendation

        console.print(Panel(
            f"[bold]Status:[/] {rec.decision_status.value}\n"
            f"[bold]Confidence:[/] {rec.confidence_score:.2%}\n"
            f"[bold]RAG Driven:[/] {rec.rag_driven}\n"
            f"[bold]AI Path:[/] {rec.ai_path_used.value}\n"
            f"[bold]Evidence Docs:[/] {len(rec.evidence_documents)}\n"
            f"[bold]Human Review Required:[/] {rec.requires_human_review}\n"
            f"[bold]Pipeline Latency:[/] {response.pipeline_latency_ms:.0f}ms",
            title="[bold cyan]Pipeline Result[/]",
            border_style="cyan"
        ))

        console.print("\n[bold cyan]── Summary ──[/]")
        console.print(rec.summary)

        console.print("\n[bold cyan]── Antiplatelet Guidance ──[/]")
        console.print(rec.antiplatelet_guidance)

        if rec.safety_flags.flags:
            console.print("\n[bold red]── Safety Flags ──[/]")
            for flag in rec.safety_flags.flags:
                console.print(f"  ⚑ {flag}")

        console.print("\n[bold cyan]── Evidence Retrieved ──[/]")
        for i, doc in enumerate(rec.evidence_documents[:3], 1):
            console.print(f"  [{i}] {doc.doc_id} | score={doc.similarity_score:.3f} | {doc.relevance_tag}")

    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/]")
        import traceback
        traceback.print_exc()


def cmd_audit(n: int = 20):
    """Display recent audit log entries."""
    console.rule("[bold cyan]CDSS Audit Log[/]")
    audit_path = Path("./logs/audit.jsonl")
    if not audit_path.exists():
        console.print("[yellow]No audit log found – run pipeline first[/]")
        return

    lines = audit_path.read_text().strip().splitlines()[-n:]
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", style="dim", width=24)
    table.add_column("Event", width=22)
    table.add_column("Correlation ID", width=12)
    table.add_column("Details", width=60)

    for line in lines:
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            event = entry.get("event", "")
            corr = str(entry.get("correlation_id", ""))[:8]
            details = ""
            if event == "cdss_request":
                details = f"user={entry.get('user_id')} patient={entry.get('patient_id')}"
            elif event == "clinical_decision":
                details = f"status={entry.get('decision_status')} conf={entry.get('confidence_score', 0):.2f}"
            elif event == "safety_gate_evaluation":
                details = f"passed={entry.get('safety_passed')} flags={len(entry.get('flags_raised', []))}"
            elif event == "rag_retrieval":
                details = f"docs={entry.get('documents_retrieved')} collection={entry.get('collection')}"
            elif event == "llm_inference":
                details = f"model={entry.get('model')} tokens={entry.get('total_tokens')} latency={entry.get('latency_ms', 0):.0f}ms"
            elif event == "human_review":
                details = f"action={entry.get('action')} reviewer={entry.get('reviewer_id')}"
            else:
                details = str(entry)[:60]
            table.add_row(ts, event, corr, details)
        except Exception:
            pass

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="CDSS Platform CLI")
    parser.add_argument("command", choices=["ingest", "test-rag", "test-pipeline", "audit"])
    parser.add_argument("--n", type=int, default=20, help="Number of audit entries (audit command)")
    args = parser.parse_args()

    # Load .env
    from dotenv import load_dotenv
    load_dotenv()

    # Ingest on startup for test commands
    if args.command in ("test-rag", "test-pipeline"):
        from app.rag.rag_engine import rag_engine
        if rag_engine.collection_count() == 0:
            console.print("[yellow]RAG collection empty – ingesting guidelines first...[/]")
            rag_engine.ingest_guidelines_directory(Path("./data/guidelines"))

    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "test-rag":
        cmd_test_rag()
    elif args.command == "test-pipeline":
        cmd_test_pipeline()
    elif args.command == "audit":
        cmd_audit(args.n)


if __name__ == "__main__":
    main()
