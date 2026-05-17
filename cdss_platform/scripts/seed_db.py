"""
CDSS Platform – Database Seed Script
Seeds initial encounter and patient data for development.
Run: python scripts/seed_db.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.models.database import init_db, AsyncSessionLocal, DBPatientEncounter
from sqlalchemy import select


async def seed():
    print("Initialising database...")
    await init_db()

    async with AsyncSessionLocal() as session:
        # Check if already seeded
        result = await session.execute(
            select(DBPatientEncounter).where(DBPatientEncounter.encounter_id == "ENC-CARD-001")
        )
        existing = result.scalar_one_or_none()

        if existing:
            print("Database already seeded.")
            return

        enc = DBPatientEncounter(
            patient_id="PAT-CARD-001",
            encounter_id="ENC-CARD-001",
            encounter_type="cardiology-consult",
            diagnoses_json=["NSTEMI", "type-2-diabetes", "chronic-kidney-disease"],
        )
        session.add(enc)
        await session.commit()
        print("Seeded: PAT-CARD-001 / ENC-CARD-001")

    print("Database seed complete.")


if __name__ == "__main__":
    asyncio.run(seed())
