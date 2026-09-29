from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 42
NUMBER_OF_SESSIONS = 1200

rng = np.random.default_rng(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "mock"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# POSSIBLE VALUES
# =========================================================

DPPS = [
    "gmh-gruppe-sn-00000006-c0a7c1a5",
    "mannstaedt-demo-002",
    "mannstaedt-demo-003",
]

SECTIONS = [
    "Overview",
    "Product Information",
    "Materials",
    "Sustainability",
    "Traceability",
    "Documents",
]

DOCUMENTS = [
    "Material Certificate.pdf",
    "Declaration of Conformity.pdf",
    "Technical Datasheet.pdf",
    "Sustainability Certificate.pdf",
]

DEVICES = [
    "desktop",
    "mobile",
    "tablet",
]

REGIONS = [
    "DE-NW",
    "DE-BY",
    "DE-BE",
    "NL-NH",
    "BE-VLG",
]

SUPPLIERS = [
    "Supplier-A",
    "Supplier-B",
    "Supplier-C",
    "Supplier-D",
    "Supplier-E",
]

MATERIALS = [
    "Steel",
    "Carbon",
    "Manganese",
    "Chromium",
    "Nickel",
]


# =========================================================
# CREATE SESSION-LEVEL ANALYTICS DATA
# =========================================================

session_rows = []

start_date = datetime(2026, 5, 1)

for session_number in range(NUMBER_OF_SESSIONS):

    session_id = f"session-{session_number + 1:05d}"

    event_date = start_date + timedelta(
        days=int(rng.integers(0, 130))
    )

    source = rng.choice(
        ["qr", "web"],
        p=[0.62, 0.38],
    )

    device = rng.choice(
        DEVICES,
        p=[0.48, 0.45, 0.07],
    )

    region = rng.choice(REGIONS)

    dpp_id = rng.choice(
        DPPS,
        p=[0.70, 0.15, 0.15],
    )

    # QR visitors are slightly more engaged in the synthetic pattern.
    if source == "qr":
        visit_duration = int(
            max(
                5,
                rng.normal(
                    loc=110,
                    scale=45,
                ),
            )
        )
    else:
        visit_duration = int(
            max(
                5,
                rng.normal(
                    loc=75,
                    scale=38,
                ),
            )
        )

    sections_viewed = int(
        np.clip(
            rng.poisson(
                3.2 if source == "qr" else 2.3
            ),
            1,
            len(SECTIONS),
        )
    )

    # Longer and deeper sessions have higher document-open probability.
    document_probability = min(
        0.85,
        0.04
        + (sections_viewed * 0.07)
        + (visit_duration / 800),
    )

    opened_document = int(
        rng.random() < document_probability
    )

    document_count = (
        int(rng.integers(1, 4))
        if opened_document
        else 0
    )

    session_rows.append(
        {
            "date": event_date.date().isoformat(),
            "session_id": session_id,
            "dpp_id": dpp_id,
            "source": source,
            "qr_entry": 1 if source == "qr" else 0,
            "device": device,
            "region": region,
            "visit_duration": visit_duration,
            "sections_viewed": sections_viewed,
            "documents_opened": document_count,
            "opened_document": opened_document,
            "data_origin": "synthetic",
        }
    )


session_df = pd.DataFrame(session_rows)


# =========================================================
# CREATE EVENT-LEVEL DATA
# =========================================================

event_rows = []

for row in session_rows:

    section_count = row["sections_viewed"]

    selected_sections = list(
        rng.choice(
            SECTIONS,
            size=section_count,
            replace=False,
        )
    )

    # Initial tab should always exist.
    if "Overview" not in selected_sections:
        selected_sections[0] = "Overview"

    for event_index, section_name in enumerate(
        selected_sections
    ):

        event_rows.append(
            {
                "date": row["date"],
                "session_id": row["session_id"],
                "dpp_id": row["dpp_id"],
                "source": row["source"],
                "event_order": event_index + 1,
                "event_name": "dpp-section-view",
                "section_name": section_name,
                "document_name": None,
                "device": row["device"],
                "region": row["region"],
                "data_origin": "synthetic",
            }
        )

    if row["opened_document"]:

        for document_index in range(
            row["documents_opened"]
        ):

            event_rows.append(
                {
                    "date": row["date"],
                    "session_id": row["session_id"],
                    "dpp_id": row["dpp_id"],
                    "source": row["source"],
                    "event_order":
                        section_count
                        + document_index
                        + 1,
                    "event_name":
                        "dpp-document-open",
                    "section_name":
                        "Documents",
                    "document_name":
                        rng.choice(DOCUMENTS),
                    "device": row["device"],
                    "region": row["region"],
                    "data_origin":
                        "synthetic",
                }
            )


event_df = pd.DataFrame(event_rows)


# =========================================================
# CREATE TRACEABILITY MOCK DATA
# =========================================================

traceability_rows = []

for i in range(500):

    product_id = (
        f"MANN-{int(rng.integers(1, 80)):04d}"
    )

    supplier = rng.choice(SUPPLIERS)

    material = rng.choice(MATERIALS)

    credential_status = rng.choice(
        [
            "valid",
            "valid",
            "valid",
            "missing",
            "expired",
        ]
    )

    missing_link = int(
        rng.random() < 0.12
    )

    base_risk = float(
        rng.uniform(0.05, 0.45)
    )

    if credential_status == "missing":
        base_risk += 0.30

    if credential_status == "expired":
        base_risk += 0.20

    if missing_link:
        base_risk += 0.25

    risk_score = round(
        min(base_risk, 1.0),
        3,
    )

    traceability_rows.append(
        {
            "product_id": product_id,
            "batch_id":
                f"B-{int(rng.integers(1000, 9999))}",
            "supplier_id": supplier,
            "material": material,
            "origin_country": rng.choice(
                [
                    "Germany",
                    "Poland",
                    "Netherlands",
                    "Sweden",
                    "France",
                ]
            ),
            "credential_status":
                credential_status,
            "missing_link":
                missing_link,
            "risk_score":
                risk_score,
            "data_origin":
                "synthetic",
        }
    )


traceability_df = pd.DataFrame(
    traceability_rows
)


# =========================================================
# SAVE
# =========================================================

session_path = (
    OUTPUT_FOLDER
    / "mannstaedt_mock_sessions.csv"
)

events_path = (
    OUTPUT_FOLDER
    / "mannstaedt_mock_events.csv"
)

traceability_path = (
    OUTPUT_FOLDER
    / "mannstaedt_mock_traceability.csv"
)


session_df.to_csv(
    session_path,
    index=False,
)

event_df.to_csv(
    events_path,
    index=False,
)

traceability_df.to_csv(
    traceability_path,
    index=False,
)


print(
    f"Created {len(session_df)} synthetic sessions."
)

print(
    f"Created {len(event_df)} synthetic events."
)

print(
    f"Created {len(traceability_df)} synthetic traceability rows."
)

print(session_path)
print(events_path)
print(traceability_path)