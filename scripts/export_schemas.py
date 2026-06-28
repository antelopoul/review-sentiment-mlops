"""
Export every versioned Pydantic model in app/schema.py to a JSON Schema file.

Output layout:
  schemas/
    index.json                       ← catalogue of all exported schemas
    sentiment_request_v1.json
    sentiment_response_v1.json
    sentiment_batch_request_v1.json
    sentiment_batch_response_v1.json
    sentiment_job_start_v1.json
    sentiment_job_status_v1.json
    sentiment_job_result_v1.json

Usage:
    python scripts/export_schemas.py               # writes to schemas/
    python scripts/export_schemas.py --out ./dist  # custom output dir
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schema import (
    SentimentRequestV1,
    SentimentResponseV1,
    SentimentBatchRequestV1,
    SentimentBatchResponseV1,
    SentimentJobStartV1,
    SentimentJobStatusV1,
    SentimentJobResultV1,
)

# Models to export — (filename_stem, model_class)
MODELS = [
    ("sentiment_request_v1",       SentimentRequestV1),
    ("sentiment_response_v1",      SentimentResponseV1),
    ("sentiment_batch_request_v1", SentimentBatchRequestV1),
    ("sentiment_batch_response_v1",SentimentBatchResponseV1),
    ("sentiment_job_start_v1",     SentimentJobStartV1),
    ("sentiment_job_status_v1",    SentimentJobStatusV1),
    ("sentiment_job_result_v1",    SentimentJobResultV1),
]


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schemas": [],
    }

    for stem, model in MODELS:
        schema = model.model_json_schema()
        # Embed a stable $id so consumers can reference the schema by URI
        schema["$id"] = f"https://example.com/schemas/{stem}.json"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"

        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(schema, indent=2))
        print(f"  wrote {path}")

        index["schemas"].append({
            "id": stem,
            "file": f"{stem}.json",
            "title": schema.get("title", stem),
        })

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"  wrote {index_path}")
    print(f"\n✓ {len(MODELS)} schemas exported to {out_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Pydantic models to JSON Schema")
    parser.add_argument("--out", default="schemas", help="Output directory (default: schemas/)")
    args = parser.parse_args()

    export(Path(args.out))
