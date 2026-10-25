"""Load FHIR transaction bundles into a FHIR server (e.g. the HAPI container).

Works with the bundled synthetic patients and with Synthea output
(``./run_synthea -p 100`` writes bundles to ``output/fhir``).

Usage:
    python scripts/load_fhir_bundles.py --server http://localhost:8080/fhir data/fhir
    python scripts/load_fhir_bundles.py --server http://localhost:8080/fhir ~/synthea/output/fhir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="a bundle .json file or a directory of them")
    parser.add_argument("--server", default="http://localhost:8080/fhir")
    args = parser.parse_args()

    files = sorted(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
    # Synthea writes practitioner/hospital bundles that other bundles reference; load first.
    files.sort(
        key=lambda f: not f.name.startswith(("hospitalInformation", "practitionerInformation"))
    )
    with httpx.Client(timeout=120, headers={"Content-Type": "application/fhir+json"}) as client:
        for f in files:
            bundle = json.loads(f.read_text(encoding="utf-8"))
            resp = client.post(args.server, json=bundle)
            status = "ok" if resp.is_success else f"FAILED {resp.status_code}"
            print(f"{f.name}: {len(bundle.get('entry', []))} resources {status}")


if __name__ == "__main__":
    main()
