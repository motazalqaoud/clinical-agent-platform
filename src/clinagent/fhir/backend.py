"""FHIR R4 data access.

``FhirBackend`` is the only thing the rest of the service knows about. Two
implementations:

* ``HttpFhirBackend`` talks to a real FHIR server (HAPI FHIR in docker-compose, or a
  hospital's endpoint) using standard REST search.
* ``InMemoryFhirBackend`` loads a transaction bundle from disk and implements the
  handful of search parameters the tools use. It keeps tests and the CI evaluation
  offline, and is deliberately strict: unknown search parameters raise instead of
  being silently ignored, so the two backends cannot quietly drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import httpx

Resource = dict[str, Any]


class FhirBackend(Protocol):
    def read(self, resource_type: str, resource_id: str) -> Resource | None: ...

    def search(self, resource_type: str, params: dict[str, str]) -> list[Resource]: ...


_SUPPORTED_PARAMS = {"patient", "status", "code", "_sort", "_count", "clinical-status"}


def _ref_matches(resource: Resource, patient_id: str) -> bool:
    for key in ("subject", "patient"):
        ref = resource.get(key, {}).get("reference", "")
        if ref in (f"Patient/{patient_id}", patient_id):
            return True
    return False


def _codes(resource: Resource) -> set[str]:
    return {c.get("code", "") for c in resource.get("code", {}).get("coding", [])}


def _date(resource: Resource) -> str:
    for key in ("effectiveDateTime", "authoredOn", "onsetDateTime", "recordedDate"):
        if key in resource:
            return str(resource[key])
    return ""


class InMemoryFhirBackend:
    def __init__(self, resources: list[Resource]):
        self._by_type: dict[str, dict[str, Resource]] = {}
        for res in resources:
            self._by_type.setdefault(res["resourceType"], {})[res["id"]] = res

    @classmethod
    def from_bundle(cls, path: str | Path) -> InMemoryFhirBackend:
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([e["resource"] for e in bundle.get("entry", [])])

    def read(self, resource_type: str, resource_id: str) -> Resource | None:
        return self._by_type.get(resource_type, {}).get(resource_id)

    def search(self, resource_type: str, params: dict[str, str]) -> list[Resource]:
        unknown = set(params) - _SUPPORTED_PARAMS
        if unknown:
            raise ValueError(f"Unsupported search parameter(s): {sorted(unknown)}")
        out = list(self._by_type.get(resource_type, {}).values())
        if "patient" in params:
            out = [r for r in out if _ref_matches(r, params["patient"])]
        if "status" in params:
            out = [r for r in out if r.get("status") == params["status"]]
        if "clinical-status" in params:
            want = params["clinical-status"]
            out = [
                r
                for r in out
                if any(c.get("code") == want for c in r.get("clinicalStatus", {}).get("coding", []))
            ]
        if "code" in params:
            wanted = {c.split("|")[-1] for c in params["code"].split(",")}
            out = [r for r in out if _codes(r) & wanted]
        if params.get("_sort", "").lstrip("-") == "date":
            out.sort(key=_date, reverse=params["_sort"].startswith("-"))
        if "_count" in params:
            out = out[: int(params["_count"])]
        return out


class HttpFhirBackend:
    """Minimal FHIR REST client (read + search, follows no paging beyond page one)."""

    def __init__(self, base_url: str, timeout: float = 10.0, token: str | None = None):
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)

    def read(self, resource_type: str, resource_id: str) -> Resource | None:
        resp = self._client.get(f"/{resource_type}/{resource_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data: Resource = resp.json()
        return data

    def search(self, resource_type: str, params: dict[str, str]) -> list[Resource]:
        # HAPI sorts Observation by "date", MedicationRequest by "authoredon".
        query = dict(params)
        if resource_type == "MedicationRequest" and query.get("_sort", "").lstrip("-") == "date":
            query["_sort"] = query["_sort"].replace("date", "authoredon")
        resp = self._client.get(f"/{resource_type}", params=query)
        resp.raise_for_status()
        return [e["resource"] for e in resp.json().get("entry", [])]


def make_backend(url_or_path: str) -> FhirBackend:
    if url_or_path.startswith(("http://", "https://")):
        return HttpFhirBackend(url_or_path)
    return InMemoryFhirBackend.from_bundle(url_or_path)
