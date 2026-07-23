import httpx
import numpy as np

from qmshe.settings import Settings, get_settings


class QdrantMultiVectorStore:
    def __init__(self, collection: str = "qmshe_objects", settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.collection = collection

    def create_collection(self, dimensions: dict[str, int], recreate: bool = False) -> None:
        with httpx.Client(timeout=30, trust_env=False) as client:
            if recreate:
                client.delete(f"{self.settings.qdrant_url}/collections/{self.collection}")
            payload = {
                "vectors": {
                    name: {"size": size, "distance": "Cosine"}
                    for name, size in dimensions.items()
                }
            }
            response = client.put(
                f"{self.settings.qdrant_url}/collections/{self.collection}", json=payload
            )
        response.raise_for_status()

    def upsert(
        self, object_ids: list[str], vectors: dict[str, np.ndarray], payloads: list[dict]
    ) -> None:
        points = []
        for index, object_id in enumerate(object_ids):
            points.append(
                {
                    "id": _stable_uint64(object_id),
                    "vector": {name: matrix[index].tolist() for name, matrix in vectors.items()},
                    "payload": {"object_id": object_id, **payloads[index]},
                }
            )
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.put(
                f"{self.settings.qdrant_url}/collections/{self.collection}/points?wait=true",
                json={"points": points},
            )
        response.raise_for_status()

    def search(self, vector_name: str, vector: np.ndarray, top_k: int, filters: dict | None = None) -> list[dict]:
        payload = {"query": vector.tolist(), "using": vector_name, "limit": top_k, "with_payload": True}
        if filters:
            payload["filter"] = filters
        with httpx.Client(timeout=30, trust_env=False) as client:
            response = client.post(
                f"{self.settings.qdrant_url}/collections/{self.collection}/points/query", json=payload
            )
        response.raise_for_status()
        return response.json()["result"]["points"]


def _stable_uint64(value: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")
