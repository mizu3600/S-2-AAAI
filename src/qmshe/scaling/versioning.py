from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class ArtifactVersion(BaseModel):
    graph_version: str
    encoder_version: str
    spectral_version: str
    index_version: str
    created_at: str

    @classmethod
    def create(cls, graph: int = 1, encoder: int = 1, spectral: int = 1, index: int = 1):
        return cls(
            graph_version=f"graph-v{graph}", encoder_version=f"text-v{encoder}",
            spectral_version=f"spec-v{spectral}", index_version=f"index-v{index}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def compatible_with(self, other: "ArtifactVersion") -> bool:
        return (
            self.graph_version == other.graph_version
            and self.encoder_version == other.encoder_version
            and self.spectral_version == other.spectral_version
        )


class VersionManifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> ArtifactVersion | None:
        if not self.path.exists():
            return None
        return ArtifactVersion.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, version: ArtifactVersion) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(version.model_dump_json(indent=2), encoding="utf-8")

    def assert_compatible(self, version: ArtifactVersion) -> None:
        current = self.load()
        if current is not None and not current.compatible_with(version):
            raise ValueError(
                f"incompatible artifact versions: current={current.model_dump()} new={version.model_dump()}"
            )
