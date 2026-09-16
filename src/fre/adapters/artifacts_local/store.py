"""Local SHA-256-addressed artifact storage."""

import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import UUID, uuid5

from fre.domain.common import ArtifactDescriptor, JsonValue, canonical_json, utc_now

ARTIFACT_NAMESPACE = UUID("3bd854a9-ebc8-4f2f-a28c-f0bc76f56852")


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects"
        self.metadata = root / "metadata"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.metadata.mkdir(parents=True, exist_ok=True)

    def put(
        self, content: bytes, *, media_type: str, metadata: dict[str, JsonValue] | None = None
    ) -> ArtifactDescriptor:
        digest = hashlib.sha256(content).hexdigest()
        path = self.objects / digest[:2] / digest[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        artifact_id = uuid5(ARTIFACT_NAMESPACE, digest)
        descriptor = ArtifactDescriptor(
            id=artifact_id,
            created_at=utc_now(),
            created_by_action=artifact_id,
            created_by_module="artifact-store",
            sha256=digest,
            media_type=media_type,
            byte_size=len(content),
            path=str(path.relative_to(self.root)),
            metadata=metadata or {},
        )
        metadata_path = self.metadata / f"{digest}.json"
        if not metadata_path.exists():
            metadata_path.write_bytes(canonical_json(descriptor))
        return descriptor

    def get(self, sha256: str) -> bytes:
        return (self.objects / sha256[:2] / sha256[2:]).read_bytes()

    def object_count(self) -> int:
        return sum(1 for path in self.objects.glob("*/*") if path.is_file())

    def descriptor(self, sha256: str) -> dict[str, object]:
        return cast(dict[str, object], json.loads((self.metadata / f"{sha256}.json").read_text()))
