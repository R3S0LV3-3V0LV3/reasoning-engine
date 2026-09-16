"""Deterministic SourceAnchor resolution against permitted module inputs."""

import hashlib

from fre.domain.common import ArtifactRef, ObjectRef
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import TaskEnvelope


class InvalidSourceAnchor(ValueError):
    pass


def validate_source_anchor(
    anchor: SourceAnchor,
    envelope: TaskEnvelope,
    available_artifacts: frozenset[str] = frozenset(),
) -> None:
    if anchor.source_kind is SourceKind.ARTIFACT:
        if (
            not isinstance(anchor.source_ref, ArtifactRef)
            or anchor.source_ref.sha256 not in available_artifacts
        ):
            raise InvalidSourceAnchor("source artifact is not available")
        return
    if not isinstance(anchor.source_ref, ObjectRef) or anchor.source_ref.object_id != str(
        envelope.task_id
    ):
        raise InvalidSourceAnchor("anchor does not reference this TaskEnvelope")
    if anchor.source_kind is SourceKind.TASK_TEXT:
        if anchor.selector != "/text" or anchor.char_start is None or anchor.char_end is None:
            raise InvalidSourceAnchor("task text requires /text and a span")
        if anchor.char_end > len(envelope.text):
            raise InvalidSourceAnchor("task text bounds are invalid")
        excerpt = envelope.text[anchor.char_start : anchor.char_end]
        if (
            anchor.excerpt_hash
            and hashlib.sha256(excerpt.encode()).hexdigest() != anchor.excerpt_hash
        ):
            raise InvalidSourceAnchor("source excerpt hash mismatch")
        return
    allowed = {
        "/explicit_constraints",
        "/requested_output",
        "/execution_permissions",
        "/user_metadata",
    }
    root = "/" + anchor.selector.strip("/").split("/")[0]
    if root not in allowed:
        raise InvalidSourceAnchor("structured source selector is not permitted")
