"""Deterministic SourceAnchor resolution against permitted module inputs."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from fre.domain.common import ArtifactRef, ObjectRef
from fre.domain.semantic import SourceAnchor, SourceKind
from fre.domain.task import TaskEnvelope


class InvalidSourceAnchor(ValueError):
    """A claimed source does not resolve to the permitted canonical input."""


_TASK_FIELD_ROOTS = frozenset(
    {"explicit_constraints", "requested_output", "execution_permissions", "user_metadata"}
)


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise InvalidSourceAnchor("selector is not an RFC 6901 JSON pointer")
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        token = ""
        index = 0
        while index < len(raw):
            if raw[index] != "~":
                token += raw[index]
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise InvalidSourceAnchor("selector contains malformed RFC 6901 escaping")
            token += "~" if raw[index + 1] == "0" else "/"
            index += 2
        tokens.append(token)
    return tuple(tokens)


def _resolve_pointer(document: Any, pointer: str) -> Any:
    value = document
    for token in _pointer_tokens(pointer):
        if isinstance(value, Mapping):
            if token not in value:
                raise InvalidSourceAnchor("selector mapping key does not exist")
            value = value[token]
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            if token == "-" or not token.isascii() or not token.isdecimal():
                raise InvalidSourceAnchor("selector contains a malformed sequence index")
            if token != "0" and token.startswith("0"):
                raise InvalidSourceAnchor("selector contains a malformed sequence index")
            item_index = int(token)
            if item_index >= len(value):
                raise InvalidSourceAnchor("selector sequence index is out of bounds")
            value = value[item_index]
        else:
            raise InvalidSourceAnchor("selector traverses beyond a scalar value")
    return value


def _validate_resolved_value(anchor: SourceAnchor, value: Any) -> None:
    has_span = anchor.char_start is not None and anchor.char_end is not None
    if not isinstance(value, str):
        if has_span or anchor.excerpt_hash is not None:
            raise InvalidSourceAnchor("non-text sources cannot have spans or excerpt hashes")
        return
    if not has_span:
        if anchor.excerpt_hash is not None:
            raise InvalidSourceAnchor("an excerpt hash requires a text span")
        return
    assert anchor.char_start is not None and anchor.char_end is not None
    if anchor.char_end > len(value):
        raise InvalidSourceAnchor("source text bounds are invalid")
    excerpt = value[anchor.char_start : anchor.char_end]
    if anchor.excerpt_hash and hashlib.sha256(excerpt.encode()).hexdigest() != anchor.excerpt_hash:
        raise InvalidSourceAnchor("source excerpt hash mismatch")


def validate_source_anchor(
    anchor: SourceAnchor,
    envelope: TaskEnvelope,
    available_artifacts: frozenset[str] = frozenset(),
) -> None:
    """Resolve and validate an anchor against an envelope's canonical JSON data."""
    if anchor.source_kind is SourceKind.ARTIFACT:
        if (
            not isinstance(anchor.source_ref, ArtifactRef)
            or anchor.source_ref.sha256 not in available_artifacts
        ):
            raise InvalidSourceAnchor("source artifact is not available")
        if anchor.selector or anchor.char_start is not None or anchor.excerpt_hash is not None:
            raise InvalidSourceAnchor("opaque artifacts cannot have selectors, spans, or excerpts")
        return

    if not isinstance(anchor.source_ref, ObjectRef):
        raise InvalidSourceAnchor("task source requires an ObjectRef")
    if (
        anchor.source_ref.object_type != "TaskEnvelope"
        or anchor.source_ref.object_id != str(envelope.task_id)
        or anchor.source_ref.revision is not None
    ):
        raise InvalidSourceAnchor("anchor does not reference this TaskEnvelope")

    tokens = _pointer_tokens(anchor.selector)
    if anchor.source_kind is SourceKind.TASK_TEXT:
        if tokens != ("text",) or anchor.char_start is None:
            raise InvalidSourceAnchor("task text requires /text and a span")
    elif not tokens or tokens[0] not in _TASK_FIELD_ROOTS:
        raise InvalidSourceAnchor("structured source selector is not permitted")

    value = _resolve_pointer(envelope.model_dump(mode="json"), anchor.selector)
    _validate_resolved_value(anchor, value)
