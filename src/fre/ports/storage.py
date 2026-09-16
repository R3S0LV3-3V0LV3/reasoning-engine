"""Persistence ports."""

from typing import Protocol
from uuid import UUID

from fre.runtime.events import StoredEvent, UncommittedEvent


class EventStore(Protocol):
    def append(
        self, run_id: UUID, expected_version: int, events: tuple[UncommittedEvent, ...]
    ) -> tuple[StoredEvent, ...]: ...

    def load(self, run_id: UUID, after_sequence: int = 0) -> tuple[StoredEvent, ...]: ...
