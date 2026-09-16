"""Clock and UUID ports."""

from datetime import datetime
from typing import Protocol
from uuid import UUID


class Clock(Protocol):
    def now(self) -> datetime: ...


class UUIDFactory(Protocol):
    def new(self) -> UUID: ...
