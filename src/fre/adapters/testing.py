"""Deterministic clock and UUID fixtures."""

from collections.abc import Iterable
from datetime import datetime
from uuid import UUID


class FakeClock:
    def __init__(self, times: Iterable[datetime]) -> None:
        self._times = iter(times)

    def now(self) -> datetime:
        return next(self._times)


class FakeUUIDFactory:
    def __init__(self, values: Iterable[UUID]) -> None:
        self._values = iter(values)

    def new(self) -> UUID:
        return next(self._values)
