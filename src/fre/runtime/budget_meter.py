"""Event-derived executable budget meter and advisory burn-rate projection."""

from fre.domain.budget import (
    RESOURCE_NAMES,
    BudgetBurnRate,
    BudgetExceeded,
    BudgetProjection,
    BudgetRemaining,
    BudgetReservation,
    BurnTrend,
    InvalidReservationSettlement,
    RateAvailability,
    ReservationConflict,
    ResourceBurnRate,
    ResourceVector,
)
from fre.domain.common import canonical_hash


def _sum(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    return ResourceVector(
        **{name: getattr(left, name) + getattr(right, name) for name in RESOURCE_NAMES}
    )


def _reserved(projection: BudgetProjection) -> ResourceVector:
    value = ResourceVector()
    for reservation in projection.reservations:
        value = _sum(value, reservation.resources)
    return value


class BudgetMeter:
    def remaining(self, projection: BudgetProjection) -> BudgetRemaining:
        if projection.plan is None:
            raise BudgetExceeded("budget has not been allocated")
        allocated = projection.plan.limits.resources()
        reserved = _reserved(projection)
        resources = ResourceVector(
            **{
                name: getattr(allocated, name)
                - getattr(projection.committed, name)
                - getattr(reserved, name)
                for name in RESOURCE_NAMES
            }
        )
        return BudgetRemaining(
            resources=resources,
            active_concurrent_actions=len(projection.reservations),
            projection_hash=canonical_hash(projection),
        )

    def consume(self, projection: BudgetProjection, usage: ResourceVector) -> BudgetProjection:
        remaining = self.remaining(projection).resources
        if any(getattr(usage, name) > getattr(remaining, name) for name in RESOURCE_NAMES):
            raise BudgetExceeded("consumption exceeds a hard ceiling")
        return projection.model_copy(
            update={
                "committed": _sum(projection.committed, usage),
                "usage_events": (*projection.usage_events, usage),
            }
        )

    def reserve(
        self, projection: BudgetProjection, reservation: BudgetReservation
    ) -> BudgetProjection:
        if any(
            item.reservation_id == reservation.reservation_id for item in projection.reservations
        ):
            raise ReservationConflict("reservation identifier already exists")
        if (
            projection.plan is None
            or len(projection.reservations) >= projection.plan.limits.max_concurrent_actions
        ):
            raise ReservationConflict("concurrency ceiling reached")
        remaining = self.remaining(projection).resources
        if any(
            getattr(reservation.resources, name) > getattr(remaining, name)
            for name in RESOURCE_NAMES
        ):
            raise ReservationConflict("reservation would oversubscribe the budget")
        return projection.model_copy(
            update={
                "reservations": tuple(
                    sorted(
                        (*projection.reservations, reservation),
                        key=lambda item: item.reservation_id,
                    )
                )
            }
        )

    def settle(
        self, projection: BudgetProjection, reservation_id: str, actual: ResourceVector
    ) -> BudgetProjection:
        reservation = next(
            (item for item in projection.reservations if item.reservation_id == reservation_id),
            None,
        )
        if reservation is None:
            raise InvalidReservationSettlement("reservation does not exist")
        if any(
            getattr(actual, name) > getattr(reservation.resources, name) for name in RESOURCE_NAMES
        ):
            raise InvalidReservationSettlement("actual usage exceeds reservation")
        remaining_reservations = tuple(
            item for item in projection.reservations if item.reservation_id != reservation_id
        )
        return projection.model_copy(
            update={
                "reservations": remaining_reservations,
                "committed": _sum(projection.committed, actual),
                "usage_events": (*projection.usage_events, actual),
            }
        )

    def release(self, projection: BudgetProjection, reservation_id: str) -> BudgetProjection:
        if not any(item.reservation_id == reservation_id for item in projection.reservations):
            raise ReservationConflict("reservation does not exist")
        return projection.model_copy(
            update={
                "reservations": tuple(
                    item
                    for item in projection.reservations
                    if item.reservation_id != reservation_id
                )
            }
        )

    def burn_rate(self, projection: BudgetProjection, window_size: int = 3) -> BudgetBurnRate:
        remaining = self.remaining(projection).resources
        events = projection.usage_events
        current_events = events[-window_size:]
        previous_events = events[-2 * window_size : -window_size]
        values: dict[str, ResourceBurnRate] = {}
        if projection.plan is None:
            raise BudgetExceeded("budget has not been allocated")
        allocated = projection.plan.limits.resources()
        for name in RESOURCE_NAMES:
            current_total = sum(getattr(item, name) for item in current_events)
            previous_total = sum(getattr(item, name) for item in previous_events)
            current_rate = current_total / len(current_events) if current_events else None
            previous_rate = previous_total / len(previous_events) if previous_events else None
            if len(current_events) < 2:
                availability = RateAvailability.INSUFFICIENT_DATA
                trend = BurnTrend.INSUFFICIENT_DATA
            elif current_rate == 0:
                availability = RateAvailability.ZERO_RATE
                trend = BurnTrend.STABLE
            else:
                availability = RateAvailability.AVAILABLE
                if previous_rate is None:
                    trend = BurnTrend.INSUFFICIENT_DATA
                elif current_rate > previous_rate:
                    trend = BurnTrend.INCREASING
                elif current_rate < previous_rate:
                    trend = BurnTrend.DECREASING
                else:
                    trend = BurnTrend.STABLE
            values[name] = ResourceBurnRate(
                availability=availability,
                committed=getattr(projection.committed, name),
                remaining=getattr(remaining, name),
                consumption_per_usage_event=current_rate,
                previous_window_rate=previous_rate,
                trend=trend,
                projected_iterations_to_exhaustion=(
                    getattr(remaining, name) / current_rate
                    if current_rate and current_rate > 0
                    else None
                ),
                ceiling_proximity=(
                    getattr(projection.committed, name) / getattr(allocated, name)
                    if getattr(allocated, name)
                    else 0.0
                ),
            )
        source_hash = canonical_hash(events)
        preimage = {
            "version": "burn-rate/1.0",
            "window_size": window_size,
            "resources": values,
            "source_event_hash": source_hash,
        }
        return BudgetBurnRate(
            version="burn-rate/1.0",
            window_size=window_size,
            resources=values,
            source_event_hash=source_hash,
            projection_hash=canonical_hash(preimage),
        )
