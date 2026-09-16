# ADR-002: Event sourcing and replay

Status: Accepted

Append-only events are authoritative. Pure, versioned reducers rebuild state;
snapshots are disposable accelerators whose hashes are verified by replay.

