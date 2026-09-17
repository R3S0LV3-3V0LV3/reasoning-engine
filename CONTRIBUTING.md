# Contributing

The Frontier Reasoning Engine is an event-sourced control plane. Changes to events, reducers,
budgets, provenance, artifacts, or replay are contract changes even when a public function
signature is unchanged.

## Required workflow

1. Create a short-lived branch from the current target branch.
2. Add a focused regression for the invariant being changed.
3. Use versioned events and schemas for material serialized-semantic changes. Existing event
   bytes, fixture hashes, and decoder behaviour must not be silently redefined.
4. Run `make quality` and `uv build` before opening a pull request.
5. Complete the pull-request compatibility, threat-model, and verification sections.
6. Merge only through a pull request after every required check passes on the current head.

Use conventional commits. Do not push directly to protected branches, bypass hooks or checks,
force-push shared branches, commit credentials, or weaken a test or invariant merely to obtain a
green build. Revert published mistakes with a checked pull request rather than rewriting history.

## Definition of done

A source change is not complete because the current tests pass. Its named behaviour must have
positive and negative evidence, replay compatibility must be explicit, and documentation must
distinguish implementation from validation and independent approval.
