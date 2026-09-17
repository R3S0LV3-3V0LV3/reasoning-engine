# Security policy

## Supported version

Security corrections are applied to the current `main` branch. This repository has not yet made
a stable package release, so no older release line is currently supported.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting facility for this repository. Do not disclose a
suspected vulnerability in a public issue or pull request before a fix is available.

Include the affected commit, reachable interface, attacker-controlled input, missing or failed
control, impact, and the smallest reproduction you can provide. Do not include real credentials,
private data, or destructive payloads.

## Security-sensitive invariants

- Model output is untrusted until schema, semantic, provenance, and budget validation succeeds.
- Event batches are validated before atomic persistence; replay never invokes a provider.
- Provider invocation cannot restore consumed hard-budget capacity.
- Artifact and provenance references must resolve to registered, hash-verified material.
- Historical event and snapshot meanings are changed only through explicit versioning.
- Pull-request code receives no repository write token and cannot bypass required checks.
