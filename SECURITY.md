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

## GitHub Actions trust boundary

Same-repository branch access is a privileged security-administrator capability, not a normal
contributor capability. Do not grant repository write, maintain, or administrator access to an
untrusted contributor. External contributions must use forks.

The repository requires maintainer approval before any external fork pull-request workflow runs.
GitHub supplies those fork runs with a read-only token, while repository workflow permissions
default to read-only and cannot approve pull requests. Repository-controlled workflows add a
second layer: they declare explicit read-only permissions, disable persisted checkout credentials,
prohibit privileged pull-request triggers and local actions, pin external actions, and perform
CodeQL analysis without uploading from pull-request code.

This access model is part of the security boundary. Adding a non-administrator same-repository
writer, weakening fork-run approval, or changing Actions token settings requires a new threat-model
review before the change is made.
