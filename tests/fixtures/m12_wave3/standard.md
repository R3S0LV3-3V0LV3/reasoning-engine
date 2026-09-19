# Frontier Reasoning Engine Context

- Profile: `STANDARD`
- Snapshot: `7`
- Packet hash: `b2d72057d9466f6da11db464afdd81aaea1dd0e5e61b704572b4341e721c90ac`

## Wave 3 semantic availability
- Availability: `UNAVAILABLE_VALIDATION` — representation plan selects view(s) with no corresponding validated artifact
- Problem spec ref: `9679dbf97ee5f63c5ab9de1d45f84224071c702a1facfe18c16240ba9337db36`
- Ledger root: `a3d3d81f139103f97338bc4d7a6e01bb266eee9810c39ccbea4b90d8a987f3fd` @ version `7`
- Blocker refs: `blocker-1`
- Representation plan ref: `0407d2fce7f31a1e282309ccc464d8ca4b097e2f81f6b447c566be4ab533fb7b`
- Representation artifact refs: `none`
- Unresolved UNKNOWNs: `1`

## Objectives
- `{"description":"Minimize cost","direction":"MIN","evaluator_ref":null,"id":"obj-1","name":"cost","priority":null,"provenance":null,"unit":null}`

## Hard constraints
- `{"description":"cost <= 10","id":"hard-1","kind":"HARD","provenance":null,"source_refs":[],"support":[],"verification_mode":"DETERMINISTIC","verification_status":"UNKNOWN","verifier_ref":"verify.cost"}`

## Soft preferences
- `{"description":"prefer option a","id":"soft-1","kind":"SOFT","provenance":null,"source_refs":[],"support":[],"verification_mode":"HUMAN","verification_status":"UNKNOWN","verifier_ref":null}`

## Decision variables
- `{"domain":["a","b"],"id":"dv-1","name":"option","provenance":null}`

## Fixed parameters
- `{"id":"fixed-1","name":"limit","provenance":{"anchors":[],"basis":"deterministic fixture","confidence":null,"origin":"WORKING_ASSUMPTION","policy_basis":"test policy","support":[],"supporting_refs":[]},"value":10}`

## Unknowns
- `{"candidate_actions":[],"decision_relevance":null,"description":"future demand","domain":null,"id":"unknown-1","impact":null,"provenance":null,"rationale":null,"resolvable":true}`

## Observables
- `{"description":"quoted cost","id":"obs-1","provenance":null,"unit":"USD"}`

## Assumptions
- `{"items":[{"decision_relevance":null,"id":"assume-1","provenance":{"anchors":[],"basis":"deterministic fixture","confidence":null,"origin":"WORKING_ASSUMPTION","policy_basis":"test policy","support":[],"supporting_refs":[]},"scope":null,"statement":"Supply is stable","why_needed":"Bound the choice"}],"statements":["Demand is stable"]}`

## Acceptance criteria
- `{"blocker_ref":null,"id":"accept-1","predicate_description":"cost is within limit","provenance":null,"required":true,"verification_mode":"DETERMINISTIC","verification_status":"UNKNOWN"}`

## Output requirements
- `{"form":"JSON","requirements":["include rationale"],"schema_ref":null}`

## Relations
- `{"kind":"DEPENDS_ON","source_id":"obj-1","target_id":"hard-1"}`

## Explicit blockers
- `{"blocker_id":"blocker-1","description":"Demand must be observed","ledger_ref":{"node_id":"00000000-0000-0000-0000-000000000009","revision":1},"resolvable":true}`

## Current representation
- `{"omitted_reasons":[],"problem_spec_hash":"9679dbf97ee5f63c5ab9de1d45f84224071c702a1facfe18c16240ba9337db36","selection_basis":["decision variables available"],"views":[{"builder_ref":"m04.decision-table","builder_version":"1.0","compatibility_score":0.9,"expected_value":"clear trade-offs","id":"view-1","kind":"DECISION_TABLE","limitations":[],"purpose":"compare options","registry_version":"wave3-m04-registry/1.0","role":"PRIMARY","score_components":[],"selection_policy_version":"wave3-m04/1.0","source_object_refs":[]}]}`

## Ledger
- `00000000-0000-0000-0000-000000000008@1` [SUPPORTED] {"fact":"current"}

## Rejections

## Budget
- Remaining: `{"candidates":0,"input_tokens":0,"iterations":3,"llm_calls":0,"output_tokens":0,"runtime_seconds":0,"tool_calls":0}`

## Next / terminal action
- observe demand
