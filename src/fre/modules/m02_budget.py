"""M02 monotone tier allocation and deterministic budget revision."""

from fre.domain.budget import (
    BudgetLimits,
    BudgetPlan,
    BudgetProjection,
    DeploymentLimits,
    DeploymentPolicyConflict,
    InvalidBudgetRevision,
    ReasoningTier,
    SearchPolicy,
    TierDefinition,
    TierPolicy,
)
from fre.domain.common import canonical_hash
from fre.domain.task import TaskSignature

TIER_ORDER = {tier: index for index, tier in enumerate(ReasoningTier)}


def default_tier_policy() -> TierPolicy:
    limits = {
        ReasoningTier.T0: (1, 1, 0, 8192, 2048, 1, 1, 60),
        ReasoningTier.T1: (2, 2, 0, 16384, 4096, 2, 1, 120),
        ReasoningTier.T2: (4, 6, 1, 65536, 16384, 4, 2, 300),
        ReasoningTier.T3: (6, 12, 2, 131072, 32768, 6, 3, 600),
        ReasoningTier.T4: (10, 24, 5, 262144, 65536, 10, 4, 1200),
        ReasoningTier.T5: (14, 48, 10, 524288, 131072, 24, 4, 1800),
    }
    search = {
        ReasoningTier.T0: (1, 1, 1, 1, 0, 0, 0),
        ReasoningTier.T1: (2, 1, 1, 1, 1, 0, 0),
        ReasoningTier.T2: (4, 2, 1, 1, 2, 0, 16),
        ReasoningTier.T3: (6, 3, 2, 2, 4, 0, 32),
        ReasoningTier.T4: (10, 4, 2, 3, 10, 1, 64),
        ReasoningTier.T5: (24, 4, 2, 3, 10, 2, 128),
    }
    thresholds = (1.0, 1.0, 0.75, 0.5, 0.35, 0.25)
    definitions: dict[ReasoningTier, TierDefinition] = {}
    for tier in ReasoningTier:
        definitions[tier] = TierDefinition(
            limits=BudgetLimits(
                **dict(
                    zip(
                        (
                            "max_iterations",
                            "max_llm_calls",
                            "max_tool_calls",
                            "max_input_tokens",
                            "max_output_tokens",
                            "max_candidates",
                            "max_concurrent_actions",
                            "max_runtime_seconds",
                        ),
                        limits[tier],
                        strict=True,
                    )
                )
            ),
            search=SearchPolicy(
                **dict(
                    zip(
                        (
                            "initial_candidate_count",
                            "max_candidate_depth",
                            "max_representation_views",
                            "synthesis_order_limit",
                            "falsifier_operator_limit",
                            "independent_validation_branches",
                            "sensitivity_samples",
                        ),
                        search[tier],
                        strict=True,
                    )
                )
            ),
            acquisition_threshold=thresholds[TIER_ORDER[tier]],
        )
    ordinal = {
        "LOW": ReasoningTier.T0,
        "MEDIUM": ReasoningTier.T1,
        "HIGH": ReasoningTier.T3,
        "CRITICAL": ReasoningTier.T4,
    }
    return TierPolicy(
        policy_version="wave2-default/1.0",
        tiers=definitions,
        consequence_floors=ordinal,
        irreversibility_floors={**ordinal, "HIGH": ReasoningTier.T2},
        ambiguity_floors={**ordinal, "HIGH": ReasoningTier.T2, "CRITICAL": ReasoningTier.T3},
        evidence_scarcity_floors={
            **ordinal,
            "HIGH": ReasoningTier.T2,
            "CRITICAL": ReasoningTier.T3,
        },
        search_space_floors={
            "CLOSED": ReasoningTier.T0,
            "BOUNDED": ReasoningTier.T1,
            "OPEN": ReasoningTier.T3,
        },
    )


class BudgetAllocator:
    @staticmethod
    def policy_hash(policy: TierPolicy, deployment: DeploymentLimits) -> str:
        return canonical_hash({"policy": policy, "deployment": deployment})

    def allocate(
        self, signature: TaskSignature, policy: TierPolicy, deployment: DeploymentLimits
    ) -> tuple[BudgetPlan, str]:
        floors = (
            policy.consequence_floors[signature.consequence.value],
            policy.irreversibility_floors[signature.irreversibility.value],
            policy.ambiguity_floors[signature.ambiguity.value],
            policy.evidence_scarcity_floors[signature.evidence_scarcity.value],
            policy.search_space_floors[signature.search_space.value],
        )
        tier = max(floors, key=TIER_ORDER.__getitem__)
        if (
            signature.consequence.value == "CRITICAL"
            and signature.irreversibility.value == "CRITICAL"
        ):
            tier = ReasoningTier.T5
        if TIER_ORDER[tier] > TIER_ORDER[deployment.maximum_tier]:
            raise DeploymentPolicyConflict("deployment maximum is below the mandatory tier floor")
        definition = policy.tiers[tier]
        limits = self._clamp(definition.limits, deployment.resource_ceilings)
        if (
            limits.max_candidates < definition.search.initial_candidate_count
            or limits.max_llm_calls < definition.search.independent_validation_branches
        ):
            raise DeploymentPolicyConflict(
                "deployment resource ceilings cannot satisfy mandatory tier obligations"
            )
        plan = BudgetPlan(
            tier=tier,
            limits=limits,
            search=definition.search,
            acquisition_threshold=definition.acquisition_threshold,
            stop_threshold=definition.stop_threshold,
            policy_version=policy.policy_version,
        )
        return plan, self.policy_hash(policy, deployment)

    @staticmethod
    def _clamp(limits: BudgetLimits, ceilings: BudgetLimits | None) -> BudgetLimits:
        if ceilings is None:
            return limits
        values = {
            name: min(getattr(limits, name), getattr(ceilings, name))
            for name in BudgetLimits.model_fields
        }
        return BudgetLimits(**values)

    def revise(
        self, projection: BudgetProjection, plan: BudgetPlan, policy_hash: str
    ) -> BudgetProjection:
        if projection.plan is None:
            raise InvalidBudgetRevision("cannot revise an unallocated budget")
        minimum = projection.committed.model_dump()
        for reservation in projection.reservations:
            for name, value in reservation.resources.model_dump().items():
                minimum[name] += value
        allocated = plan.limits.resources()
        if any(getattr(allocated, name) < value for name, value in minimum.items()):
            raise InvalidBudgetRevision("revision falls below committed plus reserved usage")
        if TIER_ORDER[plan.tier] < TIER_ORDER[projection.plan.tier]:
            raise InvalidBudgetRevision("budget revisions may not lower the committed tier")
        if (
            plan.search.independent_validation_branches
            < projection.plan.search.independent_validation_branches
        ):
            raise InvalidBudgetRevision("budget revision lowers mandatory validation floor")
        return projection.model_copy(update={"plan": plan, "policy_hash": policy_hash})
