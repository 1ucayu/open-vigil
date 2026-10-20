"""Frozen mobile-agent intent values consumed by the runtime verifier.

The mobile agent owns semantic intent binding.  Vigil receives one immutable
value snapshot for a proposed action: a slot carries a value iff its key is
present.  A slot the agent has no usable value for is simply omitted; Vigil has
no status lifecycle (no ``absent``/``unresolved``/``conflict`` protocol state).
Agent-side uncertainty stays inside the agent and never becomes a Vigil state.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

IntentBindingValue = StrictStr | StrictInt | StrictFloat | StrictBool


class IntentBindingSnapshot(RootModel[dict[str, IntentBindingValue]]):
    """Frozen value map supplied by the mobile agent for one verification.

    A slot is *bound* iff its key is present.  The model never fills omitted
    entries: consumers derive *missing* from ``slot_name not in bindings`` so a
    transport omission stays missing and never becomes a value.
    """

    model_config = ConfigDict(frozen=True)

    root: dict[str, IntentBindingValue] = Field(default_factory=dict)

    @property
    def bindings(self) -> Mapping[str, IntentBindingValue]:
        """Named view used by the verifier; JSON remains a direct slot map."""
        return self.root

    def bound_variables(self) -> dict[str, IntentBindingValue]:
        """Return every present value (a present key is a bound value)."""
        return dict(self.bindings)

    def __hash__(self) -> int:
        return hash(tuple((name, value) for name, value in sorted(self.bindings.items())))


class RevisionedIntentBindingSnapshot(BaseModel):
    """Immutable planner/execution snapshot with optimistic revision control.

    ``slot_fingerprints`` is Vigil-owned admission metadata.  It binds every
    frozen value to the stable semantics/schema of the admitted ``IntentSlot``;
    the mobile agent never authors or echoes a fingerprint in ``IntentPatch``.
    The exact-key invariant prevents a name-only value from crossing an EFSM
    revision without deterministic re-admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: StrictInt = Field(default=0, ge=0)
    values: IntentBindingSnapshot = Field(default_factory=IntentBindingSnapshot)
    slot_fingerprints: dict[str, StrictStr]

    @model_validator(mode="after")
    def validate_slot_fingerprints(self) -> RevisionedIntentBindingSnapshot:
        value_names = set(self.values.bindings)
        fingerprint_names = set(self.slot_fingerprints)
        if fingerprint_names != value_names:
            raise ValueError("slot_fingerprints must contain exactly the frozen value names")
        return self


class IntentPatch(BaseModel):
    """Agent-agnostic planner delta against one snapshot revision.

    ``updates`` maps a canonical slot name to a JSON scalar value.  A slot the
    agent cannot supply is omitted, not carried with a status; omission leaves
    the slot missing in the resulting snapshot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_revision: StrictInt = Field(ge=0)
    updates: dict[str, IntentBindingValue] = Field(default_factory=dict)

    @property
    def entries(self) -> Mapping[str, IntentBindingValue]:
        return self.updates


class AdmittedPlannerProposal(BaseModel):
    """One planner proposal bound to the EFSM and intent revisions used to admit it.

    ``agent_action_proposal`` is the canonical JSON representation chosen by the
    agent-specific adapter.  Vigil treats it as opaque correlation data; it does
    not interpret mobile-agent action fields during intent-patch admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: StrictStr
    request_id: StrictStr
    efsm_revision: StrictInt = Field(ge=0)
    agent_action_proposal: StrictStr
    binding_snapshot: RevisionedIntentBindingSnapshot


class IntentPatchAdmissionResult(BaseModel):
    """Deterministic outcome of applying an intent patch to a frozen snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: bool
    admitted_proposal: AdmittedPlannerProposal | None = None
    reasons: tuple[str, ...] = ()
