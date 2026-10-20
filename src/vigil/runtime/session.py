"""Per-episode state for the online verification loop."""

from __future__ import annotations

from pydantic import BaseModel

from vigil.models.intent_binding import IntentBindingSnapshot


class RuntimeSession(BaseModel):
    """Per-episode runtime context.

    Attributes:
        binding_snapshot: Frozen status-aware bindings supplied by the mobile agent.
    """

    binding_snapshot: IntentBindingSnapshot | None = None

    def resolve_binding_snapshot(
        self,
        supplied: IntentBindingSnapshot | None,
    ) -> IntentBindingSnapshot:
        """Freeze the first episode snapshot, including an explicit empty snapshot."""
        if self.binding_snapshot is None:
            self.binding_snapshot = supplied or IntentBindingSnapshot()
        return self.binding_snapshot
