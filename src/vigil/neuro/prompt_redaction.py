"""Configuration-driven identifier redaction for offline semantic prompts.

The assembled prompt masks application package names, output labels, raw observation
identifiers, local file paths, and evaluator labels. Stable widget aliases, resource
suffixes, normalized permissions, and action properties remain available for guard
and invariant synthesis.

Identifiers are collected from the current FSM, its observations, and caller-provided
metadata. Original evidence is never changed: deterministic admission continues to
use the unredacted registry, while resource suffixes retain their semantic meaning
in the prompt (``com.app:id/amount`` becomes ``<app>:id/amount``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vigil.neuro.guard_evidence import GuardEvidence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vigil.models.fsm import AppFSM
    from vigil.neuro.guard_evidence import ScreenEvidence
    from vigil.neuro.invariant_evidence import InvariantEvidence

# Generic backstops applied in addition to the configured/evidence identifiers.
_SCR_RE = re.compile(r"\bscr_\d+\b", re.IGNORECASE)
_ABS_PATH_RE = re.compile(r"/[^\s'\"()\]]+\.(?:png|jpe?g|webp|xml|json)", re.IGNORECASE)

_APP_MASK = "<app>"
_SCR_MASK = "<scr>"
_PATH_MASK = "<path>"


class PromptRedactor:
    """Masks configured/evidence identifiers in a prompt string.

    ``app_literals`` (packages, slugs, bundle names, auxiliary labels) -> ``<app>``;
    ``screen_ids`` and the generic ``scr_####`` pattern -> ``<scr>``;
    ``paths`` and generic absolute media/data paths -> ``<path>``.
    Longer literals are masked before shorter ones so a bundle identifier is masked before its
    embedded package, and paths are masked before package literals can fragment them.
    """

    def __init__(
        self,
        *,
        packages: Iterable[str] = (),
        screen_ids: Iterable[str] = (),
        paths: Iterable[str] = (),
        extra_identifiers: Iterable[str] = (),
    ) -> None:
        literals: set[str] = set()
        for group in (packages, extra_identifiers):
            for item in group:
                value = item.strip()
                if value:
                    literals.add(value)
        self._app_literals = sorted(literals, key=len, reverse=True)
        self._screen_ids = sorted(
            {screen_id.strip() for screen_id in screen_ids if screen_id.strip()},
            key=len,
            reverse=True,
        )
        self._paths = sorted(
            {path.strip() for path in paths if path.strip()},
            key=len,
            reverse=True,
        )

    def redact(
        self,
        text: str,
        *,
        protected_literals: Iterable[str] = (),
    ) -> str:
        """Mask configured identifiers while preserving explicit protocol fragments.

        Paths are always redacted before protection. Callers should protect rendered
        protocol fragments (for example, a complete catalog JSON object or ``alias=...``
        token), not an underlying identifier globally; this lets the same identifier be
        masked in prose while remaining byte-exact in a machine-readable field.

        Args:
            text: Assembled prompt text.
            protected_literals: Exact rendered fragments to preserve during app/screen
                identifier masking.

        Returns:
            Redacted prompt text with protected protocol fragments restored verbatim.
        """
        if not text:
            return text
        out = text
        # Paths first (exact, longest-first), then the generic absolute-path backstop, so a
        # package literal cannot fragment a path before it is masked whole.
        for path in self._paths:
            out = out.replace(path, _PATH_MASK)
        out = _ABS_PATH_RE.sub(_PATH_MASK, out)

        protected: list[tuple[str, str]] = []
        literals = sorted(
            {value for value in protected_literals if value},
            key=len,
            reverse=True,
        )
        for index, literal in enumerate(literals):
            token = f"\ue000{index:x}\ue001"
            if literal in out:
                out = out.replace(literal, token)
                protected.append((token, literal))

        # App identifiers (packages/slugs/labels). Resource-id suffixes survive because only the
        # identifier substring is replaced: ``com.app:id/foo`` -> ``<app>:id/foo``.
        for literal in self._app_literals:
            out = out.replace(literal, _APP_MASK)
        # Screen ids (explicit list, then the generic scr_#### backstop).
        for screen_id in self._screen_ids:
            out = out.replace(screen_id, _SCR_MASK)
        out = _SCR_RE.sub(_SCR_MASK, out)
        for token, literal in protected:
            out = out.replace(token, literal)
        return out


def _collect_from_screen(
    screen: ScreenEvidence,
    packages: set[str],
    screen_ids: set[str],
    paths: set[str],
) -> None:
    if screen.package_name:
        packages.add(screen.package_name)
    screen_ids.update(value for value in (screen.screen_id, screen.screenshot_screen_id) if value)
    paths.update(value for value in (screen.screenshot_path, screen.xml_tree_path) if value)


def build_prompt_redactor(
    fsm: AppFSM,
    evidence_items: Iterable[GuardEvidence | InvariantEvidence] = (),
    *,
    extra_identifiers: Iterable[str] | None = None,
) -> PromptRedactor:
    """Build a redactor from the FSM package and typed evidence identifiers.

    Works for both :class:`~vigil.neuro.guard_evidence.GuardEvidence` (``source_screen`` /
    ``target_screen`` / ``*_screen_ids``) and
    :class:`~vigil.neuro.invariant_evidence.InvariantEvidence` (``raw_screen_ids`` /
    ``observations``). ``extra_identifiers`` carries caller/config identifiers (trace package,
    output slug, directories, auxiliary labels).
    """
    packages: set[str] = set()
    screen_ids: set[str] = set()
    paths: set[str] = set()

    if fsm.app_package:
        packages.add(fsm.app_package)

    for evidence in evidence_items:
        if isinstance(evidence, GuardEvidence):
            _collect_from_screen(evidence.source_screen, packages, screen_ids, paths)
            _collect_from_screen(evidence.target_screen, packages, screen_ids, paths)
            screen_ids.update(evidence.source_screen_ids)
            screen_ids.update(evidence.target_screen_ids)
            continue
        screen_ids.update(evidence.raw_screen_ids)
        for observation in evidence.observations:
            sid = observation.get("screen_id")
            if sid:
                screen_ids.add(str(sid))
            for key in ("screenshot_path", "xml_tree_path"):
                value = observation.get(key)
                if value:
                    paths.add(str(value))

    return PromptRedactor(
        packages=packages,
        screen_ids=screen_ids,
        paths=paths,
        extra_identifiers=extra_identifiers or (),
    )
