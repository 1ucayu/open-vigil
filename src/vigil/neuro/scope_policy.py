"""Generic single-app scope policy for native UI exploration.

Vigil targets per-app FSM verification, not cross-app agent execution. The
explorer must distinguish between (a) the target app, (b) Android framework
dialogs and pickers that are legitimately part of an in-app flow, (c) the
system UI (status bar / nav bar overlays) which should never be enumerated as
app actions, (d) third-party apps the explorer accidentally landed in, and
(e) the launcher / home screen. This module performs the classification from a
package name plus a small generic config; no app-specific allowlists are
hardcoded here.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum


class ScopeCategory(StrEnum):
    """Scope of a captured screen relative to the target app."""

    IN_APP = "in_app"
    ANDROID_SYSTEM = "android_system"
    SYSTEM_UI = "system_ui"
    OUT_OF_SCOPE_EXTERNAL = "out_of_scope_external"
    LAUNCHER_OR_HOME = "launcher_or_home"


# Defaults are Android framework packages, not app-specific allowlists.
DEFAULT_ANDROID_SYSTEM_PACKAGES: frozenset[str] = frozenset(
    {
        "android",
        "com.android.documentsui",
        "com.android.packageinstaller",
        "com.google.android.packageinstaller",
        "com.android.permissioncontroller",
        "com.google.android.permissioncontroller",
        "com.android.providers.media",
        "com.android.intentresolver",
    }
)

DEFAULT_SYSTEM_UI_PACKAGES: frozenset[str] = frozenset(
    {
        "com.android.systemui",
    }
)

# Launcher patterns use fnmatch globbing so OEM launcher variants
# (com.sec.android.app.launcher, com.miui.home, etc.) classify correctly
# without enumerating every package.
DEFAULT_LAUNCHER_PATTERNS: tuple[str, ...] = (
    "com.android.launcher*",
    "com.google.android.apps.nexuslauncher",
    "com.google.android.launcher",
    "com.sec.android.app.launcher",
    "com.miui.home",
    "com.huawei.android.launcher",
    "com.oneplus.launcher",
)


@dataclass(frozen=True)
class ScopePolicy:
    """Classifies foreground packages relative to the target app.

    Attributes:
        app_package: Target app package name.
        android_system_packages: Framework packages allowed during exploration.
            Defaults to ``DEFAULT_ANDROID_SYSTEM_PACKAGES``.
        system_ui_packages: Packages classified as SYSTEM_UI (filtered from
            enumeration / fingerprint). Defaults to ``DEFAULT_SYSTEM_UI_PACKAGES``.
        launcher_patterns: Glob patterns matching launcher/home packages.
            Defaults to ``DEFAULT_LAUNCHER_PATTERNS``.
    """

    app_package: str
    android_system_packages: frozenset[str] = field(default=DEFAULT_ANDROID_SYSTEM_PACKAGES)
    system_ui_packages: frozenset[str] = field(default=DEFAULT_SYSTEM_UI_PACKAGES)
    launcher_patterns: tuple[str, ...] = field(default=DEFAULT_LAUNCHER_PATTERNS)

    def classify(self, package: str | None) -> ScopeCategory:
        """Return the scope category for ``package``.

        Empty / unknown packages are treated as OUT_OF_SCOPE_EXTERNAL because
        the explorer cannot prove they belong to the target app.
        """
        if not package:
            return ScopeCategory.OUT_OF_SCOPE_EXTERNAL
        pkg = package.strip()
        if pkg == self.app_package:
            return ScopeCategory.IN_APP
        if pkg in self.system_ui_packages:
            return ScopeCategory.SYSTEM_UI
        if pkg in self.android_system_packages:
            return ScopeCategory.ANDROID_SYSTEM
        for pattern in self.launcher_patterns:
            if fnmatch.fnmatch(pkg, pattern):
                return ScopeCategory.LAUNCHER_OR_HOME
        return ScopeCategory.OUT_OF_SCOPE_EXTERNAL

    def is_allowed(self, category: ScopeCategory) -> bool:
        """True iff the explorer should continue acting in this scope.

        ANDROID_SYSTEM is allowed so framework dialogs and pickers in app flows
        can be observed separately from the app's own states.
        """
        return category in (ScopeCategory.IN_APP, ScopeCategory.ANDROID_SYSTEM)

    def is_framework_scope(self, category: ScopeCategory) -> bool:
        """Return whether observations belong to Android framework UI."""
        return category == ScopeCategory.ANDROID_SYSTEM

    def should_filter_element(self, element_package: str | None) -> bool:
        """Filter rule for action enumeration / fingerprint contribution.

        Elements from SYSTEM_UI overlays must never affect app-level identity
        or action enumeration: a notification shade or transient toast can
        appear over any screen and would otherwise destabilize state ids.
        """
        return self.classify(element_package) == ScopeCategory.SYSTEM_UI
