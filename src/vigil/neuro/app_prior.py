"""Stage 0: App Prior Extraction.

Primary path: Androguard on a device-pulled APK. Loads manifest, resolves
string resources, and decodes all ``res/layout/*.xml`` files in a single
pass — no external ``aapt2`` dependency. ``extract_from_device_serial``
pulls the APK via adb and parses it; on failure it logs the exact cause
and returns an empty prior.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

from loguru import logger
from pydantic import BaseModel, Field

_ANDROID_NS = "http://schemas.android.com/apk/res/android"


def _android(attr: str) -> str:
    return f"{{{_ANDROID_NS}}}{attr}"


class ActivityInfo(BaseModel):
    """Parsed info about a single <activity> from AndroidManifest."""

    name: str
    label: str | None = None
    parent_activity: str | None = None
    exported: bool = False
    launch_mode: str = "standard"
    intent_actions: list[str] = Field(default_factory=list)
    is_launcher: bool = False


class WidgetDecl(BaseModel):
    """A widget declared in a layout XML file."""

    widget_id: str
    widget_class: str
    layout_file: str


class AppPrior(BaseModel):
    """Prior knowledge extracted from an Android app."""

    package_name: str
    entry_activity: str | None = None
    activities: list[ActivityInfo] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    string_constants: dict[str, str] = Field(default_factory=dict)
    string_arrays: dict[str, list[str]] = Field(default_factory=dict)
    widget_declarations: list[WidgetDecl] = Field(default_factory=list)

    def save(self, static_dir: Path) -> None:
        """Persist the prior JSON to ``static_dir/app_prior.json``."""
        static_dir = Path(static_dir)
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "app_prior.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, static_dir: Path) -> AppPrior:
        """Load an ``AppPrior`` previously saved via :meth:`save`."""
        path = Path(static_dir) / "app_prior.json"
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class AppPriorExtractor:
    """Extract structural prior knowledge from Android app metadata."""

    def extract_from_apk_file(self, apk_path: Path) -> AppPrior:
        """Load an APK via Androguard and populate an :class:`AppPrior`.

        Exercises Androguard's high-level APK object (activities, main
        activity, permissions), walks the decoded AndroidManifest XML to
        recover per-activity ``parentActivityName`` / ``launchMode`` /
        ``label`` attributes (which the high-level accessors drop), and
        iterates ``res/layout*/`` AXML files for widget declarations.
        """
        from androguard.core.apk import APK  # type: ignore[import-untyped]
        from androguard.core.axml import AXMLPrinter  # type: ignore[import-untyped]

        apk = APK(str(apk_path))
        package = apk.get_package() or ""
        manifest_axml = apk.get_android_manifest_axml()
        raw_manifest = manifest_axml.get_xml().decode("utf-8", errors="replace")

        # Parse the decoded manifest XML for full activity attributes —
        # Androguard's get_activities() returns bare names, and we need
        # parentActivityName / launchMode / label / intent filters.
        prior = self.extract_from_manifest_string(raw_manifest)
        if not prior.package_name:
            prior.package_name = package

        # Supplement permissions from Androguard in case the XML parse
        # missed any (e.g. merged from platform manifests).
        for perm in apk.get_permissions() or []:
            if perm and perm not in prior.permissions:
                prior.permissions.append(perm)

        # Strings: walk the ARSC package table directly. Androguard exposes
        # resolved entries at ``arsc.values[pkg][locale]['string']`` as a list
        # of ``[name, text]`` pairs. The default locale key is the null-byte
        # string ``'\x00\x00'`` — NOT ``'DEFAULT'`` (that label only appears
        # in ``get_resolved_strings()``'s alternate view, which keys by
        # integer resource id, not by symbolic name).
        arsc = apk.get_android_resources()
        if arsc is not None:
            # ``arsc.values`` is lazy-populated; calling get_resolved_strings()
            # triggers the full package/locale/type walk that fills it.
            try:
                arsc.get_resolved_strings()
            except Exception as exc:
                logger.debug(f"get_resolved_strings() failed: {exc}", exc_info=True)
            for pkg_name in arsc.get_packages_names() or []:
                pkg_locales = getattr(arsc, "values", {}).get(pkg_name, {})
                if not isinstance(pkg_locales, dict):
                    continue
                locale_data = pkg_locales.get("\x00\x00")
                if not (isinstance(locale_data, dict) and locale_data.get("string")):
                    locale_data = next(
                        (
                            d
                            for d in pkg_locales.values()
                            if isinstance(d, dict) and d.get("string")
                        ),
                        None,
                    )
                if not locale_data:
                    continue
                for entry in locale_data.get("string", []):
                    if (
                        isinstance(entry, list | tuple)
                        and len(entry) >= 2
                        and isinstance(entry[0], str)
                        and entry[0]
                        and isinstance(entry[1], str)
                    ):
                        prior.string_constants.setdefault(entry[0], entry[1])

        # Layouts: decode every res/layout*/*.xml file via AXMLPrinter.
        layout_count = 0
        for fn in apk.get_files():
            if not (fn.startswith("res/layout") and fn.endswith(".xml")):
                continue
            try:
                payload = apk.get_file(fn)
                if not payload:
                    continue
                axml = AXMLPrinter(payload)
                decoded = axml.get_xml().decode("utf-8", errors="replace")
            except Exception as exc:
                logger.debug(f"Failed to decode {fn}: {exc}", exc_info=True)
                continue
            layout_count += 1
            self._parse_layout_xml_string(decoded, Path(fn).stem, prior)

        logger.info(
            f"Androguard extraction: {len(prior.activities)} activities, "
            f"{len(prior.permissions)} permissions, "
            f"{len(prior.string_constants)} strings, "
            f"{len(prior.widget_declarations)} widgets from "
            f"{layout_count} layouts"
        )
        return prior

    def extract_from_manifest_string(self, xml_string: str) -> AppPrior:
        """Parse an AndroidManifest.xml from a string."""
        root = ET.fromstring(xml_string)  # noqa: S314
        tree = ET.ElementTree(root)
        return self._parse_manifest_tree(tree)

    def extract_from_device_serial(self, serial: str, package: str) -> AppPrior:
        """Pull the APK from the device and parse it with Androguard.

        Logs the exact failure and returns an empty prior when:
          - ``adb shell pm path`` returns no usable APK path.
          - ``adb pull`` fails (permission denied, device offline, etc.).
          - Androguard raises while parsing the pulled APK.

        Never raises at the outer level.
        """
        empty = AppPrior(package_name=package)
        try:
            path_result = subprocess.run(
                ["adb", "-s", serial, "shell", "pm", "path", package],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning(f"adb pm path failed ({exc})")
            return empty
        if path_result.returncode != 0:
            logger.warning(
                f"pm path returned rc={path_result.returncode}: {path_result.stderr.strip()!r}"
            )
            return empty

        apk_paths = [
            ln.removeprefix("package:").strip()
            for ln in path_result.stdout.splitlines()
            if ln.startswith("package:")
        ]
        if not apk_paths:
            logger.warning(f"pm path returned no APK entries for {package}: {path_result.stdout!r}")
            return empty

        # Prefer a ``base.apk`` entry if present (common for user-installed
        # apps); otherwise pick the first APK path (system apps on Pixel /
        # AOSP use names like ``SettingsGoogle.apk``). Split APKs without a
        # base entry aren't handled.
        remote_apk = next(
            (p for p in apk_paths if p.endswith("/base.apk")),
            apk_paths[0] if len(apk_paths) == 1 else None,
        )
        if remote_apk is None:
            logger.warning(f"Split APK layout without base.apk detected for {package}: {apk_paths}")
            return empty

        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "app.apk"
            try:
                pull_result = subprocess.run(
                    ["adb", "-s", serial, "pull", remote_apk, str(local)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                logger.warning(f"adb pull failed ({exc})")
                return empty
            if pull_result.returncode != 0 or not local.exists():
                logger.warning(
                    f"adb pull {remote_apk} failed rc={pull_result.returncode}: "
                    f"{pull_result.stderr.strip()!r}"
                )
                return empty

            try:
                return self.extract_from_apk_file(local)
            except Exception as exc:
                logger.warning(f"Androguard failed to parse {remote_apk}: {exc}")
                return empty

    def extract_resources(self, apk_dir: Path, prior: AppPrior) -> None:
        """Extract resources from an apktool-decompiled APK directory.

        Parses res/values/strings.xml and res/layout/*.xml into the prior.
        """
        strings_path = apk_dir / "res" / "values" / "strings.xml"
        if strings_path.exists():
            self._parse_strings_xml(strings_path, prior)

        layout_dir = apk_dir / "res" / "layout"
        if layout_dir.is_dir():
            for layout_file in sorted(layout_dir.glob("*.xml")):
                self._parse_layout_xml(layout_file, prior)

        logger.info(
            f"Resources extracted: {len(prior.string_constants)} strings, "
            f"{len(prior.string_arrays)} arrays, "
            f"{len(prior.widget_declarations)} widgets"
        )

    @staticmethod
    def _parse_strings_xml(path: Path, prior: AppPrior) -> None:
        """Parse res/values/strings.xml."""
        try:
            tree = ET.parse(path)  # noqa: S314
        except ET.ParseError:
            return

        root = tree.getroot()
        for elem in root.findall("string"):
            name = elem.get("name", "")
            text = elem.text or ""
            if name and text:
                prior.string_constants[name] = text

        for elem in root.findall("string-array"):
            name = elem.get("name", "")
            if not name:
                continue
            items = [item.text or "" for item in elem.findall("item")]
            if items:
                prior.string_arrays[name] = items

    @staticmethod
    def _parse_layout_xml(path: Path, prior: AppPrior) -> None:
        """Parse a single res/layout/*.xml for widget declarations."""
        try:
            tree = ET.parse(path)  # noqa: S314
        except ET.ParseError:
            return

        AppPriorExtractor._collect_layout_widgets(tree.getroot(), path.stem, prior)

    @staticmethod
    def _parse_layout_xml_string(xml_string: str, layout_name: str, prior: AppPrior) -> None:
        """In-memory sibling of :meth:`_parse_layout_xml` for Androguard-decoded
        layouts. Same widget-extraction rules, no file I/O."""
        try:
            root = ET.fromstring(xml_string)  # noqa: S314
        except ET.ParseError:
            return
        AppPriorExtractor._collect_layout_widgets(root, layout_name, prior)

    @staticmethod
    def _collect_layout_widgets(root: ET.Element, layout_name: str, prior: AppPrior) -> None:
        """Extract widget declarations from a successfully parsed layout."""
        android_ns = "http://schemas.android.com/apk/res/android"
        for elem in root.iter():
            widget_id = elem.get(f"{{{android_ns}}}id", "")
            if not widget_id:
                continue
            widget_id = widget_id.replace("@+id/", "").replace("@id/", "")
            widget_class = elem.tag.rsplit(".", 1)[-1]
            prior.widget_declarations.append(
                WidgetDecl(
                    widget_id=widget_id,
                    widget_class=widget_class,
                    layout_file=layout_name,
                )
            )

    def _parse_manifest_tree(self, tree: ET.ElementTree) -> AppPrior:
        root = tree.getroot()
        package = root.get("package", "")

        activities: list[ActivityInfo] = []
        entry_activity: str | None = None

        for elem in root.iter("activity"):
            info = self._parse_activity_element(elem, package)
            activities.append(info)
            if info.is_launcher and entry_activity is None:
                entry_activity = info.name

        for elem in root.iter("activity-alias"):
            info = self._parse_activity_element(elem, package)
            activities.append(info)
            if info.is_launcher and entry_activity is None:
                target = elem.get(_android("targetActivity"), "")
                entry_activity = self._resolve_class_name(target, package)

        permissions: list[str] = []
        for elem in root.iter("uses-permission"):
            perm = elem.get(_android("name"), "")
            if perm:
                permissions.append(perm)

        return AppPrior(
            package_name=package,
            entry_activity=entry_activity,
            activities=activities,
            permissions=permissions,
        )

    def _parse_activity_element(self, elem: ET.Element, package: str) -> ActivityInfo:
        raw_name = elem.get(_android("name"), "")
        name = self._resolve_class_name(raw_name, package)
        label = elem.get(_android("label"))
        parent = elem.get(_android("parentActivityName"))
        if parent:
            parent = self._resolve_class_name(parent, package)

        exported_str = elem.get(_android("exported"))
        exported = exported_str == "true"

        launch_mode = elem.get(_android("launchMode"), "standard")

        intent_actions: list[str] = []
        is_launcher = False
        for intent_filter in elem.findall("intent-filter"):
            actions = [
                a.get(_android("name"), "")
                for a in intent_filter.findall("action")
                if a.get(_android("name"))
            ]
            intent_actions.extend(actions)
            categories = {c.get(_android("name"), "") for c in intent_filter.findall("category")}
            has_main = "android.intent.action.MAIN" in actions
            has_launcher = "android.intent.category.LAUNCHER" in categories
            if has_main and has_launcher:
                is_launcher = True

        return ActivityInfo(
            name=name,
            label=label,
            parent_activity=parent,
            exported=exported,
            launch_mode=launch_mode,
            intent_actions=intent_actions,
            is_launcher=is_launcher,
        )

    @staticmethod
    def _resolve_class_name(name: str, package: str) -> str:
        if name.startswith("."):
            return package + name
        return name
