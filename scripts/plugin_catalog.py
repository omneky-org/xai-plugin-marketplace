#!/usr/bin/env python3
"""Shared plugin extraction for marketplace catalog tooling.

One extract path given a plugin root on disk (already resolved at a known
sha / local path):

  - load manifest (.grok-plugin/plugin.json or .claude-plugin/plugin.json)
  - read version if present
  - scan components (skills, commands, agents, mcpServers, hooks, lspServers)

Used by generate-plugin-index.py (full catalog) and bump-plugin-shas.py
(version gate at HEAD vs pin).

Fail-closed rule: a config file that is absent may fall back (e.g. to the
next manifest path or to manifest-declared hooks). A config file that is
present but unusable (too large, unreadable, not UTF-8, not JSON, wrong shape,
escapes the plugin root) raises PluginExtractionError instead, so the index
never silently omits or substitutes what the client would load.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path

MAX_STRING_LEN = 120
MAX_TEXT_READ_BYTES = 64 * 1024
MAX_JSON_BYTES = 1 << 20
FETCH_ATTEMPTS = 3

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

MANIFEST_PATHS = [
    Path(".grok-plugin/plugin.json"),
    Path(".claude-plugin/plugin.json"),
]

# Catalog plugin names: ASCII lowercase kebab-case only. ASCII-only also
# rules out Unicode lookalikes (Cyrillic, fullwidth, ...) of official names.
PLUGIN_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Invisible categories: controls, format (bidi overrides/isolates/marks,
# zero-width, tag characters, BOM, soft hyphen), and surrogates.
INVISIBLE_CATEGORIES = {"Cc", "Cf", "Cs"}
# ZWNJ / ZWJ are needed by Arabic/Persian/Indic text and emoji sequences.
# Descriptions keep them only between two visible non-ASCII characters;
# anywhere else (e.g. inside an ASCII word) they are dropped. Names never
# allow them.
JOINERS = {"\u200c", "\u200d"}


class PluginExtractionError(RuntimeError):
    """A plugin file is present but cannot be indexed faithfully."""


def plugin_name_error(name) -> str | None:
    """Return why `name` is not a valid catalog plugin name, or None if valid."""
    if not isinstance(name, str):
        return f"name must be a string, got {type(name).__name__}"
    if not PLUGIN_NAME_RE.fullmatch(name):
        return (
            f"name {ascii(name)} must be ASCII lowercase kebab-case "
            f"(letters a-z, digits, single hyphens between words)"
        )
    return None


def is_invisible(ch: str) -> bool:
    return unicodedata.category(ch) in INVISIBLE_CATEGORIES and not ch.isspace()


def _joinable(ch: str) -> bool:
    return ord(ch) > 0x7F and not ch.isspace() and not is_invisible(ch)


def strip_invisible(text: str) -> str:
    out = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if not is_invisible(ch):
            out.append(ch)
        elif (
            ch in JOINERS
            and 0 < i < last
            and _joinable(text[i - 1])
            and _joinable(text[i + 1])
        ):
            out.append(ch)
    return "".join(out)


def clean(text: str) -> str:
    text = strip_invisible(CONTROL_CHARS_RE.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_STRING_LEN:
        head = text[: MAX_STRING_LEN - 1].rstrip()
        text = head.rstrip("".join(JOINERS)).rstrip() + "\u2026"
    return text


def clean_name(name: str) -> str:
    """clean() for component identifiers; invisible characters are an error.

    Stripping would make the index show a different identifier than the one
    the client stores, so refuse instead. Names that change under NFKC
    (fullwidth, compatibility ligatures, non-breaking spaces, ...) are
    rejected too, since they render like a different identifier.
    """
    for ch in name:
        if is_invisible(ch):
            raise PluginExtractionError(
                f"component name {ascii(name)} contains invisible or "
                f"bidirectional control character U+{ord(ch):04X}"
            )
    if unicodedata.normalize("NFKC", name) != name:
        raise PluginExtractionError(
            f"component name {ascii(name)} is not NFKC-normalized "
            f"(renders like {ascii(unicodedata.normalize('NFKC', name))})"
        )
    return clean(name)


def resolve_inside(root: Path, relative) -> Path | None:
    try:
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root.resolve()):
            return candidate
    except (OSError, ValueError):
        return None
    return None


def contained(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def locate(root: Path, relative) -> Path | None:
    """Return the regular file at root/relative, or None if nothing is there.

    Anything present at that path that is not a regular file inside the root
    (directory, symlink escaping the root, ...) raises rather than reading as
    absent, so callers cannot fall back past it.
    """
    try:
        present = os.path.lexists(root / relative)
    except (OSError, ValueError):
        present = False
    if not present:
        return None
    resolved = resolve_inside(root, relative)
    if resolved is None:
        raise PluginExtractionError(f"{relative}: resolves outside the plugin root")
    if not resolved.is_file():
        raise PluginExtractionError(f"{relative}: exists but is not a regular file")
    return resolved


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    # Keys arrive already unescaped, so "sha" and "\\u0073ha" collide here.
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateKeyError(f"duplicate JSON object key {key!r}")
        obj[key] = value
    return obj


def loads_strict(text: str):
    """json.loads that rejects duplicate object keys.

    The stdlib parser silently keeps the last duplicate, so a reviewer can read
    one value while tooling and clients act on another.
    """
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def load_catalog_file(path: Path):
    """Strictly parse a marketplace catalog file (duplicate keys are errors)."""
    try:
        return loads_strict(path.read_text(encoding="utf-8"))
    except DuplicateKeyError as e:
        raise RuntimeError(f"{path}: {e}") from e


def read_text_limited(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read(MAX_TEXT_READ_BYTES)


def load_json_file(path: Path, label) -> dict:
    """Load a present JSON config file whose top level must be an object.

    Raises PluginExtractionError (never returns None) when the file is too
    large, unreadable, not UTF-8, not JSON, or not an object.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_JSON_BYTES + 1)
    except OSError as e:
        raise PluginExtractionError(
            f"{label}: cannot read file: {e.strerror or e}"
        ) from e
    if len(raw) > MAX_JSON_BYTES:
        raise PluginExtractionError(
            f"{label}: file exceeds the {MAX_JSON_BYTES}-byte limit for "
            f"indexed JSON configs"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PluginExtractionError(
            f"{label}: not valid UTF-8 ({e.reason} at byte {e.start})"
        ) from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PluginExtractionError(
            f"{label}: invalid JSON: {e.msg} at line {e.lineno} column {e.colno}"
        ) from e
    except RecursionError as e:
        raise PluginExtractionError(f"{label}: JSON is nested too deeply") from e
    if not isinstance(data, dict):
        raise PluginExtractionError(
            f"{label}: top-level JSON value must be an object, got "
            f"{type(data).__name__}"
        )
    return data


def optional_object(data: dict, key: str, label) -> dict | None:
    """data[key] when present (must be an object), else None."""
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, dict):
        raise PluginExtractionError(
            f"{label}: `{key}` must be an object, got {type(value).__name__}"
        )
    return value


def load_declared_json(root: Path, rel, what: str) -> dict:
    """Load a JSON file the manifest explicitly points at, or fail.

    A manifest path is what the installer is told to load, so an escaping,
    missing, or unreadable target must not silently drop out of the index.
    """
    path = resolve_inside(root, rel)
    if path is None:
        raise PluginExtractionError(
            f"manifest {what} path escapes the plugin root: {rel!r}"
        )
    if not path.is_file():
        raise PluginExtractionError(f"manifest {what} path not found: {rel!r}")
    try:
        return load_json_file(path, rel)
    except PluginExtractionError as e:
        raise PluginExtractionError(
            f"manifest {what} path {rel!r} is not a readable JSON object: {e}"
        ) from e


def parse_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = read_text_limited(path)
    except OSError:
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    i = 1
    while i < len(lines):
        line = lines[i]
        if line.strip() in ("---", "..."):
            break
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()
        if re.match(r"^[|>][+-]?$", value):
            block: list[str] = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith((" ", "\t")) or not lines[j].strip()):
                if lines[j].strip():
                    block.append(lines[j].strip())
                j += 1
            value = " ".join(block)
            i = j
        else:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            i += 1
        fields[key] = value
    return fields


def first_body_line(path: Path) -> str:
    try:
        text = read_text_limited(path)
    except OSError:
        return ""
    lines = text.splitlines()
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    for line in lines[i:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def make_item(name: str, description: str = "") -> dict:
    item = {"name": clean_name(name)}
    description = clean(description)
    if description:
        item["description"] = description
    return item


def load_manifest(root: Path) -> dict:
    # Only fall through to the next manifest path when this one is absent.
    for rel in MANIFEST_PATHS:
        path = locate(root, rel)
        if path is not None:
            return load_json_file(path, rel)
    return {}


def manifest_version(manifest: dict) -> str | None:
    """Return a non-empty version string from the manifest, or None."""
    value = manifest.get("version")
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def scan_skills(root: Path, manifest: dict) -> list[dict]:
    skill_dirs: dict[Path, str] = {}
    for base in [root / "skills"]:
        if base.is_dir():
            for child in sorted(base.iterdir()):
                skill_md = child / "SKILL.md"
                if skill_md.is_file() and contained(skill_md, root):
                    skill_dirs[child.resolve()] = child.name
    for entry in as_list(manifest.get("skills")):
        if isinstance(entry, str):
            candidate = resolve_inside(root, entry)
            if (
                candidate
                and (candidate / "SKILL.md").is_file()
                and contained(candidate / "SKILL.md", root)
            ):
                skill_dirs.setdefault(candidate, candidate.name)
    items = []
    for skill_dir, dirname in skill_dirs.items():
        fm = parse_frontmatter(skill_dir / "SKILL.md")
        items.append(make_item(fm.get("name") or dirname, fm.get("description", "")))
    return items


def scan_markdown_dir(
    root: Path, dirname: str, manifest_key: str, manifest: dict
) -> list[dict]:
    files: dict[Path, str] = {}
    base = root / dirname
    if base.is_dir():
        for path in sorted(base.rglob("*.md")):
            if path.is_file() and contained(path, root):
                files[path.resolve()] = path.stem
    for entry in as_list(manifest.get(manifest_key)):
        if isinstance(entry, str):
            candidate = resolve_inside(root, entry)
            if candidate and candidate.is_file() and candidate.suffix == ".md":
                files.setdefault(candidate, candidate.stem)
    items = []
    for path, stem in files.items():
        fm = parse_frontmatter(path)
        description = fm.get("description") or first_body_line(path)
        items.append(make_item(fm.get("name") or stem, description))
    return items


def transport_hint(config: dict) -> str:
    if not isinstance(config, dict):
        return ""
    transport = config.get("type") or config.get("transport")
    if not transport:
        if config.get("command"):
            transport = "stdio"
        elif config.get("url"):
            transport = "http"
    return str(transport) if transport else ""


def scan_mcp_servers(root: Path, manifest: dict) -> list[dict]:
    servers: dict[str, dict] = {}
    declared = manifest.get("mcpServers")
    data = None
    label = ".mcp.json"
    if isinstance(declared, str):
        label = declared
        data = load_declared_json(root, declared, "mcpServers")
    else:
        mcp_path = locate(root, label)
        if mcp_path is not None:
            data = load_json_file(mcp_path, label)
    if data is not None:
        nested = optional_object(data, "mcpServers", label)
        if nested is not None:
            servers.update(nested)
    if isinstance(declared, dict):
        for name, config in declared.items():
            servers.setdefault(name, config)
    return [make_item(name, transport_hint(config)) for name, config in servers.items()]


DEFAULT_HOOKS_PATH = Path("hooks") / "hooks.json"


def hook_events(data: dict, label) -> dict[str, list[str]]:
    """Map event name -> matchers for one hooks config object."""
    hooks_obj = optional_object(data, "hooks", label)
    if hooks_obj is None:
        hooks_obj = data
    events: dict[str, list[str]] = {}
    for event, entries in hooks_obj.items():
        if not isinstance(entries, list):
            continue
        events[event] = [
            e["matcher"]
            for e in entries
            if isinstance(e, dict) and isinstance(e.get("matcher"), str) and e["matcher"]
        ]
    return events


def scan_hooks(root: Path, manifest: dict) -> list[dict]:
    """Index hook events.

    An explicit manifest `hooks` value (path, inline object, or a list of
    those) is authoritative and replaces hooks/hooks.json. If the manifest
    declares hooks and a separate hooks/hooks.json also exists, which one a
    client loads depends on its semantics, so extraction fails rather than
    indexing only one of them.
    """
    declared = manifest.get("hooks")
    configs: list[tuple[dict, object]] = []
    # locate() raises if the default path is present but escapes the root or
    # is not a regular file, so it can never read as absent.
    default_path = locate(root, DEFAULT_HOOKS_PATH)
    if declared is None:
        if default_path is not None:
            configs.append(
                (load_json_file(default_path, DEFAULT_HOOKS_PATH), DEFAULT_HOOKS_PATH)
            )
    else:
        declared_paths: set[Path] = set()
        for entry in declared if isinstance(declared, list) else [declared]:
            if isinstance(entry, dict):
                configs.append((entry, "manifest hooks"))
            elif isinstance(entry, str):
                configs.append((load_declared_json(root, entry, "hooks"), entry))
                declared_paths.add(resolve_inside(root, entry))
            else:
                raise PluginExtractionError(
                    f"manifest hooks entry must be a path or object, got "
                    f"{type(entry).__name__}"
                )
        if default_path is not None and default_path not in declared_paths:
            raise PluginExtractionError(
                f"manifest declares hooks but {DEFAULT_HOOKS_PATH} also exists "
                f"and is not one of the declared paths; reference it from the "
                f"manifest or remove it so the index shows every hook that "
                f"can load"
            )
    events: dict[str, list[str]] = {}
    for config, label in configs:
        for event, matchers in hook_events(config, label).items():
            events.setdefault(event, []).extend(matchers)
    return [make_item(event, ", ".join(matchers)) for event, matchers in events.items()]


def scan_lsp_servers(root: Path, manifest: dict) -> list[dict]:
    servers: dict[str, dict] = {}
    declared = manifest.get("lspServers")
    data = None
    label = ".lsp.json"
    if isinstance(declared, str):
        label = declared
        data = load_declared_json(root, declared, "lspServers")
    else:
        lsp_path = locate(root, label)
        if lsp_path is not None:
            data = load_json_file(lsp_path, label)
    if data is not None:
        nested = optional_object(data, "lspServers", label)
        servers.update(nested if nested is not None else data)
    if isinstance(declared, dict):
        for name, config in declared.items():
            servers.setdefault(name, config)
    return [
        make_item(name)
        for name, config in servers.items()
        if isinstance(config, dict)
    ]


def extract_plugin(root: Path) -> dict:
    """Extract version + components from a plugin root on disk.

    Returns:
      {
        "version": str | omitted when missing,
        "components": { category: [items...] }
      }
    """
    if not root.is_dir():
        raise RuntimeError(f"plugin root is not a directory: {root}")

    manifest = load_manifest(root)
    categories = {
        "skills": scan_skills(root, manifest),
        "commands": scan_markdown_dir(root, "commands", "commands", manifest),
        "agents": scan_markdown_dir(root, "agents", "agents", manifest),
        "mcpServers": scan_mcp_servers(root, manifest),
        "hooks": scan_hooks(root, manifest),
        "lspServers": scan_lsp_servers(root, manifest),
    }
    components = {}
    for key in sorted(categories):
        # Keep every item: the index is what reviewers read, so nothing that
        # ships in the pinned tree may be dropped from it.
        items = sorted(categories[key], key=lambda i: i["name"])
        if items:
            components[key] = items

    record: dict = {"components": components}
    version = manifest_version(manifest)
    if version is not None:
        record["version"] = version
    return record


def run_git(cmd: list[str], url: str, sha: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to fetch {url} at {sha}: `{' '.join(cmd)}` exited "
            f"{result.returncode}: {result.stderr.strip()}"
        )


def fetch_at_sha(url: str, sha: str, dest: Path) -> None:
    """Shallow-fetch url at sha into dest and detach checkout."""
    if not SHA_RE.match(sha):
        raise RuntimeError(f"sha {sha!r} is not a 40-character lowercase hex commit")
    run_git(["git", "init", "--quiet", "--", str(dest)], url, sha)
    fetch_cmd = [
        "git",
        "-C",
        str(dest),
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "--end-of-options",
        url,
        sha,
    ]
    last_error: RuntimeError | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            run_git(fetch_cmd, url, sha)
            last_error = None
            break
        except RuntimeError as e:
            last_error = e
            if attempt < FETCH_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    if last_error is not None:
        raise last_error
    run_git(
        ["git", "-C", str(dest), "checkout", "--quiet", "--detach", sha],
        url,
        sha,
    )


def plugin_root_for_fetch(dest: Path, subdir: str | None, name: str) -> Path:
    if not subdir:
        return dest
    resolved = resolve_inside(dest, subdir)
    if resolved is None:
        raise RuntimeError(
            f"plugin '{name}': url source path escapes the repo: {subdir!r}"
        )
    if not resolved.is_dir():
        raise RuntimeError(f"plugin '{name}': url source path not found: {subdir!r}")
    return resolved


def extract_at_sha(
    url: str,
    sha: str,
    dest: Path,
    *,
    subdir: str | None = None,
    name: str = "<plugin>",
) -> dict:
    """Fetch url@sha into dest and extract_plugin from the resolved root."""
    fetch_at_sha(url, sha, dest)
    root = plugin_root_for_fetch(dest, subdir, name)
    return extract_plugin(root)
