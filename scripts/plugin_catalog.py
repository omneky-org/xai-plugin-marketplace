#!/usr/bin/env python3
"""Shared plugin extraction for marketplace catalog tooling.

One extract path given a plugin root on disk (already resolved at a known
sha / local path):

  - load manifest (.grok-plugin/plugin.json or .claude-plugin/plugin.json)
  - read version if present
  - scan components (skills, commands, agents, mcpServers, hooks, lspServers)

Used by generate-plugin-index.py (full catalog) and bump-plugin-shas.py
(version gate at HEAD vs pin).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
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

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", CONTROL_CHARS_RE.sub("", text)).strip()
    if len(text) > MAX_STRING_LEN:
        text = text[: MAX_STRING_LEN - 1].rstrip() + "\u2026"
    return text


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


def load_json_file(path: Path):
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_declared_json(root: Path, rel, what: str):
    """Load a JSON file the manifest explicitly points at, or fail.

    A manifest path is what the installer is told to load, so an escaping,
    missing, or unreadable target must not silently drop out of the index.
    """
    path = resolve_inside(root, rel)
    if path is None:
        raise RuntimeError(f"manifest {what} path escapes the plugin root: {rel!r}")
    if not path.is_file():
        raise RuntimeError(f"manifest {what} path not found: {rel!r}")
    data = load_json_file(path)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"manifest {what} path {rel!r} is not a readable JSON object "
            f"(<= {MAX_JSON_BYTES} bytes)"
        )
    return data


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
    item = {"name": clean(name)}
    description = clean(description)
    if description:
        item["description"] = description
    return item


def load_manifest(root: Path) -> dict:
    for rel in MANIFEST_PATHS:
        path = resolve_inside(root, rel)
        if path and path.is_file():
            data = load_json_file(path)
            if isinstance(data, dict):
                return data
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
    if isinstance(declared, str):
        data = load_declared_json(root, declared, "mcpServers")
    else:
        mcp_path = resolve_inside(root, ".mcp.json")
        if mcp_path and mcp_path.is_file():
            data = load_json_file(mcp_path)
    if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        servers.update(data["mcpServers"])
    if isinstance(declared, dict):
        for name, config in declared.items():
            servers.setdefault(name, config)
    return [make_item(name, transport_hint(config)) for name, config in servers.items()]


DEFAULT_HOOKS_PATH = Path("hooks") / "hooks.json"


def hook_events(data) -> dict[str, list[str]]:
    """Map event name -> matchers for one hooks config object."""
    if not isinstance(data, dict):
        return {}
    hooks_obj = data.get("hooks") if isinstance(data.get("hooks"), dict) else data
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
    configs: list = []
    if declared is None:
        hooks_path = resolve_inside(root, DEFAULT_HOOKS_PATH)
        if hooks_path and hooks_path.is_file():
            configs.append(load_json_file(hooks_path))
    else:
        declared_paths: set[Path] = set()
        for entry in declared if isinstance(declared, list) else [declared]:
            if isinstance(entry, dict):
                configs.append(entry)
            elif isinstance(entry, str):
                configs.append(load_declared_json(root, entry, "hooks"))
                declared_paths.add(resolve_inside(root, entry))
            else:
                raise RuntimeError(
                    f"manifest hooks entry must be a path or object, got "
                    f"{type(entry).__name__}"
                )
        default_path = resolve_inside(root, DEFAULT_HOOKS_PATH)
        if (
            default_path
            and default_path.is_file()
            and default_path not in declared_paths
        ):
            raise RuntimeError(
                f"manifest declares hooks but {DEFAULT_HOOKS_PATH} also exists "
                f"and is not one of the declared paths; reference it from the "
                f"manifest or remove it so the index shows every hook that "
                f"can load"
            )
    events: dict[str, list[str]] = {}
    for config in configs:
        for event, matchers in hook_events(config).items():
            events.setdefault(event, []).extend(matchers)
    return [make_item(event, ", ".join(matchers)) for event, matchers in events.items()]


def scan_lsp_servers(root: Path, manifest: dict) -> list[dict]:
    servers: dict[str, dict] = {}
    declared = manifest.get("lspServers")
    data = None
    if isinstance(declared, str):
        data = load_declared_json(root, declared, "lspServers")
    else:
        lsp_path = resolve_inside(root, ".lsp.json")
        if lsp_path and lsp_path.is_file():
            data = load_json_file(lsp_path)
    if isinstance(data, dict):
        servers.update(
            data.get("lspServers")
            if isinstance(data.get("lspServers"), dict)
            else data
        )
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
