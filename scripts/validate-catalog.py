#!/usr/bin/env python3
"""Validate the marketplace catalog index.

Enforces, for every plugin:

  - `name` is an ASCII lowercase kebab-case string (no Unicode lookalikes)
  - `name` is unique within the catalog file

and for every plugin with `"source": {"source": "url", ...}`:

  - `sha` field is present and non-empty
  - `sha` is a 40-character lowercase hex string (full commit SHA, not a
    tag, branch, or abbreviation)

Also enforces, for the whole catalog:

  - no duplicate JSON object keys (including escaped-equivalent keys), so
    the value a reviewer reads is the value tooling and clients use
  - local plugin sources contain no files that .gitattributes hides in PR
    diffs (linguist-generated, linguist-vendored, -diff)

This is the catalog-level enforcement layer for SHA pinning. Without a
pin, the installer would fall back to `git clone --branch <ref>` (or HEAD),
which means a vendor force-push or repo compromise immediately ships to
every user who installs or updates that plugin. Pinning to a specific
commit + content-verifying it at install time is the only thing that
survives that class of attack.

The runtime side (the Grok CLI plugin installer) verifies
`git rev-parse HEAD == sha` after clone — these two layers together give
us content-addressable plugin pinning.

Run locally:    python3 scripts/validate-catalog.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import plugin_catalog

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Lookup order matches the marketplace index loader in the Grok CLI.
CATALOG_PATHS = [
    Path(".grok-plugin/marketplace.json"),
    Path(".claude-plugin/marketplace.json"),
]


def validate_entry(entry: dict, idx: int) -> list[str]:
    """Return a list of human-readable error strings for a single plugin entry."""
    errors: list[str] = []
    raw_name = entry.get("name")
    name_error = plugin_catalog.plugin_name_error(raw_name)
    if name_error:
        errors.append(f"plugin at index {idx}: {name_error}")
        name = ascii(raw_name) if raw_name else f"<unnamed at index {idx}>"
    else:
        name = raw_name
    source = entry.get("source")

    # String-form sources like "./plugins/foo" are local paths; no sha needed.
    if not isinstance(source, dict):
        return errors

    if source.get("source") != "url":
        return errors

    sha = source.get("sha")
    if not sha:
        errors.append(
            f"plugin '{name}': missing `sha` field on url source "
            f"(url={source.get('url')!r}). All url-sourced plugins must "
            f"be pinned to a specific commit so a vendor force-push can't "
            f"silently ship new code to installed users."
        )
        return errors

    if not isinstance(sha, str):
        errors.append(
            f"plugin '{name}': sha must be a string, got {type(sha).__name__}"
        )
        return errors

    if not SHA_RE.match(sha):
        errors.append(
            f"plugin '{name}': sha {sha!r} is not a 40-character lowercase "
            f"hex string. Use the full commit SHA — not a tag, branch, or "
            f"abbreviated SHA."
        )

    path = source.get("path")
    if path is not None:
        if not isinstance(path, str) or not path.strip():
            errors.append(
                f"plugin '{name}': url source `path` must be a non-empty string when present."
            )
        elif (
            path.startswith("/")
            or "\\" in path
            or any(part in ("..", "") for part in path.split("/"))
        ):
            errors.append(
                f"plugin '{name}': url source `path` {path!r} must be a relative "
                f"subdirectory inside the repo (no leading '/', no '..', no backslashes)."
            )

    return errors


# Git attributes that hide file contents in GitHub PR diffs. Local plugin
# files are reviewed in this repo's diff, so none of them may carry these.
HIDING_ATTRS = ("linguist-generated", "linguist-vendored", "diff")


def local_source_path(source) -> str | None:
    if isinstance(source, str):
        return source
    if (
        isinstance(source, dict)
        and source.get("source") != "url"
        and source.get("type") != "url"
        and isinstance(source.get("path"), str)
    ):
        return source["path"]
    return None


def hidden_files(rel_root: str) -> list[str]:
    """Tracked files under rel_root whose diffs GitHub collapses or hides."""
    files = subprocess.run(
        ["git", "ls-files", "-z", "--", rel_root],
        capture_output=True,
        check=True,
    ).stdout
    if not files:
        return []
    out = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", *HIDING_ATTRS],
        input=files,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", "replace")
    fields = out.split("\0")
    hidden: set[str] = set()
    for i in range(0, len(fields) - 2, 3):
        path, attr, value = fields[i : i + 3]
        if attr == "diff" and value == "unset":
            hidden.add(path)
        elif attr != "diff" and value not in ("unspecified", "unset", "false"):
            hidden.add(path)
    return sorted(hidden)


def validate_local_source(entry: dict, idx: int) -> list[str]:
    rel = local_source_path(entry.get("source"))
    if rel is None:
        return []
    name = entry.get("name") or f"<unnamed at index {idx}>"
    try:
        hidden = hidden_files(rel)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", "replace").strip()
        return [f"plugin '{name}': cannot inspect local source {rel!r}: {detail}"]
    if not hidden:
        return []
    shown = ", ".join(hidden[:5]) + (" ..." if len(hidden) > 5 else "")
    return [
        f"plugin '{name}': local source {rel!r} contains files marked "
        f"{'/'.join(HIDING_ATTRS)} in .gitattributes, which hides them in PR "
        f"diffs: {shown}"
    ]


def validate_file(path: Path) -> list[str]:
    try:
        data = plugin_catalog.load_catalog_file(path)
    except Exception as e:
        return [f"{path}: failed to parse: {e}"]
    if not isinstance(data, dict):
        return [f"{path}: catalog must be a JSON object"]

    plugins = data.get("plugins", [])
    if not isinstance(plugins, list):
        return [f"{path}: `plugins` must be an array, got {type(plugins).__name__}"]

    errors: list[str] = []
    seen: dict[str, int] = {}
    for idx, entry in enumerate(plugins):
        if not isinstance(entry, dict):
            errors.append(f"{path}: plugin index {idx} must be an object")
            continue
        errors.extend(f"{path}: {e}" for e in validate_entry(entry, idx))
        errors.extend(f"{path}: {e}" for e in validate_local_source(entry, idx))
        name = entry.get("name")
        if isinstance(name, str):
            if name in seen:
                errors.append(
                    f"{path}: plugin '{name}' at index {idx} duplicates the "
                    f"name at index {seen[name]}"
                )
            else:
                seen[name] = idx
    return errors


def main() -> int:
    catalog_files = [p for p in CATALOG_PATHS if p.exists()]
    if not catalog_files:
        print(
            "ERROR: no catalog file found. Expected one of: "
            + ", ".join(str(p) for p in CATALOG_PATHS),
            file=sys.stderr,
        )
        return 1

    all_errors: list[str] = []
    for path in catalog_files:
        all_errors.extend(validate_file(path))

    if all_errors:
        print("Catalog validation failed:", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    summary = " + ".join(str(p) for p in catalog_files)
    print(f"Catalog OK ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
