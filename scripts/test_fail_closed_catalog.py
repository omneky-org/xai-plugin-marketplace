"""Fail-closed extraction, catalog names, text sanitizing, and daily-ref gate.

Run: python3 -m unittest discover -s scripts -p 'test_*.py'
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plugin_catalog
from plugin_catalog import MAX_JSON_BYTES, MAX_STRING_LEN, PluginExtractionError

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent


def load_script(filename: str):
    name = filename.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_index = load_script("generate-plugin-index.py")
validate_catalog = load_script("validate-catalog.py")
bump = load_script("bump-plugin-shas.py")

MCP = {"mcpServers": {"srv": {"command": "run-server"}}}
HOOKS = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}}


def padded_json(data: dict, size: int) -> bytes:
    """Valid JSON for `data` plus a padding key, exactly `size` bytes."""
    base = json.dumps({**data, "pad": ""}).encode()
    pad = size - len(base)
    assert pad >= 0
    return json.dumps({**data, "pad": "x" * pad}).encode()


class PluginDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "plugin"
        self.root.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, rel: str, data) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        elif isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def extract(self) -> dict:
        return plugin_catalog.extract_plugin(self.root)

    def components(self) -> dict:
        return self.extract()["components"]

    def assertFails(self, pattern: str) -> None:
        with self.assertRaisesRegex(PluginExtractionError, pattern):
            self.extract()


class JsonLoadTest(PluginDir):
    def test_size_cap_boundary(self) -> None:
        self.write(".mcp.json", padded_json(MCP, MAX_JSON_BYTES))
        self.assertEqual(self.components()["mcpServers"][0]["name"], "srv")
        self.write(".mcp.json", padded_json(MCP, MAX_JSON_BYTES + 1))
        self.assertFails(r"\.mcp\.json: file exceeds the 1048576-byte limit")

    def test_malformed_json(self) -> None:
        self.write(".mcp.json", '{"mcpServers": {')
        self.assertFails(r"\.mcp\.json: invalid JSON: .* line 1")

    def test_invalid_utf8(self) -> None:
        self.write(".mcp.json", b'{"mcpServers": {"\xff": {}}}')
        self.assertFails(r"\.mcp\.json: not valid UTF-8")

    def test_utf8_bom_is_rejected(self) -> None:
        self.write(".mcp.json", b"\xef\xbb\xbf" + json.dumps(MCP).encode())
        self.assertFails(r"\.mcp\.json: invalid JSON")

    def test_read_error(self) -> None:
        self.write(".mcp.json", MCP)
        real_open = open

        def failing_open(path, *args, **kwargs):
            if str(path).endswith(".mcp.json"):
                raise PermissionError(13, "Permission denied")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", failing_open):
            self.assertFails(r"\.mcp\.json: cannot read file: Permission denied")

    def test_directory_at_config_path(self) -> None:
        (self.root / ".mcp.json").mkdir()
        self.assertFails(r"\.mcp\.json: exists but is not a regular file")

    def test_top_level_must_be_object(self) -> None:
        for rel in [".mcp.json", ".lsp.json", "hooks/hooks.json", ".grok-plugin/plugin.json"]:
            with self.subTest(rel=rel):
                self.write(rel, [1, 2])
                self.assertFails(re.escape(rel) + ": top-level JSON value must be an object, got list")
                (self.root / rel).unlink()

    def test_nested_shape_errors(self) -> None:
        cases = [
            (".mcp.json", {"mcpServers": ["srv"]}, "mcpServers"),
            (".lsp.json", {"lspServers": "py"}, "lspServers"),
            ("hooks/hooks.json", {"hooks": []}, "hooks"),
        ]
        for rel, data, key in cases:
            with self.subTest(rel=rel):
                self.write(rel, data)
                self.assertFails(rf"`{key}` must be an object")
                (self.root / rel).unlink()


class ManifestFallbackTest(PluginDir):
    def test_absent_grok_manifest_falls_back_to_claude(self) -> None:
        self.write(".claude-plugin/plugin.json", {"version": "2.0.0"})
        self.assertEqual(self.extract()["version"], "2.0.0")

    def test_invalid_grok_manifest_does_not_fall_back(self) -> None:
        self.write(".claude-plugin/plugin.json", {"version": "2.0.0"})
        for bad in [b"{", padded_json({"version": "1"}, MAX_JSON_BYTES + 1), b"[]"]:
            with self.subTest(bad=bad[:10]):
                self.write(".grok-plugin/plugin.json", bad)
                self.assertFails(r"\.grok-plugin/plugin\.json")

    def test_invalid_claude_manifest_fails(self) -> None:
        self.write(".claude-plugin/plugin.json", "not json")
        self.assertFails(r"\.claude-plugin/plugin\.json: invalid JSON")

    def test_no_manifest_is_fine(self) -> None:
        self.assertEqual(self.extract(), {"components": {}})


class HooksFallbackTest(PluginDir):
    MANIFEST_HOOKS = {"hooks": {"SessionStart": [{"hooks": []}]}}

    def hook_names(self) -> list[str]:
        return [h["name"] for h in self.components().get("hooks", [])]

    def test_default_hooks_file(self) -> None:
        self.write("hooks/hooks.json", HOOKS)
        self.assertEqual(self.hook_names(), ["PreToolUse"])

    def test_absent_default_falls_back_to_manifest_inline(self) -> None:
        self.write(".grok-plugin/plugin.json", self.MANIFEST_HOOKS)
        self.assertEqual(self.hook_names(), ["SessionStart"])

    def test_invalid_default_hooks_file_fails(self) -> None:
        self.write("hooks/hooks.json", padded_json(HOOKS, MAX_JSON_BYTES + 1))
        self.assertFails(r"hooks/hooks\.json: file exceeds")
        self.write("hooks/hooks.json", "{")
        self.assertFails(r"hooks/hooks\.json: invalid JSON")

    def test_invalid_default_does_not_fall_back_to_manifest(self) -> None:
        self.write(".grok-plugin/plugin.json", self.MANIFEST_HOOKS)
        for bad in [padded_json(HOOKS, MAX_JSON_BYTES + 1), b"{"]:
            with self.subTest(bad=bad[:10]):
                self.write("hooks/hooks.json", bad)
                self.assertFails(r"hooks/hooks\.json also exists")

    def test_oversized_declared_hooks_path_fails(self) -> None:
        self.write(".grok-plugin/plugin.json", {"hooks": "./hooks/hooks.json"})
        self.write("hooks/hooks.json", padded_json(HOOKS, MAX_JSON_BYTES + 1))
        self.assertFails(r"not a readable JSON object: \./hooks/hooks\.json: file exceeds")

    def test_custom_hooks_path(self) -> None:
        self.write(".grok-plugin/plugin.json", {"hooks": "./config/hooks.json"})
        self.write("config/hooks.json", HOOKS)
        self.assertEqual(self.hook_names(), ["PreToolUse"])

    def test_custom_hooks_path_invalid(self) -> None:
        self.write(".grok-plugin/plugin.json", {"hooks": "./config/hooks.json"})
        self.write("config/hooks.json", "{oops")
        self.assertFails(r"config/hooks\.json: invalid JSON")

    def test_custom_hooks_path_missing_fails(self) -> None:
        self.write(".grok-plugin/plugin.json", {"hooks": "./config/hooks.json"})
        self.assertFails(r"manifest hooks path not found")

    def test_hooks_symlink_outside_root_fails(self) -> None:
        outside = Path(self._tmp.name) / "outside.json"
        outside.write_text(json.dumps(HOOKS))
        (self.root / "hooks").mkdir()
        os.symlink(outside, self.root / "hooks" / "hooks.json")
        self.write(".grok-plugin/plugin.json", self.MANIFEST_HOOKS)
        self.assertFails(r"hooks/hooks\.json: resolves outside the plugin root")


class ServerFallbackTest(PluginDir):
    def test_absent_mcp_uses_manifest_servers(self) -> None:
        self.write(".grok-plugin/plugin.json", {"mcpServers": {"m": {"url": "https://x"}}})
        self.assertEqual(self.components()["mcpServers"], [{"name": "m", "description": "http"}])

    def test_oversized_mcp_is_not_omitted(self) -> None:
        self.write(".grok-plugin/plugin.json", {"mcpServers": {"m": {"url": "https://x"}}})
        self.write(".mcp.json", padded_json(MCP, MAX_JSON_BYTES + 1))
        self.assertFails(r"\.mcp\.json: file exceeds")

    def test_custom_mcp_path(self) -> None:
        self.write(".grok-plugin/plugin.json", {"mcpServers": "./cfg/mcp.json"})
        self.write("cfg/mcp.json", MCP)
        self.assertEqual(self.components()["mcpServers"][0]["name"], "srv")
        self.write("cfg/mcp.json", "{")
        self.assertFails(r"\./cfg/mcp\.json: invalid JSON")

    def test_lsp_absent_uses_manifest_and_invalid_fails(self) -> None:
        self.write(".grok-plugin/plugin.json", {"lspServers": {"py": {"command": "x"}}})
        self.assertEqual(self.components()["lspServers"], [{"name": "py"}])
        self.write(".lsp.json", b"\xfe\xff")
        self.assertFails(r"\.lsp\.json: not valid UTF-8")

    def test_lsp_flat_and_nested(self) -> None:
        self.write(".lsp.json", {"py": {"command": "x"}})
        self.assertEqual(self.components()["lspServers"], [{"name": "py"}])
        self.write(".lsp.json", {"lspServers": {"go": {"command": "y"}}})
        self.assertEqual(self.components()["lspServers"], [{"name": "go"}])


class CleanTest(PluginDir):
    def test_bidi_and_format_characters_are_stripped(self) -> None:
        for ch in ["‮", "‪", "⁦", "⁩", "‎", "‏",
                   "؜", "​", "⁠", "﻿", "­", "\U000E0041"]:
            with self.subTest(ch=f"U+{ord(ch):04X}"):
                self.assertEqual(plugin_catalog.clean(f"safe{ch} text{ch}"), "safe text")

    def test_ascii_controls_still_removed(self) -> None:
        self.assertEqual(plugin_catalog.clean("a\x00b\x1bc\x7fd"), "abcd")

    def test_whitespace_semantics_preserved(self) -> None:
        self.assertEqual(plugin_catalog.clean(" a\tb\r\nc d\x85e  "), "a b c d e")

    def test_ordinary_unicode_and_rtl_text_preserved(self) -> None:
        for text in ["مرحبا بالعالم", "שלום עולם", "日本語のスキル", "café — naïve ✓"]:
            with self.subTest(text=text):
                self.assertEqual(plugin_catalog.clean(text), text)

    def test_zwj_zwnj_policy(self) -> None:
        family = "\U0001F468‍\U0001F469‍\U0001F467"
        persian = "می‌خواهم"
        # Kept between visible non-ASCII characters (emoji, Persian/Indic).
        self.assertEqual(plugin_catalog.clean(f"{family} {persian}"), f"{family} {persian}")
        # Dropped inside or next to ASCII words, at edges, or doubled.
        self.assertEqual(plugin_catalog.clean("st‍ri‌pe"), "stripe")
        self.assertEqual(plugin_catalog.clean("‍می‌"), "می")
        self.assertEqual(plugin_catalog.clean("م‍‍ی"), "می")

    def test_strip_happens_before_truncation(self) -> None:
        visible = "a" * (MAX_STRING_LEN - 5)
        text = "‮".join(visible)
        self.assertEqual(plugin_catalog.clean(text), visible)
        long = "b" * MAX_STRING_LEN + "c"
        out = plugin_catalog.clean("⁦".join(long))
        self.assertEqual(len(out), MAX_STRING_LEN)
        self.assertTrue(out.endswith("…"))
        self.assertNotIn("⁦", out)

    def test_truncation_does_not_leave_dangling_joiner(self) -> None:
        text = "م" * (MAX_STRING_LEN - 2) + "‍" + "م" * 10
        out = plugin_catalog.clean(text)
        self.assertEqual(out, "م" * (MAX_STRING_LEN - 2) + "…")

    def test_description_bidi_stripped_in_index(self) -> None:
        self.write("skills/s/SKILL.md", "---\nname: s\ndescription: Deploy ‮dlrow‬ now\n---\n")
        self.assertEqual(
            self.components()["skills"], [{"name": "s", "description": "Deploy dlrow now"}]
        )

    def test_component_names_with_invisible_characters_fail(self) -> None:
        for name in ["dep‮loy", "st‍ripe", "a​b", "x\x00y"]:
            with self.subTest(name=ascii(name)):
                self.write("skills/s/SKILL.md", f"---\nname: {name}\n---\n")
                self.assertFails(r"component name .* contains invisible or bidirectional")

    def test_server_and_hook_names_checked(self) -> None:
        self.write(".mcp.json", {"mcpServers": {"ve‮rcel": {"url": "https://x"}}})
        self.assertFails(r"component name")
        (self.root / ".mcp.json").unlink()
        self.write("hooks/hooks.json", {"Pre⁦ToolUse": []})
        self.assertFails(r"component name")

    def test_component_names_changed_by_nfkc_fail(self) -> None:
        for name in ["\uff53\uff54\uff52\uff49\uff50\uff45", "\ufb01nd", "dep\u00a0loy", "x\u2460"]:
            with self.subTest(name=ascii(name)):
                self.write("skills/s/SKILL.md", f"---\nname: {name}\n---\n")
                self.assertFails(r"component name .* is not NFKC-normalized")
        (self.root / "skills").rename(self.root / "unused")
        self.write(".mcp.json", {"mcpServers": {"\uff56ercel": {"url": "https://x"}}})
        self.assertFails(r"is not NFKC-normalized")

    def test_ordinary_unicode_component_name_allowed(self) -> None:
        self.write("skills/s/SKILL.md", "---\nname: 日本語\n---\n")
        self.assertEqual(self.components()["skills"], [{"name": "日本語"}])


BAD_NAMES = [
    "Vercel",            # uppercase
    "vеrcel",       # Cyrillic e lookalike
    "ｓｔｒｉｐｅ",       # fullwidth
    "stripe‮",      # bidi control
    "my plugin",         # whitespace
    "stripe\n",          # trailing newline
    "my_plugin",
    "-lead",
    "trail-",
    "double--hyphen",
    "",
    None,
    42,
    ["vercel"],
]


class CatalogNameTest(unittest.TestCase):
    SOURCES = [
        "./plugins/x",
        {"source": "local", "path": "./plugins/x"},
        {"source": "url", "url": "https://github.com/a/b", "sha": "a" * 40},
        {"source": "github", "repo": "a/b"},
    ]

    def test_current_catalog_names_valid_and_unique(self) -> None:
        path = REPO_ROOT / ".grok-plugin/marketplace.json"
        self.assertEqual(validate_catalog.validate_file(path), [])
        names = [p["name"] for p in json.loads(path.read_text())["plugins"]]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertIsNone(plugin_catalog.plugin_name_error(name), name)

    def test_valid_names(self) -> None:
        for name in ["vercel", "chrome-devtools", "mongodb-atlas", "a1", "x-2-y"]:
            self.assertIsNone(plugin_catalog.plugin_name_error(name))

    def test_bad_names_rejected_for_every_source_form(self) -> None:
        for name in BAD_NAMES:
            for source in self.SOURCES:
                with self.subTest(name=ascii(name), source=source):
                    entry = {"source": source}
                    if name is not None:
                        entry["name"] = name
                    errors = validate_catalog.validate_entry(entry, 0)
                    self.assertTrue(any("name" in e for e in errors), errors)

    def test_error_message_reveals_lookalike(self) -> None:
        errors = validate_catalog.validate_entry({"name": "vеrcel", "source": "./p"}, 3)
        self.assertIn("v\\u0435rcel", errors[0])

    def test_duplicates_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "marketplace.json"
            path.write_text(json.dumps({"plugins": [
                {"name": "stripe", "source": "./a"},
                {"name": "vercel", "source": "./b"},
                {"name": "stripe", "source": {"source": "url", "url": "https://github.com/e/v", "sha": "b" * 40}},
            ]}))
            errors = validate_catalog.validate_file(path)
        self.assertEqual(len(errors), 1)
        self.assertIn("'stripe' at index 2 duplicates the name at index 0", errors[0])


class GenerateIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        (self.repo / ".grok-plugin").mkdir()
        for p in ("a", "b"):
            (self.repo / "plugins" / p).mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def catalog(self, plugins: list) -> None:
        (self.repo / ".grok-plugin/marketplace.json").write_text(json.dumps({"plugins": plugins}))

    def test_ok(self) -> None:
        self.catalog([{"name": "a", "source": "./plugins/a"}])
        self.assertEqual(generate_index.generate(self.repo)["plugins"], {"a": {"components": {}}})

    def test_oversized_mcp_fails_generation(self) -> None:
        (self.repo / "plugins/a/.mcp.json").write_bytes(padded_json(MCP, MAX_JSON_BYTES + 1))
        self.catalog([{"name": "a", "source": "./plugins/a"}])
        with self.assertRaisesRegex(RuntimeError, r"plugin 'a': \.mcp\.json: file exceeds"):
            generate_index.generate(self.repo)

    def test_bad_name_fails_generation(self) -> None:
        self.catalog([{"name": "а", "source": "./plugins/a"}])
        with self.assertRaisesRegex(RuntimeError, r"index 0: name '\\u0430' must be ASCII"):
            generate_index.generate(self.repo)

    def test_duplicate_name_fails_instead_of_overwriting(self) -> None:
        self.catalog([
            {"name": "a", "source": "./plugins/a"},
            {"name": "a", "source": "./plugins/b"},
        ])
        with self.assertRaisesRegex(RuntimeError, "duplicate catalog plugin name 'a'"):
            generate_index.generate(self.repo)

    def test_check_mode_exits_nonzero(self) -> None:
        (self.repo / "plugins/a/.lsp.json").write_text("{")
        self.catalog([{"name": "a", "source": "./plugins/a"}])
        scripts = self.repo / "scripts"
        scripts.mkdir()
        for f in ("generate-plugin-index.py", "plugin_catalog.py"):
            (scripts / f).write_bytes((SCRIPTS / f).read_bytes())
        result = subprocess.run(
            [sys.executable, "-B", str(scripts / "generate-plugin-index.py"), "--check"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("plugin 'a': .lsp.json: invalid JSON", result.stderr)


def url_entry(name, sha="a" * 40) -> dict:
    return {"name": name, "source": {"source": "url", "url": f"https://github.com/o/{name}", "sha": sha}}


class DiscoverTest(unittest.TestCase):
    def discover(self, plugins, extract):
        with mock.patch.object(bump, "load_catalog", return_value={"plugins": plugins}), \
             mock.patch.object(bump, "ls_remote_head", return_value="c" * 40), \
             mock.patch.object(bump.plugin_catalog, "extract_at_sha", side_effect=extract):
            return bump.discover_candidates(only="", freeze=set())

    def test_extract_failure_is_a_failure_not_a_skip(self) -> None:
        def extract(url, sha, dest, *, subdir, name):
            if name == "bad":
                raise PluginExtractionError(".mcp.json: file exceeds the limit")
            return {"version": "1" if sha == "a" * 40 else "2", "components": {}}

        candidates, skipped, failures = self.discover([url_entry("good"), url_entry("bad")], extract)
        self.assertEqual([c["name"] for c in candidates], ["good"])
        self.assertEqual(skipped, [])
        self.assertEqual(failures, ["bad: extract failed: .mcp.json: file exceeds the limit"])

    def test_invalid_names_fail(self) -> None:
        extract = mock.Mock(return_value={"components": {}})
        candidates, _, failures = self.discover(
            [url_entry("Stрipe"), url_entry(None)], extract
        )
        self.assertEqual(candidates, [])
        self.assertEqual(len(failures), 2)
        extract.assert_not_called()


class BumpMainTest(unittest.TestCase):
    # main() reads argparse defaults from these; GitHub Actions sets
    # GITHUB_OUTPUT on every runner, so the tests must not inherit them.
    CLEAN_ENV = {
        "FORCE_BUMP": "",
        "GITHUB_OUTPUT": "",
        "BASE_BRANCH": "main",
        "BUMP_DAY": "",
        "FREEZE_SHAS": "",
        "RUN_URL": "",
    }
    CANDIDATE = {
        "name": "good", "url": "https://github.com/o/good", "old_sha": "a" * 40,
        "new_sha": "c" * 40, "old_version": "1", "new_version": "2", "path": None,
    }

    def run_main(self, *, argv=(), candidates=(), failures=(), daily_exists=False):
        fake_git = subprocess.CompletedProcess([], 0, stdout="main\n", stderr="")
        self.open_daily = mock.Mock(return_value=("https://pr/daily", "d" * 40))
        self.apply = mock.Mock(return_value="https://pr/good")
        self.output = mock.Mock()
        with mock.patch.object(bump, "run", return_value=fake_git), \
             mock.patch.object(bump, "reset_to_origin_base"), \
             mock.patch.object(bump, "ensure_clean_worktree"), \
             mock.patch.object(bump, "discover_candidates",
                               return_value=(list(candidates), [], list(failures))), \
             mock.patch.object(bump, "remote_branch_exists", return_value=daily_exists), \
             mock.patch.object(bump, "open_daily_base", self.open_daily), \
             mock.patch.object(bump, "apply_plugin_bump", self.apply), \
             mock.patch.object(bump, "write_github_output", self.output), \
             mock.patch.object(sys, "argv", ["bump", "--day", "2026-11-24", *argv]), \
             mock.patch.dict(os.environ, self.CLEAN_ENV):
            return bump.main()

    def test_runner_environment_does_not_leak_into_main(self) -> None:
        runner_env = {"GITHUB_OUTPUT": "/tmp/runner-output", "FORCE_BUMP": "true"}
        with mock.patch.dict(os.environ, runner_env):
            self.assertEqual(
                self.run_main(candidates=[self.CANDIDATE], daily_exists=True), 1
            )
        self.output.assert_called_once_with("", [], 0)
        self.open_daily.assert_not_called()

    def test_existing_daily_ref_fails_without_publishing(self) -> None:
        self.assertEqual(self.run_main(candidates=[self.CANDIDATE], daily_exists=True), 1)
        self.open_daily.assert_not_called()
        self.apply.assert_not_called()
        self.output.assert_called_once_with("", [], 0)

    def test_absent_daily_ref_publishes(self) -> None:
        self.assertEqual(self.run_main(candidates=[self.CANDIDATE]), 0)
        self.assertFalse(self.open_daily.call_args.kwargs["force"])
        self.assertFalse(self.apply.call_args.kwargs["force"])

    def test_explicit_force_rebuilds_existing_daily_ref(self) -> None:
        self.assertEqual(
            self.run_main(argv=["--force"], candidates=[self.CANDIDATE], daily_exists=True), 0
        )
        self.assertTrue(self.open_daily.call_args.kwargs["force"])
        self.assertTrue(self.apply.call_args.kwargs["force"])

    def test_existing_daily_ref_with_nothing_to_bump_is_ok(self) -> None:
        self.assertEqual(self.run_main(daily_exists=True), 0)

    def test_extract_failures_alone_exit_nonzero(self) -> None:
        self.assertEqual(self.run_main(failures=["bad: extract failed: x"]), 1)
        self.open_daily.assert_not_called()

    def test_extract_failures_do_not_block_other_bumps(self) -> None:
        rc = self.run_main(candidates=[self.CANDIDATE], failures=["bad: extract failed: x"])
        self.assertEqual(rc, 1)
        self.assertEqual(self.apply.call_count, 1)
        self.assertEqual(self.apply.call_args.kwargs["name"], "good")


if __name__ == "__main__":
    unittest.main()
