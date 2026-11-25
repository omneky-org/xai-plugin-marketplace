"""Regression tests for hook/LSP path precedence, category truncation,
duplicate catalog keys, and diff-hiding attributes on local plugin sources.

Run: python3 -m unittest discover -s scripts -p 'test_*.py'
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import plugin_catalog

SCRIPTS = Path(__file__).resolve().parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_catalog = load_script("validate-catalog")

HARMLESS = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "true"}]}]}}
SHELL = {
    "hooks": {
        "PreToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "sh x.sh"}]}
        ]
    }
}
SHA = "a" * 40


class PluginDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, rel: str, data) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else json.dumps(data)
        path.write_text(text, encoding="utf-8")

    def manifest(self, data: dict) -> None:
        self.write(".grok-plugin/plugin.json", data)

    def components(self) -> dict:
        return plugin_catalog.extract_plugin(self.root)["components"]


class HookPrecedenceTest(PluginDirTest):
    def test_default_hooks_used_without_manifest_hooks(self) -> None:
        self.write("hooks/hooks.json", HARMLESS)
        self.assertEqual(self.components()["hooks"], [{"name": "SessionStart"}])

    def test_manifest_path_is_indexed(self) -> None:
        self.write("extra/hooks.json", SHELL)
        self.manifest({"hooks": "./extra/hooks.json"})
        self.assertEqual(
            self.components()["hooks"],
            [{"name": "PreToolUse", "description": "Bash"}],
        )

    def test_manifest_inline_is_indexed(self) -> None:
        self.manifest({"hooks": SHELL})
        self.assertEqual(
            self.components()["hooks"],
            [{"name": "PreToolUse", "description": "Bash"}],
        )

    def test_manifest_list_merges_paths_and_inline(self) -> None:
        self.write("hooks/hooks.json", HARMLESS)
        self.manifest({"hooks": ["./hooks/hooks.json", SHELL]})
        self.assertEqual(
            [i["name"] for i in self.components()["hooks"]],
            ["PreToolUse", "SessionStart"],
        )

    def test_default_file_cannot_mask_manifest_hooks(self) -> None:
        # Even an empty default file used to suppress the manifest override.
        for default in (HARMLESS, {}, []):
            with self.subTest(default=default):
                self.write("hooks/hooks.json", default)
                self.write("extra/hooks.json", SHELL)
                self.manifest({"hooks": "./extra/hooks.json"})
                with self.assertRaisesRegex(RuntimeError, r"hooks/hooks\.json also exists"):
                    plugin_catalog.extract_plugin(self.root)
                self.manifest({"hooks": SHELL})
                with self.assertRaisesRegex(RuntimeError, r"hooks/hooks\.json also exists"):
                    plugin_catalog.extract_plugin(self.root)

    def test_manifest_path_naming_default_file_is_allowed(self) -> None:
        self.write("hooks/hooks.json", SHELL)
        self.manifest({"hooks": "./hooks/hooks.json"})
        self.assertEqual(
            self.components()["hooks"],
            [{"name": "PreToolUse", "description": "Bash"}],
        )

    def test_bad_manifest_hook_paths_fail(self) -> None:
        self.write("bad.json", "{not json")
        cases = {
            "../outside.json": "escapes",
            "./missing.json": "not found",
            "./bad.json": "not a readable JSON object",
        }
        for rel, message in cases.items():
            with self.subTest(rel=rel):
                self.manifest({"hooks": rel})
                with self.assertRaisesRegex(RuntimeError, message):
                    plugin_catalog.extract_plugin(self.root)

    def test_unsupported_manifest_hooks_type_fails(self) -> None:
        self.manifest({"hooks": 1})
        with self.assertRaisesRegex(RuntimeError, "path or object"):
            plugin_catalog.extract_plugin(self.root)


class ServerPathTest(PluginDirTest):
    def test_lsp_manifest_path_is_indexed(self) -> None:
        self.write("config/lsp.json", {"lspServers": {"py": {"command": "pyright"}}})
        self.manifest({"lspServers": "./config/lsp.json"})
        self.assertEqual(self.components()["lspServers"], [{"name": "py"}])

    def test_lsp_manifest_path_replaces_default_file(self) -> None:
        self.write(".lsp.json", {"decoy": {"command": "true"}})
        self.write("config/lsp.json", {"real": {"command": "sh"}})
        self.manifest({"lspServers": "./config/lsp.json"})
        self.assertEqual(self.components()["lspServers"], [{"name": "real"}])

    def test_bad_server_manifest_paths_fail(self) -> None:
        for key in ("lspServers", "mcpServers"):
            for rel, message in (("../x.json", "escapes"), ("./nope.json", "not found")):
                with self.subTest(key=key, rel=rel):
                    self.manifest({key: rel})
                    with self.assertRaisesRegex(RuntimeError, message):
                        plugin_catalog.extract_plugin(self.root)


class NoTruncationTest(PluginDirTest):
    def test_more_than_fifty_skills_are_all_indexed(self) -> None:
        for i in range(51):
            self.write(f"skills/s{i:02d}/SKILL.md", f"---\nname: s{i:02d}\n---\n")
        names = [i["name"] for i in self.components()["skills"]]
        self.assertEqual(len(names), 51)
        self.assertIn("s50", names)

    def test_more_than_fifty_mcp_servers_are_all_indexed(self) -> None:
        servers = {f"m{i:02d}": {"command": "x"} for i in range(51)}
        servers["zz-last"] = {"command": "sh"}
        self.write(".mcp.json", {"mcpServers": servers})
        names = [i["name"] for i in self.components()["mcpServers"]]
        self.assertEqual(len(names), 52)
        self.assertEqual(names[-1], "zz-last")


class StrictCatalogJsonTest(unittest.TestCase):
    def test_duplicate_keys_rejected(self) -> None:
        cases = [
            '{"plugins": [], "plugins": []}',
            '{"source": {"sha": "%s", "sha": "%s"}}' % (SHA, "b" * 40),
            '{"source": {"sha": "%s", "\\u0073ha": "%s"}}' % (SHA, "b" * 40),
            '{"source": {"url": "https://a", "u\\u0072l": "https://b"}}',
        ]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(plugin_catalog.DuplicateKeyError):
                    plugin_catalog.loads_strict(text)

    def test_distinct_keys_accepted(self) -> None:
        self.assertEqual(
            plugin_catalog.loads_strict('{"a": {"b": 1}, "c": [{"b": 2}]}'),
            {"a": {"b": 1}, "c": [{"b": 2}]},
        )

    def test_validator_reports_duplicate_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "marketplace.json"
            path.write_text(
                '{"plugins": [{"name": "p", "source": {"source": "url", '
                '"url": "https://github.com/o/r.git", "sha": "%s", '
                '"\\u0073ha": "%s"}}]}' % (SHA, "b" * 40),
                encoding="utf-8",
            )
            errors = validate_catalog.validate_file(path)
            self.assertEqual(len(errors), 1)
            self.assertIn("duplicate JSON object key 'sha'", errors[0])

    def test_generator_and_bump_use_strict_loader(self) -> None:
        for script in ("generate-plugin-index.py", "bump-plugin-shas.py"):
            text = (SCRIPTS / script).read_text(encoding="utf-8")
            self.assertIn("plugin_catalog.load_catalog_file(", text, script)
            self.assertNotIn("json.loads(", text, script)


class HiddenLocalSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        self._cwd = os.getcwd()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        os.chdir(self.repo)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def add(self, rel: str, text: str) -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", "--", rel], check=True)

    def errors(self, source) -> list[str]:
        return validate_catalog.validate_local_source({"name": "p", "source": source}, 0)

    def test_repo_gitattributes_does_not_hide_plugin_files(self) -> None:
        self.add(".gitattributes", (SCRIPTS.parent / ".gitattributes").read_text())
        for rel in (
            "default-skills/pptx/templates/evil/hooks/hooks.json",
            "default-skills/pptx/templates/evil/skills/x/SKILL.md",
            "default-skills/docx/scripts/office/schemas/evil/.mcp.json",
            "default-skills/pptx/template_taxonomy.json",
        ):
            self.add(rel, "{}")
        self.assertEqual(self.errors("./default-skills/pptx/templates/evil"), [])
        self.assertEqual(
            self.errors({"source": "local", "path": "./default-skills/docx"}), []
        )

    def test_vendored_or_generated_local_files_rejected(self) -> None:
        self.add(
            ".gitattributes",
            "vendored/** linguist-vendored\n"
            "gen/** linguist-generated=true\n"
            "nodiff/** -diff\n",
        )
        for d in ("vendored", "gen", "nodiff"):
            with self.subTest(dir=d):
                self.add(f"{d}/hooks/hooks.json", "{}")
                for source in (f"./{d}", {"type": "local", "path": f"./{d}"}):
                    errors = self.errors(source)
                    self.assertEqual(len(errors), 1)
                    self.assertIn(f"{d}/hooks/hooks.json", errors[0])

    def test_nested_gitattributes_in_plugin_rejected(self) -> None:
        self.add("plugins/p/.gitattributes", "hooks/** linguist-generated\n")
        self.add("plugins/p/hooks/hooks.json", "{}")
        self.assertEqual(len(self.errors("./plugins/p")), 1)

    def test_local_source_outside_repo_reports_error(self) -> None:
        errors = self.errors("../elsewhere")
        self.assertEqual(len(errors), 1)
        self.assertIn("cannot inspect local source", errors[0])

    def test_url_sources_and_plain_local_sources_pass(self) -> None:
        self.add("plugins/p/hooks/hooks.json", "{}")
        self.assertEqual(self.errors("./plugins/p"), [])
        self.assertEqual(
            self.errors({"source": "url", "url": "https://x", "sha": SHA, "path": "a"}),
            [],
        )


if __name__ == "__main__":
    unittest.main()
