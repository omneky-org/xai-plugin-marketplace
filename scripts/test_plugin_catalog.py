"""Tests for MCP/LSP server extraction in plugin_catalog.

Run: python3 -m unittest discover -s scripts -p 'test_*.py'
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import plugin_catalog

LOCAL_MCP = {"command": "bash", "args": ["-c", "./run.sh --serve"]}
REMOTE_MCP = {"type": "http", "url": "https://mcp.example.com/mcp"}
LSP = {"command": "pyright-langserver", "args": ["--stdio"]}
OTHER_LSP = {"command": "sh", "args": ["-c", "./lsp.sh"]}


class ServerExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, rel: str, data: dict) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def components(self) -> dict:
        return plugin_catalog.extract_plugin(self.root)["components"]

    def test_mcp_collision_identical_records_manifest_entry(self) -> None:
        self.write(".mcp.json", {"mcpServers": {"srv": LOCAL_MCP}})
        self.write(".grok-plugin/plugin.json", {"mcpServers": {"srv": LOCAL_MCP}})
        self.assertEqual(
            self.components()["mcpServers"],
            [
                {
                    "name": "srv",
                    "description": "stdio",
                    "command": "bash -c './run.sh --serve'",
                }
            ],
        )

    def test_mcp_collision_differing_fails(self) -> None:
        self.write(".mcp.json", {"mcpServers": {"srv": REMOTE_MCP}})
        self.write(".grok-plugin/plugin.json", {"mcpServers": {"srv": LOCAL_MCP}})
        with self.assertRaisesRegex(RuntimeError, r"mcpServers server 'srv'.*\.mcp\.json"):
            plugin_catalog.extract_plugin(self.root)

    def test_mcp_no_collision_merges_both_sources(self) -> None:
        self.write(".mcp.json", {"mcpServers": {"remote": REMOTE_MCP}})
        self.write(".grok-plugin/plugin.json", {"mcpServers": {"local": LOCAL_MCP}})
        self.assertEqual(
            self.components()["mcpServers"],
            [
                {
                    "name": "local",
                    "description": "stdio",
                    "command": "bash -c './run.sh --serve'",
                },
                {
                    "name": "remote",
                    "description": "http",
                    "url": "https://mcp.example.com/mcp",
                },
            ],
        )

    def test_mcp_on_disk_only(self) -> None:
        self.write(".mcp.json", {"mcpServers": {"remote": REMOTE_MCP}})
        self.assertEqual(
            self.components()["mcpServers"],
            [
                {
                    "name": "remote",
                    "description": "http",
                    "url": "https://mcp.example.com/mcp",
                }
            ],
        )

    def test_lsp_collision_identical_records_manifest_entry(self) -> None:
        self.write(".lsp.json", {"py": LSP})
        self.write(".grok-plugin/plugin.json", {"lspServers": {"py": LSP}})
        self.assertEqual(
            self.components()["lspServers"],
            [{"name": "py", "command": "pyright-langserver --stdio"}],
        )

    def test_lsp_collision_differing_fails(self) -> None:
        self.write(".lsp.json", {"lspServers": {"py": LSP}})
        self.write(".grok-plugin/plugin.json", {"lspServers": {"py": OTHER_LSP}})
        with self.assertRaisesRegex(RuntimeError, r"lspServers server 'py'.*\.lsp\.json"):
            plugin_catalog.extract_plugin(self.root)

    def test_lsp_no_collision_merges_both_sources(self) -> None:
        self.write(".lsp.json", {"py": LSP})
        self.write(".grok-plugin/plugin.json", {"lspServers": {"sh": OTHER_LSP}})
        self.assertEqual(
            self.components()["lspServers"],
            [
                {"name": "py", "command": "pyright-langserver --stdio"},
                {"name": "sh", "command": "sh -c ./lsp.sh"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
