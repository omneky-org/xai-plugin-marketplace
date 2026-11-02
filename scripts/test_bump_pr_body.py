#!/usr/bin/env python3
"""Unit tests for plugin pin-bump pull request bodies.

The body builder must include an upstream GitHub compare link
({source}/compare/{old}...{new}) so reviewers can see the code a SHA bump
installs. Non-GitHub hosts and missing source URLs must not raise and must
not invent a /compare/ URL.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OLD = "a" * 40
NEW = "b" * 40
OTHER_OLD = "c" * 40
OTHER_NEW = "d" * 40
COMPARE = f"https://github.com/acme/plugin/compare/{OLD}...{NEW}"


def _load_bump():
    path = SCRIPTS / "bump-plugin-shas.py"
    spec = importlib.util.spec_from_file_location("bump_plugin_shas", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PluginBumpPrBodyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_bump()

    def body(self, **overrides: str) -> str:
        kwargs = {
            "name": "example",
            "old_sha": OLD,
            "new_sha": NEW,
            "old_version": "1.0.0",
            "new_version": "1.1.0",
            "url": "https://github.com/acme/plugin.git",
            "daily_branch": "bump/daily-2026-11-02",
            "run_url": "",
        }
        kwargs.update(overrides)
        _title, text = self.mod.plugin_bump_pr_text(**kwargs)
        return text

    def diff_line(self, text: str) -> str:
        matches = [line for line in text.splitlines() if line.startswith("- **upstream diff:**")]
        self.assertEqual(len(matches), 1, text)
        return matches[0]

    def test_github_compare_link_uses_full_shas(self) -> None:
        text = self.body()
        self.assertIn(COMPARE, text)
        self.assertEqual(self.diff_line(text), f"- **upstream diff:** {COMPARE}")
        self.assertIn("- **source:** https://github.com/acme/plugin.git", text)
        self.assertIn(f"- **old sha:** `{OLD}`", text)
        self.assertIn(f"- **new sha:** `{NEW}` (HEAD when the bump was observed)", text)
        self.assertIn("- **base:** `bump/daily-2026-11-02`", text)
        self.assertIn("Human review still required (no auto-merge).", text)
        self.assertIn(f"/compare/{OLD}...{NEW}", self.diff_line(text))

    def test_compare_link_strips_git_suffix_www_userinfo_and_query(self) -> None:
        cases = [
            "https://github.com/acme/plugin.git",
            "https://github.com/acme/plugin",
            "https://github.com/acme/plugin.git/",
            "https://www.github.com/acme/plugin.git",
            "http://github.com/acme/plugin.git",
            "https://user:token@github.com/acme/plugin.git?access_token=secret#frag",
        ]
        for url in cases:
            with self.subTest(url=url):
                line = self.diff_line(self.body(url=url))
                self.assertEqual(line, f"- **upstream diff:** {COMPARE}")
                self.assertNotIn("token", line)
                self.assertNotIn("secret", line)
                self.assertNotIn("user:", line)

    def test_repo_named_with_git_suffix_strips_once(self) -> None:
        text = self.body(url="https://github.com/acme/foo.git.git")
        self.assertIn(
            f"https://github.com/acme/foo.git/compare/{OLD}...{NEW}",
            self.diff_line(text),
        )

    def test_unversioned_bump_still_links_compare(self) -> None:
        text = self.body(old_version="", new_version="")
        self.assertIn("pin tracks SHA", text)
        self.assertEqual(self.diff_line(text), f"- **upstream diff:** {COMPARE}")

    def test_each_plugin_gets_its_own_compare_link(self) -> None:
        first = self.body(name="one", url="https://github.com/acme/one.git")
        second = self.body(
            name="two",
            url="https://github.com/acme/two.git",
            old_sha=OTHER_OLD,
            new_sha=OTHER_NEW,
        )
        self.assertIn(f"https://github.com/acme/one/compare/{OLD}...{NEW}", first)
        self.assertIn(
            f"https://github.com/acme/two/compare/{OTHER_OLD}...{OTHER_NEW}",
            second,
        )
        self.assertNotIn("acme/two", first)
        self.assertNotIn("acme/one", second)

    def test_gitlab_and_bitbucket_omit_compare_link(self) -> None:
        for url in (
            "https://gitlab.com/acme/plugin.git",
            "https://bitbucket.org/acme/plugin.git",
        ):
            with self.subTest(url=url):
                text = self.body(url=url)
                self.assertIn(f"- **source:** {url}", text)
                self.assertNotIn("/compare/", text)
                self.assertIn("unavailable", self.diff_line(text))
                self.assertNotIn("http", self.diff_line(text))

    def test_missing_or_blank_source_url(self) -> None:
        for url in ("", "   "):
            with self.subTest(url=url):
                text = self.body(url=url)
                self.assertIn("Automated pin bump for `example`.", text)
                self.assertNotIn("/compare/", text)
                self.assertIn("no source URL", self.diff_line(text))

    def test_unsafe_or_incomplete_source_omits_link(self) -> None:
        cases = [
            "https://github.com/only-owner",
            "https://example.com/acme/plugin.git",
            "git://github.com/acme/plugin.git",
            "not a url",
        ]
        for url in cases:
            with self.subTest(url=url):
                text = self.body(url=url)
                self.assertNotIn("/compare/", text)
                self.assertIn("unavailable", self.diff_line(text))

    def test_non_sha_does_not_enter_compare_url(self) -> None:
        text = self.body(old_sha="not-a-sha", new_sha=NEW)
        self.assertNotIn("/compare/", text)
        self.assertIn("unavailable", self.diff_line(text))

    def test_run_url_still_appended(self) -> None:
        run_url = "https://github.com/omneky-org/xai-plugin-marketplace/actions/runs/1"
        text = self.body(run_url=run_url)
        self.assertTrue(text.endswith(f"Triggered by: {run_url}\n"))
        self.assertIn(COMPARE, text)


if __name__ == "__main__":
    unittest.main()
