"""Smoke tests for the CLI entry points."""

from __future__ import annotations

import pytest

from realm_retrieve.cli import main


def test_help_runs(capsys):
    # argparse calls sys.exit(0) on --help; we treat that as success.
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "realm-retrieve" in captured.out


def test_version_command(capsys):
    rc = main(["version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "realm-retrieve" in captured.out


def test_quickstart_runs(capsys):
    rc = main(["quickstart"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ReaLM-Retrieve" in captured.out
    assert "EM" in captured.out and "F1" in captured.out


def test_no_args_prints_help(capsys):
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    # Should be a help-like output
    assert "quickstart" in captured.out
