#!/usr/bin/env python3
"""End-to-end CPU-only ReaLM-Retrieve demo.

Run with::

    python examples/quickstart.py

or::

    make quickstart

This script walks through the *same loop* as the full pipeline
(segment → RSUS → policy → retrieve → answer) using lightweight stand-ins
defined in :mod:`realm_retrieve.toy`, so it runs in under 2 seconds on a
laptop with no GPU.
"""

from __future__ import annotations

from realm_retrieve.cli import _cmd_quickstart  # re-use the CLI logic
import argparse


if __name__ == "__main__":
    raise SystemExit(_cmd_quickstart(argparse.Namespace()))
