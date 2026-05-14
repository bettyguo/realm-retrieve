"""Command-line entry points for ReaLM-Retrieve.

Two scripts are exposed via ``pyproject.toml``:

    realm-retrieve      → the top-level CLI (sub-commands).
    realm-quickstart    → a shortcut that runs the toy end-to-end demo.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from realm_retrieve import __version__


def _cmd_quickstart(_: argparse.Namespace) -> int:
    from realm_retrieve.toy import (
        ToyPipeline,
        ToyReasoningModel,
        ToyRetriever,
        demo_corpus,
        demo_questions,
    )

    print("ReaLM-Retrieve · CPU quickstart")
    print("=" * 60)

    retriever = ToyRetriever(demo_corpus())
    reasoner = ToyReasoningModel(seed=42)
    pipeline = ToyPipeline(retriever, reasoner, rsus_threshold=0.5)

    f1_sum = 0.0
    retrieval_sum = 0
    correct = 0
    questions = demo_questions()

    for i, (q, gold) in enumerate(questions, start=1):
        result = pipeline.answer(q, gold=gold)
        f1_sum += result.f1
        retrieval_sum += result.retrievals
        correct += int(result.correct)
        check = "OK" if result.correct else "  "
        print(
            f"[{i}/{len(questions)}] {check}  "
            f"rsus={['%.2f' % s for s in result.rsus_per_step]}  "
            f"retr={result.retrievals}  "
            f"→ {result.answer!r:25s} (gold {result.gold!r})"
        )

    n = len(questions)
    print("-" * 60)
    print(
        f"EM  {correct / n * 100:5.1f}  |  "
        f"F1  {f1_sum / n * 100:5.1f}  |  "
        f"retrievals/q  {retrieval_sum / n:.2f}"
    )
    print("=" * 60)
    print("Swap ToyRetriever → ColBERTRetriever and ToyReasoningModel →")
    print("VLLMReasoningModel to scale this up to the full pipeline.")
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"realm-retrieve {__version__}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="realm-retrieve",
        description="Adaptive retrieval for Large Reasoning Models (SIGIR 2026).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = p.add_subparsers(dest="command", required=False)

    pq = sub.add_parser("quickstart", help="Run the CPU-only toy pipeline demo.")
    pq.set_defaults(func=_cmd_quickstart)

    pv = sub.add_parser("version", help="Print the installed version.")
    pv.set_defaults(func=_cmd_version)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


def quickstart(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``realm-quickstart`` script."""
    return _cmd_quickstart(argparse.Namespace())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
