"""Baseline retrieval methods for evaluation."""

BASELINE_NAMES = [
    "no_retrieval",
    "single_rag",
    "ircot",
    "flare",
    "self_rag",
    "search_r1",
    "naive_interleave",
]

from scripts.baselines import no_retrieval  # noqa: F401, E402
from scripts.baselines import single_rag    # noqa: F401, E402
from scripts.baselines import ircot         # noqa: F401, E402
from scripts.baselines import flare         # noqa: F401, E402
from scripts.baselines import self_rag      # noqa: F401, E402
from scripts.baselines import search_r1     # noqa: F401, E402
