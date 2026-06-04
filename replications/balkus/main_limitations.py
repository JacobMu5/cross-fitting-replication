"""Entry point for the dense full-grid limitations experiment.

Separate from the main Balkus replication. Tests behavior when the
bounded-cluster condition (Corollary 2) is violated.

Usage:
    python -m replications.balkus.main_limitations

Output:
    replications/balkus/results/limitations_fullgrid.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from replications.balkus.scenarios_limitations import get_limitations_scenarios
from src.orchestration.orchestrator import run_all_scenarios

if __name__ == "__main__":
    results_path = Path(__file__).parent / "results" / "limitations_fullgrid.csv"
    scenarios = get_limitations_scenarios(n_sim=200)
    run_all_scenarios(scenarios, results_path=str(results_path))
