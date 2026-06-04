"""Entry point for Balkus replication.

Uses DML estimator faithful to the R code in hard_cluster.R.

Usage:
    python -m replications.balkus.main

Output:
    replications/balkus/results/balkus_results.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from replications.balkus.scenarios import get_balkus_scenarios
from src.orchestration.orchestrator import run_all_scenarios

if __name__ == "__main__":
    results_path = Path(__file__).parent / "results" / "balkus_results.csv"
    scenarios = get_balkus_scenarios(n_sim=200)
    run_all_scenarios(scenarios, results_path=str(results_path))
