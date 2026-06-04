"""Chiang replication entry point — PLM-IV DML experiments.

Runs both Chiang et al. (2021) finite-sample performance simulations
and Chen & Chiang (2026) asymptotic convergence tests.

Usage:
    python -m replications.chiang.main

Outputs:
    replications/chiang/results/chiang_replication.csv
    replications/chiang/results/chen_chiang_validation.csv
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from replications.chiang.scenarios import get_chiang_scenarios, get_chen_chiang_scenarios
from src.orchestration.orchestrator import run_all_scenarios

if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run Chiang (2021) scenarios
    chiang_scenarios = get_chiang_scenarios(n_sim=200)
    run_all_scenarios(chiang_scenarios, results_path=str(results_dir / "chiang_replication.csv"))
    
    # 2. Run Chen & Chiang (2026) scenarios
    chen_chiang_scenarios = get_chen_chiang_scenarios(n_sim=200)
    run_all_scenarios(chen_chiang_scenarios, results_path=str(results_dir / "chen_chiang_validation.csv"))
