"""Balayages de paramètres : moyennes sur [-1, 1]^2 du bruit, biais², variance et erreur attendue."""
import json
from pathlib import Path

from bias_variance import estimate_bias_variance, mean_over_input_space
from models import get_factory

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_sweep(cache_name, method, configurations, X_test, n_samples):
    RESULTS_DIR.mkdir(exist_ok=True)
    cache_path = RESULTS_DIR / (cache_name + ".json")
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    rows = []
    for complexity, n_train, beta in configurations:
        result = estimate_bias_variance(get_factory(method, complexity), X_test,
                                        n_samples=n_samples, n_train=n_train, beta=beta)
        row = {"method": method, "complexity": complexity, "n_train": n_train, "beta": beta,
               **mean_over_input_space(result)}
        rows.append(row)
        print(f"  {method} c={complexity} N={n_train} beta={beta} -> erreur={row['expected_error']:.4f} "
              f"biais2={row['squared_bias']:.4f} var={row['variance']:.4f}", flush=True)

    cache_path.write_text(json.dumps(rows, indent=1))
    return rows
