"""Laws battery: trained models show emergent laws, random matrices do not."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from server.laws import matrix_measures, run_battery, verdicts


def test_nulls_and_trained():
    rng = np.random.default_rng(0)
    g = matrix_measures(rng.normal(0, 0.02, (256, 128)))
    assert g["spike_frac"] == 0 and abs(g["r"] - 0.53) < 0.06, g  # pure noise: no spikes, GOE spacing
    low = rng.normal(0, 0.02, (256, 128)) + 0.01 * rng.normal(0, 1, (256, 3)) @ rng.normal(0, 1, (3, 128))
    assert matrix_measures(low)["spikes"] >= 1  # a planted low-rank signal shows up as spikes
    b = run_battery("microopus:words", "microopus:words", pathlib.Path("."))
    v = verdicts(b)
    assert v["spikes"]["holds"] and v["balance"]["holds"], v
    json.dumps(b)
    print("laws ok", {k: x["holds"] for k, x in v.items()})


if __name__ == "__main__":
    test_nulls_and_trained()
