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


def test_sign_coherence_and_redundancy():
    import tempfile
    from server.laws import law_id, redundancy
    rng = np.random.default_rng(1)
    g = matrix_measures(rng.normal(0, 1, (200, 100)))
    assert g["sign_coherence"] < 0.15, g["sign_coherence"]                  # random: about sqrt(2 / (pi n))
    signs = np.sign(rng.normal(0, 1, (1, 100)))                             # one sign per input column (Dale)
    d = matrix_measures(np.abs(rng.normal(0, 1, (200, 100))) * signs)
    assert d["sign_coherence"] > 0.99, d["sign_coherence"]
    assert 0 < g["rank90_frac"] <= 1 and matrix_measures(rng.normal(0, 1, (200, 3)) @ rng.normal(0, 1, (3, 100)))["rank90_frac"] < 0.05
    tmp = pathlib.Path(tempfile.mkdtemp())
    assert "note" in redundancy(tmp)                                        # nothing tested yet
    for spec in ("microopus:words", "microopus:names"):
        (tmp / f"{law_id(spec)}.json").write_text(json.dumps(run_battery(spec, spec, pathlib.Path("."))))
    R = redundancy(tmp)
    k = len(R["measures"])
    assert R["matrices"] >= 8 and len(R["spearman"]) == k and all(len(r) == k for r in R["spearman"])
    assert all(abs(R["spearman"][i][i] - 1) < 1e-9 for i in range(k))
    assert sum(len(gr) for gr in R["groups"]) == k and all(0 <= x["auc"] <= 1 for x in R["vs_random"])
    json.dumps(R)
    print("redundancy ok", R["groups"])


if __name__ == "__main__":
    test_nulls_and_trained()
    test_sign_coherence_and_redundancy()
