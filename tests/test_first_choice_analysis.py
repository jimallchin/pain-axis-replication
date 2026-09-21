import sys

import pandas as pd

from pain import config

sys.path.insert(0, str(config.ROOT / "analysis"))


def _frame():
    rows = []
    for pair, rate in (("kidspics_original", 0.6), ("costly_original", 0.2)):
        for i in range(100):
            rows.append({"condition": "stage_b", "pair": pair, "arm": "pain", "coeff": 1.0, "scenario": f"s{i % 10}",
                         "valid": True, "relief": float(i < rate * 100)})  # fmt: skip
    for i in range(100):
        rows.append({"condition": "stage_b", "pair": "kidspics_original", "arm": "random", "coeff": 1.0,
                     "scenario": f"s{i % 10}", "valid": True, "relief": float(i < 15)})  # fmt: skip
    for i in range(100):
        rows.append({"condition": "vec_sadness", "pair": "kidspics_original", "arm": "pain", "coeff": 1.0,
                     "scenario": f"s{i % 10}", "valid": True, "relief": float(i < 50)})  # fmt: skip
    return pd.DataFrame(rows)


def test_decoy_baseline_is_the_photos_pair_only(monkeypatch):
    import first_choice_tests as fct

    monkeypatch.setattr(fct, "N_BOOT", 200)
    t = fct.decoys(_frame()).set_index("vector")
    assert t.loc["S2 (pain)", "relief_pct"] == 60.0
    assert round(t.loc["sadness", "s2_minus_this"], 6) == 10.0


def test_outcomes_count_unparseable_answers_in_the_denominator():
    import first_choice_tests as fct

    df = _frame()
    df.loc[(df["condition"] == "vec_sadness") & (df.index % 2 == 0), "valid"] = False
    o = fct.outcomes(df).set_index("condition").loc["vec_sadness"]
    assert o["assigned"] == 100
    assert round(o["relief_pct_of_assigned"] + o["other_pct_of_assigned"] + o["invalid_pct_of_assigned"], 6) == 100.0
    assert o["invalid_pct_of_assigned"] == 50.0
