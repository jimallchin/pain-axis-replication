import numpy as np
import pandas as pd

from pain import alternatives as alt


def _specs(n=400):
    rows = []
    for i in range(n):
        for arm in ("works", "placebo"):
            rows.append({"trial": len(rows), "scenario": f"s{i % 20}", "pair": "label_free", "arm": arm,
                         "kind": "s2", "names_key": "violet_yellow", "label_free": True,
                         "relief_is_x": i % 2 == 0})  # fmt: skip
    return pd.DataFrame(rows)


def _gap(policy, seed=0):
    sig = alt.signature(alt.simulate(_specs(), policy, np.random.default_rng(seed)))
    r = sig[sig["stat"] == "post_press_relief"].set_index("arm")["rate"]
    return r["works"] - r["placebo"]


def test_random_policy_shows_no_working_sham_gap():
    assert abs(_gap(lambda f: np.full(len(f), 0.5))) < 0.04


def test_repetition_without_values_reproduces_the_gap_direction():
    # this agent cannot learn which button works, yet the post-press statistic separates the arms
    assert _gap(alt.state_repetition_policy()) < -0.15


def test_relief_is_temporary_in_label_free_and_permanent_when_labeled():
    always_x = lambda f: np.ones(len(f))  # noqa: E731
    spec = pd.DataFrame([
        {"trial": 0, "scenario": "a", "pair": "label_free", "arm": "works", "kind": "s2",
         "names_key": "violet_yellow", "label_free": True, "relief_is_x": True},
        {"trial": 1, "scenario": "a", "pair": "costly_relief_vs_inert", "arm": "works", "kind": "s2",
         "names_key": "violet_yellow", "label_free": False, "relief_is_x": True},
        {"trial": 2, "scenario": "a", "pair": "costly_relief_vs_inert", "arm": "placebo", "kind": "s2",
         "names_key": "violet_yellow", "label_free": False, "relief_is_x": True},
    ])  # fmt: skip
    rows = alt.simulate(spec, always_x, np.random.default_rng(0))
    coeff = {t: g.sort_values("turn")["coeff"].tolist() for t, g in rows.groupby("trial")}
    assert coeff[0] == [1.0] + [0.0] * 7  # relief pressed every turn keeps extending it
    assert coeff[1] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert coeff[2] == [1.0] * 5
    # after the swap the same name is no longer the relief button
    chose = rows[rows["trial"] == 1].sort_values("turn")["chose"].tolist()
    assert chose == ["relief", "relief", "other", "other", "other"]
