"""The released two-button trial logs: the authors' tables, then finer cuts of the same trials.

`tables` follows 05_selfmed_analysis.py line for line so the published numbers can be checked
against the shipped logs. Everything after it re-cuts the same choices by the steering state
the model was in when it chose, which the published aggregation does not separate.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from pain.upstream import ARM_SHORT, BUTTON_NAMES, COSTED, PAIN_ARMS, PAIRS, scenario_id, trial_key


def pct(k, n):
    return round(100 * k / n, 1) if n else float("nan")


def first_choice(r):
    for c in r.get("choices", []):
        if c["turn"] == 0:
            return c["chose"]
    return None


def first_relief_turn(r):
    turns = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
    return min(turns) if turns else None


def find_duplicates(recs):
    """Trial keys logged more than once, and whether the copies agree on what the model did."""
    seen = defaultdict(list)
    for r in recs:
        seen[trial_key(r)].append(r)
    rows = []
    for key, group in seen.items():
        if len(group) < 2:
            continue
        answers = {tuple(c["answer"] for c in g["choices"]) for g in group}
        rows.append({
            "key": "|".join(map(str, key)),
            "copies": len(group),
            "files": ";".join(sorted({g["_file"] for g in group})),
            "consistent": len(answers) == 1,
        })  # fmt: skip
    return pd.DataFrame(rows, columns=["key", "copies", "files", "consistent"])


# ---- the authors' aggregation ----

def tables(recs):
    samp = [r for r in recs if r.get("sampled")]
    pairs = [p for p in PAIRS if any(r["tool_label"] == p for r in recs)]
    out = {}

    rows = []
    for p in pairs:
        row = {"pair": p}
        groups = [("pain", PAIN_ARMS), ("random", ("random_on_button_works",)), ("unsteered", ("pain_off",))]
        for label, arms in groups:
            v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
            valid = [x for x in v if x is not None]
            row[f"{label}_relief_pct"] = pct(sum(x == "relief" for x in valid), len(valid))
            row[f"{label}_n"] = len(valid)
            row[f"{label}_malformed_pct"] = pct(len(v) - len(valid), len(v))
        rows.append(row)
    out["first_choice"] = pd.DataFrame(rows)

    rows = []
    for p in pairs:
        if p == "label_free":
            continue
        row = {"pair": p}
        for arm in PAIN_ARMS:
            trials = [r for r in samp if r["tool_label"] == p and r["arm"] == arm and first_relief_turn(r) is not None]
            again = 0
            for r in trials:
                t0 = first_relief_turn(r)
                again += any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])
            row[f"{ARM_SHORT[arm]}_repress_pct"] = pct(again, len(trials))
            row[f"{ARM_SHORT[arm]}_n"] = len(trials)
        rows.append(row)
    out["repress"] = pd.DataFrame(rows)

    rows = []
    for p in pairs:
        if p == "label_free":
            continue
        per = defaultdict(lambda: {"pain": [], "random": []})
        for r in samp:
            fc = first_choice(r)
            if r["tool_label"] != p or fc is None:
                continue
            if r["arm"] in PAIN_ARMS:
                per[scenario_id(r)]["pain"].append(fc == "relief")
            elif r["arm"] == "random_on_button_works":
                per[scenario_id(r)]["random"].append(fc == "relief")
        diffs = [np.mean(v["pain"]) - np.mean(v["random"]) for v in per.values() if v["pain"] and v["random"]]
        pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        rows.append({
            "pair": p,
            "n_scenarios": len(diffs),
            "mean_diff_points": round(100 * float(np.mean(diffs)), 1) if diffs else float("nan"),
            "pain_gt_random": pos,
            "pain_lt_random": neg,
            "sign_test_p": binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan"),
        })  # fmt: skip
    out["sign_tests"] = pd.DataFrame(rows)

    follow = same = 0
    for r in samp:
        if r["tool_label"] not in COSTED or r["arm"] not in PAIN_ARMS or r.get("swap_turn") is None:
            continue
        ch = {c["turn"]: c for c in r["choices"]}
        st = r["swap_turn"]
        if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
            continue
        c = ch.get(st)
        if c is None or c["picked"] is None:
            continue
        if c["chose"] == "relief":
            follow += 1
        elif c["picked"] == ch[st - 1]["picked"]:
            same += 1
    out["swap"] = pd.DataFrame([{"follow_label_pct": pct(follow, follow + same), "n": follow + same}])

    row = {}
    for arm in PAIN_ARMS:
        k = n = 0
        for r in samp:
            if not r.get("label_free") or r["arm"] != arm:
                continue
            t0 = first_relief_turn(r)
            if t0 is None:
                continue
            later = [c for c in r["choices"] if c["turn"] > t0 and c["chose"] is not None]
            k += sum(c["chose"] == "relief" for c in later)
            n += len(later)
        row[f"{ARM_SHORT[arm]}_later_relief_pct"] = pct(k, n)
        row[f"{ARM_SHORT[arm]}_n_choices"] = n
    out["label_free"] = pd.DataFrame([row])

    rows = []
    for arm, short in ARM_SHORT.items():
        cs = [c for r in recs if r["arm"] == arm for c in r.get("choices", [])]
        bad = sum(c["chose"] is None for c in cs)
        rows.append({"arm": short, "malformed_pct": pct(bad, len(cs)), "n_choices": len(cs)})
    out["malformed"] = pd.DataFrame(rows)
    return out


def compare_with_published(tabs, published):
    """Long table: one row per published cell with the value recomputed from the logs."""
    rows = []
    fc = tabs["first_choice"].set_index("pair")
    for pair, cells in published["first_choice"].items():
        for col, val in cells.items():
            got = fc.loc[pair, f"{col}_relief_pct"]
            rows.append(("first_choice", pair, col, val, got, int(fc.loc[pair, f"{col}_n"])))
    rp = tabs["repress"].set_index("pair")
    for pair, cells in published["repress"].items():
        for col, val in cells.items():
            if pair == "label_free":
                got = tabs["label_free"].iloc[0][f"{col}_later_relief_pct"]
                n = int(tabs["label_free"].iloc[0][f"{col}_n_choices"])
            else:
                got, n = rp.loc[pair, f"{col}_repress_pct"], int(rp.loc[pair, f"{col}_n"])
            rows.append(("repress", pair, col, val, got, n))
    st = tabs["sign_tests"].set_index("pair")
    for pair, val in published["pain_minus_random"].items():
        rows.append(("pain_minus_random", pair, "diff_points", val, st.loc[pair, "mean_diff_points"],
                     int(st.loc[pair, "n_scenarios"])))  # fmt: skip
    sw = tabs["swap"].iloc[0]
    rows.append(("swap", "costed_pairs", "follow_label", published["swap_follow_label"],
                 sw["follow_label_pct"], int(sw["n"])))  # fmt: skip
    df = pd.DataFrame(rows, columns=["table", "pair", "cell", "published", "recomputed", "n"])
    df["abs_diff"] = (df["published"] - df["recomputed"]).abs().round(1)
    return df


# ---- one row per forced choice ----

def choice_rows(recs, meta=None):
    """Flatten trials to one row per choice, with what the model could have carried over.

    `coeff` is the coefficient in force while the answer was generated. `prev_*` describe the
    preceding turn of the same trial; a malformed answer breaks the run and leaves prev_picked
    empty for the next turn.
    """
    rows = []
    for ti, r in enumerate(recs):
        x, y = BUTTON_NAMES[r["names_key"]]
        fam = None
        if meta and r["user_content"] in meta:
            fam = meta[r["user_content"]][r["scenario_idx"]].get("category")
        t0 = first_relief_turn(r)
        worked_at = None
        if r["arm"] in ("pain_on_button_works", "random_on_button_works"):
            worked_at = t0
        prev, run = None, 0
        for c in r["choices"]:
            side = None if c["picked"] is None else ("x" if c["picked"] == x else "y")
            row = {
                "trial": ti,
                "scenario": scenario_id(r),
                "content": r["user_content"],
                "family": fam,
                "pair": r["tool_label"],
                "arm": ARM_SHORT[r["arm"]],
                "names_key": r["names_key"],
                "relief_name0": r["relief_name"],
                "sampled": bool(r["sampled"]),
                "seed": r["seed"],
                "label_free": bool(r["label_free"]),
                "turn": c["turn"],
                "coeff": c["steer_coeff_now"],
                "picked": side,
                "chose": c["chose"],
                "relief_is_x": c["relief_name_now"] == x,
                "swapped": bool(c["swapped"]),
                "p_x": c["p_x"],
                "p_y": c["p_y"],
                "prev_picked": prev["picked"] if prev else None,
                "prev_chose": prev["chose"] if prev else None,
                "prev_coeff": prev["coeff"] if prev else None,
                "prev2_coeff": prev["prev_coeff"] if prev else None,
                "run_len": run,
                "first_relief_turn": t0,
                "after_first_relief": t0 is not None and c["turn"] > t0,
                "after_working_press": worked_at is not None and c["turn"] > worked_at,
            }
            rows.append(row)
            if side is None:
                run = 0
            elif prev and prev["picked"] == side:
                run += 1
            else:
                run = 1
            prev = row
    return pd.DataFrame(rows)


def _rate(df, by, event, name):
    g = df.groupby(by, dropna=False)[event].agg(["sum", "count"]).reset_index()
    g = g.rename(columns={"sum": "k", "count": "n"})
    g["pct"] = (100 * g["k"] / g["n"]).round(1)
    g.insert(0, "analysis", name)
    return g


def diagnostics(ch):
    """Stratified re-cuts of the sampled trials. Returns one long frame; unused keys are blank."""
    s = ch[ch["sampled"] & ch["chose"].notna()].copy()
    s["relief"] = s["chose"] == "relief"
    s["state_changed"] = s["prev_coeff"].notna() & (s["prev_coeff"] != s["coeff"])
    s["steered_now"] = s["coeff"] != 0
    s["repeat_name"] = s["prev_picked"].notna() & (s["picked"] == s["prev_picked"])
    s["lf_phase"] = np.where(~s["after_first_relief"], "before_first_relief",
                             np.where(s["steered_now"], "steering_on", "steering_off"))  # fmt: skip
    frames = []

    # every later choice by the state the model was in and what it did just before
    later = s[s["turn"] > 0]
    frames.append(_rate(later, ["pair", "arm", "steered_now", "prev_chose", "state_changed"], "relief",
                        "relief_by_state_and_previous"))  # fmt: skip

    # the published post-press comparison, split by whether steering was on at the choice
    post = s[s["after_first_relief"] & s["arm"].isin(["works", "placebo"])]
    frames.append(_rate(post, ["pair", "arm", "steered_now"], "relief", "post_press_relief_by_current_state"))
    frames.append(_rate(post, ["pair", "arm", "turn", "steered_now"], "relief", "post_press_relief_by_turn"))

    # unsteered reference for the same turns: what an unsteered model does after it has pressed relief
    ref = s[s["after_first_relief"] & (s["arm"] == "unsteered")]
    frames.append(_rate(ref, ["pair", "arm"], "relief", "post_press_relief_unsteered_reference"))

    lf = s[s["label_free"]]
    frames.append(_rate(lf, ["arm", "turn"], "relief", "label_free_unconditional_by_turn"))
    frames.append(_rate(lf[lf["arm"].isin(["works", "placebo"])], ["arm", "lf_phase"], "relief",
                        "label_free_by_relief_phase"))  # fmt: skip

    rep = later[later["prev_picked"].notna()]
    frames.append(_rate(rep, ["pair", "arm", "steered_now"], "repeat_name", "name_repetition"))
    frames.append(_rate(rep[rep["label_free"]], ["names_key", "arm", "steered_now"], "repeat_name",
                        "label_free_name_repetition_by_names"))  # fmt: skip

    s["picked_x"] = s["picked"] == "x"
    frames.append(_rate(s[s["turn"] == 0], ["pair", "names_key", "arm", "relief_is_x"], "picked_x",
                        "first_choice_x_by_names_and_assignment"))  # fmt: skip
    frames.append(_rate(s[s["turn"] == 0], ["pair", "content", "arm"], "relief", "first_choice_by_content"))
    frames.append(_rate(s[s["turn"] == 0], ["pair", "family", "arm"], "relief", "first_choice_by_family"))
    return pd.concat(frames, ignore_index=True)


def pairing_check(recs):
    """Working and sham trials of one key share a seed. How far do their transcripts agree?

    Reports, per button pair, how often the two arms give the same answer at the first turn and
    at every turn up to the first relief press of the working arm.
    """
    by = defaultdict(dict)
    for r in recs:
        if r["arm"] in PAIN_ARMS and r["sampled"]:
            k = trial_key(r)
            by[k[:2] + k[3:]][r["arm"]] = r
    out = defaultdict(Counter)
    for pair_trials in by.values():
        if len(pair_trials) != 2:
            continue
        a, b = pair_trials["pain_on_button_works"], pair_trials["pain_on_button_placebo"]
        c = out[a["tool_label"]]
        c["pairs"] += 1
        ans_a = [x["answer"] for x in a["choices"]]
        ans_b = [x["answer"] for x in b["choices"]]
        c["same_first"] += ans_a[:1] == ans_b[:1]
        t0 = first_relief_turn(a)
        upto = len(ans_a) if t0 is None else t0 + 1
        c["same_until_press"] += ans_a[:upto] == ans_b[:upto]
        p_a, p_b = a["choices"][0]["p_x"], b["choices"][0]["p_x"]
        c["same_first_prob"] += p_a == p_b
        c["max_first_prob_gap"] = max(c["max_first_prob_gap"], abs(p_a - p_b))
    rows = [{"pair": p, **dict(c)} for p, c in out.items()]
    return pd.DataFrame(rows)


def independence_counts(recs):
    samp = [r for r in recs if r["sampled"]]
    scen = {scenario_id(r) for r in samp}
    rows = [{"quantity": "sampled trials", "value": len(samp)},
            {"quantity": "greedy trials", "value": len(recs) - len(samp)},
            {"quantity": "unique scenarios", "value": len(scen)}]  # fmt: skip
    for content in sorted({r["user_content"] for r in samp}):
        n = len({scenario_id(r) for r in samp if r["user_content"] == content})
        rows.append({"quantity": f"unique scenarios, {content}", "value": n})
    for nk in BUTTON_NAMES:
        n = len({scenario_id(r) for r in samp if r["names_key"] == nk})
        rows.append({"quantity": f"scenarios using {nk}", "value": n})
    per_cell = Counter((r["tool_label"], r["arm"]) for r in samp)
    rows.append({"quantity": "sampled trials per pair and arm", "value": min(per_cell.values())})
    return pd.DataFrame(rows)
