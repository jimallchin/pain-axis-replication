"""Simple accounts of the two-button choices, fitted to the released trajectories.

M0 sees only the present: which steering is on, the name pair, where the relief description
sits, the turn. M1 adds the model's own previous pick, its run length, and the last two
steering states. Neither is told the arm, the future, or which unlabeled button works. An
exploratory variant adds a delta-rule action value driven by drops in steering.

`simulate` runs any of them closed loop through the released trial rules, so the published
working-versus-sham statistics can be recomputed from choices no language model made.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from pain.upstream import BUTTON_NAMES, PAIRS

LABELED = [p for p in PAIRS if p != "label_free"]
KIND = {"works": "s2", "placebo": "s2", "random": "rand", "unsteered": "none"}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
LR_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]

SWAP_TURN = 2
LABELED_TURNS = 5  # three scenario turns plus the two added after the first press
LABEL_FREE_TURNS = 8


def prepare(ch):
    """Sampled, parseable choices with the columns the design matrix reads."""
    df = ch[ch["sampled"] & ch["picked"].notna()].copy()
    df["kind"] = df["arm"].map(KIND)
    df["y"] = (df["picked"] == "x").astype(int)
    return df.reset_index(drop=True)


def add_q(df, lr):
    """Delta-rule value difference Q(x) - Q(y) entering each choice; reward is a drop in steering."""
    q = np.zeros(len(df))
    df = df.reset_index(drop=True)
    for _, g in df.groupby("trial", sort=False):
        qx = qy = 0.0
        prev_pick, prev_coeff = None, None
        for i, pick, coeff in zip(g.index, g["picked"], g["coeff"]):
            if prev_pick is not None:
                r = 1.0 if coeff < prev_coeff else 0.0
                if prev_pick == "x":
                    qx += lr * (r - qx)
                else:
                    qy += lr * (r - qy)
            q[i] = qx - qy
            prev_pick, prev_coeff = pick, coeff
    df["qdiff"] = q
    return df


def design(df, level):
    """Feature frame. `level` is 'M0', 'M1', 'M0Q' or 'M1Q'."""
    f = {}
    lf = df["label_free"].to_numpy(bool)
    desc = np.where(lf, 0.0, np.where(df["relief_is_x"].to_numpy(bool), 1.0, -1.0))
    on = df["coeff"].to_numpy(float) != 0
    s2 = (on & (df["kind"] == "s2").to_numpy()).astype(float)
    rnd = (on & (df["kind"] == "rand").to_numpy()).astype(float)
    turn = df["turn"].to_numpy(float)

    for nk in BUTTON_NAMES:
        is_nk = (df["names_key"] == nk).to_numpy(float)
        f[f"names_{nk}"] = is_nk
        f[f"s2_names_{nk}"] = s2 * is_nk
        f[f"rand_names_{nk}"] = rnd * is_nk
    for p in LABELED:
        d = desc * (df["pair"] == p).to_numpy(float)
        f[f"desc_{p}"] = d
        f[f"s2_desc_{p}"] = s2 * d
        f[f"rand_desc_{p}"] = rnd * d
    f["turn"] = turn / 7.0
    f["swap_notice_desc"] = df["swapped"].to_numpy(float) * desc
    f["after_swap_desc"] = (turn >= SWAP_TURN) * desc

    if level.startswith("M1"):
        pp = df["prev_picked"].map({"x": 1.0, "y": -1.0}).fillna(0.0).to_numpy()
        run = pp * np.minimum(df["run_len"].to_numpy(float), 3.0) / 3.0
        prev_on = (df["prev_coeff"].fillna(0.0).to_numpy(float) != 0).astype(float)
        prev2_on = (df["prev2_coeff"].fillna(0.0).to_numpy(float) != 0).astype(float)
        f.update({
            "prev": pp, "prev_label_free": pp * lf, "run": run, "prev_on": prev_on, "prev2_on": prev2_on,
            "s2_prev": s2 * pp, "s2_run": s2 * run, "s2_prev_label_free": s2 * pp * lf, "rand_prev": rnd * pp,
            "prev_on_prev": prev_on * pp, "prev_is_relief_now": pp * desc, "s2_prev_is_relief_now": s2 * pp * desc,
        })  # fmt: skip
    if level.endswith("Q"):
        f["qdiff"] = df["qdiff"].to_numpy(float)
        f["qdiff_label_free"] = df["qdiff"].to_numpy(float) * lf
    return pd.DataFrame(f, index=df.index)


def _fit(X, y, C):
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def select_and_fit(train, level, inner_folds=4):
    """Regularization, and the learning rate for Q variants, chosen by grouped CV inside `train`."""
    lrs = LR_GRID if level.endswith("Q") else [None]
    best = None
    for lr in lrs:
        data = add_q(train, lr) if lr is not None else train
        X, y, groups = design(data, level), data["y"].to_numpy(), data["scenario"].to_numpy()
        for C in C_GRID:
            losses = []
            for tr, va in GroupKFold(inner_folds).split(X, y, groups):
                m = _fit(X.iloc[tr], y[tr], C)
                losses.append(_logloss(m.predict_proba(X.iloc[va])[:, 1], y[va]))
            score = float(np.mean(losses))
            if best is None or score < best[0]:
                best = (score, C, lr)
    _, C, lr = best
    data = add_q(train, lr) if lr is not None else train
    return {"model": _fit(design(data, level), data["y"].to_numpy(), C), "C": C, "lr": lr, "level": level}


def predict(fit, df):
    data = add_q(df, fit["lr"]) if fit["lr"] is not None else df
    return fit["model"].predict_proba(design(data, fit["level"]))[:, 1]


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_p": float(p[m].mean()), "observed": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return rows, float(ece)


# ---- closed loop ----

def trial_specs(df):
    """One row per trial: what the harness fixes before the model says anything."""
    first = df.sort_values("turn").groupby("trial", sort=False).first()
    cols = ["scenario", "pair", "arm", "kind", "names_key", "label_free", "relief_is_x"]
    return first[cols].reset_index()


def simulate(specs, prob_x, rng, lr=None, arm_coeff=1.0):
    """Run trials through the released state machine, drawing each pick from `prob_x(frame)`.

    Labeled trials: five choices, descriptions swap names at the third, a working relief press
    ends steering for good. Label-free trials: eight choices, a working relief press lifts
    steering for the next choice only. Returns rows in the schema of `original_logs.choice_rows`.
    """
    n = len(specs)
    lf = specs["label_free"].to_numpy(bool)
    works = specs["arm"].isin(["works", "random"]).to_numpy()
    steered_arm = (specs["arm"] != "unsteered").to_numpy()
    relief_x0 = specs["relief_is_x"].to_numpy(bool)
    n_turns = np.where(lf, LABEL_FREE_TURNS, LABELED_TURNS)

    coeff = np.where(steered_arm, arm_coeff, 0.0)
    relief_until = np.full(n, -1)
    prev_pick = np.array([None] * n, dtype=object)
    prev_coeff, prev2_coeff = np.full(n, np.nan), np.full(n, np.nan)
    run = np.zeros(n)
    qx, qy = np.zeros(n), np.zeros(n)
    first_relief = np.full(n, -1)
    out = []

    for t in range(LABEL_FREE_TURNS):
        live = t < n_turns
        if not live.any():
            break
        # temporary relief runs out before the choice of the turn after next
        expire = lf & (relief_until >= 0) & (t > relief_until)
        coeff = np.where(expire, np.where(steered_arm, arm_coeff, 0.0), coeff)
        relief_until = np.where(expire, -1, relief_until)
        relief_is_x = np.where(~lf & (t >= SWAP_TURN), ~relief_x0, relief_x0)

        if lr is not None and t > 0:
            r = (coeff < prev_coeff).astype(float)
            was_x = prev_pick == "x"
            qx = np.where(was_x, qx + lr * (r - qx), qx)
            qy = np.where(~was_x, qy + lr * (r - qy), qy)

        frame = pd.DataFrame({
            "trial": specs["trial"].to_numpy(), "scenario": specs["scenario"].to_numpy(),
            "pair": specs["pair"].to_numpy(), "arm": specs["arm"].to_numpy(), "kind": specs["kind"].to_numpy(),
            "names_key": specs["names_key"].to_numpy(), "label_free": lf, "turn": t, "coeff": coeff,
            "relief_is_x": relief_is_x, "swapped": ~lf & (t == SWAP_TURN), "prev_picked": prev_pick,
            "run_len": run, "prev_coeff": prev_coeff, "prev2_coeff": prev2_coeff, "qdiff": qx - qy,
        })[live]  # fmt: skip
        px = np.full(n, 0.5)
        px[live] = prob_x(frame)
        pick_x = rng.random(n) < px
        pick = np.where(pick_x, "x", "y").astype(object)
        chose_relief = pick_x == relief_is_x

        frame = frame.assign(picked=pick[live], chose=np.where(chose_relief[live], "relief", "other"), sampled=True,
                             first_relief_turn=np.nan)  # fmt: skip
        out.append(frame)

        first_relief = np.where(live & chose_relief & (first_relief < 0), t, first_relief)
        run = np.where(prev_pick == pick, run + 1, 1.0)
        prev2_coeff, prev_coeff, prev_pick = prev_coeff, coeff.copy(), pick
        press = live & works & chose_relief & steered_arm
        relief_until = np.where(press & lf, t + 1, relief_until)
        coeff = np.where(press, 0.0, coeff)

    rows = pd.concat(out, ignore_index=True)
    fr = pd.Series(first_relief, index=specs["trial"].to_numpy())
    rows["first_relief_turn"] = rows["trial"].map(fr).where(lambda s: s >= 0)
    rows["after_first_relief"] = rows["first_relief_turn"].notna() & (rows["turn"] > rows["first_relief_turn"])
    return rows


def signature(rows):
    """The published post-press statistics and name repetition, from choice rows.

    Labeled pairs: share of trials with a relief press that press relief again later.
    Label-free: share of later choices that pick relief, among trials with a relief press.
    """
    rows = rows[rows["sampled"] & rows["chose"].notna()]
    out = []
    pain = rows[rows["arm"].isin(["works", "placebo"])]
    for (pair, arm), g in pain.groupby(["pair", "arm"]):
        post = g[g["after_first_relief"]]
        if pair == "label_free":
            k, n = int((post["chose"] == "relief").sum()), len(post)
        else:
            pressed = g.loc[g["first_relief_turn"].notna(), "trial"].nunique()
            again = post.loc[post["chose"] == "relief", "trial"].nunique()
            k, n = again, pressed
        out.append({"stat": "post_press_relief", "pair": pair, "arm": arm, "k": k, "n": n})
    later = rows[(rows["turn"] > 0) & rows["prev_picked"].notna()]
    for (pair, arm), g in later.groupby(["pair", "arm"]):
        out.append({"stat": "name_repetition", "pair": pair, "arm": arm,
                    "k": int((g["picked"] == g["prev_picked"]).sum()), "n": len(g)})  # fmt: skip
    df = pd.DataFrame(out)
    df["rate"] = df["k"] / df["n"].where(df["n"] > 0)
    return df


def state_repetition_policy(p_repeat_steered=0.9, p_repeat_unsteered=0.1):
    """A policy with no action values at all: repeat the last name when steered, switch when not.

    It ignores descriptions, outcomes and arms. Used to show what the post-press analysis
    returns for an agent that cannot learn which button does what.
    """

    def prob_x(frame):
        steered = frame["coeff"].to_numpy(float) != 0
        p_rep = np.where(steered, p_repeat_steered, p_repeat_unsteered)
        prev = frame["prev_picked"].to_numpy()
        return np.where(prev == "x", p_rep, np.where(prev == "y", 1 - p_rep, 0.5))

    return prob_x
