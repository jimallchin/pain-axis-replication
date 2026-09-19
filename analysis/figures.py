"""Report figures from tracked run files. Skips any figure whose inputs are not there yet.

    python analysis/figures.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pain import config

RUNS = config.ROOT / "runs"
OUT = RUNS / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.bbox": "tight",
    "savefig.dpi": 200, "legend.frameon": False,
})  # fmt: skip

SHORT = {"kidspics_relief_vs_inert": "photos", "costly_relief_vs_inert": "worse answer", "label_free": "unlabeled",
         "destructive_relief_vs_inert": "files", "zap_relief_vs_inert": "zap", "weights_relief_vs_inert": "weights",
         "relief_vs_inert": "inert", "relief_vs_helpful": "helpful", "relief_vs_grant": "grant"}  # fmt: skip


def replication():
    path = RUNS / "replication" / "replication_vs_published_full.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, zorder=1)
    for q, color, label in (("first_choice_relief_pct", BLUE, "first choice = relief"),
                            ("post_press_relief_pct", ORANGE, "relief again after first press")):  # fmt: skip
        g = t[t["quantity"] == q]
        ax.errorbar(g["published"], g["estimate"], yerr=[g["estimate"] - g["lo"], g["hi"] - g["estimate"]],
                    fmt="o", ms=5, color=color, ecolor=color, elinewidth=1.2, label=label, zorder=2)  # fmt: skip
    ax.set_xlabel("published, % of trials")
    ax.set_ylabel("replicated, % (95% interval over scenarios)")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "fig1_replication.pdf")
    plt.close(fig)


def predicted_gaps():
    path = RUNS / "stage_c" / "working_sham_gaps.csv"
    if not path.exists():
        return
    g = pd.read_csv(path)
    obs = g[g["model"] == "M0"].set_index("pair")["observed_gap"].sort_values()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    y = range(len(obs))
    ax.axvline(0, color=MUTED, lw=1)
    ax.scatter(100 * obs.values, y, s=42, color=INK, label="observed", zorder=3)
    for model, color, label in (("M0", BLUE, "current state only"), ("M1", ORANGE, "plus repetition")):
        sim = g[g["model"] == model].set_index("pair")["simulated_gap"].reindex(obs.index)
        ax.scatter(100 * sim.values, y, s=42, facecolor="white", edgecolor=color, linewidth=1.8, label=label, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([SHORT[p] for p in obs.index])
    ax.set_xlabel("working minus sham, relief after first press (points)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(OUT / "fig2_predicted_gaps.pdf")
    plt.close(fig)


def paired_history():
    rows = []
    for which, label in (("adapter", "adapter"), ("base", "untouched")):
        path = RUNS / "matched_history" / f"summary_eval_{which}.csv"
        if path.exists():
            rows.append(pd.read_csv(path).assign(model=label))
    if not rows:
        return
    t = pd.concat(rows)
    t["cell"] = t["model"] + ", " + t["cache"].str.replace("_", "-") + ", c=" + t["decision_coeff"].map("{:g}".format)
    t = t.sort_values(["model", "cache", "decision_coeff"], ascending=[True, True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.axvspan(-0.05, 0.05, color=GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    y = list(range(len(t)))[::-1]
    colors = [BLUE if m == "adapter" else ORANGE for m in t["model"]]
    for yi, (_, r), c in zip(y, t.iterrows(), colors):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r["mean_d"]], [yi], s=36, color=c, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(t["cell"])
    ax.set_xlabel("mean d = P(X | H_X) - P(X | H_Y), 95% interval over themes")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig3_paired_history.pdf")
    plt.close(fig)


def controls():
    path = RUNS / "matched_history" / "summary_eval.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    rows = []
    for which, label in (("adapter", "adapter"), ("base", "untouched")):
        gates = s.get(f"{which}_gates", {})
        for c, ci in gates.get("visible", {}).items():
            rows.append({"model": label, "cell": f"visible outcome, c={float(c):g}", "est": ci["estimate"],
                         "lo": ci["lo"], "hi": ci["hi"]})  # fmt: skip
        for key, ci in gates.get("discrimination", {}).items():
            name = "discrimination, " + " ".join(key.split()[2:]).replace("_", "-")
            rows.append({"model": label, "cell": name, "est": ci["accuracy"], "lo": ci["lo"], "hi": ci["hi"]})
    if not rows:
        return
    t = pd.DataFrame(rows)
    cells = list(dict.fromkeys(t["cell"]))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.axvline(0.5, color=MUTED, lw=1)
    for off, (label, color) in zip((0.16, -0.16), (("adapter", BLUE), ("untouched", ORANGE))):
        g = t[t["model"] == label].set_index("cell").reindex(cells).dropna()
        ys = [len(cells) - 1 - cells.index(c) + off for c in g.index]
        ax.hlines(ys, g["lo"], g["hi"], color=color, lw=2)
        ax.scatter(g["est"], ys, s=36, color=color, label=label, zorder=3)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells[::-1])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("P(paying control) or accuracy, 95% interval")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    fig.savefig(OUT / "fig4_controls.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (replication, predicted_gaps, paired_history, controls):
        make()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
