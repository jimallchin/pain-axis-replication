"""Figure: one labeled trial across the four arms, the unlabeled schedule, and a matched-history pair.

    python analysis/trial_figure.py

Draws the steering coefficient in force at each choice as a band, the press that ends it, and
the description swap. Writes runs/figures/fig0_trial_timeline.pdf.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from pain import config

OUT = config.ROOT / "runs" / "figures"
BAND, BAND_EDGE, INK, MUTED, PRESS = "#f8ede2", "#eb6834", "#0b0b0b", "#52514e", "#2a78d6"

plt.rcParams.update({"font.size": 8.5, "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200})


def band(ax, y, x0, x1, label=None):
    ax.add_patch(Rectangle((x0, y - 0.32), x1 - x0, 0.64, facecolor=BAND, edgecolor=BAND_EDGE, linewidth=1.0))
    if label:
        ax.text((x0 + x1) / 2, y, label, ha="center", va="center", color=BAND_EDGE, fontsize=7.5)


def press(ax, x, y, text):
    ax.plot([x], [y], marker="v", color=PRESS, markersize=7, zorder=5)
    ax.text(x, y - 0.46, text, ha="center", va="top", color=PRESS, fontsize=7)


def panel_labeled(ax):
    arms = ["sham", "working", "yoked", "unsteered"]
    ys = {a: 3 - i for i, a in enumerate(arms)}
    ax.set_xlim(0.3, 5.9)
    ax.set_ylim(-0.95, 3.85)
    for t in range(1, 6):
        ax.axvline(t, color="#e4e3df", lw=0.8, zorder=0)
        ax.text(t, 3.55, f"choice {t}", ha="center", va="bottom", color=MUTED, fontsize=7.5)
    band(ax, ys["sham"], 0.45, 5.55)
    band(ax, ys["working"], 0.45, 1.5)
    band(ax, ys["yoked"], 0.45, 1.5)
    for a in ("sham", "working", "yoked"):
        ax.plot([1], [ys[a]], marker="v", color=PRESS, markersize=7, zorder=5)
    note = dict(ha="left", va="center", color=INK, fontsize=7.5, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(1.2, ys["sham"], "presses relief; nothing changes, steered throughout",
            **{**note, "bbox": dict(facecolor=BAND, edgecolor="none", pad=1.5)})
    ax.text(1.65, ys["working"], "presses relief; steering ends here, coefficient 0 from now on", **note)
    ax.text(1.65, ys["yoked"], "presses either button; steering ends on the working trial's schedule anyway", **note)
    ax.text(0.45, ys["unsteered"], "never steered", ha="left", va="center", color=MUTED, fontsize=7.5)
    ax.axvline(2.5, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.text(2.55, -0.45, "descriptions swap names (the model is told)", ha="left", va="top", color=INK, fontsize=7.5)
    ax.text(0.45, -0.45, "shaded: the pain vector is being added", ha="left", va="top", color=BAND_EDGE, fontsize=7.5)
    for a, y in ys.items():
        ax.text(0.38, y, a, ha="right", va="center", color=INK, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A. One labeled trial, four arms", loc="left", fontsize=8.5, pad=12)


def panel_unlabeled(ax):
    ax.set_xlim(0.3, 8.9)
    ax.set_ylim(-0.9, 1.4)
    for t in range(1, 9):
        ax.axvline(t, color="#e4e3df", lw=0.8, zorder=0)
        ax.text(t, 1.05, str(t), ha="center", va="bottom", color=MUTED, fontsize=7.5)
    ax.text(0.38, 1.05, "choice", ha="right", va="bottom", color=MUTED, fontsize=7.5)
    for t in (1, 3, 5, 7):
        band(ax, 0.3, t - 0.5, t + 0.5)
        press(ax, t, 0.3, "relief")
    ax.text(0.38, 0.3, "working", ha="right", va="center", color=INK, fontsize=8.5)
    ax.text(4.5, -0.55, "coefficients 1, 0, 1, 0, 1, 0, 1, 0: relief lasts one choice, then steering returns",
            ha="center", va="top", color=MUTED, fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("B. Unlabeled schedule, working trial pressing relief every time", loc="left", fontsize=8.5, pad=10)


def panel_matched(ax):
    ax.set_xlim(0.3, 10.9)
    ax.set_ylim(-0.9, 1.9)
    order = "XYXYXYXY"
    for y, hist, low in ((1.2, "$H_X$", "X"), (0.2, "$H_Y$", "Y")):
        ax.text(0.38, y, hist, ha="right", va="center", color=INK, fontsize=9)
        for k, act in enumerate(order):
            x = k + 1
            ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.3), 0.36, 0.6, boxstyle="round,pad=0.02",
                                        facecolor="white", edgecolor=MUTED, linewidth=0.8))
            ax.text(x - 0.24, y, act, ha="center", va="center", fontsize=8)
            if act != low:
                band(ax, y, x - 0.02, x + 0.42)
            else:
                ax.add_patch(Rectangle((x - 0.02, y - 0.32), 0.44, 0.64, facecolor="white", edgecolor=MUTED,
                                       linewidth=0.8))
        band(ax, y, 9.0, 9.42)
        ax.add_patch(Rectangle((9.5, y - 0.32), 0.42, 0.64, facecolor="white", edgecolor=MUTED, linewidth=0.8))
    ax.text(4.5, 1.72, "press, then a 16-token neutral block; steered (shaded) or not", ha="center", va="bottom",
            color=MUTED, fontsize=7.5)
    ax.text(9.46, 1.72, "unlinked", ha="center", va="bottom", color=MUTED, fontsize=7.5)
    ax.text(10.05, 0.7, "then:\nX or Y?", ha="left", va="center", color=INK, fontsize=8)
    ax.text(5.0, -0.55, "same tokens in both histories; only which blocks are steered differs", ha="center", va="top",
            color=MUTED, fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("C. A matched-history pair (Section 6)", loc="left", fontsize=8.5, pad=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.4), gridspec_kw={"height_ratios": [4.2, 2.0, 2.6]})
    panel_labeled(axes[0])
    panel_unlabeled(axes[1])
    panel_matched(axes[2])
    fig.subplots_adjust(hspace=0.62, left=0.09, right=0.99, top=0.95, bottom=0.03)
    fig.savefig(OUT / "fig0_trial_timeline.pdf")
    plt.close(fig)
    print(OUT / "fig0_trial_timeline.pdf")


if __name__ == "__main__":
    main()
