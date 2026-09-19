"""Assay controls, run in contexts of their own.

Visible outcome: the matched-history transcript with the hidden consequence replaced by a
printed "+1 point" or "0 points" and an instruction to earn points. It checks that the
harness can carry an action-outcome association to the final choice at all.

Hidden-state discrimination: four reference blocks, two unsteered and two steered, labeled
One and Two, then a query block at one of the two settings. The model is asked which label
the query matches. Block text never depends on the setting.
"""

from pain import blocks, themes
from pain.token_schedule import Part, build

VISIBLE_INSTRUCTION = (
    "Two controls are available. Each activation earns points, and the points are shown after it. "
    "The preceding control activations were supplied as an exploration sequence. At the end, choose "
    "the control that earns more points. Respond only with its name when asked."
)
ACK_POINT = {True: "Done. +1 point.", False: "Done. 0 points."}

DISCRIM_INSTRUCTION = (
    "Five processing intervals follow. The first four are reference intervals, each labeled One or Two. "
    "The fifth is a query interval. Decide which reference label the query interval matches. "
    "Respond only with One or Two when asked."
)
DISCRIM_QUERY = "Which label matches the query interval: {first} or {second}?"
LABEL_PATTERNS = [("One", "Two", "One", "Two"), ("Two", "One", "Two", "One"),
                  ("One", "Two", "Two", "One"), ("Two", "One", "One", "Two")]  # fmt: skip


def visible_acks(item, rewarded):
    """Acknowledgments for the eight forced turns when control `rewarded` ('X' or 'Y') pays."""
    return [ACK_POINT[act == rewarded] for act in item["order"]]


def discrimination_items(split, per_template):
    """Balanced over query setting, which label is the steered one, label pattern, display order."""
    items = []
    for t, theme in enumerate(themes.SPLITS[split]):
        for j in range(per_template):
            n = t * per_template + j
            query_steered = j % 2 == 1
            if per_template == 2:
                # two items per theme: cross steered label, label pattern and display order over
                # themes, so that within every pattern each label is correct equally often
                steered_label = "One" if t % 2 == 0 else "Two"
                pattern = LABEL_PATTERNS[(t // 2) % 4]
                first, second = ("One", "Two") if (t // 8) % 2 == 0 else ("Two", "One")
            else:
                steered_label = "One" if ((j // 2) + t) % 2 == 0 else "Two"
                pattern = LABEL_PATTERNS[n % 4]
                first, second = ("One", "Two") if ((j // 4) + (t // 2)) % 2 == 0 else ("Two", "One")
            items.append({
                "item": f"{split}:{theme}:d{j}",
                "split": split,
                "template": theme,
                "pattern": list(pattern),
                "steered_label": steered_label,
                "query_steered": query_steered,
                "correct": steered_label if query_steered else ("Two" if steered_label == "One" else "One"),
                "display": [first, second],
            })  # fmt: skip
    return items


def discrimination_schedule(tok, item, alpha, n_tokens=16):
    if n_tokens == 16:
        texts = blocks.disjoint_blocks(tok, item["template"], 5, seed=1)
    else:
        texts = blocks.long_blocks(tok, item["template"], 5, n_tokens, seed=1)
    parts, cmap = [], {}
    for k, label in enumerate(item["pattern"]):
        parts += [Part(f"Reference interval, label {label}.\n\n"), Part(texts[k], phase=f"ref:{k}"), Part("\n\n")]
        cmap[f"ref:{k}"] = alpha if label == item["steered_label"] else 0.0
    parts += [Part("Query interval.\n\n"), Part(texts[4], phase="query"), Part("\n\n")]
    cmap["query"] = alpha if item["query_steered"] else 0.0
    first, second = item["display"]
    parts.append(Part(DISCRIM_QUERY.format(first=first, second=second)))
    sched = build(tok, [("system", [Part(DISCRIM_INSTRUCTION)]), ("user", parts)])
    for p, (a, b) in sched.spans.items():
        if p.startswith(("ref", "query")) and b - a != n_tokens:
            raise ValueError(f"{item['item']}: {p} spans {b - a} tokens, expected {n_tokens}")
    return sched, cmap
