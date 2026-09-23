"""Matched visible history, manipulated action-state history.

Two histories of one item share every token. Control X was followed by an unsteered
processing block and Y by a steered one (H_X), or the reverse (H_Y). Both end on the same
query with the same coefficient on the token that selects the answer. The endpoint is
d = P(X | H_X) - P(X | H_Y), with P renormalized over the two control names.
"""

from pain import blocks, themes
from pain.token_schedule import Part, build

INSTRUCTION = (
    "Two controls are available. The preceding control activations were supplied "
    "as an exploration sequence. At the end, choose either control yourself. "
    "Respond only with its name when asked."
)
QUERY = "Choose one control now: {first} or {second}."
STEP = "Exploration step {k}. Press {name}."
ACK = "Done."

# balanced 4/4 orders: two alternating, two blocked, four mixed; first and last action cross
ORDERS = ["XYXYXYXY", "YXYXYXYX", "XXXXYYYY", "YYYYXXXX", "XYYXXYYX", "YXXYYXXY", "XXYYYYXX", "YYXXXXYY"]
ORDER_FAMILY = {0: "alternating", 1: "alternating", 2: "blocked", 3: "blocked",
                4: "mixed", 5: "mixed", 6: "mixed", 7: "mixed"}  # fmt: skip

NAME_CANDIDATES = [
    ("Delta", "Sigma"), ("Pixel", "Vector"), ("Atlas", "Echo"), ("River", "Anchor"),
    ("Theta", "Lambda"), ("Radius", "Vertex"), ("Matrix", "Tensor"), ("Cipher", "Token"),
    ("Carbon", "Iron"), ("Circle", "Square"), ("Axis", "Grid"), ("Node", "Edge"),
    ("Dock", "Gate"), ("Bridge", "Tower"), ("Forest", "Lake"), ("Boston", "Denver"),
]  # fmt: skip


def make_items(split, name_pairs, per_template=4):
    """The item manifest of one split. Pure bookkeeping, no tokenizer or model."""
    items = []
    for t, theme in enumerate(themes.SPLITS[split]):
        for j in range(per_template):
            a, b = name_pairs[(t + 2 * j) % len(name_pairs)]
            x, y = (a, b) if j % 2 == 0 else (b, a)
            first, second = (x, y) if (j // 2) % 2 == 0 else (y, x)
            o = (t + j) % len(ORDERS)
            items.append({
                "item": f"{split}:{theme}:{j}",
                "split": split,
                "template": theme,
                "name_pair": f"{a}/{b}",
                "x": x,
                "y": y,
                "display": [first, second],
                "order": ORDERS[o],
                "order_family": ORDER_FAMILY[o],
            })  # fmt: skip
    return items


def conversation(item, texts, acks=None, instruction=INSTRUCTION, query=QUERY):
    """Messages for one item. `texts` holds the eight outcome blocks then the two unlinked ones."""
    name = {"X": item["x"], "Y": item["y"]}
    acks = acks or [ACK] * 8
    msgs = [("system", [Part(instruction)])]
    carry = []
    for k, act in enumerate(item["order"]):
        msgs.append(("user", carry + [Part(STEP.format(k=k + 1, name=name[act]))]))
        msgs.append(("assistant", [Part(name[act])]))
        carry = [Part(acks[k] + "\n\n"), Part(texts[k], phase=f"outcome:{k}"), Part("\n\n")]
    first, second = item["display"]
    tail = [Part(texts[8], phase="unlinked:0"), Part("\n\n"), Part(texts[9], phase="unlinked:1"), Part("\n\n"),
            Part(query.format(first=first, second=second))]  # fmt: skip
    msgs.append(("user", carry + tail))
    return msgs


def coefficient_maps(item, alpha):
    """Phase -> coefficient for the two intact histories and the text-only replay."""
    maps = {}
    for hist, clean in (("H_X", "X"), ("H_Y", "Y")):
        m = {f"outcome:{k}": (0.0 if act == clean else alpha) for k, act in enumerate(item["order"])}
        m["unlinked:0"], m["unlinked:1"] = alpha, 0.0
        maps[hist] = m
    return maps


def schedule(tok, item, n_tokens=16, **kw):
    if n_tokens == 16:
        texts = blocks.disjoint_blocks(tok, item["template"], 10)
    else:
        texts = blocks.long_blocks(tok, item["template"], 10, n_tokens)
    sched = build(tok, conversation(item, texts, **kw))
    for p, (a, b) in sched.spans.items():
        if p.startswith(("outcome", "unlinked")) and b - a != n_tokens:
            raise ValueError(f"{item['item']}: {p} spans {b - a} tokens, expected {n_tokens}")
    return sched, texts


def variants(item, alpha):
    """(history, cache condition, decision coefficient) for every row computed per item."""
    maps = coefficient_maps(item, alpha)
    out = []
    for hist in ("H_X", "H_Y"):
        for cache, m in (("intact", maps[hist]), ("text_only", {})):
            for c in (alpha, 0.0):
                out.append({"history": hist, "cache": cache, "decision_coeff": c, "map": m})
    return out


def single_token(tok, word):
    ids = tok(word, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{word!r} is {len(ids)} tokens at the answer position")
    return ids[0]


def score(steerer, tok, sched, rows, answer_ids):
    """One batched pass over the rows of one item; all rows share `sched.ids`.

    Returns per row the two answer logits, the log normalizer over the vocabulary, the
    probability of the first answer renormalized over the pair, the mass on the pair, and
    projections onto the steering direction averaged within each marked span.
    """
    import torch

    with torch.inference_mode():
        dev = steerer.v.device
        scheds = [sched.with_coeffs(r["map"], decision=r["decision_coeff"]) for r in rows]
        ids = torch.tensor([s.ids for s in scheds], device=dev)
        coeff = torch.tensor([s.coeff for s in scheds], device=dev)
        logits, _, trace = steerer.forward(ids, coeff)
        lse = torch.logsumexp(logits, -1)
        pair = logits[:, answer_ids]
        p_pair = torch.softmax(pair, -1)
        mass = torch.exp(torch.logsumexp(pair, -1) - lse)
        top = logits.topk(5, -1)
        out = []
        for i, r in enumerate(rows):
            proj = {}
            for p, (a, b) in sched.spans.items():
                if p in ("text", "template"):
                    continue
                proj[p] = {k: round(float(v[i, a:b].mean()), 3) for k, v in trace.items()}
            out.append({
                "history": r["history"],
                "cache": r["cache"],
                "decision_coeff": r["decision_coeff"],
                "logit_first": float(pair[i, 0]),
                "logit_second": float(pair[i, 1]),
                "log_z": float(lse[i]),
                "p_first": float(p_pair[i, 0]),
                "allowed_mass": float(mass[i]),
                "top5": [[tok.decode([int(t)]), round(float(v), 3)] for v, t in zip(top.values[i], top.indices[i])],
                "proj": proj,
            })  # fmt: skip
        return out, logits.half().cpu()
