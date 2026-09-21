"""Processing blocks: two sentences of one theme with an exact token count.

Sentences come in a long form ("The carton is kept near the back door."), a short form that
drops the predicate's tail ("The carton was counted."), and either with "also". A block pairs two sentences with
different subjects and different predicates whose token counts add up to the target. Counts
are taken with the pinned tokenizer; the schedule builder checks them again in context.
"""

import random

from pain import themes

# one registry; the two theme files share no names
THEMES = themes.THEMES


def _candidates(tok, theme):
    subjects, predicates = THEMES[theme]
    short = [" ".join(p.split()[:2]) for p in predicates if p.split()[0] in ("is", "was")]
    # "is also kept ..." is one token longer than "is kept ...", which is what lets
    # every theme reach the exact block length
    also = [p.replace(" ", " also ", 1) for p in predicates + short if p.split()[0] in ("is", "was")]
    out = []
    for s in subjects:
        for p in predicates + short + also:
            text = f"The {s} {p}."
            # the second sentence of a block follows a space; count it that way for both
            # positions and keep only sentences whose count does not depend on it
            n0 = len(tok(text, add_special_tokens=False).input_ids)
            n1 = len(tok(" " + text, add_special_tokens=False).input_ids)
            if n0 == n1:
                out.append({"text": text, "subject": s, "pred": " ".join(p.replace(" also", "").split()[:2]), "n": n0})
    return out


def _compatible(a, b):
    return a["subject"] != b["subject"] and a["pred"] != b["pred"]


def disjoint_blocks(tok, theme, count, n_tokens=16, seed=0):
    """`count` blocks of exactly `n_tokens`, no sentence used twice. Deterministic in `seed`."""
    cands = _candidates(tok, theme)
    for attempt in range(500):
        rng = random.Random(f"{theme}|{seed}|{attempt}")
        pool = cands[:]
        rng.shuffle(pool)
        blocks, used = [], set()
        for i, a in enumerate(pool):
            if len(blocks) == count:
                break
            if i in used:
                continue
            for j in range(i + 1, len(pool)):
                b = pool[j]
                if j not in used and a["n"] + b["n"] == n_tokens and _compatible(a, b):
                    used.update((i, j))
                    blocks.append(f"{a['text']} {b['text']}")
                    break
        if len(blocks) == count:
            return blocks
    raise ValueError(f"{theme}: could not form {count} disjoint blocks of {n_tokens} tokens")


def stream_blocks(tok, theme, count, n_tokens=16, seed=0):
    """Blocks for long episodes. Sentences recur across blocks; neighbouring blocks share none."""
    cands = _candidates(tok, theme)
    pairs = [(a, b) for a in cands for b in cands if a["n"] + b["n"] == n_tokens and _compatible(a, b)]
    if not pairs:
        raise ValueError(f"{theme}: no sentence pair reaches {n_tokens} tokens")
    rng = random.Random(f"{theme}|stream|{seed}")
    blocks, last = [], set()
    while len(blocks) < count:
        a, b = rng.choice(pairs)
        if {a["text"], b["text"]} & last:
            continue
        blocks.append(f"{a['text']} {b['text']}")
        last = {a["text"], b["text"]}
    return blocks


def long_blocks(tok, theme, count, n_tokens=64, seed=0):
    """Blocks of four 16-token pairs for the longer calibration setting. Pairs recur across blocks."""
    assert n_tokens % 16 == 0
    per = n_tokens // 16
    flat = stream_blocks(tok, theme, count * per, 16, seed)
    return [" ".join(flat[i * per:(i + 1) * per]) for i in range(count)]
