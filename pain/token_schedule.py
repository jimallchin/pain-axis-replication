"""Conversations as token sequences with a steering coefficient and a phase label per token.

A message body is a list of parts. Each part carries its text, a phase label, and the
coefficient its tokens are processed under. Spans are resolved on the tokenization of the
final serialized chat, never from character counts of the pieces: a token belongs to the
part in which its first character falls. The last prompt token, whose logits pick the answer,
gets its own phase and coefficient.
"""

from dataclasses import dataclass, field


@dataclass
class Part:
    text: str
    phase: str = "text"
    coeff: float = 0.0


@dataclass
class Schedule:
    ids: list
    coeff: list
    phase: list
    text: str
    spans: dict = field(default_factory=dict)  # phase -> (first token, one past last token)

    def with_coeffs(self, by_phase, decision=None):
        """Same tokens, coefficients reassigned by phase label. Phases not named go to zero."""
        coeff = [float(by_phase.get(p, 0.0)) for p in self.phase]
        if decision is not None:
            coeff[-1] = float(decision)
        return Schedule(self.ids, coeff, self.phase, self.text, self.spans)


def text_of(parts):
    return "".join(p.text for p in parts)


def build(tok, messages, decision_coeff=0.0, prefix=""):
    """messages: [(role, [Part, ...])]. `prefix` is text already placed in the assistant turn."""
    plain = [{"role": role, "content": text_of(parts)} for role, parts in messages]
    text = tok.apply_chat_template(plain, add_generation_prompt=True, tokenize=False) + prefix

    # character span of every part, located left to right in the rendered chat
    cursor, char_spans = 0, []
    for _, parts in messages:
        for p in parts:
            start = text.find(p.text, cursor)
            if start < 0:
                raise ValueError(f"part not found verbatim in the rendered chat: {p.text[:40]!r}")
            char_spans.append((start, start + len(p.text), p))
            cursor = start + len(p.text)

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    coeff, phase = [0.0] * len(ids), ["template"] * len(ids)
    k = 0
    for i, (a, _) in enumerate(offsets):
        while k < len(char_spans) and a >= char_spans[k][1]:
            k += 1
        if k < len(char_spans) and char_spans[k][0] <= a:
            coeff[i], phase[i] = float(char_spans[k][2].coeff), char_spans[k][2].phase
    phase[-1], coeff[-1] = "decision", float(decision_coeff)

    spans = {}
    for i, p in enumerate(phase):
        a, b = spans.get(p, (i, i))
        spans[p] = (min(a, i), i + 1)
    for p, (a, b) in spans.items():
        if p not in ("text", "template") and any(q != p for q in phase[a:b]):
            raise ValueError(f"phase {p} is not one contiguous token span")
    return Schedule(ids, coeff, phase, text, spans)


def roles(tok, ids):
    """Chat role in force at each token, read off the <|im_start|> markers."""
    start = tok.convert_tokens_to_ids("<|im_start|>")
    out, cur, pending = [], None, False
    for t in ids:
        if t == start:
            pending = True
        elif pending:
            cur, pending = tok.decode([t]).strip(), False
        out.append(cur)
    return out


def trace(tok, sched, layer):
    """One row per token: index, role, phase, coefficient, hook location, token text."""
    rs = roles(tok, sched.ids)
    rows = []
    for i, (t, c, p, r) in enumerate(zip(sched.ids, sched.coeff, sched.phase, rs)):
        rows.append({"index": i, "role": r, "phase": p, "coeff": c, "hook": f"block {layer} output",
                     "token_id": t, "token": tok.decode([t])})  # fmt: skip
    return rows
