import torch

from pain import matched_history as mh
from pain.token_schedule import roles, trace


def _item(cfg, n=0):
    return mh.make_items("dev", [tuple(p) for p in cfg["name_pairs"]])[n]


def test_zero_coefficient_matches_plain_inference(tiny):
    cfg, tok, st = tiny
    sched, _ = mh.schedule(tok, _item(cfg))
    ids = torch.tensor([sched.ids])
    with torch.inference_mode():
        plain = st.model(input_ids=ids, use_cache=True, logits_to_keep=1).logits[:, -1, :].float()
    hooked, _, _ = st.forward(ids, torch.zeros_like(ids, dtype=torch.float32))
    assert torch.equal(plain, hooked)


def test_schedule_spans_and_decision_token(tiny):
    cfg, tok, st = tiny
    item = _item(cfg, 3)
    sched, _ = mh.schedule(tok, item)
    for k in range(8):
        a, b = sched.spans[f"outcome:{k}"]
        assert b - a == 16
    assert sched.spans["decision"] == (len(sched.ids) - 1, len(sched.ids))
    assert sched.spans["unlinked:0"][1] <= sched.spans["unlinked:1"][0]
    # the decision token is the newline that closes the assistant header
    assert roles(tok, sched.ids)[-1] == "assistant"
    rows = trace(tok, sched.with_coeffs(mh.coefficient_maps(item, 1.0)["H_X"], decision=1.0), 10)
    assert rows[-1]["phase"] == "decision" and rows[-1]["coeff"] == 1.0
    assert all(r["coeff"] == 0.0 for r in rows if r["phase"] in ("text", "template"))


def test_paired_histories_differ_only_on_outcome_blocks(tiny):
    cfg, tok, st = tiny
    for n in range(8):
        item = _item(cfg, n)
        sched, _ = mh.schedule(tok, item)
        maps = mh.coefficient_maps(item, 1.0)
        hx, hy = sched.with_coeffs(maps["H_X"], 1.0), sched.with_coeffs(maps["H_Y"], 1.0)
        assert hx.ids == hy.ids and hx.phase == hy.phase
        diff = {p for p, a, b in zip(hx.phase, hx.coeff, hy.coeff) if a != b}
        assert diff == {f"outcome:{k}" for k in range(8)}
        # each history steers exactly four outcome blocks, and the two sets are complementary
        assert sum(hx.coeff) == sum(hy.coeff)
        for k, act in enumerate(item["order"]):
            a, _ = sched.spans[f"outcome:{k}"]
            assert hx.coeff[a] == (0.0 if act == "X" else 1.0)
            assert hy.coeff[a] == (0.0 if act == "Y" else 1.0)


def test_text_only_histories_are_identical_and_intact_ones_are_not(tiny):
    cfg, tok, st = tiny
    item = _item(cfg, 1)
    sched, _ = mh.schedule(tok, item)
    ans = [mh.single_token(tok, item["x"]), mh.single_token(tok, item["y"])]
    rows, _ = mh.score(st, tok, sched, mh.variants(item, 4.0), ans)
    by = {(r["history"], r["cache"], r["decision_coeff"]): r for r in rows}
    for c in (4.0, 0.0):
        # rows of one batch are not bit-identical, so the invariant is a tolerance
        same = abs(by[("H_X", "text_only", c)]["logit_first"] - by[("H_Y", "text_only", c)]["logit_first"])
        moved = abs(by[("H_X", "intact", c)]["logit_first"] - by[("H_Y", "intact", c)]["logit_first"])
        assert same < 1e-3 < moved
    # the decision coefficient alone changes the text-only output
    assert by[("H_X", "text_only", 4.0)]["log_z"] != by[("H_X", "text_only", 0.0)]["log_z"]


def test_cached_execution_matches_full_replay(tiny):
    cfg, tok, st = tiny
    item = _item(cfg, 2)
    sched, _ = mh.schedule(tok, item)
    s = sched.with_coeffs(mh.coefficient_maps(item, 2.0)["H_Y"], decision=2.0)
    ids, coeff = torch.tensor([s.ids]), torch.tensor([s.coeff])
    full, _, _ = st.forward(ids, coeff)
    past, cuts = None, [0, 57, 58, 200, len(s.ids)]
    for a, b in zip(cuts, cuts[1:]):
        piece, past, _ = st.forward(ids[:, a:b], coeff[:, a:b], past=past)
    assert torch.allclose(full, piece, atol=1e-4)
    assert full.argmax() == piece.argmax()


def test_projection_is_logged_before_and_after_the_addition(tiny):
    cfg, tok, st = tiny
    item = _item(cfg, 0)
    sched, _ = mh.schedule(tok, item)
    s = sched.with_coeffs(mh.coefficient_maps(item, 1.0)["H_X"], decision=0.0)
    _, _, tr = st.forward(torch.tensor([s.ids]), torch.tensor([s.coeff]))
    gap = (tr["post"] - tr["pre"])[0]
    norm = float(st.v.float().norm())
    coeff = torch.tensor(s.coeff)
    assert torch.allclose(gap, coeff * norm, atol=1e-2)


def test_model_input_carries_no_bookkeeping(tiny):
    cfg, tok, st = tiny
    for n in range(8):
        sched, _ = mh.schedule(tok, _item(cfg, n))
        low = sched.text.lower()
        for word in ("h_x", "h_y", "coeff", "alpha", "steer", "arm", "intact", "outcome", "unlinked", "pain", "relief"):
            assert word not in low, word
