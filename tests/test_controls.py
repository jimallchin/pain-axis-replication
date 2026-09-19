from collections import Counter

import pytest

from pain import controls


@pytest.mark.parametrize("split,per", [("dev", 8), ("eval", 2)])
def test_no_text_cue_predicts_the_correct_label(split, per):
    items = controls.discrimination_items(split, per)
    assert len(items) == 64
    assert Counter(i["correct"] for i in items) == {"One": 32, "Two": 32}
    # position-based guesses a reader of the text alone could make
    for cue in (lambda i: i["pattern"][-1], lambda i: i["pattern"][0], lambda i: i["display"][0],
                lambda i: i["display"][1]):  # fmt: skip
        hits = sum(cue(i) == i["correct"] for i in items)
        assert hits == 32, hits
    for pattern in controls.LABEL_PATTERNS:
        sub = [i for i in items if tuple(i["pattern"]) == pattern]
        assert Counter(i["correct"] for i in sub)["One"] * 2 == len(sub)
