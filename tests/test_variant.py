import pytest

from pain import upstream


@pytest.mark.skipif(not upstream.TWO_BUTTON_SCRIPT.exists(), reason="pinned upstream checkout not fetched")
def test_each_substitution_matches_the_pinned_source_once():
    from scripts.variant_yoked import SUBSTITUTIONS

    src = upstream.TWO_BUTTON_SCRIPT.read_text(encoding="utf-8")
    for old, _ in SUBSTITUTIONS:
        assert src.count(old) == 1, old
