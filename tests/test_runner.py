import json

import pytest

pytest.importorskip("torch")
from pain.runner import RecordFile  # noqa: E402


def test_resume_skips_done_items_and_refuses_a_changed_stamp(tmp_path):
    path = tmp_path / "r.jsonl"
    rf = RecordFile(path, stamp={"config_hash": "a"})
    rf.write({"item": "x", "v": 1})
    rf.f.close()

    again = RecordFile(path, stamp={"config_hash": "a"})
    assert again.done == {"x"}
    with pytest.raises(ValueError):
        again.write({"item": "x", "v": 2})
    again.f.close()

    with pytest.raises(ValueError, match="written under"):
        RecordFile(path, stamp={"config_hash": "b"})


def test_duplicate_items_in_a_file_are_refused(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps({"item": "x"}) for _ in range(2)) + "\n")
    with pytest.raises(ValueError, match="twice"):
        RecordFile(path)
