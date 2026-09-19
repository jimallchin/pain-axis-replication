import os

import pytest

os.environ.setdefault("PAIN_DEVICE", "cpu")


@pytest.fixture(scope="session")
def tiny():
    """Small checkpoint with a random steering direction, float32 on CPU."""
    pytest.importorskip("torch")
    from pain import config, runner

    cfg = config.load(config.ROOT / "configs" / "tiny.yaml")
    tok, steerer, _ = runner.open_model(cfg, "base")
    return cfg, tok, steerer
