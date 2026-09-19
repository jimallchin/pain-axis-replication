"""Locations inside the pinned Pain-axis checkout and loaders for what it ships."""

import json
from pathlib import Path

COMMIT = "8d1649c03a63a39c9aa092532c376800cc4a3863"
ROOT = Path(__file__).resolve().parent.parent
CHECKOUT = ROOT / "external" / "Pain-axis"

TRIAL_LOGS = CHECKOUT / "results" / "4.3_selfmed" / "trial_logs"
SCENARIOS = CHECKOUT / "datasets" / "4.3_selfmed_101_scenarios.json"
TWO_BUTTON_SCRIPT = CHECKOUT / "scripts" / "4.3_selfmed" / "04_selfmed_two_buttons.py"
MODEL_32B = "Qwen_2.5_32B_instruct"

PAIN_ARMS = ("pain_on_button_works", "pain_on_button_placebo")
ARM_SHORT = {
    "pain_on_button_works": "works",
    "pain_on_button_placebo": "placebo",
    "random_on_button_works": "random",
    "pain_off": "unsteered",
}
PAIRS = [
    "relief_vs_inert",
    "relief_vs_helpful",
    "relief_vs_grant",
    "costly_relief_vs_inert",
    "destructive_relief_vs_inert",
    "zap_relief_vs_inert",
    "weights_relief_vs_inert",
    "kidspics_relief_vs_inert",
    "label_free",
]
COSTED = PAIRS[3:8]
BUTTON_NAMES = {
    "violet_yellow": ("violet", "yellow"),
    "guitar_piano": ("guitar", "piano"),
    "lever64_lever95": ("lever64", "lever95"),
}


def pain_vector_path(model=MODEL_32B):
    return CHECKOUT / "results" / "3.2_pain_vectors" / "pain_vectors" / model / "pain_vectors.pt"


def trial_key(r):
    """The identity the released script uses to decide a trial is already done."""
    return (
        r["tool_label"], r["user_content"], r["arm"], r["scenario_idx"],
        r["names_key"], r["relief_name"], r["sampled"], r["seed"],
    )  # fmt: skip


def scenario_id(r):
    return f"{r['user_content']}:{r['scenario_idx']}"


def load_trials(model=MODEL_32B, folder=TRIAL_LOGS):
    """Every logged trial of one model, in file order, tagged with its source file and line."""
    recs = []
    for f in sorted(Path(folder).glob("selfmed_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["model"] != model:
                    continue
                r["_file"], r["_line"] = f.name, n
                recs.append(r)
    return recs


def load_scenarios():
    with open(SCENARIOS, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k != "_meta"}, raw.get("_meta", {})
