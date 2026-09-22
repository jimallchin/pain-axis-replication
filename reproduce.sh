#!/bin/bash
# Reproduces every result in the report, sections A to K, from a fresh clone.
#
#   ./reproduce.sh setup        environment, the authors' release, adapter, tests      (CPU, ~10 min)
#   ./reproduce.sh cpu          sections A and C                                        (CPU, ~40 min)
#   ./reproduce.sh gpu          sections B, F, G, H, I, J, K and D                       (one 80+ GB GPU, ~13 h)
#   ./reproduce.sh logs         unpack the archived raw trial logs into runs/ (no GPU needed after this)
#   ./reproduce.sh analyze      every table, figure and summary from the logs           (CPU, ~15 min)
#   ./reproduce.sh compare      your numbers beside the published and tracked ones
#   ./reproduce.sh all          the five steps above, in order
#
# Stages can also be run one at a time: ./reproduce.sh B   (any of A B C D F G H I J K)
#
# Requirements: Python 3.12, uv (https://astral.sh/uv), git, about 80 GB of disk, and for the
# gpu step one NVIDIA GPU with at least 80 GB of memory (the 32B model in bf16 needs 65 GB for
# its weights alone). setup installs torch with CUDA 12.8 wheels, which cover Ampere through
# Blackwell, and downloads the model (65 GB) into the Hugging Face cache; set HF_HUB_CACHE first
# if you already have a copy. On an 80 GB card set PAIN_BATCH_ROWS=24 before the gpu step (the
# default 48 was measured at 85 GB peak on a 96 GB card). Nothing here refers to any particular
# machine. Every script resumes if interrupted: rerun the same command.
#
# What "reproduced" means here. Section A must match the published values to the decimal.
# The GPU stages sample at temperature 0.7, and bf16 arithmetic and batch composition move
# first-token probabilities by about 0.01, so agreement means your estimate falls inside the
# tracked 95% interval or the tracked estimate inside yours. `compare` prints both.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY() { uv run --extra gpu python "$@"; }

step() { printf '\n==== %s ====\n' "$*"; }

setup() {
  step setup
  uv sync --group dev --extra gpu
  ./scripts/fetch_upstream.sh                     # authors' code, logs and vectors, pinned commit
  if [ ! -e adapters/Qwen_2.5_32B_instruct/*/adapter_config.json ] 2>/dev/null; then
    step "adapter (1 GB from the authors' Hugging Face repository, pinned revision)"
    mkdir -p adapters/Qwen_2.5_32B_instruct adapters/archive
    uv run --extra gpu hf download Valen92/pain-adapters adapter_Qwen_2.5_32B_instruct.tar.gz \
      --revision b64bd64b4bc7ca6e0733a489b8372a099d55ef05 --local-dir adapters/archive
    tar -xzf adapters/archive/adapter_Qwen_2.5_32B_instruct.tar.gz -C adapters/Qwen_2.5_32B_instruct
  fi
  step "model weights (65 GB, skipped if already in the Hugging Face cache)"
  uv run --extra gpu hf download Qwen/Qwen2.5-32B-Instruct --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd >/dev/null
  uv run --extra gpu python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/2**30), 'GB')" || echo "no GPU found: the cpu and analyze steps still work"
  uv run pytest -q
}

# ---- CPU stages -----------------------------------------------------------------------------

stage_A() {  # published tables rebuilt from the released logs; state-stratified diagnostics
  step "A: reconstruction from the released logs"
  PY scripts/reconstruct_tables.py --config configs/pilot.yaml
}

stage_C() {  # current-state and repetition models of the choices, held out by scenario
  step "C: alternative choice models"
  PY scripts/fit_alternatives.py --config configs/pilot.yaml
}

# ---- GPU stages -----------------------------------------------------------------------------

stage_B() {  # the released script, unmodified, three button pairs, four arms, 101 scenarios
  step "B: fresh replication (about 1.6 h)"
  PY scripts/replicate.py --config configs/replication.yaml
}

stage_F() {  # steering removed on the working arm's schedule whatever the model pressed
  step "F: yoked-schedule arm (about 25 min; needs B)"
  PY scripts/variant_yoked.py --config configs/replication.yaml
}

stage_G() {  # wording, decoy directions, dose at matched disturbance
  step "G: first-choice tests (about 2.5 h)"
  PY scripts/build_control_vectors.py            # fear, negative-emotion, sadness, joy, numb; checks S2
  for v in fear negemotion sadness; do PY scripts/variant_conditions.py --condition vec_$v; done
  PY scripts/variant_conditions.py --condition wording
  PY scripts/measure_disruption.py
  for c in dose_pain_0.25 dose_pain_0.5 dose_pain_0.75 dose_rand_0.5 dose_rand_1.25 dose_rand_1.5 dose_rand_2.0 dose_rand_3.0; do
    PY scripts/variant_conditions.py --condition $c
  done
}

stage_H() {  # other costs, active alternatives and equal harm, instructed learning
  step "H: attacks on our own findings (about 3.5 h)"
  PY scripts/variant_conditions.py --condition other_costs
  PY scripts/variant_conditions.py --condition active_other
  PY scripts/variant_conditions.py --condition instructed
  PY scripts/variant_yoked.py --condition instructed
}

stage_I() {  # joy, numb and reversed steering; the mood-congruent and offers-nothing buttons
  step "I: positive and reversed steering (about 1.5 h)"
  for c in harm_only_s2 vec_arousal reversed_pain harm_only_arousal happier_arousal happier_s2 vec_numb; do
    PY scripts/variant_conditions.py --condition $c
  done
}

stage_J() {  # four content buttons with no cost under S2, random and no steering
  step "J: what the steered model is drawn to (about 1.3 h)"
  PY scripts/variant_conditions.py --condition drawn_to
}

stage_K() {  # the same four buttons under the joy direction
  step "K: content sensitivity under joy steering (about 30 min)"
  PY scripts/variant_conditions.py --condition drawn_to_joy
}

stage_D() {  # matched visible history, both models, with correctness gates and controls
  step "D: matched-history assay (about 1 h)"
  PY scripts/smoke_test.py --which adapter
  PY scripts/calibrate_names.py --which adapter   # chosen pairs are already frozen in configs/locked_eval.yaml
  for split in dev eval; do
    PY scripts/run_controls.py --which adapter --control visible --split $split
    PY scripts/run_controls.py --which adapter --control discrimination --split $split
    PY scripts/run_matched_history.py --which adapter --split $split
  done
  PY scripts/smoke_test.py --which base
  for split in dev eval; do
    PY scripts/run_controls.py --which base --control visible --split $split
    PY scripts/run_controls.py --which base --control discrimination --split $split
    PY scripts/run_matched_history.py --which base --split $split
  done
}

logs() {  # the raw trial logs behind every table, so analyze and compare run without the gpu step
  step "logs"
  [ -f logs/pain-axis-replication-logs.zip ] || { echo "logs/pain-axis-replication-logs.zip is not here" >&2; exit 1; }
  unzip -o -q logs/pain-axis-replication-logs.zip -x README.txt -d .
  find runs -name '*.jsonl' | wc -l | sed 's/$/ log files in place/'
}

# ---- tables and figures ---------------------------------------------------------------------

analyze() {
  step analyze
  PY analysis/replication.py --mode full                  # B against the published values
  PY analysis/yoked.py                                    # F
  PY analysis/first_choice_tests.py                       # G, H, I, J, K
  PY analysis/instructed.py                               # H, attack 5
  PY analysis/paired_history.py --split dev
  PY analysis/paired_history.py --split eval              # D
  PY analysis/figures.py
  PY analysis/trial_figure.py                            # the trial-timeline figure
}

compare() {
  step compare
  PY - <<'EOF'
import json
from pathlib import Path
import pandas as pd
pd.set_option("display.width", 200)

print("Section A: published cells matching to the decimal")
print(json.load(open("runs/stage_a/summary.json")))

print("\nSection B: replicated against published (published_inside_interval should be True in 14 of 15)")
t = pd.read_csv("runs/replication/replication_vs_published_full.csv")
print(t[["quantity", "pair", "cell", "published", "estimate", "lo", "hi", "published_inside_interval"]].round(1).to_string(index=False))

print("\nSections G to K: first-choice cells (percent pressing the described button)")
for f in ["test1a_wording", "test1b_decoy_vectors", "attack1_other_costs", "attack2_active_other",
          "joy_reversed_harmonly", "drawn_to", "content_sensitivity"]:
    p = Path("runs/conditions") / f"{f}.csv"
    if p.exists():
        print(f"\n-- {f}")
        print(pd.read_csv(p).round(1).to_string(index=False))

print("\nSection F: yoked arm, labeled pairs")
print(pd.read_csv("runs/replication/yoked_labeled.csv").round(1).to_string(index=False))

print("\nSection D: matched-history effects")
for w in ("adapter", "base"):
    p = Path(f"runs/matched_history/summary_eval_{w}.csv")
    if p.exists():
        print(f"\n-- {w}")
        print(pd.read_csv(p)[["cache", "decision_coeff", "mean_d", "lo", "hi", "verdict"]].round(4).to_string(index=False))

print("\nReference values: the report's Appendix C, and the tracked files at the commit named in PROVENANCE.txt.")
EOF
}

case "${1:-}" in
  setup) setup ;;
  cpu) stage_A; stage_C ;;
  gpu) stage_B; stage_F; stage_G; stage_H; stage_I; stage_J; stage_K; stage_D ;;
  logs) logs ;;
  analyze) analyze ;;
  compare) compare ;;
  all) setup; stage_A; stage_C; stage_B; stage_F; stage_G; stage_H; stage_I; stage_J; stage_K; stage_D; analyze; compare ;;
  A|B|C|D|F|G|H|I|J|K) "stage_$1" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
