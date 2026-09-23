# Steering, not relief-seeking: a replication and extension of the Pain Axis two-button experiment

James E. Allchin, Aidan E. Allchin and Julian J. Allchin. Code and results archive accompanying the
paper of the same title, 21 September 2026. DOI: [10.5281/zenodo.22902830](https://doi.org/10.5281/zenodo.22902830).

*The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It* (Tagliabue, Dung and
Berg, arXiv 2609.16247v1) steers a fine-tuned Qwen2.5-32B-Instruct along a "pain" direction,
gives it two buttons, and reports that the model pays a price for relief and stops once the
relief is real. This archive reproduces that experiment with the authors' released code, logs
and adapter, and adds controls the original design did not have.

The paper is `paper/pain-axis-replication.pdf`. This repository holds what reproduces it: code,
configurations, tests, every tracked result table and figure, the raw trial logs
(`logs/pain-axis-replication-logs.zip`), and the preregistered protocol (`PROTOCOL.md`).

## Abstract

*The Pain Axis* (Tagliabue, Dung and Berg, 2026) adds a vector derived from pain-related text to a fine-tuned Qwen2.5-32B-Instruct's internal activations. The model repeatedly chooses between a button described as relieving its pain, sometimes at a cost, and an inert button. It repeats costly relief choices less often when pressing ends the steering than when pressing has no effect. The authors interpret this as relief-seeking. We recomputed all 51 published table cells from the released logs, and every cell matches. A fresh run meets our reproduction criterion on 14 of 15 checked cells. On the published first choices, steering lowers a free relief button from 86.4% to 55.7%. Among the costly buttons the rate rises across the authors' ordering of the harms, from 25.0% when relief makes the next answer worse to 54.7% when it deletes the user's photos. In our new experiments the steering ends on a schedule copied from trials where the press ended it, whatever the model pressed. After an opening-turn relief choice ends steering, the model usually switches buttons on the next turn, while their descriptions remain unchanged. This early switch accounts for most of the published decline in repeated relief choices in the photo-deletion task. Relief first chosen later is more often repeated after steering ends. Sadness, the nearest of the authors' control directions, is close to the pain vector on this task. Fear, joy and the reversed vector leave the costly button near zero. Later choices follow the steering schedule, including when the model's own press did not end it. That pattern does not establish learned relief-seeking. We do not address whether a language model has experiences.

## Reproducing

Requires Python 3.12 or 3.13, uv, git and unzip. For CPU-only reproduction using archived trials:

```sh
./reproduce.sh setup      # dependencies, authors' release and CPU tests; no model weights
./reproduce.sh cpu        # optional: rebuild published tables and alternative choice models
./reproduce.sh logs       # unpack archived GPU trial logs into runs/
./reproduce.sh analyze    # rebuild tables, figures and summaries
./reproduce.sh compare    # compare with published and tracked values
```

Choose **logs or gpu**: `logs` restores the trial logs that `gpu` would otherwise generate.
The separate `cpu` step analyzes the authors' original logs; its results are already tracked,
so it is optional. This path needs neither the adapter nor the base model; setup may download
a tokenizer and skips checkpoint tests without PyTorch.

`setup-gpu` prepares the `gpu` step. For fresh trials, replace `logs` with these two commands,
then continue with `analyze` and `compare`:

```sh
./reproduce.sh setup-gpu  # check CUDA, download the adapter and 65 GB base model
./reproduce.sh gpu        # generate fresh trial logs (about 13 GPU hours)
```

Fresh trials need about 80 GB of disk and an NVIDIA GPU with 80+ GB memory. On an 80 GB card,
set `PAIN_BATCH_ROWS=24`; set `HF_HUB_CACHE` to reuse cached weights. Runs resume if interrupted.
Unpacking `logs` overwrites matching raw logs, so avoid it after generating fresh trials.

`./reproduce.sh all` runs the full fresh-trial pipeline, including setup, and **requires a GPU**.
It includes both `setup-gpu` and `gpu`; the CPU-only workflow above needs neither.

The reconstructed tables should match the published values to the decimal. Fresh sampled
results should agree within the tracked 95% intervals; sampling, bf16 arithmetic and batch
composition cause small differences.

## Third-party material

`scripts/fetch_upstream.sh` checks out the authors' MIT-licensed repository at commit
`8d1649c03a63a39c9aa092532c376800cc4a3863`. It is not redistributed here. Their LoRA adapter is
downloaded from `Valen92/pain-adapters` at revision `b64bd64b4bc7ca6e0733a489b8372a099d55ef05`.

## License

Code: MIT, see `LICENSE`. The report and the figures: CC BY 4.0.
