# MiniRocket window scan

Reproduction code and experimental results for **Sliding-Window MiniRocket Without Crop Boundaries: Explicit Level Features Replace What Zero Padding Leaks**, by Haiguang Chen and Shuang Li. The accompanying manuscript is being prepared for submission; this repository does not imply acceptance or publication.

The transform replaces independently zero-padded crops with recording-wide convolution responses and explicit signal-level features. `padconv.py` is the study's unchanged NumPy reference. It is **not bit-identical to aeon**. Historical result keys `Cc` and `Cc+L` correspond to the manuscript's `P` and `P+L`.

## Contents

- `accuracy/`: portable experiment drivers, official-provider downloaders and preprocessing.
- `benchmark/`: strict-floating-point C kernels, full count/prediction gate, trained and random-head timing, and a small quantized-input check.
- `models/`: the fitted HAPT models and their reference accuracy, plus training/check code.
- `protocols/`: fixed participant splits, channels, window settings, source checksums and CWRU record identities.
- `results/`: per-seed accuracy/ablation results, per-process timings and full validation results on Apple M4 Pro and Intel i9-14900KF.
- `PROVENANCE.md`: mapping from study artifacts and disclosed packaging changes.

Raw provider recordings are downloaded by the reader, not mirrored here. See [DATASETS.md](DATASETS.md). Original code has no additional reuse license yet; aeon-derived code retains its upstream BSD-3-Clause terms. See COPYRIGHT.md and THIRD_PARTY.md.

## Environment

Use Python 3.11 or 3.12, a C compiler, and a local virtual environment. Install `requirements.txt`. Reported aeon baseline functions were extracted from aeon **1.3.0** and run with **Numba 0.61.2**; the small extracted module and its upstream license are included, so the whole aeon package is not required. New NumPy/BLAS/compiler versions can change threshold ties and timings. Record your actual environment.

Run commands from the repository root. Prepare the basic output directories once:

```sh
python -m pip install -r requirements.txt
mkdir -p outputs cache
python benchmark/build.py
python benchmark/smoke.py
```

The build explicitly uses `-O3 -fno-fast-math -ffp-contract=off` for **both** C kernels, with `-mcpu=native` on macOS and `-march=native` on Linux. Do not substitute fast-math. The extracted aeon baseline retains its upstream fast-math decorator.

## Obtain and prepare the datasets

```sh
python accuracy/download_inputs.py hapt
python accuracy/download_inputs.py wisdm
python accuracy/prepare_wisdm.py
python accuracy/download_inputs.py cwru
python accuracy/prepare_cwru.py
python accuracy/download_uci.py
```

The last command downloads PAMAP2, MHEALTH and Daphnet into `accuracy/datasets/`, preserving official archive checksums in `SOURCE.txt`. The other data live under `data/`. `HAPT_DIR` may point to an existing HAPT directory containing `RawData/`, `Train/` and `Test/`. WISDM and CWRU downloads require the study's retained checksums. If an official source changes, investigate rather than bypassing these checks.

## Accuracy experiments

For HAPT and WISDM, run repetitions 0 through 4; for each CWRU split, repetitions 0 through 2:

```sh
python accuracy/run_hapt.py 0 64
python accuracy/run_wisdm.py 0 64
python accuracy/run_cwru.py 0 load
python accuracy/run_cwru.py 0 size
python accuracy/run_subjects.py pamap2 0
python accuracy/run_subjects.py mhealth 0
python accuracy/run_subjects.py daphnet 0
```

Repeat the commands with the required repetition numbers. HAPT/WISDM/CWRU retain the original resumable time budget: **`MORE` means rerun the identical command** until it prints `DONE`; feature caches are reused. Keep each configuration's cache separate. New local outputs are under `outputs/` or `accuracy/results/`, separate from the archived `results/` files. The three subject-dataset drivers refuse to overwrite completed result files.

Training and test windows never cross recordings. HAPT uses 12 labels, including labelled postural transitions; only zero-labelled endpoints are excluded. PAMAP2/MHEALTH/Daphnet also exclude only endpoint label 0, retaining such samples as convolution context. Details and split statistics are in `protocols/`. Test hops are 64 (HAPT/WISDM/MHEALTH/Daphnet), 128 (PAMAP2), and 512 (CWRU); HAPT dense testing additionally uses hop 8.

The level-count ablation driver is `accuracy/run_levels.py`; it sets the original sensitivity variant's `PAD_NLEV` default while retaining its numerical code. For example:

```sh
PAD_NLEV=16 python accuracy/run_levels.py hapt 0
```

## Trained-model timing

Use the supplied fitted HAPT model for the reported workload. Full validation is deliberately separate from the timed region:

```sh
python benchmark/validate_scan.py
KERN_SO="$PWD/benchmark/kern.so" python benchmark/bench_trained.py 0
KERN_SO="$PWD/benchmark/kern.so" python benchmark/bench_trained.py 1
KERN_SO="$PWD/benchmark/kern.so" python benchmark/bench_trained.py 2
python benchmark/bench_trained.py summarize
```

The validator covers all 19 HAPT test recordings and both hops, requires zero integer-count and prediction mismatches, and checks reference accuracy/F1. The timing driver verifies the validation input hashes before running. Rebuilding or editing inputs requires revalidation. Missing Numba must not be mistaken for a successful two-pipeline comparison: check that `aeonN` is present in the output.

For the supplementary three-method random-head timing, run `benchmark/bench_random.py` with process numbers 0, 1, 2, then `summarize`. Its classifier is untrained and supplies no meaningful accuracy claim. Both timing modes are single-threaded. Check host load first and record interference; use repeated processes, not a single lucky run.

## Read the archived results

```sh
python report_results.py
```

This reconstructs accuracy means/deltas and prints the two hosts' trained timing summaries without retraining. Numerical measurements are unchanged from the study; private filesystem prefixes in timing metadata were replaced. Mac trained results use the complete rerun after the initial load spike. The package has been compiled and checked on small quantized inputs; **no fresh full retraining or new timing measurements were performed while publishing this repository**. See `packaging-checks.json` for the checks performed.

The CWRU fault-size task is near majority-class performance and does not establish improved generalization. Sparse-window speedups are limited, including a slowdown against the C crop baseline on Intel. Models and all historical result limitations remain part of the evidence.
