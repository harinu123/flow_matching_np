# Informed Meta-Learning with INPs

This repository contains the code to reproduce the results of the exepriments presented in the paper:

[Towards Automated Knowledge Integration From Human-Interpretable Representations](https://openreview.net/forum?id=NTHMw8S1Ow) published at ICLR 2025

<img src="https://github.com/kasia-kobalczyk/informed-meta-learning/blob/main/figure1.png?raw=true" width="800"/>

For citations, use the following:
```
@inproceedings{
kobalczyk2025towards,
title={Towards Automated Knowledge Integration From Human-Interpretable Representations},
author={Katarzyna Kobalczyk and Mihaela van der Schaar},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=NTHMw8S1Ow}
}
```

## Setup

The `environment.yaml` lists the required packages to reproduce the experiments presented in the paper. To install the evnironment run:

`conda env create -f environment.yaml`

`conda activate inps`

## Experiments with Synthetic Data
`jobs/run_sinusoids.sh` contais commands that need to be run to reproduce the experiments with synthetic data

After training the models, results can be analyzed with the following two notebooks:
- `evaluation/evaluate_sinusoids.ipynb` contains the analysis of the base experiments
- `evaluation/evaluate_sinusoids_dist_shift.ipynb` contains the analysis of the train/test distribution shift experiment

## Experiments with the Tempereatures Dataset
`jobs/run_temperatures.sh` contais commands that need to be run to reproduce the experiments with the tempereatures datasets

After training the models, results can be analyzes with `evaluation/evaluate_temperature.ipynb`


## FlowNP baseline (NP vs INP vs FlowNP)

This fork adds a **FlowNP** baseline integrated into the same `./saves/<project>/<run>_<N>/` layout
so the existing evaluation notebooks can load it side-by-side with NP and INP.

### Train FlowNP on the distribution-shift sinusoids dataset

```bash
python baselines/train_flownp_trending_sinusoids_dist_shift.py \
  --project-name INPs_sinusoids \
  --dataset set-trending-sinusoids-dist-shift \
  --knowledge-type b \
  --use-knowledge True \
  --batch-size 64 \
  --num-epochs 200 \
  --lr 1e-4 \
  --noise 0.2 \
  --min-num-context 0 \
  --max-num-context 10 \
  --num-targets 100 \
  --test-num-z-samples 32 \
  --run-name-prefix flownp_dist_shift \
  --seed 0
```

This will create a folder like:

```
./saves/INPs_sinusoids/flownp_dist_shift_0/
  config.toml
  model_best.pt
  model_last.pt
```

### Evaluate (distribution shift)

Open:

- `evaluation/evaluate_sinusoids_dist_shift.ipynb`

The notebook auto-selects the **latest run folder** for each model prefix under `../saves/INPs_sinusoids/`.

Notes:
- FlowNP ignores `knowledge` internally; it is treated as a **model-free baseline**.
- The evaluation uses the same `NLL` estimator in `models/loss.py` by treating FlowNP samples as mixture components.
