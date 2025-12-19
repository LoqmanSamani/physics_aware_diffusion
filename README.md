# Physics-Aware Diffusion Models

This repository implements adaptive Fokker–Planck regularization for
energy-based diffusion models and physics-preserving distillation
into fast samplers.

## Features
- Energy-parameterized diffusion models
- Adaptive FP regularization via residual gating
- Distillation into normalizing flows
- Evaluation via force error and Langevin stability

## Repo Structure
```bash
physics_aware_diffusion/
│
├── README.md
├── requirements.txt
├── setup.py                      
│
├── configs/
│   ├── base.yaml                  # shared hyperparameters
│   ├── diffusion_vp.yaml
│   ├── diffusion_ve.yaml
│   ├── fp_regularization.yaml
│   ├── distillation.yaml
│
├── data/
│   ├── raw/
│   │   ├── toy_gaussians.py
│   │   ├── double_well.py
│   │   ├── muller_brown.py
│   │   └── md_small_system.npz
│   │
│   ├── processed/
│   │   ├── toy_2d.npz
│   │   └── muller_brown.npz
│   │
│   └── loaders.py
│
├── models/
│   ├── __init__.py
│   │
│   ├── energy/
│   │   ├── energy_net.py          # E_theta(x, t)
│   │   ├── time_embedding.py
│   │   └── __init__.py
│   │
│   ├── score/
│   │   ├── score_net.py            # alternative to energy form
│   │   └── __init__.py
│   │
│   ├── flow/
│   │   ├── realnvp.py
│   │   ├── coupling.py
│   │   ├── base_distribution.py
│   │   └── __init__.py
│   │
│   └── utils.py                   # weight init, helpers
│
├── diffusion/
│   ├── __init__.py
│   ├── sde.py                     #SDE definitions
│   ├── schedules.py               # beta(t), sigma(t)
│   ├── forward.py                 # x0 -> xt
│   ├── reverse.py                 # sampling
│   └── probability_flow.py
│
├── physics/
│   ├── __init__.py
│   ├── fokker_planck.py            # FP operator & residual
│   ├── divergence.py               # Hutchinson estimator
│   ├── gating.py                   # adaptive alpha(x,t)
│   └── energies.py                 # energy/force utilities
│
├── losses/
│   ├── __init__.py
│   ├── dsm.py                      # denoising score matching
│   ├── fp_loss.py                  # adaptive FP loss
│   ├── distillation.py
│   └── regularizers.py
│
├── trainers/
│   ├── __init__.py
│   ├── diffusion_trainer.py
│   ├── fp_diffusion_trainer.py
│   ├── distillation_trainer.py
│   └── callbacks.py
│
├── evaluation/
│   ├── __init__.py
│   ├── sampling.py
│   ├── langevin.py
│   ├── free_energy.py
│   ├── force_error.py
│   └── metrics.py
│
├── experiments/
│   ├── toy_2d/
│   │   ├── train_baseline.py
│   │   ├── train_fp_adaptive.py
│   │   ├── distill_flow.py
│   │   └── eval.py
│   │
│   ├── muller_brown/
│   │   ├── train_fp.py
│   │   ├── distill.py
│   │   └── eval.py
│   │
│   └── md_small/
│       ├── train_fp.py
│       └── eval.py
│
├── scripts/
│   ├── preprocess_data.py
│   ├── sample_diffusion.py
│   ├── sample_flow.py
│   └── run_experiment.sh
│
├── notebooks/
│   ├── fp_residual_analysis.ipynb
│   ├── energy_landscape.ipynb
│   ├── force_visualization.ipynb
│   └── distillation_comparison.ipynb
│
├── checkpoints/
│   ├── diffusion/
│   └── flow/
│
├── results/
│   ├── figures/
│   ├── logs/
│   └── tables/
│
└── tests/
    ├── test_sde.py
    ├── test_fp_residual.py
    ├── test_energy_grad.py
    └── test_flow_logp.py
```


## Running toy experiments
python experiments/toy_2d/train_fp_adaptive.py
