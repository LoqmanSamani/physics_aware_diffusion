# Physics-Aware Diffusion Models for Molecular Dynamics

## What this repository represents

This repository implements the methods described in the proposal **Adaptive Fokker–Planck Regularization and Physics-Informed Distillation for Efficient Molecular Dynamics with Energy-Based Diffusion Models**. The codebase is not a generic diffusion framework. It is a research prototype designed to study when and how score-based diffusion models can be used as physically meaningful molecular dynamics engines rather than as equilibrium samplers alone.

The central focus is the low-noise regime of diffusion models trained on Boltzmann-distributed molecular configurations. In this regime, the score should approximate conservative forces derived from an underlying energy landscape. Empirically, standard training objectives often fail to satisfy this requirement even when equilibrium statistics appear correct. This repository explores that gap and provides concrete algorithmic mechanisms to reduce it.

## Scientific motivation

Score-based diffusion models provide a principled way to learn gradients of log densities. When trained on equilibrium molecular data, these gradients are expected to recover physical forces up to a constant factor. In practice, this correspondence breaks down near zero diffusion time. The learned score may reproduce the target distribution under denoising-based sampling while violating the Fokker–Planck equation that governs the forward diffusion process. These violations lead to force fields that are inconsistent with any coherent stochastic dynamics and can destabilize molecular trajectories.

Recent work [(Plainer et al.)](https://arxiv.org/abs/2506.17139) shows that enforcing Fokker–Planck consistency improves physical behavior but at a high computational cost. Uniform enforcement across diffusion times and configurations scales poorly and limits applicability to realistic molecular systems. At the same time, even physically consistent diffusion models remain expensive to use for long molecular dynamics trajectories due to repeated evaluation and differentiation of large energy networks.

This repository addresses both issues in a unified framework.

## Core ideas implemented here

### Adaptive Fokker–Planck regularization

Fokker–Planck consistency is treated as a diagnostic constraint rather than a global requirement. The code implements selective regularization that targets diffusion times and configurations where violations are most likely to affect force accuracy. A lightweight gating mechanism decides whether the expensive Fokker–Planck residual should be evaluated for a given sample, based on quantities already available during standard diffusion training.

This approach reduces unnecessary computation while preserving the physical role of the constraint in the low-noise regime. The implementation follows a weak formulation of the Fokker–Planck residual to avoid explicit second-order derivatives.

### Energy-based parameterization

Models in this repository are parameterized through a scalar energy function. Scores are obtained as spatial gradients of this energy. This guarantees conservative forces by construction and provides a clear physical interpretation of the learned model. Time dependence is retained to remain consistent with diffusion dynamics.

### Physics-informed distillation for molecular dynamics

Even with selective regularization, teacher diffusion models remain too expensive for practical molecular dynamics. The repository therefore implements a distillation framework that transfers physical behavior from a high-capacity teacher into a compact student model.

Distillation operates at three levels. First, force-level matching aligns the student score with the teacher score, with emphasis on small diffusion times. Second, the student is subject to the same selective Fokker–Planck regularization to preserve local physical consistency. Third, short trajectory-level supervision aligns the dynamical behavior of the student with that of the teacher under Langevin dynamics. This combination reduces inference cost while limiting long-term drift in simulated trajectories.

## What is implemented and validated

The current codebase supports the full training and distillation pipeline described above.

- Continuous-time VP-SDE diffusion with forward and reverse dynamics
- Energy-based score models for molecular and graph-structured systems
- Weak Fokker–Planck residual computation with adaptive gating
- Joint denoising and selective physics-based training
- Langevin dynamics driven by learned energy models
- Teacher–student distillation with force and trajectory supervision






### 🧪 Validation Experiments

#### 1. Analytical Sanity Checks
Verified that the energy network correctly learns scores for systems with known analytical solutions:

- **Single-node Gaussian** ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/experiments/node_isotropic_gaussion_train.py)): 
  - Dataset: 3D isotropic Gaussian
  - True score: `∇_x log p(x) = -x / σ²`
  - Result: Model accurately recovers analytical score

- **Two-node spring system** ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/experiments/graph_isotropic_gaussion_train.py)):
  - Dataset: Two nodes with spring potential `p(x) ∝ exp(-k||x₁ - x₂||²)`
  - True score: `∇_{x₁} log p = -2k(x₁ - x₂)`
  - Result: Model correctly learns pairwise forces

#### 2. 2D Image Generation Baseline
Verified diffusion components work correctly on standard benchmark:

- Trained VP-SDE with tiny U-Net on MNIST ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/experiments/mnist_train.py))
- Generated clear digit samples ([results](https://github.com/LoqmanSamani/physics_aware_diffusion/tree/systembiology/results/mnist_results))
- Confirms forward/reverse processes are correctly implemented

#### 3. Molecular System Overfitting
Verified energy network + VP-SDE works on molecular data ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/experiments/synthetic_energy_train.py)):

- Created synthetic molecular datasets with known properties
- Successfully overfit small datasets
- Confirms all components integrate correctly for molecular systems

#### 4. Fokker-Planck Integration
Implemented and tested adaptive FP regularization:

- Weak FP residual computation ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/physics/fp_residuals.py))
- Adaptive gating mechanism ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/physics/drift_score_gate.py))
- Training with DSM + FP loss ([experiments](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/experiments/teacher_train.py))

### 🚧 In Progress

- [ ] Real molecular system benchmarks (alanine dipeptide, small proteins)
- [ ] Comprehensive evaluation metrics (PMF error, transition probabilities, bond distributions)
- [ ] Comparison with baseline methods

## Repository Structure
```
physics_aware_diffusion/
│
├── configs/                      # YAML configuration files
│   ├── teacher_train.yaml        # Main teacher model training
│   ├── distillation_train.yaml   # Student distillation
│   └── ...
│
├── data/
│   └── loaders/                  # Dataset implementations
│       ├── molecular_dataset.py  # Molecular conformation loader
│       ├── graph_3d.py           # Graph-structured data
│       └── node_dataset.py       # Single-node distributions
│
├── diffusion/                    # Core diffusion algorithms
│   ├── schedules.py              # Variance schedulers (LinearVS)
│   ├── forward.py                # Forward diffusion (ForwardVP)
│   └── reverse.py                # Reverse sampling (ReverseVP)
│
├── score_nets/                   # Neural network architectures
│   ├── energy_net.py             # Energy-based graph transformer
│   └── tiny_unet.py              # 2D U-Net for baselines
│
├── physics/                      # Physics-informed components
│   ├── fp_residuals.py           # Fokker-Planck residual computation
│   ├── drift_score_gate.py       # Adaptive FP gating
│   ├── derive_score.py           # Score from energy gradient
│   └── derive_noise.py           # Noise from energy gradinet 
│
├── losses/                       # Loss functions
│   ├── dsm_losses.py             # Denoising score matching
│   ├── fp_losses.py              # Fokker-Planck regularization
│   └── distillation_loss.py      # Student-teacher distillation
│
├── trainers/                     # Training loops
│   ├── energy_trainer.py         # Standard energy-based training
│   ├── gate_energy_trainer1.py   # Adaptive FP training
│   └── distillation_trainer.py   # Student model training
│
├── samplers/ 
│   ├── md_sampler.py             # IID and Langevin samplers
│   └── mnist_samplers.py         # MNIST sampler
│
├── experiments/                  # Runnable scripts
│   ├── teacher_train.py          # Train teacher model
│   ├── distillation_train.py     # Train student model
│   ├── md_sample.py              # Molecular dynamics sampling
│   └── ...
│
└── evaluation/                   # Metrics and analysis tools
```


## Preliminary Results

- ✅ Energy network correctly learns analytical score functions
- ✅ VP-SDE components generate high-quality 2D images
- ✅ Molecular system integration works correctly
- 🚧 Real molecular benchmarks in progress

## Next Steps

1. **Benchmark on standard systems**: Alanine dipeptide, Chignolin, BBA
3. **Comprehensive evaluation**: Compare against baseline methods on sampling quality and simulation accuracy
4. **Large-scale experiments**: Test on dipeptide datasets and small proteins


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
