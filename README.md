# Energy-Based Diffusion Models with Adaptive Fokker-Planck Regularization

## Overview

Energy-based diffusion models trained on molecular dynamics (MD) simulations can generate equilibrium molecular configurations, potentially reducing the need for expensive MD simulations. However, these models face several critical limitations:

1. **Training instability** – Standard training procedures often fail to converge reliably
2. **Expensive inference** – Generating MD trajectories requires running the diffusion model for each configuration, scaling linearly with trajectory length
3. **Physical inconsistency** – The learned score function at small diffusion times violates the Fokker-Planck equation, producing correct equilibrium distributions but incorrect dynamics

Recent work has shown that enforcing Fokker-Planck consistency during training improves physical validity, but at significant computational cost.

## Research Goals

This project develops methods to make energy-based diffusion models both **physically accurate** and **computationally efficient** through:

1. **Adaptive Fokker-Planck regularization** – Apply the expensive Fokker-Planck constraint only when violations exceed a threshold, rather than at every training step
2. **Progressive distillation** – Compress the physically consistent model into a faster student model that:
   - Uses fewer diffusion steps for independent sampling
   - Takes larger timesteps during MD simulation
   - Preserves both equilibrium statistics and conservative forces

The goal is to achieve substantial speedups while maintaining thermodynamic and dynamic correctness.

## Current Implementation Status

### ✅ Core Components (Complete)

- **VP-SDE diffusion framework**: Forward process, reverse sampling, variance scheduler
- **Energy network**: Graph transformer with conservative parameterization for molecular systems
- **Fokker-Planck regularization**: Weak residual formulation with adaptive gating
- **Training infrastructure**: Combined DSM + FP loss with batch processing
- **Sampling methods**: Both independent (iid) sampling and Langevin dynamics simulation

### 🧪 Validation Experiments (Complete)

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

- Trained VP-SDE with tiny U-Net on MNIST
- Generated clear digit samples ([results](https://github.com/LoqmanSamani/physics_aware_diffusion/tree/systembiology/results/mnist_results))
- Confirms forward/reverse processes are correctly implemented

#### 3. Molecular System Overfitting
Verified energy network + VP-SDE works on molecular data:

- Created synthetic molecular datasets with known properties
- Successfully overfit small datasets
- Confirms all components integrate correctly for molecular systems

#### 4. Fokker-Planck Integration
Implemented and tested adaptive FP regularization:

- Weak FP residual computation ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/physics/fp_residuals.py))
- Adaptive gating mechanism ([code](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/physics/drift_score_gate.py))
- Training with FP loss ([experiments](https://github.com/LoqmanSamani/physics_aware_diffusion/blob/systembiology/experiments/teacher_train.py))

### 🚧 In Progress

- [ ] Real molecular system benchmarks (alanine dipeptide, small proteins)
- [ ] Distillation pipeline for acceleration
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
│   └── derive_noise.py           # Noise prediction utilities
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
├── experiments/                  # Runnable scripts
│   ├── teacher_train.py          # Train teacher model
│   ├── distillation_train.py     # Train student model
│   ├── md_sample.py              # Molecular dynamics sampling
│   └── ...
│
└── evaluation/                   # Metrics and analysis tools
```

## Quick Start

### Installation
```bash
git clone https://github.com/LoqmanSamani/physics_aware_diffusion
cd physics_aware_diffusion
pip install -r requirements.txt
```

### Train Teacher Model
```bash
python experiments/teacher_train.py --config configs/teacher_train.yaml
```

### Generate Samples
```bash
python experiments/md_sample.py --checkpoint checkpoints/teacher_model.pt
```

## Key Technical Details

### Conservative Energy Parameterization
Following recent work on energy-based diffusion models, we parameterize the score as:
```
s_θ(x,t) = ∇_x log p_θ(x,t) = ∇_x E_θ(x,t)
```

where `E_θ` is the learned energy function. This ensures:
- Conservative forces for MD simulation
- Gradient flow through molecular geometry
- Physical consistency between sampling and simulation

### Weak Fokker-Planck Residual
The Fokker-Planck equation for diffusion processes is:
```
∂_t log p_t(x) = 0.5 g²(t)[div_x(s) + ||s||²] - ⟨f, s⟩ - div_x(f)
```

We use the weak formulation with Gaussian perturbations to avoid expensive second-order derivatives while maintaining unbiased estimation.

### Adaptive Gating
Instead of applying FP regularization at every training step, we:
1. Monitor FP residual during training
2. Apply regularization only when residual exceeds threshold
3. Reduce computational cost while maintaining physical accuracy

## Preliminary Results

- ✅ Energy network correctly learns analytical score functions
- ✅ VP-SDE components generate high-quality 2D images
- ✅ Molecular system integration works correctly
- 🚧 Real molecular benchmarks in progress

## Next Steps

1. **Benchmark on standard systems**: Alanine dipeptide, Chignolin, BBA
2. **Implement distillation**: Compress teacher into fast student model
3. **Comprehensive evaluation**: Compare against baseline methods on sampling quality and simulation accuracy
4. **Large-scale experiments**: Test on dipeptide datasets and small proteins

## Contact

This work is part of my research portfolio for PhD applications. I have an MSc in Computational Biology and am seeking research positions/internships in machine learning for molecular simulation.

**GitHub**: [LoqmanSamani](https://github.com/LoqmanSamani)  
**Project**: [physics_aware_diffusion](https://github.com/LoqmanSamani/physics_aware_diffusion)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon:
- *"Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models"* (Plainer et al., NeurIPS 2025)
- Methods for conservative score parameterization and Fokker-Planck regularization

## References
```bibtex
@inproceedings{plainer2025consistent,
  title={Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models},
  author={Plainer, Michael and Wu, Hao and Klein, Leon and G{\"u}nnemann, Stephan and No{\'e}, Frank},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
