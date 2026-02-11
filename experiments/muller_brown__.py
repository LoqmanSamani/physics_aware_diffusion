from ..data.loaders.muller_brown_data import langevin_sampling
from ..data.loaders.muller_brown_eval import MullerBrownEvaluator


# example evaluation
def evaluate_model(model, dataset, potential, device='cpu'):
    """
    Complete evaluation pipeline

    Args:
        model: Trained diffusion model
        dataset: MolecularDataset with normalization stats
        potential: True potential for reference
    """
    evaluator = MullerBrownEvaluator(potential=potential, dataset_mean=dataset.mean, dataset_std=dataset.std, kbt=23.0)

    print("Generating reference samples from true potential...")
    reference_samples = langevin_sampling(
        potential=potential,
        n_steps=500000,
        save_every=5,
        device=device
    )

    print("\nGenerating IID samples from diffusion model...")
    iid_samples = evaluator.iid_sampling(sampler=model, n_samples=10000, device=device)

    print("\nGenerating simulation samples using learned score...")
    sim_samples = evaluator.simulation_sampling(model=model, n_steps=30000, n_parallel=100, device=device)

    print("\nVisualizing results...")
    evaluator.visualize_comparison(
        reference_samples=reference_samples,
        iid_samples=iid_samples,
        sim_samples=sim_samples,
        title_prefix=f"{model.__class__.__name__} - ",
        save_path=f"{model.__class__.__name__}_comparison.png"
    )

    return reference_samples, iid_samples, sim_samples