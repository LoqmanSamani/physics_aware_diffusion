import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class MolecularVisualizer:
    """visualize molecules and diffusion process using matplotlib and plotly"""
    def __init__(self, use_plotly: bool = False):
        self.use_plotly = use_plotly
        self.atom_colors = {
            'default': '#1f77b4',
            'H': '#FFFFFF',
            'C': '#909090',
            'N': '#3050F8',
            'O': '#FF0D0D',
            'S': '#FFFF30',
        }

    def plot_molecule(
            self,
            pos: torch.Tensor,
            edge_index: Optional[torch.Tensor] = None,
            atom_types: Optional[List[str]] = None,
            title: str = "Molecule",
            save_path: Optional[str] = None,
            show: bool = True,
            ax: Optional[plt.Axes] = None
    ):
        """
        plot a single molecule in 3D.

        arguments:
            pos: (N, 3) atom positions
            edge_index: (2, E) edge connectivity
            atom_types: List of atom type strings (e.g., ['C', 'H', 'N'])
            title: Plot title
            save_path: If provided, save figure to this path
            show: Whether to display the plot
            ax: Existing matplotlib axis to plot on
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()

        if self.use_plotly:
            return self._plot_molecule_plotly(pos, edge_index, atom_types, title, save_path)
        else:
            return self._plot_molecule_matplotlib(pos, edge_index, atom_types, title, save_path, show, ax)

    def _plot_molecule_matplotlib(
            self,
            pos: np.ndarray,
            edge_index: Optional[torch.Tensor],
            atom_types: Optional[List[str]],
            title: str,
            save_path: Optional[str],
            show: bool,
            ax: Optional[plt.Axes]
    ):
        """matplotlib version of molecule plotting"""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            created_fig = False
        if atom_types is not None:
            colors = [self.atom_colors.get(at, self.atom_colors['default']) for at in atom_types]
        else:
            colors = [self.atom_colors['default']] * len(pos)
        if edge_index is not None:
            edge_index = edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                if src < dst:
                    ax.plot(
                        [pos[src, 0], pos[dst, 0]],
                        [pos[src, 1], pos[dst, 1]],
                        [pos[src, 2], pos[dst, 2]],
                        'gray', alpha=0.5, linewidth=1.5
                    )

        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=colors, s=200, alpha=0.9,
            edgecolors='black', linewidths=1.5
        )
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        max_range = np.abs(pos).max()
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        if save_path and created_fig:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show and created_fig:
            plt.show()
        return ax

    def _plot_molecule_plotly(
            self,
            pos: np.ndarray,
            edge_index: Optional[torch.Tensor],
            atom_types: Optional[List[str]],
            title: str,
            save_path: Optional[str]
    ):
        """plotly version (interactive 3D)"""
        fig = go.Figure()
        if edge_index is not None:
            edge_index = edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                if src < dst:
                    fig.add_trace(go.Scatter3d(
                        x=[pos[src, 0], pos[dst, 0], None],
                        y=[pos[src, 1], pos[dst, 1], None],
                        z=[pos[src, 2], pos[dst, 2], None],
                        mode='lines',
                        line=dict(color='gray', width=4),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        if atom_types is not None:
            colors = [self.atom_colors.get(at, self.atom_colors['default']) for at in atom_types]
        else:
            colors = [self.atom_colors['default']] * len(pos)
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                line=dict(color='black', width=2)
            ),
            text=[f'Atom {i}' for i in range(len(pos))],
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
            showlegend=False
        ))
        max_range = np.abs(pos).max() * 1.2
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-max_range, max_range], title='X (Å)'),
                yaxis=dict(range=[-max_range, max_range], title='Y (Å)'),
                zaxis=dict(range=[-max_range, max_range], title='Z (Å)'),
                aspectmode='cube'
            ),
            width=800,
            height=700
        )
        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    def plot_batch(
            self,
            positions: List[torch.Tensor],
            edge_indices: Optional[List[torch.Tensor]] = None,
            titles: Optional[List[str]] = None,
            ncols: int = 4,
            figsize: Tuple[int, int] = (16, 12),
            save_path: Optional[str] = None
    ):
        """
        plot multiple molecules in a grid.

        arguments:
            positions: List of (N_i, 3) position tensors
            edge_indices: List of edge_index tensors
            titles: List of titles for each molecule
            ncols: Number of columns in grid
            figsize: Figure size
            save_path: If provided, save figure
        """
        n_molecules = len(positions)
        nrows = (n_molecules + ncols - 1) // ncols
        fig = plt.figure(figsize=figsize)
        for idx, pos in enumerate(positions):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
            edge_index = edge_indices[idx] if edge_indices else None
            title = titles[idx] if titles else f"Molecule {idx}"
            self.plot_molecule(
                pos, edge_index, title=title,
                show=False, ax=ax
            )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_diffusion_process(
            self,
            positions: List[torch.Tensor],
            edge_index: torch.Tensor,
            timesteps: List[float],
            title: str = "Diffusion Process",
            save_path: Optional[str] = None,
            show: bool = True
    ):
        """
        visualize the diffusion trajectory of a molecule.

        arguments:
            positions: List of (N, 3) tensors showing molecule at different timesteps
            edge_index: Edge connectivity (same for all timesteps)
            timesteps: List of timestep values (e.g., [0.0, 0.25, 0.5, 0.75, 1.0])
            title: Overall title
            save_path: If provided, save figure
            show: Whether to display
        """
        n_steps = len(positions)
        ncols = min(5, n_steps)
        nrows = (n_steps + ncols - 1) // ncols
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
        for idx, (pos, t) in enumerate(zip(positions, timesteps)):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

            self.plot_molecule(
                pos, edge_index,
                title=f"t = {t:.3f}",
                show=False, ax=ax
            )

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()

    def animate_diffusion(
            self,
            positions: List[torch.Tensor],
            edge_index: torch.Tensor,
            timesteps: Optional[List[float]] = None,
            save_path: str = "diffusion_animation.gif",
            fps: int = 10,
            title: str = "Diffusion Process"
    ):
        """
        create an animated GIF of the diffusion process.
        arguments:
            positions: List of (N, 3) tensors at different timesteps
            edge_index: Edge connectivity
            timesteps: Timestep values for each frame
            save_path: Path to save GIF
            fps: Frames per second
            title: Animation title
        """
        if timesteps is None:
            timesteps = np.linspace(0, 1, len(positions))
        positions_np = [p.detach().cpu().numpy() for p in positions]
        edge_index_np = edge_index.cpu().numpy()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        all_pos = np.concatenate(positions_np, axis=0)
        max_range = np.abs(all_pos).max() * 1.1

        def update(frame):
            ax.clear()
            pos = positions_np[frame]
            t = timesteps[frame]
            for i in range(edge_index_np.shape[1]):
                src, dst = edge_index_np[:, i]
                if src < dst:
                    ax.plot(
                        [pos[src, 0], pos[dst, 0]],
                        [pos[src, 1], pos[dst, 1]],
                        [pos[src, 2], pos[dst, 2]],
                        'gray', alpha=0.5, linewidth=1.5
                    )
            ax.scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                c='#1f77b4', s=200, alpha=0.9,
                edgecolors='black', linewidths=1.5
            )
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(f"{title}\nt = {t:.3f}", fontsize=14, fontweight='bold')
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            return ax,

        anim = FuncAnimation(
            fig, update,
            frames=len(positions),
            interval=1000 // fps,
            blit=False
        )
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
        plt.close()

    def plot_training_progress(
            self,
            train_losses: List[float],
            val_losses: Optional[List[float]] = None,
            save_path: Optional[str] = None,
            log_scale: bool = False
    ):
        """
        plot training and validation losses.

        arguments:
            train_losses: List of training losses per epoch
            val_losses: Optional list of validation losses
            save_path: If provided, save figure
            log_scale: Use log scale for y-axis
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
        if val_losses is not None:
            ax.plot(epochs, val_losses, label='Val Loss', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def compare_molecules(
            self,
            original: torch.Tensor,
            generated: torch.Tensor,
            edge_index: torch.Tensor,
            title_orig: str = "Original",
            title_gen: str = "Generated",
            save_path: Optional[str] = None
    ):
        """
        side-by-side comparison of original vs generated molecules.
        arguments:
            original: (N, 3) original molecule positions
            generated: (N, 3) generated molecule positions
            edge_index: Edge connectivity
            title_orig: Title for original
            title_gen: Title for generated
            save_path: If provided, save figure
        """
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        self.plot_molecule(original, edge_index, title=title_orig, show=False, ax=ax1)
        ax2 = fig.add_subplot(122, projection='3d')
        self.plot_molecule(generated, edge_index, title=title_gen, show=False, ax=ax2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def visualize_training_samples(
        model,
        dataset,
        n_samples: int = 8,
        save_dir: str = "./visualizations"
):
    """
    visualize samples from the dataset during training.

    arguments:
        model: The diffusion model (not used here, for generation later)
        dataset: The molecular dataset
        n_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    visualizer = MolecularVisualizer(use_plotly=False)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    positions = []
    edge_indices = []
    titles = []
    for idx in indices:
        data = dataset[int(idx)]
        info = dataset.get_molecule_info(int(idx))

        positions.append(data.pos)
        edge_indices.append(data.edge_index)
        titles.append(f"{info['geometry'].replace('_', ' ').title()} ({info['num_atoms']} atoms)")
    visualizer.plot_batch(
        positions, edge_indices, titles,
        save_path=f"{save_dir}/training_samples.png"
    )


def visualize_generation_process(
        model,
        dataset,
        device: str = "cuda",
        n_steps: int = 10,
        save_dir: str = "./visualizations"
):
    """
    visualize the denoising/generation process.
    arguments:
        model: Trained diffusion model
        dataset: Dataset (for structure reference)
        device: Device to run on
        n_steps: Number of timesteps to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    visualizer = MolecularVisualizer(use_plotly=False)
    data = dataset[0]
    num_atoms = data.pos.shape[0]
    edge_index = data.edge_index
    xt = torch.randn(num_atoms, 3).to(device)
    positions = []
    timesteps = []
    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            t = 1.0 - (step / n_steps)
            positions.append(xt.cpu().clone())
            timesteps.append(t)
            # Denoise step (pseudo-code, adapt to your model)
            # xt = denoise_step(model, xt, t, data.x, edge_index)
    visualizer.visualize_diffusion_process(
        positions, edge_index, timesteps,
        title="Generation Process (Denoising)",
        save_path=f"{save_dir}/generation_process.png"
    )
    visualizer.animate_diffusion(
        positions, edge_index, timesteps,
        save_path=f"{save_dir}/generation_animation.gif",
        fps=2
    )


class TrainingVisualizer:
    """helper class to visualize during training"""

    def __init__(self, save_dir: str = "./training_vis"):
        self.visualizer = MolecularVisualizer(use_plotly=False)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []

    def log_losses(self, train_loss: float, val_loss: Optional[float] = None):
        """log losses for plotting"""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def plot_progress(self, epoch: int):
        """plot training progress"""
        self.visualizer.plot_training_progress(
            self.train_losses,
            self.val_losses if self.val_losses else None,
            save_path=f"{self.save_dir}/loss_epoch_{epoch}.png"
        )

    def visualize_samples(self, dataset, epoch: int, n_samples: int = 8):
        """visualize dataset samples"""
        visualize_training_samples(
            None, dataset, n_samples,
            save_dir=f"{self.save_dir}/epoch_{epoch}"
        )