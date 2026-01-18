import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from typing import Callable, Optional, Dict
from torch.nn.utils import clip_grad_norm_
from contextlib import contextmanager
from tqdm import tqdm
import os



class MDDistillationTrainer(nn.Module):
    """knowledge distillation trainer with fokker-planck regularization applied through gating"""
    def __init__(
            self,
            teacher_enet: nn.Module,
            student_enet: nn.Module,
            forward_vp: nn.Module,
            data_loader,
            optimizer: torch.optim.Optimizer,
            fp_gate: Callable,
            fp_loss: Callable, # fokker-planck loss function
            dist_loss: Callable, # distillation loss function
            score_fn: Callable, # computes score from energy
            fp_residual: Callable, # computes weak fokker-planck residuals
            val_loader: None,
            epochs: int,
            device: torch.device | None = None,
            grad_acc: int = 1,
            checkpoint: int = 5,
            log_freq: int = 1,
            store_path: str = "./checkpoints",
            warmup_steps: int = 0,
            rotation_augment: bool = False, # if true molecules will be randomly augmented through  training
            fp_alpha: float = 5e-4,
            mix_precision: bool = True,
            lambda_t: Callable[[torch.Tensor], torch.Tensor] = lambda t: torch.exp(-t), # time-dependent weighting λ(t)
            gate_params: Optional[Dict] = None,
            use_cached_trajectories: bool = True,
            traj_loss_weight: float = 1.0,
            traj_freq: int = 10,
            eps_time: float = 1e-5,
            *args
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_enet = teacher_enet.to(self.device)
        self.student_enet = student_enet.to(self.device)
        self.forward_vp = forward_vp.to(self.device)
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.fp_gate = fp_gate
        self.fp_loss = fp_loss
        self.dist_loss = dist_loss
        self.score_fn = score_fn
        self.fp_residual = fp_residual
        self.epochs = epochs
        self.grad_acc = grad_acc
        self.checkpoint = checkpoint
        self.log_freq = log_freq
        self.store_path = store_path
        self.warmup_steps = warmup_steps
        self.rotation_augment = rotation_augment
        self.lambda_t = lambda_t
        self.fp_alpha = fp_alpha
        self.mix_precision = mix_precision
        self.use_cached_trajectories = use_cached_trajectories
        self.traj_loss_weight = traj_loss_weight
        self.traj_freq = traj_freq
        self.eps_time = eps_time
        self.global_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        self.best_loss = float('inf')
        self.use_amp = self.mix_precision and self.device.type == "cuda"
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        self.gate_params = gate_params or {
            "k": 1.2, "snr_min": 0.5, "t_scale": 0.1,
            "sharpness": 5.0, "eps": 0.1
        }
        self.losses = {
            'total_losses': [], 'fp_losses': [], 'force_losses': [],
            'traj_losses': [], 'val_losses': []
        }

    def forward(self):
        self.teacher_enet.eval()
        self.student_enet.train()
        for epoch in range(self.epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            mean_losses = self.train_batch(pbar, epoch)
            self.losses['total_losses'].append(mean_losses[0])
            self.losses['force_losses'].append(mean_losses[1])
            self.losses['fp_losses'].append(mean_losses[2])
            self.losses['traj_losses'].append(mean_losses[3])
            if (epoch + 1) % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1}/{self.epochs} | LR: {lr:.2e} | "
                      f"Total Loss: {mean_losses[0]:.4f} | Force Loss: {mean_losses[1]:.4f} | "
                      f"FP Loss: {mean_losses[2]:.4f} | Traj Loss: {mean_losses[3]:.4f}")
                if self.val_loader is not None:
                    val_loss = self.validate()
                    self.losses['val_losses'].append(val_loss)
                    print(f" | Val Loss: {val_loss:.4f}")
            if (epoch + 1) % self.checkpoint == 0:
                self.save_checkpoint(epoch + 1, mean_losses[0])
            if mean_losses[0] < self.best_loss:
                self.best_loss = mean_losses[0]
                self.save_checkpoint(epoch + 1, mean_losses[0], is_best=True)
        return self.losses

    def train_batch(self, pbar, epoch):
        """train one epoch of student model"""
        total_losses = []
        force_losses = []
        fp_losses = []
        traj_losses = []
        for step, batch in enumerate(pbar):
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            # extract cached trajectory data if available
            compute_traj = False
            if hasattr(batch, "mol_idx") and epoch % self.traj_freq == 0:
                mol_indices = batch.mol_idx.tolist()
                traj_data = self.data_loader.dataset_ref.get_trajectories_for_batch(mol_indices)
                teacher_trajectories = traj_data["trajectories"]
                traj_params_list = traj_data["params"]
                compute_traj = True
            else:
                teacher_trajectories = None
                traj_params_list = None
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    step_losses = self.train_step(
                        x0, atom_features, edge_index, batch_idx, num_molecules,
                        teacher_trajectories = teacher_trajectories,
                        traj_params_list = traj_params_list,
                        compute_traj_loss = compute_traj
                    )
                self.scaler.scale(step_losses[0]).backward()
            else:
                step_losses = self.train_step(
                    x0, atom_features, edge_index, batch_idx, num_molecules,
                    teacher_trajectories = teacher_trajectories,
                    traj_params_list = traj_params_list,
                    compute_traj_loss = compute_traj
                )
                step_losses[0].backward()
            if (step + 1) % self.grad_acc == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.student_enet.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self.update_learning_rate()
            pbar.set_postfix({
                'total': f'{step_losses[0].item() * self.grad_acc:.4f}',
                'force': f'{step_losses[1].item() * self.grad_acc:.4f}',
                'fp': f'{step_losses[2].item() * self.grad_acc:.4f}',
                'traj': f'{step_losses[3].item() * self.grad_acc:.4f}'
            })
            total_losses.append(step_losses[0].item() * self.grad_acc)
            force_losses.append(step_losses[1].item() * self.grad_acc)
            fp_losses.append(step_losses[2].item() * self.grad_acc)
            traj_losses.append(step_losses[3].item() * self.grad_acc)

        if self.global_step >= self.warmup_steps:
            self.scheduler.step()

        return (sum(total_losses) / len(total_losses),
                sum(force_losses) / len(force_losses),
                sum(fp_losses) / len(fp_losses),
                sum(traj_losses) / len(traj_losses))

    def train_step(self, x0: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch_idx: torch.Tensor, num_molecules: int, teacher_trajectories: list,
                   traj_params_list: list, compute_traj_loss):
        # force loss
        # ------------------------------------------------
        noise = torch.randn_like(x0)
        x0 = x0.clone()
        noise = noise.clone()
        if self.rotation_augment:
            with torch.amp.autocast('cuda', enabled=False):
                R = self.random_rotation_matrix(num_molecules)
                for mol_idx in range(num_molecules):
                    mask = batch_idx == mol_idx
                    x0[mask] = x0[mask] @ R[mol_idx].T
                    noise[mask] = noise[mask] @ R[mol_idx].T
        t_mol = self.sample_time(num_molecules, self.eps_time)
        t_atom = t_mol[batch_idx]
        lambda_val = self.lambda_t(t_atom.mean())
        xt, _ = self.forward_vp(x0, noise, t_atom)
        xt = xt.detach()
        with self.freeze_params(self.teacher_enet):
            xt_teacher = xt.clone().requires_grad_(True)
            true_logp = self.teacher_enet(xt_teacher, atom_features, edge_index, t_atom, batch_idx)
            true_score = self.score_fn(true_logp, xt_teacher)
        xt_student = xt.clone().requires_grad_(True)
        pred_logp = self.student_enet(xt_student, atom_features, edge_index, t_atom, batch_idx)
        pred_score = self.score_fn(pred_logp, xt_student)
        force_loss = lambda_val * self.dist_loss(pred_score, true_score, batch_idx)

        # fp regularization
        # ---------------------------------------------------
        fp_loss = torch.tensor(0.0, device=self.device)
        fp_mask, activate_fp = self.fp_gate(
            x0, t_atom, self.forward_vp.vs,
            k=self.gate_params['k'], snr_min=self.gate_params['snr_min'],
            t_scale=self.gate_params['t_scale'], sharpness=self.gate_params['sharpness'],
            eps=self.gate_params['eps']
        )
        if activate_fp:
            xt_active, atom_feat_active, edge_idx_active, batch_active, t_active = \
                self.subset_graph_data(
                    xt_student, atom_features, edge_index, t_atom, batch_idx,
                    fp_mask, torch.unique(batch_idx[fp_mask])
                )
            seed1 = self.global_step * 1000
            seed2 = self.global_step * 1001
            r1 = self.fp_residual(
                self.student_enet, xt_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs, seed1
            )
            r2 = self.fp_residual(
                self.student_enet, xt_active, atom_feat_active, edge_idx_active,
                t_active, batch_active, self.forward_vp.vs, seed2
            )
            fp_lambda_val = self.lambda_t(t_active.mean())
            fp_loss = fp_lambda_val * self.fp_loss(r1, r2, batch_active, alpha=self.fp_alpha)

        # trajectory loss
        # -------------------------------------------------------------------
        traj_loss = torch.tensor(0.0, device=self.device)
        if compute_traj_loss and teacher_trajectories is not None:
            if self.use_cached_trajectories:
                # process each molecule's trajectory separately
                mol_traj_losses = []
                for mol_idx in range(num_molecules):
                    # get atoms belonging to this molecule
                    mol_mask = batch_idx == mol_idx
                    mol_x0 = x0[mol_mask]
                    mol_features = atom_features[mol_mask]
                    # get edges for this molecule
                    # need to filter edge_index for this molecule
                    edge_mask = mol_mask[edge_index[0]] & mol_mask[edge_index[1]]
                    mol_edge_index = edge_index[:, edge_mask]
                    # remap edge indices to local molecule indices
                    atom_indices = torch.where(mol_mask)[0]
                    old_to_new = torch.zeros(x0.shape[0], dtype=torch.long, device=self.device)
                    old_to_new[atom_indices] = torch.arange(len(atom_indices), device=self.device)
                    mol_edge_index = old_to_new[mol_edge_index]
                    # create single-molecule batch index
                    mol_batch_idx = torch.zeros(mol_x0.shape[0], dtype=torch.long, device=self.device)
                    # get cached teacher trajectory and params
                    true_traj = teacher_trajectories[mol_idx].to(self.device)
                    params = traj_params_list[mol_idx]
                    # simulate student trajectory with same parameters
                    pred_traj = self.simulate_langevin(
                        self.student_enet, mol_x0, mol_features, mol_edge_index,
                        mol_batch_idx,
                        num_steps=params.num_steps,
                        dt=params.dt,
                        temp=params.temp,
                        friction=params.friction,
                        mass=params.mass,
                        kb=params.kb,
                        t_eval=params.t_eval
                    )
                    mol_loss = ((pred_traj - true_traj) ** 2).mean()
                    mol_traj_losses.append(mol_loss)
                traj_loss = self.traj_loss_weight * torch.stack(mol_traj_losses).mean()

        total_loss = (force_loss + fp_loss + traj_loss) / self.grad_acc
        force_loss = force_loss / self.grad_acc
        fp_loss = fp_loss / self.grad_acc
        traj_loss = traj_loss / self.grad_acc
        return total_loss, force_loss, fp_loss, traj_loss

    def simulate_langevin(self, energy_net, x0: torch.Tensor, atom_features: torch.Tensor,
                          edge_index: torch.Tensor, batch_idx: torch.Tensor, num_steps: int,
                          dt: float, temp: float, friction: float, mass: float, kb: float, t_eval: float):
        """
        langevin dynamics simulation using learned forces
        dx = v dt
        M dv = -∇U(x) dt - γM v dt + √(2γk_B T) dw
        arguments:
            x0: (num_atoms, 3) initial coordinates
            atom_features: atom features
            edge_index: connectivity
            num_steps: number of simulation steps
            dt: timestep (must be consistent with friction units)
            temp: temperature in Kelvin
            friction: friction coefficient
            mass: particle mass
            kb: boltzmann constant (J/K)
            save_frequency: save every N steps
        returns:
            trajectory: (num_saved, num_atoms, 3) coordinates
            velocities: (num_saved, num_atoms, 3) velocities
        """
        traj = []
        kb_t = kb * temp
        assert kb_t > 0, "temperature must be positive"
        x = x0.clone()
        kb_t = torch.tensor(kb_t, device=x.device)
        mass = torch.tensor(mass, device=x.device)
        v = torch.randn_like(x) * torch.sqrt((kb_t / mass).detach().clone())  # maxwell-boltzmann
        alpha = torch.exp(torch.tensor(-friction * dt, device=x.device))
        sigma = torch.sqrt(kb_t * (1 - alpha ** 2) / mass)
        #steps = tqdm(range(num_steps), desc="MD-Trajectory")
        #for _ in steps:
        for _ in range(num_steps):
            # get forces from the trained energy model
            forces = self.get_forces(energy_net, x, atom_features, edge_index, batch_idx, float(kb_t), t_eval)
            # langevin integrator
            # update velocity
            v = (alpha * v + (1 - alpha) / (friction * mass) * forces + sigma * torch.randn_like(v))
            # update position
            x = x + dt * v
            traj.append(x.clone())
            #x = x.detach()
            #v = v.detach()
        traj_ = torch.stack(traj)
        return traj_

    def get_forces(self, energy_net, x: torch.Tensor, atom_features: torch.Tensor, edge_index: torch.Tensor,
                   batch_idx: torch.Tensor, t_eval, kb_t) -> torch.Tensor:
        """
        extract forces from the score at t ≈ 0
            - s_θ(x, t=0) = ∇_x log p(x) = -∇_x U(x) / k_B T
            - so forces F = -∇_x U(x) = k_B T * s_θ(x, 0)
        arguments:
            x: (num_atoms, 3) or (batch_size*num_atoms, 3) coordinates
            atom_features: atom features
            edge_index: connectivity
            batch_idx: batch assignment (if batched)
            t_eval: small timestep to evaluate (here we use the same t as
                    used in time sampling (epsilon = 1e-5 by default))
            kb_t: boltzmann constant times temperature here we assume kB_T = 1
        returns:
            forces with same shape as x
        """
        # continuous diffusion time at t ≈ 0
        t_atom = torch.full((x.shape[0], ), t_eval, device=self.device) # (number of atoms, )
        x = x.detach()
        x.requires_grad_(True)
        # computes energy and score
        logp = energy_net(x, atom_features, edge_index, t_atom, batch_idx)
        score = self.score_fn(logp, x)
        forces = (score * kb_t).detach()
        return forces

    def validate(self):
        """validation without fp regularization"""
        self.student_enet.eval()
        val_losses = []
        for batch in self.val_loader:
            x0 = batch.coords.to(self.device)
            atom_features = batch.atom_features.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            batch_idx = batch.batch.to(self.device)
            num_molecules = batch.num_graphs
            noise = torch.randn_like(x0)
            t_mol = self.sample_time(num_molecules, self.eps_time)  # (num_molecules,)
            t_atom = t_mol[batch_idx]  # (num_atoms,)
            lambda_val = self.lambda_t(t_atom.mean())
            xt, true_score = self.forward_vp(x0, noise, t_atom)
            # freeze both teacher and student energy nets
            with self.freeze_params(self.teacher_enet):
                xt = xt.clone().requires_grad_(True)
                true_logp = self.teacher_enet(xt, atom_features, edge_index, t_atom, batch_idx)
                true_score = self.score_fn(true_logp, xt)
            with self.freeze_params(self.student_enet):
                xt_ = xt.clone().requires_grad_(True)
                pred_logp = self.student_enet(xt_, atom_features, edge_index, t_atom, batch_idx)
                pred_score = self.score_fn(pred_logp, xt_)
            force_loss = lambda_val * self.dist_loss(pred_score, true_score, batch_idx)
            val_losses.append(force_loss.item())
        self.student_enet.train()
        return sum(val_losses) / len(val_losses)

    def subset_graph_data(self, xt, atom_features, edge_index, t, batch_idx, active_atom_mask, active_mol_indices):
        xt_active = xt[active_atom_mask]
        atom_feat_active = atom_features[active_atom_mask]
        batch_active = batch_idx[active_atom_mask]
        old_to_new_mol = torch.full((batch_idx.max() + 1,), -1, dtype=torch.long, device=batch_idx.device)
        old_to_new_mol[active_mol_indices] = torch.arange(len(active_mol_indices), device=batch_idx.device)
        batch_active = old_to_new_mol[batch_active]
        edge_mask = active_atom_mask[edge_index[0]] & active_atom_mask[edge_index[1]]
        edge_idx_active = edge_index[:, edge_mask]
        old_to_new_atom = torch.full((xt.shape[0],), -1, dtype=torch.long, device=xt.device)
        old_to_new_atom[active_atom_mask] = torch.arange(active_atom_mask.sum(), device=xt.device)
        edge_idx_active = old_to_new_atom[edge_idx_active]
        t_active = t[active_atom_mask]
        return xt_active, atom_feat_active, edge_idx_active, batch_active, t_active

    def random_rotation_matrix(self, batch_size: int) -> torch.Tensor:
        """generate random 3D rotation matrices using QR decomposition"""
        M = torch.randn(batch_size, 3, 3, device=self.device)
        Q, R = torch.linalg.qr(M)
        det = torch.det(Q)
        Q = Q * det.view(-1, 1, 1).sign()
        return Q

    def update_learning_rate(self):
        """linear warmup"""
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            lr = self.base_lr * (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        checkpoint = {
            'current_epoch': epoch,
            'student_energy_net_state': self.student_enet.state_dict(),
            'teacher_energy_net_state': self.teacher_enet.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'current_loss': loss,
            'losses': self.losses,
            'variance_scheduler': self.forward_vp.vs.state_dict(),
            'max_epochs': self.epochs,
            'base_larning_rate': self.base_lr,
            'global_step': self.global_step
        }
        filename = "dist_best.pth" if is_best else f"dist_epoch_{epoch}.pth"
        filepath = os.path.join(self.store_path, filename)
        os.makedirs(self.store_path, exist_ok=True)
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Best model saved at epoch {epoch} with loss {loss:.4f}")
        else:
            print(f"Checkpoint saved at epoch {epoch} with loss {loss: .4f}")

    def sample_time(self, batch_size: int, eps: float = 1e-5) -> torch.Tensor:
        return eps + (1 - eps) * torch.rand(batch_size, device=self.device)

    @contextmanager
    def freeze_params(self, module):
        old_requires_grad = []
        for p in module.parameters():
            old_requires_grad.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(module.parameters(), old_requires_grad):
                p.requires_grad_(rg)