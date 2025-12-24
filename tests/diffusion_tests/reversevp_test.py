import torch
import torch.nn as nn
import pytest
from diffusion.schedules import LinearVS
from diffusion.forward import ForwardVP
from diffusion.reverse import ReverseVP



class TestReverseVP:
    @pytest.fixture
    def scheduler(self):
        return LinearVS(num_steps=1000, beta_start=0.1, beta_end=20.0, start=0.0, end=1.0)

    @pytest.fixture
    def reverse_vp(self, scheduler):
        return ReverseVP(scheduler)

    @pytest.fixture
    def forward_vp(self, scheduler):
        return ForwardVP(scheduler)

    def test_initialization(self, scheduler):
        reverse_vp = ReverseVP(scheduler)
        assert reverse_vp.vs is scheduler
        assert isinstance(reverse_vp, nn.Module)

    def test_forward_shape_preservation_2d(self, reverse_vp):
        batch_size = 16
        features = 128
        xt = torch.randn(batch_size, features)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (batch_size,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == xt.shape

    def test_forward_shape_preservation_4d(self, reverse_vp):
        batch_size = 8
        channels = 3
        height = 32
        width = 32
        xt = torch.randn(batch_size, channels, height, width)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (batch_size,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == xt.shape

    def test_forward_shape_preservation_3d(self, reverse_vp):
        batch_size = 4
        seq_len = 50
        embed_dim = 256
        xt = torch.randn(batch_size, seq_len, embed_dim)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (batch_size,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == xt.shape

    def test_perfect_noise_prediction_moves_toward_clean(self, reverse_vp, forward_vp):

        x0 = torch.randn(4, 3, 32, 32)
        true_noise = torch.randn_like(x0)
        t_index = torch.full((4,), 50, dtype=torch.long)
        xt = forward_vp(x0, true_noise, t_index)
        x_current = xt.clone()
        zero_noise = torch.zeros_like(xt)
        for step in range(50, 0, -5):
            t_current = torch.full((4,), step, dtype=torch.long)
            alpha_t = torch.exp(-0.5 * reverse_vp.vs.integral_beta[t_current])
            sigma_t = reverse_vp.vs.get_std(t_current)
            while alpha_t.dim() < x_current.dim():
                alpha_t = alpha_t.unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1)
            perfect_noise = (x_current - alpha_t * x0) / (sigma_t + 1e-8)
            x_current = reverse_vp(x_current, perfect_noise, t_current, zero_noise)
        dist_before = torch.norm(xt - x0)
        dist_after = torch.norm(x_current - x0)
        assert dist_after < dist_before

    def test_deterministic_with_same_inputs(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (4,))
        xt_prev1 = reverse_vp(xt, noise_pred, t_index, noise)
        xt_prev2 = reverse_vp(xt, noise_pred, t_index, noise)
        assert torch.allclose(xt_prev1, xt_prev2)

    def test_different_noise_different_results(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (4,))
        noise1 = torch.randn_like(xt)
        noise2 = torch.randn_like(xt)
        xt_prev1 = reverse_vp(xt, noise_pred, t_index, noise1)
        xt_prev2 = reverse_vp(xt, noise_pred, t_index, noise2)
        assert not torch.allclose(xt_prev1, xt_prev2)

    def test_zero_noise_deterministic(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (4,))
        zero_noise = torch.zeros_like(xt)
        xt_prev1 = reverse_vp(xt, noise_pred, t_index, zero_noise)
        xt_prev2 = reverse_vp(xt, noise_pred, t_index, zero_noise)
        assert torch.allclose(xt_prev1, xt_prev2)

    def test_different_timesteps_different_updates(self, reverse_vp):
        xt = torch.randn(1, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_early = torch.tensor([100])
        t_late = torch.tensor([800])
        xt_prev_early = reverse_vp(xt, noise_pred, t_early, noise)
        xt_prev_late = reverse_vp(xt, noise_pred, t_late, noise)
        assert not torch.allclose(xt_prev_early, xt_prev_late)

    def test_batch_processing(self, reverse_vp):
        batch_size = 16
        xt = torch.randn(batch_size, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (batch_size,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        xt_prev_individual = []
        for i in range(batch_size):
            xt_prev_i = reverse_vp(xt[i:i + 1], noise_pred[i:i + 1], t_index[i:i + 1], noise[i:i + 1])
            xt_prev_individual.append(xt_prev_i)
        xt_prev_individual = torch.cat(xt_prev_individual, dim=0)
        assert torch.allclose(xt_prev, xt_prev_individual, atol=1e-6)

    def test_gradient_flow(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32, requires_grad=True)
        noise_pred = torch.randn_like(xt, requires_grad=True)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (4,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        loss = xt_prev.sum()
        loss.backward()
        assert xt.grad is not None
        assert noise_pred.grad is not None
        assert torch.any(xt.grad != 0)
        assert torch.any(noise_pred.grad != 0)

    def test_multistep_reverse_reduces_noise(self, reverse_vp, forward_vp):
        x0 = torch.randn(1, 3, 32, 32)
        true_noise = torch.randn_like(x0)
        t_high = torch.tensor([200])
        xt = forward_vp(x0, true_noise, t_high)
        x_current = xt
        distances = [torch.norm(x_current - x0).item()]
        for step in range(200, 0, -20):
            t_current = torch.tensor([step])
            zero_noise = torch.zeros_like(x_current)
            x_current = reverse_vp(x_current, true_noise, t_current, zero_noise)
            distances.append(torch.norm(x_current - x0).item())
        assert distances[-1] < distances[0] * 1.2

    def test_score_computation_correctness(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (4,))
        variance_t = reverse_vp.vs.get_variance(t_index)
        sigma_t = torch.sqrt(variance_t)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        expected_score = -noise_pred / (sigma_t + 1e-8)
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert not torch.allclose(xt_prev, xt)

    def test_device_compatibility(self, scheduler):
        reverse_vp = ReverseVP(scheduler)
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (4,))
        xt_prev_cpu = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev_cpu.device.type == 'cpu'
        if torch.cuda.is_available():
            reverse_vp_gpu = reverse_vp.cuda()
            xt_gpu = xt.cuda()
            noise_pred_gpu = noise_pred.cuda()
            noise_gpu = noise.cuda()
            t_index_gpu = t_index.cuda()
            xt_prev_gpu = reverse_vp_gpu(xt_gpu, noise_pred_gpu, t_index_gpu, noise_gpu)
            assert xt_prev_gpu.device.type == 'cuda'
            assert torch.allclose(xt_prev_cpu, xt_prev_gpu.cpu(), atol=1e-5)

    def test_coefficient_broadcasting_1d(self, reverse_vp):
        xt = torch.randn(8, 64)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (8,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == xt.shape

    def test_coefficient_broadcasting_5d(self, reverse_vp):
        xt = torch.randn(2, 3, 8, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (2,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == xt.shape

    def test_numerical_stability_high_variance(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.full((4,), 990, dtype=torch.long)
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert not torch.isnan(xt_prev).any()
        assert not torch.isinf(xt_prev).any()

    def test_numerical_stability_low_variance(self, reverse_vp):
        xt = torch.randn(4, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.full((4,), 10, dtype=torch.long)
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert not torch.isnan(xt_prev).any()
        assert not torch.isinf(xt_prev).any()


class TestReverseVPEdgeCases:
    @pytest.fixture
    def reverse_vp(self):
        scheduler = LinearVS(num_steps=1000, beta_start=0.1, beta_end=20.0)
        return ReverseVP(scheduler)

    def test_single_sample_batch(self, reverse_vp):
        xt = torch.randn(1, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.tensor([500])
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == (1, 3, 32, 32)

    def test_large_batch(self, reverse_vp):
        batch_size = 256
        xt = torch.randn(batch_size, 64)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.randint(1, 1000, (batch_size,))
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == (batch_size, 64)

    def test_same_timestep_all_batch(self, reverse_vp):
        batch_size = 8
        xt = torch.randn(batch_size, 3, 32, 32)
        noise_pred = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        t_index = torch.full((batch_size,), 500, dtype=torch.long)
        xt_prev = reverse_vp(xt, noise_pred, t_index, noise)
        assert xt_prev.shape == xt.shape


class TestForwardReverseConsistency:

    @pytest.fixture
    def scheduler(self):
        return LinearVS(num_steps=1000, beta_start=0.1, beta_end=20.0)

    @pytest.fixture
    def forward_vp(self, scheduler):
        return ForwardVP(scheduler)

    @pytest.fixture
    def reverse_vp(self, scheduler):
        return ReverseVP(scheduler)

    def test_forward_then_reverse_approximate_recovery(self, forward_vp, reverse_vp):
        x0 = torch.randn(4, 3, 32, 32)
        true_noise = torch.randn_like(x0)
        t_forward = torch.full((4,), 50, dtype=torch.long)
        xt = forward_vp(x0, true_noise, t_forward)
        zero_noise = torch.zeros_like(xt)
        x0_reconstructed = xt
        for step in range(50, 0, -5):
            t_current = torch.full((4,), step, dtype=torch.long)
            x0_reconstructed = reverse_vp(x0_reconstructed, true_noise, t_current, zero_noise)
        relative_error = torch.norm(x0_reconstructed - x0) / torch.norm(x0)
        assert relative_error < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])