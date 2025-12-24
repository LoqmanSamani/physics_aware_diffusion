import torch
import torch.nn as nn
import pytest
from diffusion.schedules import LinearVS
from diffusion.forward import ForwardVP


class TestForwardVP:
    @pytest.fixture
    def scheduler(self):
        return LinearVS(num_steps=1000, beta_start=0.1, beta_end=20.0)

    @pytest.fixture
    def forward_vp(self, scheduler):
        return ForwardVP(scheduler)

    def test_initialization(self, scheduler):
        forward_vp = ForwardVP(scheduler)
        assert forward_vp.vs is scheduler
        assert isinstance(forward_vp, nn.Module)

    def test_forward_shape_preservation_2d(self, forward_vp):
        batch_size = 16
        features = 128
        x0 = torch.randn(batch_size, features)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (batch_size,))
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == x0.shape

    def test_forward_shape_preservation_4d(self, forward_vp):
        batch_size = 8
        channels = 3
        height = 32
        width = 32
        x0 = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (batch_size,))
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == x0.shape

    def test_forward_shape_preservation_3d(self, forward_vp):
        batch_size = 4
        seq_len = 50
        embed_dim = 256
        x0 = torch.randn(batch_size, seq_len, embed_dim)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (batch_size,))
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == x0.shape

    def test_t_zero_returns_original(self, forward_vp):
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.zeros(4, dtype=torch.long)  # t=0
        xt = forward_vp(x0, noise, t_index)
        assert torch.allclose(xt, x0, atol=1e-4)

    def test_t_large_mostly_noise(self, forward_vp):
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.full((4,), 999, dtype=torch.long)
        xt = forward_vp(x0, noise, t_index)
        variance = forward_vp.vs.get_variance(t_index)
        assert torch.all(variance > 0.9)
        signal_coeff = torch.sqrt(1.0 - variance)
        assert torch.all(signal_coeff < 0.5)

    def test_different_t_different_results(self, forward_vp):
        x0 = torch.randn(1, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_early = torch.tensor([100])
        t_late = torch.tensor([800])
        xt_early = forward_vp(x0, noise, t_early)
        xt_late = forward_vp(x0, noise, t_late)
        assert not torch.allclose(xt_early, xt_late)
        dist_early = torch.norm(xt_early - x0)
        dist_late = torch.norm(xt_late - x0)
        assert dist_late > dist_early

    def test_linear_combination_property(self, forward_vp):
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (4,))
        xt = forward_vp(x0, noise, t_index)
        variance = forward_vp.vs.get_variance(t_index)
        signal_coeff = torch.sqrt(1.0 - variance)
        noise_coeff = torch.sqrt(variance)
        signal_coeff = signal_coeff.view(-1, 1, 1, 1)
        noise_coeff = noise_coeff.view(-1, 1, 1, 1)
        expected = signal_coeff * x0 + noise_coeff * noise
        assert torch.allclose(xt, expected, atol=1e-6)

    def test_variance_preserving_property(self, forward_vp):
        x0 = torch.randn(1000, 128)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (1000,))
        xt = forward_vp(x0, noise, t_index)
        var_x0 = torch.var(x0)
        var_xt = torch.var(xt)
        assert torch.abs(var_x0 - 1.0) < 0.1
        assert torch.abs(var_xt - 1.0) < 0.2

    def test_deterministic_with_same_inputs(self, forward_vp):
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (4,))
        xt1 = forward_vp(x0, noise, t_index)
        xt2 = forward_vp(x0, noise, t_index)
        assert torch.allclose(xt1, xt2)

    def test_batch_processing(self, forward_vp):
        batch_size = 16
        x0 = torch.randn(batch_size, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (batch_size,))
        xt = forward_vp(x0, noise, t_index)
        xt_individual = []
        for i in range(batch_size):
            xt_i = forward_vp(x0[i:i + 1], noise[i:i + 1], t_index[i:i + 1])
            xt_individual.append(xt_i)
        xt_individual = torch.cat(xt_individual, dim=0)
        assert torch.allclose(xt, xt_individual, atol=1e-6)

    def test_device_compatibility(self, scheduler):
        forward_vp = ForwardVP(scheduler)
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (4,))
        xt_cpu = forward_vp(x0, noise, t_index)
        assert xt_cpu.device.type == 'cpu'
        if torch.cuda.is_available():
            forward_vp_gpu = forward_vp.cuda()
            x0_gpu = x0.cuda()
            noise_gpu = noise.cuda()
            t_index_gpu = t_index.cuda()
            xt_gpu = forward_vp_gpu(x0_gpu, noise_gpu, t_index_gpu)
            assert xt_gpu.device.type == 'cuda'
            assert torch.allclose(xt_cpu, xt_gpu.cpu(), atol=1e-5)

    def test_gradient_flow(self, forward_vp):
        x0 = torch.randn(4, 3, 32, 32, requires_grad=True)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (4,))
        xt = forward_vp(x0, noise, t_index)
        loss = xt.sum()
        loss.backward()
        assert x0.grad is not None
        assert torch.any(x0.grad != 0)

    def test_coefficient_broadcasting_1d(self, forward_vp):
        x0 = torch.randn(8, 64)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (8,))
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == x0.shape

    def test_coefficient_broadcasting_5d(self, forward_vp):
        x0 = torch.randn(2, 3, 8, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (2,))
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == x0.shape

    def test_zero_noise_same_t(self, forward_vp):
        x0 = torch.randn(4, 3, 32, 32)
        noise = torch.zeros_like(x0)
        t_index = torch.randint(100, 900, (4,))
        xt = forward_vp(x0, noise, t_index)
        variance = forward_vp.vs.get_variance(t_index)
        signal_coeff = torch.sqrt(1.0 - variance)
        signal_coeff = signal_coeff.view(-1, 1, 1, 1)
        expected = signal_coeff * x0
        assert torch.allclose(xt, expected, atol=1e-6)

    def test_same_t_for_all_batch(self, forward_vp):
        batch_size = 8
        x0 = torch.randn(batch_size, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.full((batch_size,), 500, dtype=torch.long)
        xt = forward_vp(x0, noise, t_index)
        variance = forward_vp.vs.get_variance(t_index)
        assert torch.all(variance == variance[0])
        assert xt.shape == x0.shape


class TestForwardVPEdgeCases:
    @pytest.fixture
    def forward_vp(self):
        scheduler = LinearVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        return ForwardVP(scheduler)

    def test_single_sample_batch(self, forward_vp):
        x0 = torch.randn(1, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.tensor([500])
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == (1, 3, 32, 32)

    def test_large_batch(self, forward_vp):
        batch_size = 256
        x0 = torch.randn(batch_size, 64)
        noise = torch.randn_like(x0)
        t_index = torch.randint(0, 1000, (batch_size,))
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == (batch_size, 64)

    def test_t_index_boundary(self, forward_vp):
        x0 = torch.randn(3, 3, 32, 32)
        noise = torch.randn_like(x0)
        t_index = torch.tensor([0, 500, 999])
        xt = forward_vp(x0, noise, t_index)
        assert xt.shape == x0.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])