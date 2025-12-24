import pytest
import torch
from diffusion.schedules import LinearVS, SigmoidVS


class TestLinearVS:
    def test_initialization(self):
        scheduler = LinearVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        assert scheduler.num_steps == 1000
        assert scheduler.start == 0.0
        assert scheduler.end == 1.0
        assert scheduler.betas.shape == (1000,)
        assert scheduler.integral_beta.shape == (1000,)

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            LinearVS(num_steps=0)
        with pytest.raises(ValueError, match="num_steps must be positive"):
            LinearVS(num_steps=-10)

    def test_invalid_beta_range(self):
        with pytest.raises(ValueError, match="Must satisfy 0 < beta_start < beta_end"):
            LinearVS(beta_start=0.02, beta_end=1e-4)
        with pytest.raises(ValueError, match="Must satisfy 0 < beta_start < beta_end"):
            LinearVS(beta_start=0.0, beta_end=0.02)
        with pytest.raises(ValueError, match="Must satisfy 0 < beta_start < beta_end"):
            LinearVS(beta_start=0.01, beta_end=0.01)

    def test_beta_schedule_monotonic(self):
        scheduler = LinearVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        betas = scheduler.betas
        assert torch.all(betas[1:] >= betas[:-1])

    def test_beta_bounds(self):
        beta_start, beta_end = 1e-4, 0.02
        scheduler = LinearVS(num_steps=1000, beta_start=beta_start, beta_end=beta_end)
        assert torch.isclose(scheduler.betas[0], torch.tensor(beta_start), atol=1e-6)
        assert torch.isclose(scheduler.betas[-1], torch.tensor(beta_end), atol=1e-6)

    def test_variance_at_zero(self):
        scheduler = LinearVS(num_steps=1000)
        variance = scheduler.get_variance(torch.tensor([0]))
        assert torch.isclose(variance, torch.tensor(0.0), atol=1e-6)

    def test_variance_monotonic(self):
        scheduler = LinearVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        variances = scheduler.get_variance(time_steps)
        assert torch.all(variances[1:] >= variances[:-1])

    def test_variance_bounded(self):
        scheduler = LinearVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        variances = scheduler.get_variance(time_steps)
        assert torch.all(variances >= 0.0)
        assert torch.all(variances <= 1.0)

    def test_std_squared_equals_variance(self):
        scheduler = LinearVS(num_steps=1000)
        time_steps = torch.tensor([100, 500, 900])
        variance = scheduler.get_variance(time_steps)
        std = scheduler.get_std(time_steps)
        assert torch.allclose(std ** 2, variance, atol=1e-6)

    def test_drift_coefficient_negative(self):
        scheduler = LinearVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        drift = scheduler.get_drift_coeff(time_steps)
        assert torch.all(drift <= 0.0)

    def test_diffusion_coefficient_positive(self):
        scheduler = LinearVS(num_steps=1000)
        time_steps = torch.arange(1, 1000)
        diffusion = scheduler.get_diffusion_coeff(time_steps)
        assert torch.all(diffusion >= 0.0)

    def test_batch_indexing(self):
        scheduler = LinearVS(num_steps=1000)
        time_steps = torch.tensor([0, 100, 500, 999])
        variances = scheduler.get_variance(time_steps)
        assert variances.shape == (4,)

    def test_device_compatibility(self):
        scheduler = LinearVS(num_steps=100)
        time_steps = torch.tensor([10, 50, 90])
        variance_cpu = scheduler.get_variance(time_steps)
        assert variance_cpu.device.type == 'cpu'
        if torch.cuda.is_available():
            scheduler_gpu = scheduler.cuda()
            time_steps_gpu = time_steps.cuda()
            variance_gpu = scheduler_gpu.get_variance(time_steps_gpu)
            assert variance_gpu.device.type == 'cuda'

    def test_integral_beta_formula(self):
        beta_start, beta_end = 1e-4, 0.02
        scheduler = LinearVS(num_steps=1000, beta_start=beta_start, beta_end=beta_end)
        for idx in [100, 500, 900]:
            t = scheduler.t[idx]
            expected = beta_start * t + 0.5 * (beta_end - beta_start) * t ** 2
            actual = scheduler.integral_beta[idx]
            assert torch.isclose(actual, expected, atol=1e-5)


class TestSigmoidVS:

    def test_initialization(self):
        scheduler = SigmoidVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        assert scheduler.num_steps == 1000
        assert scheduler.start == 0.0
        assert scheduler.end == 1.0
        assert scheduler.betas.shape == (1000,)
        assert scheduler.integral_beta.shape == (1000,)

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            SigmoidVS(num_steps=0)
        with pytest.raises(ValueError, match="num_steps must be positive"):
            SigmoidVS(num_steps=-10)

    def test_invalid_beta_range(self):
        with pytest.raises(ValueError, match="Must satisfy 0 < beta_start < beta_end"):
            SigmoidVS(beta_start=0.02, beta_end=1e-4)
        with pytest.raises(ValueError, match="Must satisfy 0 < beta_start < beta_end"):
            SigmoidVS(beta_start=0.0, beta_end=0.02)

    def test_beta_schedule_sigmoid_shape(self):
        scheduler = SigmoidVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        betas = scheduler.betas
        first_diff = betas[100] - betas[0]
        middle_diff = betas[600] - betas[500]
        last_diff = betas[999] - betas[899]
        assert middle_diff > first_diff
        assert middle_diff > last_diff

    def test_beta_bounds(self):
        beta_start, beta_end = 1e-4, 0.02
        scheduler = SigmoidVS(num_steps=1000, beta_start=beta_start, beta_end=beta_end)
        assert torch.all(scheduler.betas >= beta_start)
        assert torch.all(scheduler.betas <= beta_end)

    def test_variance_at_zero(self):
        scheduler = SigmoidVS(num_steps=1000)
        variance = scheduler.get_variance(torch.tensor([0]))
        assert variance < 1e-4

    def test_variance_monotonic(self):
        scheduler = SigmoidVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        variances = scheduler.get_variance(time_steps)
        assert torch.all(variances[1:] >= variances[:-1])

    def test_variance_bounded(self):
        scheduler = SigmoidVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        variances = scheduler.get_variance(time_steps)
        assert torch.all(variances >= 0.0)
        assert torch.all(variances <= 1.0)

    def test_std_squared_equals_variance(self):
        scheduler = SigmoidVS(num_steps=1000)
        time_steps = torch.tensor([100, 500, 900])
        variance = scheduler.get_variance(time_steps)
        std = scheduler.get_std(time_steps)
        assert torch.allclose(std ** 2, variance, atol=1e-6)

    def test_drift_coefficient_negative(self):
        scheduler = SigmoidVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        drift = scheduler.get_drift_coeff(time_steps)
        assert torch.all(drift <= 0.0)

    def test_diffusion_coefficient_positive(self):
        scheduler = SigmoidVS(num_steps=1000)
        time_steps = torch.arange(0, 1000)
        diffusion = scheduler.get_diffusion_coeff(time_steps)
        assert torch.all(diffusion >= 0.0)

    def test_batch_indexing(self):
        scheduler = SigmoidVS(num_steps=1000)
        time_steps = torch.tensor([0, 100, 500, 999])
        variances = scheduler.get_variance(time_steps)
        assert variances.shape == (4,)


class TestLinearVSSigmoidVSComparison:

    def test_both_reach_similar_final_variance(self):
        linear = LinearVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        sigmoid = SigmoidVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        linear_final = linear.get_variance(torch.tensor([999]))
        sigmoid_final = sigmoid.get_variance(torch.tensor([999]))
        assert torch.abs(linear_final - sigmoid_final) / linear_final < 0.1

    def test_sigmoid_slower_start(self):
        linear = LinearVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        sigmoid = SigmoidVS(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        early_time = torch.tensor([100])
        linear_var = linear.get_variance(early_time)
        sigmoid_var = sigmoid.get_variance(early_time)
        assert sigmoid_var < linear_var

    def test_same_interface(self):
        linear = LinearVS(num_steps=100)
        sigmoid = SigmoidVS(num_steps=100)
        time_steps = torch.tensor([10, 50, 90])
        for scheduler in [linear, sigmoid]:
            _ = scheduler.get_variance(time_steps)
            _ = scheduler.get_std(time_steps)
            _ = scheduler.get_drift_coeff(time_steps)
            _ = scheduler.get_diffusion_coeff(time_steps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])