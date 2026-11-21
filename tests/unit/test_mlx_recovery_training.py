"""Optional MLX export parity and optimization tests (requires Metal access)."""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch

from real2sim.policy.recovery_mlp import RecoveryMLP


@pytest.fixture(scope="module")
def metal_device():
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("Optional MLX dependency is not installed")
    # Probe in a child: a failed Metal import can leave a partially initialized
    # extension that aborts Python if a second parametrized case imports it.
    probe = subprocess.run([sys.executable, "-c", "import mlx.core"],
                           capture_output=True, timeout=30)
    if probe.returncode:
        pytest.skip("MLX requires an accessible Metal device")


@pytest.mark.parametrize("use_act", [True, False])
def test_mlx_export_and_training_match_runtime(use_act, metal_device):
    import mlx.core as mx

    from real2sim.policy.mlx_recovery_training import MLXRecoveryTrainer

    torch.manual_seed(4)
    model = RecoveryMLP(np.zeros(3), np.ones(3), np.zeros(2), np.ones(2),
                        use_act=use_act, hidden_sizes=(8, 4))
    trainer = MLXRecoveryTrainer(model, .01)
    batch = {"observations": torch.randn(16, 3), "base_actions": torch.rand(16, 7) * .01,
             "features": torch.randn(16, 2), "target_actions": torch.ones(16, 7) * .01}
    initial = trainer.step(batch)
    for _ in range(30):
        final = trainer.step(batch)
    assert final < initial
    trainer.export()
    with torch.no_grad():
        expected = model(batch["observations"], batch["base_actions"], batch["features"]).numpy()
    actual = trainer.forward(trainer.network, {k: mx.array(v.numpy()) for k, v in batch.items()})
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6, rtol=1e-5)
    assert np.isfinite(expected).all()
