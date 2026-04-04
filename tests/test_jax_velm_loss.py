import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from velm.model import VELM
from velm.model.config import CONFIGS


def test_velm_training_loss_smoke():
    key = jax.random.PRNGKey(1)
    cfg_name = "smoke"
    cfg = CONFIGS[cfg_name]
    K = cfg["chunk_size_k"]

    # instantiate tiny VELM
    model = VELM(config_name=cfg_name, vocab_size=32, key=key)

    # create a small sequence of chunks (S, K)
    key, sub = jax.random.split(key)
    S = 8
    token_ids = jax.random.randint(sub, (S, K), 0, 32)

    # compute training loss
    key, sub = jax.random.split(key)
    loss, metrics = model.training_loss(token_ids, key=sub, num_samples=2)
    assert loss.shape == ()
    assert "energy_loss" in metrics
    assert jnp.isfinite(loss)
