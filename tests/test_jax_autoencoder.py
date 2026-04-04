import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from velm.model.autoencoder import CALMAutoencoder, batch_ae_loss, reconstruction_accuracy


def test_calmautoencoder_batch_loss_and_reconstruction():
    key = jax.random.PRNGKey(0)
    # tiny model for CI
    ae = CALMAutoencoder(vocab_size=32, chunk_size=2, hidden_dim=16, latent_dim=8, ffn_intermediate=32, key=key)

    # create a small batch of random chunks
    key, sub = jax.random.split(key)
    batch_tokens = jax.random.randint(sub, (4, 2), 0, 32)

    # compute batch loss
    key, sub = jax.random.split(key)
    loss, metrics = batch_ae_loss(ae, batch_tokens, key=sub)
    assert loss.shape == ()
    assert "recon_loss" in metrics and "kl_loss" in metrics

    # reconstruction accuracy runs and is in [0,1]
    acc = reconstruction_accuracy(ae, batch_tokens)
    assert acc >= 0.0 and acc <= 1.0
