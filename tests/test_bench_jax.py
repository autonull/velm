import pytest

pytest.skip("JAX backend deprecated in favor of PyTorch core", allow_module_level=True)
jax = pytest.importorskip("jax")
import time


def test_jax_autoencoder_benchmark():
    key = jax.random.PRNGKey(0)
    ae = CALMAutoencoder(vocab_size=512, chunk_size=4, hidden_dim=64, latent_dim=32, ffn_intermediate=128, key=key)

    # create a single batch and time encode/decode
    key, sub = jax.random.split(key)
    tokens = jax.random.randint(sub, (8, 4), 0, 512)

    # JAX warmup (compilation might take time; run once)
    _ = ae.encode(tokens[0])

    iters = 5
    t0 = time.perf_counter()
    for i in range(iters):
        _ = ae.encode(tokens[i % tokens.shape[0]])
    dt = (time.perf_counter() - t0) / iters

    # generous threshold for CI
    assert dt < 5.0
