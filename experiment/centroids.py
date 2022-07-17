import jax
import jax.numpy as jnp


@jax.jit
def avg_dist(v, centroids):
    distances = jnp.linalg.norm(v[:, jnp.newaxis, :] - centroids, axis=-1)
    return jnp.mean(distances, axis=-1)
