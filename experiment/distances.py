import jax
import jax.numpy as jnp


@jax.jit
def avg_distance(v, centroids, label):
    distance = jnp.linalg.norm(v[:, jnp.newaxis, :] - centroids, axis=-1)
    where = jnp.arange(centroids.shape[0]) != label[:, jnp.newaxis]
    avg_dist = jnp.mean(distance, axis=-1, where=where)
    return avg_dist
