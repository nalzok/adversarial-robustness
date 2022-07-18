import jax
import jax.numpy as jnp


@jax.jit
def avg_distance(v, centroids, label):
    distance = jnp.linalg.norm(v[:, jnp.newaxis, :] - centroids, axis=-1)
    where = jnp.arange(centroids.shape[0]) != label[:, jnp.newaxis]
    avg_dist = jnp.mean(distance, axis=-1, where=where)
    return avg_dist


@jax.jit
def min_distance(v, centroids, label):
    distance = jnp.linalg.norm(v[:, jnp.newaxis, :] - centroids, axis=-1)
    initial = jnp.inf
    where = jnp.arange(centroids.shape[0]) != label[:, jnp.newaxis]
    min_dist = jnp.min(distance, axis=-1, initial=initial, where=where)
    return min_dist
