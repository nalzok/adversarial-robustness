import jax
import jax.numpy as jnp


@jax.jit
def dispersion(v, centroids, label):
    squares = jnp.sum((v - centroids[label])**2, axis=-1)
    sum_of_squares = jnp.sum(squares, axis=-1)
    return sum_of_squares


@jax.jit
def pushpull_distance(v, centroids, label):
    distance = jnp.linalg.norm(v[:, jnp.newaxis, :] - centroids, axis=-1)
    multiplier = jnp.where(jnp.arange(centroids.shape[0]) == label[:, jnp.newaxis], 1, -0.1)
    pushpull_dist = jnp.mean(distance * multiplier, axis=-1)
    return pushpull_dist


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


@jax.jit
def avg_cosine(v, centroids, label):
    distance = jnp.sum(v[:, jnp.newaxis, :] * centroids, axis=-1)
    where = jnp.arange(centroids.shape[0]) != label[:, jnp.newaxis]
    avg_dist = jnp.mean(distance, axis=-1, where=where)
    return avg_dist


@jax.jit
def min_cosine(v, centroids, label):
    distance = jnp.sum(v[:, jnp.newaxis, :] * centroids, axis=-1)
    initial = jnp.inf
    where = jnp.arange(centroids.shape[0]) != label[:, jnp.newaxis]
    min_dist = jnp.min(distance, axis=-1, initial=initial, where=where)
    return min_dist
