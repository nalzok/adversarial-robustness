from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax

from .resnet import ResNet18
from .distances import avg_distance


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]
    centroids: jnp.ndarray


def create_train_state(key, num_classes, learning_rate, specimen):
    net = ResNet18(num_classes=num_classes)
    (_, embedding), variables = net.init_with_output(key, specimen, True)
    tx = optax.adam(learning_rate)
    state = TrainState.create(
            apply_fn=net.apply,
            params=variables['params'],
            tx=tx,
            batch_stats=variables['batch_stats'],
            centroids=jnp.zeros((num_classes, np.prod(embedding.shape[1:], dtype=int)))
    )
    return state


@jax.jit
def train_step(state, image, label):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (logits, embedding), new_model_state = state.apply_fn(
            variables, image, True, mutable=['batch_stats']
        )
        embedding = embedding.reshape(embedding.shape[0], -1)

        predictive_term = optax.softmax_cross_entropy_with_integer_labels(logits, label)
        dist_regularizer = avg_distance(embedding, state.centroids, label)
        loss = (predictive_term - 0 * image.shape[0]/64 * dist_regularizer).sum()

        return loss, (embedding, new_model_state)

    (loss, (embedding, new_model_state)), grads = loss_fn(state.params)

    num_classes = state.centroids.shape[0]
    vbincount = jax.vmap(lambda X: jnp.bincount(label, weights=X, length=num_classes), in_axes=1, out_axes=1)
    count = jnp.bincount(label, length=num_classes)
    centroids = vbincount(embedding)/count[:, jnp.newaxis]

    momentum = 1 - 0.8 * count/image.shape[0]   # TODO: improve this heuristic
    new_centroids = momentum[:, jnp.newaxis] * state.centroids + (1-momentum[:, jnp.newaxis]) * centroids

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
        centroids=new_centroids
    )

    return state, loss


@jax.jit
def test_step(state, image, label):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits, _ = state.apply_fn(variables, image, False)
    prediction = jnp.argmax(logits, axis=-1)
    hit = jnp.sum(prediction == label)

    return hit