from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax

from .resnet import ResNet18


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(key, num_classes, learning_rate, specimen):
    net = ResNet18(num_classes=num_classes)
    variables = net.init(key, specimen, True)
    tx = optax.adam(learning_rate)
    state = TrainState.create(
            apply_fn=net.apply,
            params=variables['params'],
            tx=tx,
            batch_stats=variables['batch_stats'],
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def train_step(state, image, image_adv, label):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables, image_adv, True, mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, label)

        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )

    return state, loss


@jax.jit
def test_step(state, image, label):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, image, False)
    prediction = jnp.argmax(logits, axis=-1)
    hit = jnp.sum(prediction == label)

    return hit
