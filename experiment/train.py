from typing import Any, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax

from .resnet import ResNet18


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]


def create_train_state(key: Any, num_classes: int, learning_rate: float, specimen: jnp.ndarray) -> TrainState:
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


def create_finetune_state(state: TrainState, learning_rate: float) -> TrainState:
    tx = optax.adam(learning_rate)
    opt_state = tx.init(state.params)   # resets momentum, etc.
    state = state.replace(step=0, tx=tx, opt_state=opt_state)
    return state



@partial(jax.pmap, axis_name='batch')
def train_step(state: TrainState, image: jnp.ndarray, image_adv: jnp.ndarray, label: jnp.ndarray) \
        -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        _, embedding = state.apply_fn(variables, image, False)
        (logits, embedding_adv), new_model_state = state.apply_fn(
            variables, image_adv, True, mutable=['batch_stats']
        )

        predictive = optax.softmax_cross_entropy_with_integer_labels(logits, label)
        consistency = jnp.sum((embedding_adv - embedding)**2, axis=-1)
        loss = predictive.sum() + consistency.sum()

        return loss, (predictive.sum(), consistency.sum(), new_model_state)

    (_, (preductive, consistency, new_model_state)), grads = loss_fn(state.params)
    grads = jax.lax.psum(grads, axis_name='batch')

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
    )

    return state, preductive, consistency


@jax.pmap
def update_batch_stats_step(state: TrainState, image_adv: jnp.ndarray) -> TrainState:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    _, new_model_state = state.apply_fn(
        variables, image_adv, True, mutable=['batch_stats']
    )
    state = state.replace(
        batch_stats=new_model_state['batch_stats'],
    )
    return state


cross_replica_mean: Callable = jax.pmap(lambda x: jax.lax.pmean(x, 'batch'), 'batch')

 
@jax.pmap
def test_step(state: TrainState, image: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits, _ = state.apply_fn(variables, image, False)
    prediction = jnp.argmax(logits, axis=-1)
    hit = jnp.sum(prediction == label)

    return hit
