from functools import partial

import jax
import jax.numpy as jnp
import optax


@partial(jax.pmap, static_broadcasted_argnums=(4, 5))
def pgd_untargeted(key, state, image, label, epsilon: float, iters: int):
    # key_epsilon, key = jax.random.split(key)
    # epsilon = jax.random.uniform(key_epsilon, (1,), minval=0, maxval=epsilon)

    image_adv = image + jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)

    lower = jnp.clip(image - epsilon, 0, 1)
    upper = jnp.clip(image + epsilon, 0, 1)
    image_adv = jnp.clip(image_adv, lower, upper)

    @jax.grad
    def loss_fn(image_adv):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logit, _ = state.apply_fn(variables, image_adv, False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logit, label)
        return loss.sum()

    for k in range(iters):
        grad = loss_fn(image_adv)
        delta = 2*epsilon/(k+1) * jnp.sign(grad)
        image_adv += delta * jnp.sign(grad)
        image_adv = jnp.clip(image_adv, lower, upper)

    return image_adv
