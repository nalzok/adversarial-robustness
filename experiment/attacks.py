from functools import partial

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=('epsilon', 'max_steps', 'step_size'))
def pgd_untargeted(key, state, image, label, /, *, epsilon, max_steps, step_size):
    image_adv = image + jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)

    lower = jnp.clip(image - epsilon, 0, 1)
    upper = jnp.clip(image + epsilon, 0, 1)
    image_adv = jnp.clip(image_adv, lower, upper)

    @jax.grad
    def loss_fn(image_adv):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logit = state.apply_fn(variables, image_adv, False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logit, label)
        return loss.sum()

    for _ in range(max_steps):
        sign_grad = jnp.sign(loss_fn(image_adv))
        image_adv += step_size * sign_grad      # we want to increase the loss
        image_adv = jnp.clip(image_adv, lower, upper)

    return image_adv
