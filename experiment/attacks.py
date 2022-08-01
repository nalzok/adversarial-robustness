from functools import partial

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnames=('epsilon', 'max_steps', 'step_size', 'randomize'), donate_argnums=(2,))
def pgd_untargeted(key, state, image, label, /, *, epsilon, max_steps, step_size, randomize):
    if randomize:
        key_epsilon, key = jax.random.split(key)
        epsilon = jax.random.uniform(key_epsilon, (1,), minval=0, maxval=epsilon)

    image_adv = image + jax.random.uniform(key, image.shape, minval=-epsilon, maxval=epsilon)

    lower = jnp.clip(image - epsilon, 0, 1)
    upper = jnp.clip(image + epsilon, 0, 1)
    image_adv = jnp.clip(image_adv, lower, upper)

    @partial(jax.grad, has_aux=True)
    def loss_fn(image_adv):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        logit = state.apply_fn(variables, image_adv, False)
        prediction = jnp.argmax(logit, axis=-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logit, label)
        return loss.sum(), prediction

    for _ in range(max_steps):
        grad, prediction = loss_fn(image_adv)
        mask = prediction != label
        image_adv += step_size * mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * jnp.sign(grad)
        image_adv = jnp.clip(image_adv, lower, upper)

    return image_adv
