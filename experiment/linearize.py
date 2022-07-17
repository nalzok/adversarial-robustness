import jax
import jax.numpy as jnp
import numpy as np


def linearize(state, net, X0):
    Js = []
    output = X0
    for f in net.steps:
        output, J = jax.vmap(
            lambda tangent: jax.jvp(
                lambda X: net.apply(state.params, X, method=f), (output,), (tangent,)
            ),
        )(jnp.eye(np.prod(output.shape)).reshape(-1, *output.shape))
        output = output[0]
        Js.append(J)

    return Js, output
