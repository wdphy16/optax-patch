#!/usr/bin/env python3

import optax
from jax import numpy as jnp

from optax_patch.update import apply_updates_conj
from optax_patch.weight_norm import weight_norm

param = jnp.ones((2, 2)) * 2
# param = {"a": jnp.ones((2, 2)) * 2, "b": jnp.ones((2, 2)) * 2}
grad = jnp.ones((2, 2))
# grad = {"a": jnp.ones((2, 2)), "b": jnp.ones((2, 2))}
optimizer = optax.chain(optax.sgd(learning_rate=0.1, momentum=0.9), weight_norm())
state = optimizer.init(param)
print("state")
print(state)

new_grads, new_state = optimizer.update(grad, state, param)
print("new_grads")
print(new_grads)
print("new_state")
print(new_state)

new_param = apply_updates_conj(param, new_grads)
print("new_param")
print(new_param)
