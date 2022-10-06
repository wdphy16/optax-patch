from typing import NamedTuple, Optional, Union

import chex
import jax
from jax import numpy as jnp
from jax.tree_util import tree_leaves, tree_map, tree_structure, tree_unflatten
from netket.jax import PRNGKey, dtype_real
from optax._src import base, numerics
from optax._src.transform import AddNoiseState

ScalarOrSchedule = Union[float, base.Schedule]


def add_rel_noise(
    eta: float, gamma: float = 1, seed: Optional[int] = None
) -> base.GradientTransformation:
    def init_fn(_):
        return AddNoiseState(count=jnp.zeros((), jnp.int32), rng_key=PRNGKey(seed))

    def update_fn(updates, state, params=None):
        num_keys = len(tree_leaves(updates))
        treedef = tree_structure(updates)
        count = numerics.safe_int32_increment(state.count)
        noise_amp = eta / count**gamma
        keys = jax.random.split(state.rng_key, num_keys + 1)

        def f(grad, key):
            mul = 1 + noise_amp.astype(grad.dtype) * jax.random.normal(
                key, shape=grad.shape, dtype=grad.dtype
            )
            return grad * mul

        updates = tree_map(f, updates, tree_unflatten(treedef, keys[1:]))
        return updates, AddNoiseState(count=count, rng_key=keys[0])

    return base.GradientTransformation(init_fn, update_fn)


class RandomizeState(NamedTuple):
    rng_key: chex.PRNGKey


def randomize(seed: Optional[int] = None) -> base.GradientTransformation:
    def init_fn(_):
        return RandomizeState(rng_key=PRNGKey(seed))

    def update_fn(updates, state, params=None):
        num_keys = len(tree_leaves(updates))
        treedef = tree_structure(updates)
        keys = jax.random.split(state.rng_key, num_keys + 1)

        def f(grad, key):
            if jnp.isrealobj(grad):
                grad = jnp.sign(grad)
            else:
                grad /= jnp.abs(grad)
            return grad * jax.random.uniform(
                key, shape=grad.shape, dtype=dtype_real(grad.dtype)
            )

        updates = tree_map(f, updates, tree_unflatten(treedef, keys[1:]))
        return updates, RandomizeState(rng_key=keys[0])

    return base.GradientTransformation(init_fn, update_fn)
