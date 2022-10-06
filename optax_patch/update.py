from jax import numpy as jnp
from jax.tree_util import tree_map
from optax._src import base


def apply_updates_conj(params: base.Params, updates: base.Updates) -> base.Params:
    return tree_map(
        lambda p, u: jnp.asarray(p + u.conj()).astype(jnp.asarray(p).dtype),
        params,
        updates,
    )
