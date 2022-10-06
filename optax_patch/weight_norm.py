from typing import NamedTuple, Optional, Tuple

import chex
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_map
from optax._src import base

from .update import apply_updates_conj

Param = chex.Array
Update = chex.Array


def _l2_norm(x):
    x = x.reshape((-1, x.shape[-1]))
    norm = jnp.sqrt(((x.conj() * x).real).sum(axis=0))
    return norm


def _tree_zip_map(f, tree, *rest, is_leaf=None):
    leaves, treedef = tree_flatten(tree, is_leaf)
    all_leaves = (leaves,) + tuple(treedef.flatten_up_to(r) for r in rest)
    all_outputs = tuple(f(*xs) for xs in zip(*all_leaves))
    return tuple(treedef.unflatten(ys) for ys in zip(*all_outputs))


class WeightNormState(NamedTuple):
    directions: base.Params
    scales: base.Params


def weight_norm(decay: float = 0, eps: float = 1e-8) -> base.GradientTransformation:
    def _split_param(param: Param) -> Tuple[Param, Optional[Param]]:
        if param.ndim > 1:
            scale = _l2_norm(param)
            direction = param / (scale + eps)
            return direction, scale
        else:
            return param, None

    def _merge_param(direction: Param, scale: Optional[Param]) -> Param:
        if direction.ndim > 1:
            return direction * scale
        else:
            return direction

    def _split_grad(
        grad: Update, direction: Param, scale: Param
    ) -> Tuple[Update, Optional[Update]]:
        if direction.ndim > 1:
            norm = _l2_norm(direction) + eps
            direction /= norm
            scale_grad = jnp.sum(
                grad * direction, axis=tuple(range(direction.ndim - 1))
            )
            direction_grad = scale / norm * (grad - scale_grad * direction)
            if decay:
                direction_grad += decay * direction
            return direction_grad, scale_grad
        else:
            return grad, None

    def init_fn(params):
        directions, scales = _tree_zip_map(_split_param, params)
        return WeightNormState(directions=directions, scales=scales)

    def update_fn(updates, state, params):
        direction_grads, scale_grads = _tree_zip_map(
            _split_grad, updates, state.directions, state.scales
        )

        wn_params = {"directions": state.directions, "scales": state.scales}
        wn_updates = {"directions": direction_grads, "scales": scale_grads}
        new_wn_params = apply_updates_conj(wn_params, wn_updates)

        new_directions = new_wn_params["directions"]
        new_scales = new_wn_params["scales"]
        new_params = tree_map(_merge_param, new_directions, new_scales)

        new_updates = tree_map(lambda n, p: (n - p).conj(), new_params, params)

        return new_updates, WeightNormState(
            directions=new_directions, scales=new_scales
        )

    return base.GradientTransformation(init_fn, update_fn)
