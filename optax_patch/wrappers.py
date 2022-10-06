from typing import Callable, Union

from flax.core import FrozenDict, freeze, unfreeze
from jax.tree_util import tree_map
from optax._src import base, wrappers

from . import base as _base


def masked(
    inner: _base.GradientTransformation,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]],
) -> _base.GradientTransformation:
    def mask_pytree(pytree, mask_tree):
        return tree_map(lambda m, p: p if m else None, mask_tree, pytree)

    def init_fn(params):
        params = unfreeze(params)
        mask_tree = mask(params) if callable(mask) else mask
        masked_params = mask_pytree(params, mask_tree)
        return wrappers.MaskedState(inner_state=inner.init(masked_params))

    def update_fn(updates, state, params=None, loss=None):
        is_frozen = False
        if isinstance(updates, FrozenDict):
            is_frozen = True
            updates = unfreeze(updates)

        params = unfreeze(params)
        mask_tree = mask(updates) if callable(mask) else mask
        masked_updates = mask_pytree(updates, mask_tree)
        masked_params = None if params is None else mask_pytree(params, mask_tree)

        new_masked_updates, new_inner_state = inner.update(
            masked_updates, state.inner_state, masked_params
        )

        new_updates = tree_map(
            lambda m, new_u, old_u: new_u if m else old_u,
            mask_tree,
            new_masked_updates,
            updates,
        )

        if is_frozen:
            new_updates = freeze(new_updates)

        return new_updates, wrappers.MaskedState(inner_state=new_inner_state)

    return _base.AdaptiveGradientTransformation(init_fn, update_fn)
