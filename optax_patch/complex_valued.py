from jax.tree_util import tree_map
from optax.contrib import _complex_valued as complex_valued

from . import base as _base


def split_real_and_imaginary(
    inner: _base.GradientTransformation,
) -> _base.GradientTransformation:
    def init_fn(params):
        params = tree_map(complex_valued._complex_to_real_pair, params)
        inner_state = inner.init(params)
        return complex_valued.SplitRealAndImaginaryState(inner_state)

    def update_fn(updates, state, params=None, loss=None):
        inner_state = state.inner_state
        updates = tree_map(complex_valued._complex_to_real_pair, updates)
        params = tree_map(complex_valued._complex_to_real_pair, params)
        updates, inner_state = inner.update(updates, inner_state, params, loss)
        updates = tree_map(
            complex_valued._real_pair_to_complex,
            updates,
            is_leaf=lambda x: isinstance(x, complex_valued.SplitRealAndImaginaryArrays),
        )
        return updates, complex_valued.SplitRealAndImaginaryState(inner_state)

    return _base.AdaptiveGradientTransformation(init_fn, update_fn)
