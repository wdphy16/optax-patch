from typing import NamedTuple

from jax.tree_util import tree_map
from optax._src import base


class ExponentialMovingAverageState(NamedTuple):
    params: base.Params


def exponential_moving_average(decay: float) -> base.GradientTransformation:
    def init_fn(params):
        return ExponentialMovingAverageState(params=params)

    def update_fn(updates, state, params):
        def f(param_old, param_new):
            return decay * param_old + (1 - decay) * param_new

        new_params = tree_map(f, state.params, params)
        return updates, ExponentialMovingAverageState(params=new_params)

    return base.GradientTransformation(init_fn, update_fn)
