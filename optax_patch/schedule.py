from typing import NamedTuple

import chex
import jax
from jax import lax
from jax import numpy as jnp
from jax.tree_util import tree_map
from optax._src import base, numerics

from . import base as _base


class ReduceOnPlateauState(NamedTuple):
    losses: chex.Array
    count: int
    lr: float
    t_stat: float


# Assume `y` is real
@jax.jit
def get_slope_t_stat(y, eps=1e-16):
    N = y.size
    x = jnp.arange(N)
    x_mean = x.mean()
    y_mean = jnp.nanmean(y)
    x_var = (x**2).mean() - x_mean**2
    y_var = jnp.nanmean(y**2) - y_mean**2
    cov = jnp.nanmean(x * y) - x_mean * y_mean
    t_stat = jnp.sqrt(N - 2) * cov / jnp.sqrt(x_var * y_var + eps)
    return t_stat


def reduce_on_plateau(
    lr_init: float,
    lr_min: float,
    lr_max: float,
    lr_decay: float,
    lr_grow: float,
    window: int,
    threshold: float,
) -> _base.AdaptiveGradientTransformation:
    assert 0 < lr_min <= lr_init <= lr_max
    assert 0 < lr_decay <= 1
    assert lr_grow >= 1
    assert window > 0
    assert threshold > 0

    def init_fn(_):
        return ReduceOnPlateauState(
            losses=jnp.zeros((window,)),
            count=jnp.zeros((), dtype=jnp.int32),
            lr=float(lr_init),
            t_stat=0.0,
        )

    def update_fn(updates, state, _, loss):
        losses = jnp.concatenate(
            [state.losses[1:], jnp.expand_dims(loss.real, axis=0)], axis=0
        )

        def try_reduce_lr(_):
            count = jnp.zeros((), dtype=jnp.int32)

            t_stat = get_slope_t_stat(losses)
            need_decay = t_stat > -threshold

            lr = state.lr
            lr = need_decay * lr_decay * lr + (1 - need_decay) * lr_grow * lr
            lr = jnp.maximum(lr, lr_min)
            lr = jnp.minimum(lr, lr_max)

            return count, lr, t_stat

        count = numerics.safe_int32_increment(state.count)
        count, lr, t_stat = lax.cond(
            count >= window,
            try_reduce_lr,
            lambda _: (count, state.lr, state.t_stat),
            None,
        )

        updates = tree_map(lambda g: -jnp.asarray(lr, dtype=g.dtype) * g, updates)

        return updates, ReduceOnPlateauState(losses, count, lr, t_stat)

    return _base.AdaptiveGradientTransformation(init_fn, update_fn)


class AdjustableLRState(NamedTuple):
    lr: float


def adjustable_lr(lr_init: float) -> base.GradientTransformation:
    assert lr_init > 0

    def init_fn(_):
        return AdjustableLRState(lr=lr_init)

    def update_fn(updates, state, _):
        updates = tree_map(lambda g: -jnp.asarray(state.lr, dtype=g.dtype) * g, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
