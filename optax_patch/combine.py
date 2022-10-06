import inspect
from typing import Callable, Hashable, Mapping, Union

from jax.tree_util import tree_leaves, tree_map
from optax._src import base, combine

from . import base as _base
from . import wrappers as _wrappers


def chain(*args: _base.GradientTransformation) -> _base.GradientTransformation:
    init_fns, update_fns = zip(*args)

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None, loss=None):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in "
                "chain! Make sure you have called init first!"
            )

        new_state = []
        for s, fn in zip(state, update_fns):
            if "loss" in inspect.signature(fn).parameters:
                updates, new_s = fn(updates, s, params, loss)
            else:
                updates, new_s = fn(updates, s, params)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return _base.AdaptiveGradientTransformation(init_fn, update_fn)


def multi_transform(
    transforms: Mapping[Hashable, _base.GradientTransformation],
    param_labels: Union[base.PyTree, Callable[[base.PyTree], base.PyTree]],
) -> _base.GradientTransformation:
    def make_mask(labels, group):
        return tree_map(lambda label: label == group, labels)

    def init_fn(params):
        labels = param_labels(params) if callable(param_labels) else param_labels

        label_set = set(tree_leaves(labels))
        if not label_set.issubset(transforms.keys()):
            raise ValueError(
                "Some parameters have no corresponding transformation.\n"
                f"Parameter labels: {sorted(label_set)}\n"
                f"Transforms keys: {sorted(transforms.keys())}\n"
            )

        inner_states = {
            group: _wrappers.masked(tx, make_mask(labels, group)).init(params)
            for group, tx in transforms.items()
        }
        return combine.MultiTransformState(inner_states)

    def update_fn(updates, state, params=None, loss=None):
        labels = param_labels(updates) if callable(param_labels) else param_labels

        new_inner_states = {}
        for group, tx in transforms.items():
            masked_tx = _wrappers.masked(tx, make_mask(labels, group))
            updates, new_inner_states[group] = masked_tx.update(
                updates, state.inner_states[group], params, loss
            )
        return updates, combine.MultiTransformState(new_inner_states)

    return _base.AdaptiveGradientTransformation(init_fn, update_fn)
