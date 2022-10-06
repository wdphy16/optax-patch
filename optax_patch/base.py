from typing import Callable, NamedTuple, Optional, Tuple, Union

import chex
from optax._src import base

AdaptiveTransformUpdateFn = Callable[
    [base.Updates, base.OptState, Optional[base.Params], Optional[chex.Array]],
    Tuple[base.Updates, base.OptState],
]


class AdaptiveGradientTransformation(NamedTuple):
    init: base.TransformInitFn
    update: AdaptiveTransformUpdateFn


GradientTransformation = Union[
    base.GradientTransformation, AdaptiveGradientTransformation
]
