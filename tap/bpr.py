import jax.numpy as jnp
from jaxtyping import Array, Float

_CAP_EPS: float = 1e-8
_HUGE_COST: float = 1e30


def bpr_cost(
    flow: Float[Array, "E"],
    fft: Float[Array, "E"],
    capacity: Float[Array, "E"],
    alpha: Float[Array, "E"],
    beta: Float[Array, "E"],
) -> Float[Array, "E"]:
    """Vectorized BPR travel time ``t(x) = fft * (1 + alpha * (x / capacity) ** beta)``.

    Blocked links (``capacity <= 0``) receive a finite ``_HUGE_COST``. Inputs may be
    scalars or ``(E,)`` arrays (they broadcast). Differentiable w.r.t. all arguments;
    note the ``beta`` gradient involves ``log(flow / capacity)`` and is only finite for
    ``flow > 0`` (sample away from zero flow in gradient checks).
    """
    blocked = capacity <= _CAP_EPS
    safe_cap = jnp.where(blocked, 1.0, capacity)
    ratio = flow / safe_cap
    cost = fft * (1.0 + alpha * ratio**beta)
    return jnp.where(blocked, _HUGE_COST, cost)


def bpr_marginal(
    flow: Float[Array, "E"],
    fft: Float[Array, "E"],
    capacity: Float[Array, "E"],
    alpha: Float[Array, "E"],
    beta: Float[Array, "E"],
) -> Float[Array, "E"]:
    """Marginal (system-optimal) link cost.

    The derivative of the *total* link cost ``x * t(x)`` w.r.t. ``x``::

        d/dx [x * t(x)] = fft * (1 + alpha * (beta + 1) * (x / capacity) ** beta)

    Using this in place of :func:`bpr_cost` inside the assignment loop yields the
    system-optimal flow (minimizing total travel time) rather than user equilibrium.
    Same blocked-link / differentiability semantics as :func:`bpr_cost`.
    """
    blocked = capacity <= _CAP_EPS
    safe_cap = jnp.where(blocked, 1.0, capacity)
    ratio = flow / safe_cap
    cost = fft * (1.0 + alpha * (beta + 1.0) * ratio**beta)
    return jnp.where(blocked, _HUGE_COST, cost)
