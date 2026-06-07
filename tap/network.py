from dataclasses import dataclass, field

import jax
from jaxtyping import Array, Float, Int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrafficNetwork:
    """A static road network as a struct-of-arrays JAX pytree.

    Directed-link representation: ``edge_index`` is ``(2, E)`` with row 0 = source node,
    row 1 = destination node (PyG convention). Differentiable leaves are the float arrays;
    ``edge_index`` is an integer leaf (traced/batched but not differentiated); ``n_nodes``
    is pytree-static. BPR cost: ``t(x) = fft * (1 + bpr_alpha * (x / capacity) ** bpr_beta)``
    (TNTP ``B`` / ``power`` map to ``bpr_alpha`` / ``bpr_beta``).
    """

    edge_index: Int[Array, "2 E"]
    fft: Float[Array, "E"]
    capacity: Float[Array, "E"]
    bpr_alpha: Float[Array, "E"]
    bpr_beta: Float[Array, "E"]
    n_nodes: int = field(metadata={"static": True})
    length: Float[Array, "E"] | None = None
    speed: Float[Array, "E"] | None = None
    coords: Float[Array, "N 2"] | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Demand:
    """Origin-destination trip demand over a set of zones (a JAX pytree).

    ``od_matrix[i, j]`` is the demand from zone ``i`` to zone ``j``; ``zone_nodes[i]`` is
    the network node id of zone ``i``.
    """

    od_matrix: Float[Array, "Z Z"]
    zone_nodes: Int[Array, "Z"]
