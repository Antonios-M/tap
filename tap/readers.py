import jax.numpy as jnp
from jaxtyping import Array, Float
from numpy.typing import ArrayLike

from tap.network import Demand, TrafficNetwork

# --------------------------------------------------------------------------- #
# Casting helpers (eager — used by constructors only)                           #
# --------------------------------------------------------------------------- #


def _link_array(x: ArrayLike, n_links: int, name: str) -> Float[Array, "E"]:
    """Cast ``x`` to a float ``(E,)`` array, broadcasting a scalar if needed."""
    arr = jnp.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = jnp.full((n_links,), arr)
    if arr.shape != (n_links,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({n_links},); got {arr.shape}"
        )
    return arr


def _maybe_link_array(
    x: ArrayLike | None, n_links: int, name: str
) -> Float[Array, "E"] | None:
    return None if x is None else _link_array(x, n_links, name)


# --------------------------------------------------------------------------- #
# Constructors                                                                  #
# --------------------------------------------------------------------------- #


def from_edges(
    edge_index: ArrayLike,
    capacity: ArrayLike,
    *,
    fft: ArrayLike | None = None,
    length: ArrayLike | None = None,
    speed: ArrayLike | None = None,
    bpr_alpha: ArrayLike = 0.15,
    bpr_beta: ArrayLike = 4.0,
    n_nodes: int | None = None,
    coords: ArrayLike | None = None,
    validate: bool = True,
) -> TrafficNetwork:
    """Build a :class:`~tap.network.TrafficNetwork`, casting inputs to JAX arrays.

    Parameters
    ----------
    edge_index : array-like, shape (2, E)
        ``[src; dst]`` node ids per directed link.
    capacity : scalar or array-like (E,)
        Link capacities (veh/h). May be 0 for a fully blocked link.
    fft : scalar or array-like (E,), optional
        Free-flow travel time. If omitted, both ``length`` and ``speed`` must be given
        and ``fft = length / speed`` is used.
    length, speed : scalar or array-like (E,), optional
        Link length and free-flow speed. Stored if provided; used to derive ``fft``
        when ``fft`` is not given.
    bpr_alpha, bpr_beta : scalar or array-like (E,)
        BPR parameters (TNTP ``B`` / ``power``). Defaults: 0.15 / 4.0.
    n_nodes : int, optional
        Node count. Derived from ``coords`` or ``edge_index`` if omitted.
    coords : array-like, shape (n_nodes, 2), optional
        Node coordinates.
    validate : bool
        Run eager shape/value validation (disable inside hot paths).
    """
    edge_index = jnp.asarray(edge_index, dtype=jnp.int32)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E); got {edge_index.shape}")
    n_links = int(edge_index.shape[1])

    capacity = _link_array(capacity, n_links, "capacity")
    bpr_alpha = _link_array(bpr_alpha, n_links, "bpr_alpha")
    bpr_beta = _link_array(bpr_beta, n_links, "bpr_beta")
    length = _maybe_link_array(length, n_links, "length")
    speed = _maybe_link_array(speed, n_links, "speed")

    if fft is not None:
        fft = _link_array(fft, n_links, "fft")
    elif length is not None and speed is not None:
        fft = length / speed
    else:
        raise ValueError("provide either `fft` or both `length` and `speed`")

    if coords is not None:
        coords = jnp.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (n_nodes, 2); got {coords.shape}")

    if n_nodes is None:
        if coords is not None:
            n_nodes = int(coords.shape[0])
        elif n_links > 0:
            n_nodes = int(edge_index.max()) + 1
        else:
            raise ValueError("cannot infer n_nodes: pass n_nodes or coords")

    net = TrafficNetwork(
        edge_index=edge_index,
        fft=fft,
        capacity=capacity,
        bpr_alpha=bpr_alpha,
        bpr_beta=bpr_beta,
        n_nodes=int(n_nodes),
        length=length,
        speed=speed,
        coords=coords,
    )
    if validate:
        validate_network(net)
    return net


def make_demand(
    od_matrix: ArrayLike, zone_nodes: ArrayLike, *, validate: bool = True
) -> Demand:
    """Build a :class:`~tap.network.Demand`, casting inputs to JAX arrays."""
    od_matrix = jnp.asarray(od_matrix, dtype=float)
    zone_nodes = jnp.asarray(zone_nodes, dtype=jnp.int32)
    demand = Demand(od_matrix=od_matrix, zone_nodes=zone_nodes)
    if validate:
        validate_demand(demand)
    return demand


# --------------------------------------------------------------------------- #
# Validation (eager — NOT for use under jit/vmap; forces concretization)        #
# --------------------------------------------------------------------------- #


def validate_network(net: TrafficNetwork) -> None:
    """Raise ``ValueError`` if *net* is structurally or numerically invalid (eager)."""
    e = int(net.edge_index.shape[1])
    if net.edge_index.shape != (2, e):
        raise ValueError(f"edge_index shape {net.edge_index.shape} != (2, {e})")
    for name in ("fft", "capacity", "bpr_alpha", "bpr_beta"):
        arr = getattr(net, name)
        if arr.shape != (e,):
            raise ValueError(f"{name} shape {arr.shape} != ({e},)")
    for name in ("length", "speed"):
        arr = getattr(net, name)
        if arr is not None and arr.shape != (e,):
            raise ValueError(f"{name} shape {arr.shape} != ({e},)")
    if net.coords is not None and net.coords.shape != (net.n_nodes, 2):
        raise ValueError(f"coords shape {net.coords.shape} != ({net.n_nodes}, 2)")

    if e > 0:
        if bool((net.edge_index < 0).any()) or bool(
            (net.edge_index >= net.n_nodes).any()
        ):
            raise ValueError(f"edge_index node ids out of range [0, {net.n_nodes})")
        if bool((net.capacity < 0).any()):
            raise ValueError("capacity must be non-negative")
        if bool((net.fft < 0).any()):
            raise ValueError("fft must be non-negative")
        if net.speed is not None and bool((net.speed <= 0).any()):
            raise ValueError("speed must be strictly positive")


def validate_demand(demand: Demand) -> None:
    """Raise ``ValueError`` if *demand* is structurally or numerically invalid (eager)."""
    z = int(demand.zone_nodes.shape[0])
    if demand.od_matrix.shape != (z, z):
        raise ValueError(f"od_matrix shape {demand.od_matrix.shape} != ({z}, {z})")
    if z > 0 and bool((demand.od_matrix < 0).any()):
        raise ValueError("od_matrix must be non-negative")
