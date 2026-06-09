import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int


def dense_cost_matrix(
    edge_index: Int[Array, "2 E"],
    costs: Float[Array, "E"],
    n_nodes: int,
) -> tuple[Float[Array, "N N"], Int[Array, "N N"]]:
    """Build the weighted adjacency matrix of the network, plus an edge-index lookup.

    The network is an edge list: ``edge_index[:, k] = (from_node, to_node)`` and
    ``costs[k]`` is the weight (travel time) of directed link ``k``. Returns two
    ``(n_nodes, n_nodes)`` arrays:

    - ``cost_matrix[i, j]`` — weight of the cheapest direct edge ``i -> j``; ``inf`` where
      there is no edge, ``0`` on the diagonal.
    - ``link_index[i, j]`` — index of the edge realising that weight (``n_links``, i.e.
      out of range, where there is no edge). Lets flow routed over the pair ``(i, j)`` be
      scattered back onto the originating link.

    Parallel edges collapse to their minimum weight; ``link_index`` breaks ties to the
    lowest index. Both are assembled with scatter-``min`` (``.at[].min``), hence
    independent of edge ordering.
    """
    from_node, to_node = edge_index[0], edge_index[1]
    n_links = costs.shape[0]
    inf = jnp.asarray(jnp.inf, dtype=costs.dtype)

    cost_matrix = (
        jnp.full((n_nodes, n_nodes), inf, dtype=costs.dtype)
        .at[from_node, to_node]
        .min(costs)
    )
    nodes = jnp.arange(n_nodes)
    cost_matrix = cost_matrix.at[nodes, nodes].set(jnp.asarray(0.0, dtype=costs.dtype))

    # Representative edge per pair: lowest-index link attaining the cell minimum.
    cheapest_for_pair = cost_matrix[from_node, to_node]
    is_cheapest = costs == cheapest_for_pair
    candidate = jnp.where(is_cheapest, jnp.arange(n_links), n_links)
    link_index = (
        jnp.full((n_nodes, n_nodes), n_links, dtype=jnp.int32)
        .at[from_node, to_node]
        .min(candidate.astype(jnp.int32))
    )
    return cost_matrix, link_index


def floyd_warshall(
    cost_matrix: Float[Array, "N N"],
) -> tuple[Float[Array, "N N"], Int[Array, "N N"]]:
    """All-pairs shortest paths (Floyd-Warshall) with a predecessor matrix.

    ``cost_matrix`` is a weighted adjacency matrix (``inf`` = no edge, ``0`` diagonal; see
    :func:`dense_cost_matrix`). Returns:

    - ``distance[i, j]`` — shortest-path distance, ``inf`` if ``j`` is unreachable from
      ``i``.
    - ``predecessor[i, j]`` — node before ``j`` on a shortest ``i -> j`` path (``-1`` if
      none or ``i == j``); follow it back from ``j`` to recover the path.

    The ``k``-loop is a ``lax.fori_loop`` (so it traces under ``jit``/``vmap``); each step
    relaxes every pair at once via the ``i -> k -> j`` outer sum. Ties keep the
    earliest-found path (strict ``<``), so every pair gets one deterministic path.
    """
    n = cost_matrix.shape[-1]
    nodes = jnp.arange(n)

    # Predecessor seeded from direct edges only: prior node of i -> j is i itself.
    has_direct_edge = jnp.isfinite(cost_matrix) & (nodes[:, None] != nodes[None, :])
    predecessor_init = jnp.where(has_direct_edge, nodes[:, None].astype(jnp.int32), -1)

    def relax_through(k: int, carry):
        distance, predecessor = carry
        through_k = (
            distance[:, k][:, None] + distance[k, :][None, :]
        )  # cost of i -> k -> j
        is_shorter = through_k < distance
        distance = jnp.where(is_shorter, through_k, distance)
        # On improvement, j inherits its predecessor from the k -> j sub-path.
        predecessor = jnp.where(is_shorter, predecessor[k, :][None, :], predecessor)
        return distance, predecessor

    distance, predecessor = lax.fori_loop(
        0, n, relax_through, (cost_matrix, predecessor_init)
    )
    predecessor = predecessor.at[nodes, nodes].set(-1)
    return distance, predecessor


def aon_load(
    edge_index: Int[Array, "2 E"],
    costs: Float[Array, "E"],
    n_nodes: int,
    od_matrix: Float[Array, "Z Z"],
    zone_nodes: Int[Array, "Z"],
) -> tuple[Float[Array, "E"], Float[Array, ""]]:
    """All-or-nothing assignment: route each demand pair onto one shortest path.

    A road-network primitive: with edge weights ``costs`` fixed, every origin-destination
    pair routes *all* of its demand down a single shortest path (no splitting).
    ``od_matrix[i, j]`` is the demand from zone ``i`` to zone ``j`` and ``zone_nodes[i]``
    the network node of zone ``i``. Returns:

    - ``link_flows`` ``(E,)`` — demand accumulated onto each edge.
    - ``total_cost`` — ``sum_ij od_matrix[i, j] * distance(i, j)`` over reachable pairs.

    After :func:`floyd_warshall`, all paths are reconstructed in parallel: start the
    ``(Z, Z)`` pair grid at the destination nodes and follow ``predecessor`` back toward
    the origins for a fixed ``n_nodes - 1`` steps (an upper bound on path length),
    scattering each pair's demand onto the traversed edge with ``segment_sum`` at every
    step. Fixed step count and no data-dependent branching, so it ``jit``/``vmap``s.
    """
    n_links = costs.shape[0]
    cost_matrix, link_index = dense_cost_matrix(edge_index, costs, n_nodes)
    distance, predecessor = floyd_warshall(cost_matrix)

    # (Z, Z) grids of origin / current node per pair; walk backward from the destination.
    origin_node = jnp.broadcast_to(zone_nodes[:, None], od_matrix.shape)
    current_node = jnp.broadcast_to(zone_nodes[None, :], od_matrix.shape)

    def step_back(_, carry):
        current_node, link_flows = carry
        prev_node = predecessor[origin_node, current_node]  # predecessor toward origin
        on_route = (current_node != origin_node) & (prev_node >= 0)  # path still active
        # Edge prev_node -> current_node (index guarded to 0 when off-route; masked below).
        edge = link_index[jnp.where(on_route, prev_node, 0), current_node]
        demand = jnp.where(on_route, od_matrix, 0.0)
        link_flows = link_flows + jax.ops.segment_sum(
            demand.reshape(-1), edge.reshape(-1), num_segments=n_links
        )
        return jnp.where(on_route, prev_node, current_node), link_flows

    link_flows = jnp.zeros((n_links,), dtype=od_matrix.dtype)
    _, link_flows = lax.fori_loop(
        0, max(n_nodes - 1, 0), step_back, (current_node, link_flows)
    )

    # sum demand * shortest distance over reachable pairs.
    route_cost = distance[zone_nodes[:, None], zone_nodes[None, :]]
    reachable_cost = jnp.where(jnp.isfinite(route_cost), route_cost, 0.0)
    total_cost = jnp.sum(od_matrix * reachable_cost)
    return link_flows, total_cost
