import jax
import jax.numpy as jnp

from tap.shortest_path import aon_load, dense_cost_matrix, floyd_warshall

# Triangle: links 0:(0->1), 1:(1->2), 2:(0->2). Going 0->1->2 costs 2, direct 0->2 costs 3,
# so the shortest 0->2 path detours through node 1.
TRI_EDGES = jnp.array([[0, 1, 0], [1, 2, 2]])
TRI_COSTS = jnp.array([1.0, 1.0, 3.0])


def test_dense_cost_matrix_and_parallel_links() -> None:
    # Two parallel 0->1 links (costs 5 and 2); the matrix keeps the cheaper one.
    edges = jnp.array([[0, 0], [1, 1]])
    d, link_of = dense_cost_matrix(edges, jnp.array([5.0, 2.0]), n_nodes=2)
    assert float(d[0, 1]) == 2.0
    assert float(d[0, 0]) == 0.0 and not jnp.isfinite(d[1, 0])
    assert int(link_of[0, 1]) == 1  # the cheaper (index-1) link is the representative


def test_floyd_warshall_distances_and_predecessors() -> None:
    d, _ = dense_cost_matrix(TRI_EDGES, TRI_COSTS, n_nodes=3)
    dist, pred = floyd_warshall(d)
    assert float(dist[0, 2]) == 2.0  # via node 1, not the direct cost-3 link
    assert float(dist[0, 1]) == 1.0
    assert int(pred[0, 2]) == 1  # node before 2 on the 0->2 shortest path
    assert int(pred[0, 1]) == 0
    assert int(pred[0, 0]) == -1  # no self predecessor


def test_aon_loads_detour_path() -> None:
    # Demand 5 from zone 0 (node 0) to zone 2 (node 2) routes over links 0 and 1.
    od = (
        jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).at[0, 2].set(5.0)
    )
    zone_nodes = jnp.array([0, 1, 2])
    y, sptt = aon_load(TRI_EDGES, TRI_COSTS, 3, od, zone_nodes)
    assert jnp.allclose(y, jnp.array([5.0, 5.0, 0.0]))
    assert float(sptt) == 10.0  # 5 trips * distance 2


def test_aon_takes_direct_link_when_cheaper() -> None:
    # Drop the direct 0->2 cost below the detour: now all flow goes on link 2.
    costs = jnp.array([1.0, 1.0, 1.5])
    od = jnp.zeros((3, 3)).at[0, 2].set(5.0)
    zone_nodes = jnp.array([0, 1, 2])
    y, sptt = aon_load(TRI_EDGES, costs, 3, od, zone_nodes)
    assert jnp.allclose(y, jnp.array([0.0, 0.0, 5.0]))
    assert float(sptt) == 7.5


def test_aon_parallel_network_picks_cheaper_link() -> None:
    edges = jnp.array([[0, 0], [1, 1]])  # two 0->1 links
    costs = jnp.array([1.0, 2.0])
    od = jnp.array([[0.0, 6.0], [0.0, 0.0]])
    zone_nodes = jnp.array([0, 1])
    y, sptt = aon_load(edges, costs, 2, od, zone_nodes)
    assert jnp.allclose(y, jnp.array([6.0, 0.0]))  # all on the cheaper link
    assert float(sptt) == 6.0


def test_aon_conserves_flow_multiple_od() -> None:
    # Two OD pairs sharing link 0 (0->1): 0->1 demand 4 and 0->2 demand 5.
    od = jnp.zeros((3, 3)).at[0, 1].set(4.0).at[0, 2].set(5.0)
    zone_nodes = jnp.array([0, 1, 2])
    y, _ = aon_load(TRI_EDGES, TRI_COSTS, 3, od, zone_nodes)
    # Link 0 carries both flows (4 + 5), link 1 carries the 0->2 flow (5).
    assert jnp.allclose(y, jnp.array([9.0, 5.0, 0.0]))


def test_aon_jit_and_vmap_batch() -> None:
    od = jnp.zeros((3, 3)).at[0, 2].set(5.0)
    zone_nodes = jnp.array([0, 1, 2])

    def solve(costs):
        return aon_load(TRI_EDGES, costs, 3, od, zone_nodes)[0]

    jitted = jax.jit(solve)
    assert jnp.allclose(jitted(TRI_COSTS), solve(TRI_COSTS))

    # Batch of two cost vectors: detour vs. direct-cheaper -> different AON flows.
    batch = jnp.stack([TRI_COSTS, jnp.array([1.0, 1.0, 1.5])])
    out = jax.vmap(solve)(batch)
    expected = jnp.stack([solve(TRI_COSTS), solve(jnp.array([1.0, 1.0, 1.5]))])
    assert jnp.allclose(out, expected)
