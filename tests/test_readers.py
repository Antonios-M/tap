import jax.numpy as jnp
import pytest

from tap.readers import from_edges, make_demand

# A tiny 2-link parallel network: node 0 -> node 1 via two independent links.
PARALLEL_EDGES = [[0, 0], [1, 1]]


# --------------------------------------------------------------------------- #
# Construction & casting                                                         #
# --------------------------------------------------------------------------- #


def test_from_edges_basic_shapes_and_dtypes() -> None:
    net = from_edges(edge_index=PARALLEL_EDGES, capacity=[100.0, 100.0], fft=[1.0, 2.0])
    assert net.edge_index.shape[1] == 2  # E
    assert net.n_nodes == 2  # inferred from edge_index max + 1
    assert net.edge_index.shape == (2, 2)
    assert jnp.issubdtype(net.edge_index.dtype, jnp.integer)
    for arr in (net.fft, net.capacity, net.bpr_alpha, net.bpr_beta):
        assert arr.shape == (2,)
        # Precision follows the JAX config: float32 by default, float64 under x64.
        assert jnp.issubdtype(arr.dtype, jnp.floating)


def test_scalar_bpr_params_broadcast() -> None:
    net = from_edges(edge_index=PARALLEL_EDGES, capacity=100.0, fft=1.0)
    assert jnp.allclose(net.bpr_alpha, 0.15)
    assert jnp.allclose(net.bpr_beta, 4.0)


def test_fft_derived_from_length_and_speed() -> None:
    net = from_edges(
        edge_index=PARALLEL_EDGES,
        capacity=100.0,
        length=[100.0, 200.0],
        speed=[10.0, 10.0],
    )
    assert jnp.allclose(net.fft, jnp.array([10.0, 20.0]))
    assert net.length is not None and net.speed is not None


def test_requires_fft_or_length_speed() -> None:
    with pytest.raises(ValueError, match=r"fft.*length.*speed"):
        from_edges(edge_index=PARALLEL_EDGES, capacity=100.0)


def test_n_nodes_from_coords() -> None:
    net = from_edges(
        edge_index=PARALLEL_EDGES,
        capacity=100.0,
        fft=1.0,
        coords=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    )
    assert net.n_nodes == 3
    coords = net.coords
    assert coords is not None
    assert coords.shape == (3, 2)


def test_edge_index_ordering() -> None:
    net = from_edges(edge_index=[[0, 1, 2], [1, 2, 0]], capacity=100.0, fft=1.0)
    assert list(net.edge_index[0]) == [0, 1, 2]  # sources
    assert list(net.edge_index[1]) == [1, 2, 0]  # destinations


# --------------------------------------------------------------------------- #
# Network validation                                                            #
# --------------------------------------------------------------------------- #


def test_bad_edge_index_shape_rejected() -> None:
    with pytest.raises(ValueError, match="edge_index"):
        from_edges(edge_index=[[0, 1, 2]], capacity=1.0, fft=1.0)


def test_wrong_length_array_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        from_edges(edge_index=PARALLEL_EDGES, capacity=[1.0, 2.0, 3.0], fft=1.0)


def test_node_id_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        from_edges(edge_index=[[0], [5]], capacity=1.0, fft=1.0, n_nodes=2)


def test_negative_capacity_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        from_edges(edge_index=[[0], [1]], capacity=[-1.0], fft=1.0)


def test_zero_capacity_allowed() -> None:
    # A fully blocked link is valid (capacity == 0); BPR guards against div-by-zero.
    net = from_edges(edge_index=[[0], [1]], capacity=[0.0], fft=1.0)
    assert float(net.capacity[0]) == 0.0


# --------------------------------------------------------------------------- #
# Demand                                                                         #
# --------------------------------------------------------------------------- #


def test_make_demand_and_totals() -> None:
    demand = make_demand(od_matrix=[[0.0, 10.0], [5.0, 0.0]], zone_nodes=[0, 1])
    assert demand.zone_nodes.shape[0] == 2  # Z
    assert float(demand.od_matrix.sum()) == 15.0


def test_demand_non_square_rejected() -> None:
    with pytest.raises(ValueError, match="od_matrix"):
        make_demand(od_matrix=[[1.0, 2.0, 3.0]], zone_nodes=[0])
