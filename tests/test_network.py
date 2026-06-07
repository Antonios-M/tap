import jax
import jax.numpy as jnp

from tap.network import TrafficNetwork
from tap.readers import from_edges

# A tiny 2-link parallel network: node 0 -> node 1 via two independent links.
PARALLEL_EDGES = [[0, 0], [1, 1]]


def _parallel_net() -> TrafficNetwork:
    return from_edges(
        edge_index=PARALLEL_EDGES, capacity=[100.0, 100.0], fft=[1.0, 2.0]
    )


def test_pytree_roundtrip_static_vs_dynamic() -> None:
    net = _parallel_net()
    leaves, treedef = jax.tree_util.tree_flatten(net)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.n_nodes == net.n_nodes  # static field preserved in treedef
    assert jnp.allclose(rebuilt.capacity, net.capacity)
    # n_nodes is static (lives in treedef, not leaves). Leaves are the array fields
    # only: edge_index, fft, capacity, bpr_alpha, bpr_beta (length/speed/coords None).
    assert len(leaves) == 5
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_jit_over_network() -> None:
    net = _parallel_net()

    @jax.jit
    def total_capacity(n: TrafficNetwork):
        return n.capacity.sum()

    assert float(total_capacity(net)) == 200.0


def test_vmap_over_batch_of_networks() -> None:
    # Stack two networks (same shapes) along a leading batch axis.
    net = _parallel_net()
    batch = jax.tree_util.tree_map(
        lambda x: (
            jnp.stack([x, x * 1.0])
            if jnp.issubdtype(x.dtype, jnp.floating)
            else jnp.stack([x, x])
        ),
        net,
    )

    @jax.vmap
    def fft_sum(n: TrafficNetwork):
        return n.fft.sum()

    out = fft_sum(batch)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([3.0, 3.0]))


def test_grad_wrt_capacity_flows_through_leaf() -> None:
    net = _parallel_net()

    def cost_proxy(n: TrafficNetwork):
        # A smooth scalar function of the differentiable capacity leaf.
        return jnp.sum(1.0 / n.capacity)

    # Differentiating w.r.t. a whole TrafficNetwork requires allow_int=True because
    # edge_index is an integer leaf (it receives a float0 tangent, i.e. no gradient).
    grads = jax.grad(cost_proxy, allow_int=True)(net)
    assert grads.capacity.shape == (2,)
    assert bool((grads.capacity != 0).all())
    assert grads.edge_index.dtype == jax.dtypes.float0
