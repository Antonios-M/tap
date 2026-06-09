import jax
import jax.numpy as jnp

from tap.bpr import _HUGE_COST, bpr_cost, bpr_marginal

ALPHA = 0.15
BETA = 4.0


def test_bpr_cost_free_flow_and_capacity() -> None:
    fft = jnp.array([1.0, 2.0])
    cap = jnp.array([100.0, 100.0])
    a = jnp.full(2, ALPHA)
    b = jnp.full(2, BETA)
    # At zero flow the cost is exactly the free-flow time.
    assert jnp.allclose(bpr_cost(jnp.zeros(2), fft, cap, a, b), fft)
    # At flow == capacity the multiplier is (1 + alpha).
    at_cap = bpr_cost(cap, fft, cap, a, b)
    assert jnp.allclose(at_cap, fft * (1.0 + ALPHA))


def test_bpr_cost_matches_formula() -> None:
    flow = jnp.array([50.0, 120.0])
    fft = jnp.array([1.0, 3.0])
    cap = jnp.array([100.0, 80.0])
    a, b = jnp.full(2, ALPHA), jnp.full(2, BETA)
    expected = fft * (1.0 + ALPHA * (flow / cap) ** BETA)
    assert jnp.allclose(bpr_cost(flow, fft, cap, a, b), expected)


def test_bpr_marginal_is_so_marginal_cost() -> None:
    flow = jnp.array([50.0, 120.0])
    fft = jnp.array([1.0, 3.0])
    cap = jnp.array([100.0, 80.0])
    a, b = jnp.full(2, ALPHA), jnp.full(2, BETA)
    expected = fft * (1.0 + ALPHA * (BETA + 1.0) * (flow / cap) ** BETA)
    assert jnp.allclose(bpr_marginal(flow, fft, cap, a, b), expected)
    # Marginal >= cost for positive flow (the congestion externality term).
    assert bool(
        (bpr_marginal(flow, fft, cap, a, b) >= bpr_cost(flow, fft, cap, a, b)).all()
    )


def test_blocked_link_returns_huge_finite_cost() -> None:
    flow = jnp.array([10.0, 10.0])
    fft = jnp.array([1.0, 1.0])
    cap = jnp.array([0.0, 100.0])  # link 0 fully blocked
    a, b = jnp.full(2, ALPHA), jnp.full(2, BETA)
    out = bpr_cost(flow, fft, cap, a, b)
    assert jnp.allclose(out[0], jnp.asarray(_HUGE_COST, out.dtype))
    assert jnp.isfinite(out).all()  # huge, but not inf (safe for min-plus sums)


def test_differentiable_and_matches_analytic_capacity_grad() -> None:
    fft, cap = 2.0, 100.0
    flow = 60.0

    def cost_of_cap(c):
        return bpr_cost(
            jnp.asarray(flow),
            jnp.asarray(fft),
            c,
            jnp.asarray(ALPHA),
            jnp.asarray(BETA),
        )

    g = jax.grad(cost_of_cap)(jnp.asarray(cap))
    # d/dc [fft*(1+alpha*(x/c)^beta)] = -fft*alpha*beta*x^beta / c^(beta+1)
    analytic = -fft * ALPHA * BETA * flow**BETA / cap ** (BETA + 1.0)
    assert jnp.allclose(g, analytic, rtol=1e-4)
    assert float(g) < 0.0  # more capacity -> lower cost


def test_jit_and_vmap() -> None:
    flow = jnp.array([10.0, 20.0])
    fft = jnp.array([1.0, 1.0])
    cap = jnp.array([100.0, 100.0])
    a, b = jnp.full(2, ALPHA), jnp.full(2, BETA)

    jitted = jax.jit(bpr_cost)
    assert jnp.allclose(jitted(flow, fft, cap, a, b), bpr_cost(flow, fft, cap, a, b))

    batch = jnp.stack([flow, flow * 2.0])
    out = jax.vmap(lambda f: bpr_cost(f, fft, cap, a, b))(batch)
    assert out.shape == (2, 2)
    assert jnp.allclose(out[0], bpr_cost(flow, fft, cap, a, b))
