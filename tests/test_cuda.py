import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial
import pytest
import graphgp as gp

rng = jr.key(99)

# These tests mostly ensure the CUDA extension matches the pure JAX implementation.
# They require GraphGP to be installed as well.


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["f32", "f64"])
def setup_graph(request):
    dtype = request.param
    with jax.enable_x64(dtype == jnp.float64):
        n_points = 1000
        n_dim = 3
        n0 = 100
        k = 10

        points = jr.normal(rng, (n_points, n_dim)).astype(dtype)
        graph = gp.build_graph(points, n0=n0, k=k, cuda=True)
        covariance = gp.extras.matern_kernel(
            p=0, variance=1.0, cutoff=1.0, r_min=1e-4, r_max=10, n_bins=1000, jitter=1e-5
        )
        covariance = (covariance[0].astype(dtype), covariance[1].astype(dtype))

        yield graph, covariance, points


def _rtol(dtype):
    return 1e-6 if dtype == jnp.float32 else 1e-12

def _atol(dtype):
    return 1e-3 if dtype == jnp.float32 else 1e-12


def test_forward(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)

    jax_values = gp.generate(graph, covariance, xi, cuda=False)
    cuda_values = gp.generate(graph, covariance, xi, cuda=True)

    assert jnp.allclose(jax_values, cuda_values, rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."


def test_xi_jvp(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)

    jax_jvp = jax.jvp(
        Partial(gp.generate, graph, covariance, cuda=False),
        (xi,),
        (jnp.ones_like(xi),),
    )
    cuda_jvp = jax.jvp(
        Partial(gp.generate, graph, covariance, cuda=True),
        (xi,),
        (jnp.ones_like(xi),),
    )

    assert jnp.allclose(jax_jvp[0], cuda_jvp[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."
    assert jnp.allclose(jax_jvp[1], cuda_jvp[1], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA xi JVP do not match."


def test_xi_vjp(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)
    v = jr.normal(rng, (points.shape[0],)).astype(dtype)

    jax_vjp = jax.vjp(Partial(gp.generate, graph, covariance, cuda=False), xi)
    cuda_vjp = jax.vjp(Partial(gp.generate, graph, covariance, cuda=True), xi)

    jax_vjp_result = jax_vjp[1](v)[0]
    cuda_vjp_result = cuda_vjp[1](v)[0]

    assert jnp.allclose(jax_vjp[0], cuda_vjp[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."
    assert jnp.allclose(jax_vjp_result, cuda_vjp_result, rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA xi VJP do not match."


def test_full_jvp(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)

    def forward(xi, cov_vals, *, cuda):
        return gp.generate(graph, (covariance[0], cov_vals), xi, cuda=cuda)

    jax_jvp = jax.jvp(
        Partial(forward, cuda=False),
        (xi, covariance[1]),
        (jnp.ones_like(xi), jnp.ones_like(covariance[1])),
    )
    cuda_jvp = jax.jvp(
        Partial(forward, cuda=True),
        (xi, covariance[1]),
        (jnp.ones_like(xi), jnp.ones_like(covariance[1])),
    )

    assert jnp.allclose(jax_jvp[0], cuda_jvp[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."
    assert jnp.allclose(jax_jvp[1], cuda_jvp[1], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA full JVP do not match."


def test_full_vjp(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)
    v = jr.normal(rng, (points.shape[0],)).astype(dtype)

    def forward(xi, cov_vals, *, cuda):
        return gp.generate(graph, (covariance[0], cov_vals), xi, cuda=cuda)

    jax_vjp = jax.vjp(Partial(forward, cuda=False), xi, covariance[1])
    cuda_vjp = jax.vjp(Partial(forward, cuda=True), xi, covariance[1])

    jax_vjp_result = jax_vjp[1](v)
    cuda_vjp_result = cuda_vjp[1](v)

    assert jnp.allclose(jax_vjp[0], cuda_vjp[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."
    assert jnp.allclose(jax_vjp_result[0], cuda_vjp_result[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA full VJP do not match for xi."
    assert jnp.allclose(jax_vjp_result[1], cuda_vjp_result[1], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA full VJP do not match for cov."


def test_cov_jvp(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)

    def forward(cov_vals, *, cuda):
        return gp.generate(graph, (covariance[0], cov_vals), xi, cuda=cuda)

    jax_jvp = jax.jvp(
        Partial(forward, cuda=False),
        (covariance[1],),
        (jnp.ones_like(covariance[1]),),
    )
    cuda_jvp = jax.jvp(
        Partial(forward, cuda=True),
        (covariance[1],),
        (jnp.ones_like(covariance[1]),),
    )

    assert jnp.allclose(jax_jvp[0], cuda_jvp[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."
    assert jnp.allclose(jax_jvp[1], cuda_jvp[1], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA cov JVP do not match."


def test_cov_vjp(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)
    v = jr.normal(rng, (points.shape[0],)).astype(dtype)

    def forward(cov_vals, *, cuda):
        return gp.generate(graph, (covariance[0], cov_vals), xi, cuda=cuda)

    jax_vjp = jax.vjp(Partial(forward, cuda=False), covariance[1])
    cuda_vjp = jax.vjp(Partial(forward, cuda=True), covariance[1])

    jax_vjp_result = jax_vjp[1](v)[0]
    cuda_vjp_result = cuda_vjp[1](v)[0]

    assert jnp.allclose(jax_vjp[0], cuda_vjp[0], rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA forward do not match."
    assert jnp.allclose(jax_vjp_result, cuda_vjp_result, rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA cov VJP do not match."


def test_vmap(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    n_samples = 5
    xi = jr.normal(rng, (n_samples, points.shape[0])).astype(dtype)

    jax_values = jax.vmap(Partial(gp.generate, graph, covariance, cuda=False))(xi)
    cuda_values = jax.vmap(Partial(gp.generate, graph, covariance, cuda=True))(xi)

    assert jnp.allclose(jax_values, cuda_values, rtol=_rtol(dtype), atol=_atol(dtype)), "JAX and CUDA vmap do not match."


def test_jit(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    n_samples = 5
    xi = jr.normal(rng, (points.shape[0],)).astype(dtype)

    vjp_func = jax.vjp(Partial(gp.generate, graph, covariance, cuda=True), xi)[1]

    result = jax.jit(jax.vmap(vjp_func))(jnp.ones((n_samples, points.shape[0]), dtype=dtype))
    assert jnp.all(jnp.isfinite(result[0])), "Jitted vmap VJP produced non-finite results."


def test_xi_adjoint(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype

    k1, k2, k3, k4 = jr.split(rng, 4)
    xi = jr.normal(k2, (points.shape[0],)).astype(dtype)
    xi_tangent = jr.normal(k3, (points.shape[0],)).astype(dtype)
    values_tangent = jr.normal(k4, (points.shape[0],)).astype(dtype)
    func = Partial(gp.generate, graph, covariance, cuda=True)

    val1 = jnp.dot(values_tangent, jax.jvp(func, (xi,), (xi_tangent,))[1])
    val2 = jnp.dot(xi_tangent, jax.vjp(func, xi)[1](values_tangent)[0])
    rtol = _rtol(dtype)
    atol = _atol(dtype)
    assert jnp.isclose(val1, val2, rtol=rtol, atol=atol), (
        f"Adjoint test failed: {val1:.5e} != {val2:.5e} within rtol={rtol} and atol={atol}"
    )


def test_full_adjoint(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype

    k1, k2, k3, k4 = jr.split(rng, 4)
    xi = jr.normal(k1, (points.shape[0],)).astype(dtype)
    cov_vals = covariance[1]
    xi_tangent = jr.normal(k2, (points.shape[0],)).astype(dtype)
    cov_tangent = jr.normal(k3, cov_vals.shape).astype(dtype)
    values_tangent = jr.normal(k4, (points.shape[0],)).astype(dtype)

    def forward(xi, cov_vals):
        return gp.generate(graph, (covariance[0], cov_vals), xi, cuda=True)

    val1 = jnp.dot(
        values_tangent,
        jax.jvp(forward, (xi, cov_vals), (xi_tangent, cov_tangent))[1],
    )
    vjp_result = jax.vjp(forward, xi, cov_vals)[1](values_tangent)
    val2 = jnp.dot(xi_tangent, vjp_result[0]) + jnp.dot(cov_tangent, vjp_result[1])
    rtol = _rtol(dtype)
    atol = _atol(dtype)
    assert jnp.isclose(val1, val2, rtol=rtol, atol=atol), (
        f"Full adjoint test failed: {val1:.5e} != {val2:.5e} within rtol={rtol} and atol={atol}"
    )


def test_inverse(setup_graph):
    graph, covariance, points = setup_graph
    dtype = points.dtype
    xi = jr.normal(rng, (graph.points.shape[0],)).astype(dtype)
    values = gp.generate(graph, covariance, xi, cuda=True)
    xi_back = gp.generate_inv(graph, covariance, values, cuda=True)
    values_back = gp.generate(graph, covariance, xi_back, cuda=True)
    assert jnp.allclose(values, values_back, rtol=_rtol(dtype), atol=_atol(dtype)), (
        "Values from xi and inverted xi do not match."
    )


def test_logdet(setup_graph):
    graph, covariance, points = setup_graph
    logdet1 = gp.generate_logdet(graph, covariance, cuda=True)
    logdet2 = gp.generate_logdet(graph, covariance, cuda=False)
    assert jnp.isclose(logdet1, logdet2, rtol=_rtol(points.dtype), atol=_atol(points.dtype)), (
        "Log-determinants from JAX and CUDA do not match."
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_query_neighbors_radius(dtype):
    """The radius-limited query equals a brute force (half of the k nearest inside the radius) and,
    with an infinite radius, the unbounded CUDA query."""
    import numpy as np
    import graphgp_cuda

    with jax.enable_x64(dtype == jnp.float64):
        n_points, n_dim, k, n0 = 4000, 3, 8, 30
        points, split_dims, _ = gp.build_tree(jr.normal(rng, (n_points, n_dim)).astype(dtype), cuda=True)
        query_indices = jnp.arange(n0, n_points, dtype=split_dims.dtype)
        p = np.asarray(points, dtype=np.float64)
        pairwise = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        i, j = np.indices(pairwise.shape)
        pairwise[i <= j] = np.inf
        pairwise = pairwise[n0:]
        true_neighbors = np.argsort(pairwise, axis=-1)[:, :k]
        true_distances = np.take_along_axis(pairwise, true_neighbors, axis=-1)
        radius = 0.5 * (true_distances[:, k // 2 - 1] + true_distances[:, k // 2])
        neighbors = graphgp_cuda.query_neighbors_radius(
            points, split_dims, query_indices, query_indices, jnp.asarray(radius**2, dtype=dtype), k=k
        )
        expected = np.where(true_distances < radius[:, None], true_neighbors, -1)
        assert np.array_equal(np.asarray(neighbors), expected), "radius-limited neighbors differ from brute force"
        unbounded = graphgp_cuda.query_neighbors(points, split_dims, query_indices, query_indices, k=k)
        inf = graphgp_cuda.query_neighbors_radius(
            points, split_dims, query_indices, query_indices, jnp.full(query_indices.shape, jnp.inf, dtype=dtype), k=k
        )
        assert np.array_equal(np.asarray(inf), np.asarray(unbounded)), "infinite radius differs from query_neighbors"


def _generate_by_steps(graph, cov_bins, cov_vals, xi_tree, pad=0):
    """Generation in the graph's order as one ``graphgp_cuda.refine_step`` per level, optionally with every
    level padded by ``pad`` invalid targets (neighbors -1, a point that coincides with itself k times)."""
    import graphgp_cuda

    dtype = graph.points.dtype
    n0, k = int(graph.offsets[0]), graph.neighbors.shape[1]
    initial = gp.generate_dense(graph.points[:n0], (cov_bins, cov_vals), xi_tree[:n0])
    values = jnp.zeros(graph.points.shape[0], dtype).at[:n0].set(initial)
    for start, end in zip(graph.offsets[:-1], graph.offsets[1:]):
        start, end = int(start), int(end)
        targets, neighbors, xi = graph.points[start:end], graph.neighbors[start - n0 : end - n0], xi_tree[start:end]
        valid = None
        if pad:
            targets = jnp.concatenate([targets, jnp.zeros((pad, targets.shape[1]), dtype)])
            neighbors = jnp.concatenate([neighbors, jnp.full((pad, k), -1, neighbors.dtype)])
            xi = jnp.concatenate([xi, jnp.ones(pad, dtype)])
            valid = jnp.arange(end - start + pad) < end - start
        new = graphgp_cuda.refine_step(graph.points, values, targets, neighbors, xi, cov_bins, cov_vals, valid=valid)
        values = values.at[start:end].set(new[: end - start])
    return values


def test_refine_step(setup_graph):
    """``graphgp_cuda.refine_step`` applied level by level in the graph's order matches the pure JAX
    generation, and its derivatives with respect to the excitations and the covariance values are the
    transposes of each other (the kernel carries its own vjp)."""
    graph, covariance, _points = setup_graph
    dtype = graph.points.dtype
    n = graph.points.shape[0]
    xi = jr.normal(rng, (n,)).astype(dtype)
    with jax.enable_x64(dtype == jnp.float64):
        expected = gp.generate(graph, covariance, xi, cuda=False, fast_jit=False)[graph.indices]
        xi_tree = xi[graph.indices]
        cov_bins, cov_vals = covariance
        run = lambda xi_tree, cov_vals: _generate_by_steps(graph, cov_bins, cov_vals, xi_tree)  # noqa: E731

        values = jax.jit(run)(xi_tree, cov_vals)
        assert jnp.allclose(values, expected, rtol=_rtol(dtype), atol=_atol(dtype))
        v_xi = jr.normal(jr.key(1), xi.shape).astype(dtype)
        v_cov = (jr.normal(jr.key(2), cov_vals.shape) * 1e-2).astype(dtype)
        w = jr.normal(jr.key(3), (n,)).astype(dtype)
        _, jv = jax.jvp(run, (xi_tree, cov_vals), (v_xi, v_cov))
        _, vjp = jax.vjp(run, xi_tree, cov_vals)
        ct_xi, ct_cov = vjp(w)
        lhs, rhs = jnp.vdot(jv, w), jnp.vdot(v_xi, ct_xi) + jnp.vdot(v_cov, ct_cov)
        assert jnp.allclose(lhs, rhs, rtol=100 * _rtol(dtype))


def test_refine_step_padding(setup_graph):
    """Targets with ``valid == 0`` return 0, are not factorized (their window is singular here) and leave
    the values and the gradients of the others unchanged."""
    import graphgp_cuda

    graph, covariance, _points = setup_graph
    dtype = graph.points.dtype
    n, k = graph.points.shape[0], graph.neighbors.shape[1]
    with jax.enable_x64(dtype == jnp.float64):
        xi_tree = jr.normal(rng, (n,)).astype(dtype)[graph.indices]
        cov_bins, cov_vals = covariance

        targets = jnp.zeros((5, graph.points.shape[1]), dtype)
        neighbors = jnp.full((5, k), -1, graph.neighbors.dtype)
        out = graphgp_cuda.refine_step(
            graph.points, jnp.ones(n, dtype), targets, neighbors, jnp.ones(5, dtype), cov_bins, cov_vals,
            valid=jnp.zeros(5, bool),
        )
        assert jnp.all(out == 0)

        def loss(xi_tree, cov_vals, pad):
            return jnp.sum(jnp.sin(_generate_by_steps(graph, cov_bins, cov_vals, xi_tree, pad=pad)))

        v0, g0 = jax.value_and_grad(loss, argnums=(0, 1))(xi_tree, cov_vals, 0)
        v1, g1 = jax.value_and_grad(loss, argnums=(0, 1))(xi_tree, cov_vals, 7)
        assert jnp.allclose(v1, v0, rtol=_rtol(dtype))
        for a, b in zip(g1, g0):
            assert jnp.all(jnp.isfinite(a))
            assert jnp.allclose(a, b, rtol=100 * _rtol(dtype), atol=_atol(dtype))


def test_refine_step_vmap(setup_graph):
    """Batches of source values, excitations and covariance values (leading axis) match a Python loop."""
    import graphgp_cuda

    graph, covariance, _points = setup_graph
    dtype = graph.points.dtype
    n0 = int(graph.offsets[0])
    start, end = int(graph.offsets[0]), int(graph.offsets[1])
    with jax.enable_x64(dtype == jnp.float64):
        cov_bins, cov_vals = covariance
        source_values = jr.normal(jr.key(4), (3, graph.points.shape[0])).astype(dtype)
        xi = jr.normal(jr.key(5), (3, end - start)).astype(dtype)
        scale = jnp.array([1.0, 2.0, 0.5], dtype)

        def step(source_values, xi, scale):
            return graphgp_cuda.refine_step(
                graph.points, source_values, graph.points[start:end], graph.neighbors[start - n0 : end - n0], xi,
                cov_bins, scale * cov_vals,
            )

        batched = jax.vmap(step)(source_values, xi, scale)
        looped = jnp.stack([step(source_values[i], xi[i], scale[i]) for i in range(3)])
        assert jnp.allclose(batched, looped, rtol=_rtol(dtype), atol=_atol(dtype))
