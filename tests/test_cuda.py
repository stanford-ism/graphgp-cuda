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
