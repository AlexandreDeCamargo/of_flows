from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController
import jax
import jax.numpy as jnp
import functools


RTOL = 1e-8
ATOL = 1e-8


def set_tolerances(rtol=None, atol=None):
    """Set the default ODE ``rtol`` / ``atol`` used by both ``fwd_ode`` and ``rev_ode``.

    Pass either or both; an argument left as ``None`` keeps its current value."""
    global RTOL, ATOL
    if rtol is not None:
        RTOL = rtol
    if atol is not None:
        ATOL = atol
    return RTOL, ATOL


def get_tolerances():
    """Return the current default ``(rtol, atol)``."""
    return RTOL, ATOL


@functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
def forward(model, x, t):
    return model(x, t)


def fwd_ode(flow_model, x_and_logpx, solver, rtol=None, atol=None):
    """Forward solve t0 -> t1.  `rtol`/`atol` default to the module-level tolerances."""
    t0, t1 = 0., 1.
    dt0 = t1 - t0

    vector_field = lambda t, x, args: forward(flow_model, x, t * jnp.ones((x.shape[0], 1)))
    term = ODETerm(vector_field)
    saveat = SaveAt(ts=jnp.array([0., 1.]))
    stepsize_controller = PIDController(
        rtol=RTOL if rtol is None else rtol,
        atol=ATOL if atol is None else atol,
    )

    sol = diffeqsolve(term, solver, t0, t1, dt0, x_and_logpx,
                      stepsize_controller=stepsize_controller,
                      saveat=saveat)
    data_dim = 3
    z_t = sol.ys[:, :, :data_dim]
    logp_diff_t = sol.ys[:, :, data_dim:data_dim + 1]
    score_t = sol.ys[:, :, data_dim + 1:]
    z_t1, logp_diff_t1, score_t1 = z_t[-1], logp_diff_t[-1], score_t[-1]
    return z_t1, logp_diff_t1, score_t1


def rev_ode(flow_model, z_and_logpz, solver, rtol=None, atol=None):
    """Reverse solve t1 -> t0.  `rtol`/`atol` default to the module-level tolerances."""
    t0, t1 = 0., 1.
    dt0 = t1 - t0

    vector_field = lambda t, x, args: forward(flow_model, x, t * jnp.ones((x.shape[0], 1)))
    term = ODETerm(vector_field)
    saveat = SaveAt(ts=jnp.array([1., 0.]))
    stepsize_controller = PIDController(
        rtol=RTOL if rtol is None else rtol,
        atol=ATOL if atol is None else atol,
    )

    sol = diffeqsolve(term, solver, t1, t0, -dt0, z_and_logpz,
                      stepsize_controller=stepsize_controller,
                      saveat=saveat)
    data_dim = 3
    z_t = sol.ys[:, :, :data_dim]
    logp_diff_t = sol.ys[:, :, data_dim:data_dim + 1]
    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    return z_t0, logp_diff_t0
