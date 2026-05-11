from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController
import jax
import jax.numpy as jnp
import functools


@functools.partial(jax.vmap, in_axes=(None,0,0), out_axes=0)
def forward(model,x,t):
    return model(x,t)



def fwd_ode(flow_model,x_and_logpx,solver):
    t0 = 0.
    t1 = 1.
    dt0 = t1 - t0
    

    vector_field = lambda t,x,args: forward(flow_model,x,t*jnp.ones((x.shape[0],1)))
    term = ODETerm(vector_field)
    solver = solver
    saveat = SaveAt(ts=jnp.array([0.,1.]))
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8)

    sol = diffeqsolve(term, solver, t0, t1, dt0, x_and_logpx,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat)
    data_dim = 3
    z_t, logp_diff_t, score_t = sol.ys[:, :,:data_dim],sol.ys[:, :, data_dim:data_dim+1],sol.ys[:, :, data_dim+1:]
    z_t1, logp_diff_t1, score_t1 = z_t[-1], logp_diff_t[-1], score_t[-1]

    return z_t1, logp_diff_t1, score_t1

# def rev_ode(flow_model, z_and_logpz, solver):
    
#     t0 = 0.
#     t1 = 1.
#     dt0 = t1 - t0
#     vector_field = lambda t,x,args: forward(flow_model,x,t*jnp.ones((x.shape[0],1)))
#     term = ODETerm(vector_field)
#     solver = solver
#     saveat = SaveAt(ts=jnp.array([1., 0.]))
#     stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

#     sol = diffeqsolve(term, solver, t1, t0, -dt0, z_and_logpz,
#                      stepsize_controller=stepsize_controller,
#                      saveat=saveat)
#     data_dim = 3
#     z_t, logp_diff_t, score_diff_t = sol.ys[:, :, :data_dim], sol.ys[:, :, data_dim:data_dim+1], sol.ys[:, :, data_dim+1:]
#     z_t0, logp_diff_t0, score_diff_t0 = z_t[-1], logp_diff_t[-1], score_diff_t[-1]

#     return z_t0, logp_diff_t0, score_diff_t0


def rev_ode(flow_model, z_and_logpz, solver):
    t0 = 0.
    t1 = 1.
    dt0 = t1 - t0
  
    vector_field = lambda t,x,args: forward(flow_model,x,t*jnp.ones((x.shape[0],1)))
    term = ODETerm(vector_field)
    solver = solver
    saveat = SaveAt(ts=jnp.array([1., 0.]))
    stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

    sol = diffeqsolve(term, solver, t1, t0, -dt0, z_and_logpz,
                     stepsize_controller=stepsize_controller,
                     saveat=saveat)
    data_dim = 3
    # z_t, logp_diff_t, _ = sol.ys[:-1, :, :data_dim], sol.ys[:-1, :, data_dim:data_dim+1], sol.ys[:, :, data_dim+1:]
    # z_t0, logp_diff_t0 = sol.ys[:-1, :, :data_dim], sol.ys[:-1, :, data_dim:data_dim+1]
    # return sol.ys
    z_t, logp_diff_t, score_diff_t = sol.ys[:, :, :data_dim], sol.ys[:, :, data_dim:data_dim+1], sol.ys[:, :, data_dim+1:]
    z_t0, logp_diff_t0, score_diff_t0 = z_t[-1], logp_diff_t[-1], score_diff_t[-1]
    return z_t0, logp_diff_t0

    # z_t, logp_diff_t, _ = sol.ys[:-1, :, :data_dim], sol.ys[:-1, :, data_dim:data_dim+1], sol.ys[:, :, data_dim+1:]
    # z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    # return z_t0, logp_diff_t0


def forward_drf(model, z_and_logpz_score):
    """
    Discrete radial flow forward pass — drop-in replacement for fwd_ode.

    Propagates positions, log-probability, and score through K sequential
    radial layers using the exact change-of-variables formula:

        log ρ_{k+1}(x_{k+1}) = log ρ_k(x_k) - log|det J_k(x_k)|

        score_{k+1} = J_k^{-T} (score_k - ∇_{x_k} log|det J_k(x_k)|)

    The correction term ∇ log|det J_k| is computed via second-order AD.

    Parameters
    ----------
    model : DiscreteRadialFlow
    z_and_logpz_score : (batch, 7) array  —  [x(3), log_p(1), score(3)]

    Returns
    -------
    x     : (batch, 3)
    log_p : (batch, 1)
    score : (batch, 3)
    """
    def _single(state):
        x     = state[:3]
        log_p = state[3]
        score = state[4:]

        for layer in model.layers:
            J = jax.jacrev(layer)(x)  # (3, 3)

            # ∇_{x} log|det J(x)|  —  needed for exact score transport
            def _log_det(xi):
                return jnp.log(jnp.abs(jnp.linalg.det(jax.jacrev(layer)(xi))))
            grad_logdet = jax.grad(_log_det)(x)  # (3,)

            # score_{k+1} = J^{-T} (score_k - grad_logdet)
            score = jnp.linalg.solve(J.T, score - grad_logdet)

            log_p = log_p - jnp.log(jnp.abs(jnp.linalg.det(J)))
            x     = layer(x)

        return x, log_p.reshape(1,), score

    xs, log_ps, scores = jax.vmap(_single)(z_and_logpz_score)
    return xs, log_ps, scores

