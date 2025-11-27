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
    z_t, logp_diff_t, _ = sol.ys[:-1, :, :data_dim], sol.ys[:-1, :, data_dim:data_dim+1], sol.ys[:, :, data_dim+1:]
    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    return z_t0, logp_diff_t0




