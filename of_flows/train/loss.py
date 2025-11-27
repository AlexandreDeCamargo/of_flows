import jax.numpy as jnp
from typing import NamedTuple
from functionals.kinetic import ThomasFermi, Weizsacker
from functionals.exchange_correlation import LDA, B88Exchange, CorrelationVWN, CorrelationPW92
from functionals.hartree import CoulombPotential
from functionals.external import NuclearPotential
from functionals.core_correction import KatoCondition
from functionals.functional import CompositeFunctional
from ode_solver.eqx_ode import fwd_ode
import jax
class F_values(NamedTuple):
    """Container for energy components."""
    energy: float
    kin: float
    vnuc: float
    hart: float
    xc: float
    cc: float

FUNCTIONAL_CLASSES = {
    # Kinetic
    'tf': ThomasFermi,
    'w': Weizsacker, 
    'tf_w': lambda: CompositeFunctional(ThomasFermi(), Weizsacker()),
    
    # Exchange
    'lda': LDA,
    'b88_x': B88Exchange,
    'lda_b88_x': lambda: CompositeFunctional(LDA(), B88Exchange()),

    # Correlation
    'vwn_c': CorrelationVWN,
    'pw92_c': CorrelationPW92,
    
    # Hartree
    'coulomb': CoulombPotential,
    
    # External
    'np': NuclearPotential,
    
    # Core correction
    'kato': KatoCondition,
}

def create_loss_function(
    kinetic_name: str = 'tf',
    exchange_name: str = 'lda',
    correlation_name: str = 'vwn_c',
    hartree_name: str = 'coulomb',
    external_name: str = 'np',
    core_correction_name: str = 'none'
):
    """
    Factory function to create a loss function with specific functionals.
    
    Parameters
    ----------
    kinetic_name : str
        Name of kinetic functional ('tf', 'w')
    exchange_name : str
        Name of exchange functional ('lda', 'b88_x')
    correlation_name : str
        Name of correlation functional ('vwn_c', 'pw92_c')
    hartree_name : str
        Name of Hartree functional ('coulomb')
    external_name : str
        Name of external potential functional ('np')
    core_correction_name : str
        Name of core correction functional ('kato', 'none')
    
    Returns
    -------
    grad_loss : callable
    """
   
    t_functional = FUNCTIONAL_CLASSES[kinetic_name]()
    x_functional = FUNCTIONAL_CLASSES[exchange_name]()
    c_functional = FUNCTIONAL_CLASSES[correlation_name]()
    h_functional = FUNCTIONAL_CLASSES[hartree_name]()
    n_functional = FUNCTIONAL_CLASSES[external_name]()
    
    if core_correction_name != 'none':
        cc_functional = FUNCTIONAL_CLASSES[core_correction_name]()
    else:
        cc_functional = None
    
    def grad_loss(model, z_and_logpz, solver, Ne, mol):
        """
        Compute the loss function.
        """
        x, log_px, _score = fwd_ode(model, z_and_logpz, solver)
        
        bs = int(x.shape[0] / 2)
        
        den_all, x_all, score_all = jnp.exp(log_px), x, _score
        score, scorep = score_all[:bs], score_all[bs:]
        den, denp = den_all[:bs], den_all[bs:]
        x, xp = x_all[:bs], x_all[bs:]
        
        # Kinetic energy
        t_e = t_functional(den, score, Ne)
        
        # External (nuclear) potential
        n_e = n_functional(x, Ne, mol)
        
        # Hartree (electron-electron) potential
        h_e = h_functional(x, xp, Ne)
        
        # Exchange energy
        x_e = x_functional(den, score, Ne)
        
        # Correlation energy
        c_e = c_functional(den, score, Ne)
        xc_e = x_e + c_e
        
        # Core correction
        if cc_functional is not None:
            cc_e = cc_functional(x, den, score, Ne, mol)
        else:
            cc_e = 0.0
        
        # Total energy
        e = t_e + n_e + h_e + xc_e + cc_e
        energy = jnp.mean(e)
        
        f_values = F_values(
            energy=energy,
            kin=jnp.mean(t_e),
            vnuc=jnp.mean(n_e),
            hart=jnp.mean(h_e),
            xc=jnp.mean(xc_e),
            cc=jnp.mean(cc_e) if isinstance(cc_e, jnp.ndarray) else cc_e
        )
        
        return energy, f_values
    
    return grad_loss