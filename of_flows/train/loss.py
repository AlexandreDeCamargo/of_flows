import jax.numpy as jnp
from typing import NamedTuple
from functionals.kinetic import ThomasFermi, Weizsacker
from functionals.exchange_correlation import LDA, B88Exchange, CorrelationVWN, CorrelationPW92
from functionals.hartree import CoulombPotential, CoulombPotentialAllPairs
from functionals.external import NuclearPotential
from functionals.core_correction import KatoCondition, HutcheonCuspCondition
from functionals.functional import CompositeFunctional
from ode_solver.eqx_ode import fwd_ode, forward_drf, forward_rdm
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
    'coulomb_allpairs': CoulombPotentialAllPairs,
    
    # External
    'np': NuclearPotential,
    
    # Core correction
    'kato': KatoCondition,
    'hutcheon': HutcheonCuspCondition,
}

def _build_kinetic(kinetic_name: str, lam: float):
    """Instantiate the kinetic functional, injecting λ into any Weizsäcker term."""
    if kinetic_name == 'w':
        return Weizsacker(lambda_0=lam)
    if kinetic_name == 'tf_w':
        return CompositeFunctional(ThomasFermi(), Weizsacker(lambda_0=lam))
    return FUNCTIONAL_CLASSES[kinetic_name]()   # 'tf' — lam irrelevant


def create_loss_function(
    kinetic_name: str = 'tf',
    lam: float = 1.0,
    exchange_name: str = 'lda',
    correlation_name: str = 'none',
    hartree_name: str = 'coulomb',
    external_name: str = 'np',
    core_correction_name: str = 'none'
):
    """
    Factory function to create a loss function with specific functionals.

    Parameters
    ----------
    kinetic_name : str
        Name of kinetic functional ('tf', 'w', 'tf_w')
    lam : float
        Weizsäcker prefactor λ in TF-λW  (ignored when kinetic_name='tf')
    exchange_name : str
        Name of exchange functional ('lda', 'b88_x')
    correlation_name : str
        Name of correlation functional ('vwn_c', 'pw92_c', 'none')
    hartree_name : str
        Name of Hartree functional ('coulomb')
    external_name : str
        Name of external potential functional ('np')
    core_correction_name : str
        Name of core correction functional ('kato', 'hutcheon', 'none')

    Returns
    -------
    grad_loss : callable
    """
    t_functional = _build_kinetic(kinetic_name, lam)
    x_functional = FUNCTIONAL_CLASSES[exchange_name]()
    h_functional = FUNCTIONAL_CLASSES[hartree_name]()
    n_functional = FUNCTIONAL_CLASSES[external_name]()
    
    if core_correction_name != 'none':
        cc_functional = FUNCTIONAL_CLASSES[core_correction_name]()
    else:
        cc_functional = None
    
    if correlation_name != 'none':
        c_functional = FUNCTIONAL_CLASSES[correlation_name]()
    else:
        c_functional = None
    
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
        if c_functional is not None: 
            c_e = c_functional(den, score, Ne)
        else: 
            c_e = 0.0 
        
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


def create_loss_function_rdm(
    kinetic_name: str = 'tf_w',
    lam: float = 1.0,
    exchange_name: str = 'lda',
    correlation_name: str = 'none',
    hartree_name: str = 'coulomb',
    external_name: str = 'np',
    core_correction_name: str = 'none'
):
    """Loss function factory for RezendeRadialFlow (uses forward_rdm)."""
    t_functional = _build_kinetic(kinetic_name, lam)
    x_functional = FUNCTIONAL_CLASSES[exchange_name]()
    h_functional = FUNCTIONAL_CLASSES[hartree_name]()
    n_functional = FUNCTIONAL_CLASSES[external_name]()
    c_functional  = FUNCTIONAL_CLASSES[correlation_name]()  if correlation_name  != 'none' else None
    cc_functional = FUNCTIONAL_CLASSES[core_correction_name]() if core_correction_name != 'none' else None

    def grad_loss_rdm(model, z_and_logpz, Ne, mol):
        x, log_px, _score = forward_rdm(model, z_and_logpz)

        bs = int(x.shape[0] / 2)
        den_all, score_all = jnp.exp(log_px), _score
        score, scorep = score_all[:bs], score_all[bs:]
        den,   denp   = den_all[:bs],   den_all[bs:]
        x,     xp     = x[:bs],          x[bs:]

        t_e  = t_functional(den, score, Ne)
        n_e  = n_functional(x, Ne, mol)
        h_e  = h_functional(x, xp, Ne)
        x_e  = x_functional(den, score, Ne)
        c_e  = c_functional(den, score, Ne)  if c_functional  is not None else 0.0
        cc_e = cc_functional(x, den, score, Ne, mol) if cc_functional is not None else 0.0

        xc_e   = x_e + c_e
        e      = t_e + n_e + h_e + xc_e + cc_e
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

    return grad_loss_rdm


def create_loss_function_drf(
    kinetic_name: str = 'tf_w',
    lam: float = 1.0,
    exchange_name: str = 'lda',
    correlation_name: str = 'none',
    hartree_name: str = 'coulomb',
    external_name: str = 'np',
    core_correction_name: str = 'none'
):
    """
    Loss function factory for DiscreteRadialFlow models.

    Same functionals as create_loss_function but uses forward_drf instead
    of the ODE-based fwd_ode, so no solver argument is needed at call time.
    """
    t_functional = _build_kinetic(kinetic_name, lam)
    x_functional = FUNCTIONAL_CLASSES[exchange_name]()
    h_functional = FUNCTIONAL_CLASSES[hartree_name]()
    n_functional = FUNCTIONAL_CLASSES[external_name]()

    c_functional  = FUNCTIONAL_CLASSES[correlation_name]()  if correlation_name  != 'none' else None
    cc_functional = FUNCTIONAL_CLASSES[core_correction_name]() if core_correction_name != 'none' else None

    def grad_loss_drf(model, z_and_logpz, Ne, mol):
        """
        Compute energy loss for a DiscreteRadialFlow model.

        Parameters
        ----------
        model : DiscreteRadialFlow
        z_and_logpz : (batch, 7)  —  [z(3), log_p0(1), score_p0(3)]
        Ne : int – number of electrons
        mol : dict – {'coords': ..., 'z': ...}
        """
        x, log_px, _score = forward_drf(model, z_and_logpz)

        bs = int(x.shape[0] / 2)

        den_all   = jnp.exp(log_px)
        score_all = _score

        score, scorep = score_all[:bs], score_all[bs:]
        den,   denp   = den_all[:bs],   den_all[bs:]
        x,     xp     = x[:bs],          x[bs:]

        t_e = t_functional(den, score, Ne)
        n_e = n_functional(x, Ne, mol)
        h_e = h_functional(x, xp, Ne)
        x_e = x_functional(den, score, Ne)

        c_e  = c_functional(den, score, Ne)  if c_functional  is not None else 0.0
        cc_e = cc_functional(x, den, score, Ne, mol) if cc_functional is not None else 0.0

        xc_e   = x_e + c_e
        e      = t_e + n_e + h_e + xc_e + cc_e
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

    return grad_loss_drf